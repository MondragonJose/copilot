"""Async ingest worker with retry/backoff and dead-letter per Blueprint §4.

Pipeline stages tracked in ``jobs.stage``::

    queued → parse → chunk → persist → done

On failure the worker increments ``attempts``, stores the error, and
reverts to ``failed``.  Once ``attempts >= max_attempts`` the job goes
``dead`` with the last error and stage preserved.
"""

from __future__ import annotations

from collections.abc import Callable

from core._utils import exponential_backoff
from core.errors import RCError
from core.interfaces import Embedder, Parser, Retriever
from core.models import Chunk, Paper
from ingest.chunking import chunk_document
from ingest.persist import persist_document
from retrieval.db import Pool

MAX_BACKOFF = 3600.0
BASE_DELAY = 10.0


async def process_job(
    pool: Pool,
    retriever: Retriever,
    embedder: Embedder,
    router: Parser,
    job_id: str,
    *,
    chunker: Callable[[Paper, str], list[Chunk]] = chunk_document,
) -> dict[str, object]:
    """Run a single ingest job through the pipeline.

    Returns a result dict with at least ``"result"`` (one of ``done``,
    ``failed``, ``not_found``).
    """
    job_row = await _fetch_job_or_none(pool, job_id)
    if job_row is None:
        return {"result": "not_found"}

    pdf_path = _extract_pdf_path(job_row)
    if pdf_path is None:
        await _mark_dead(pool, job_id, "parse", "No pdf_path in job payload")
        return {"result": "failed", "error": "No pdf_path"}

    new_attempts, max_attempts = _bump_attempts(job_row)
    await _mark_running(pool, job_id, new_attempts)

    stage: str | None = None

    try:
        await _set_stage(pool, job_id, "parse")
        stage = "parse"
        paper, full_text = await router.parse(pdf_path)

        await _set_stage(pool, job_id, "chunk")
        stage = "chunk"
        chunks = chunker(paper, full_text)

        await _set_stage(pool, job_id, "persist")
        stage = "persist"
        count = await persist_document(pool, retriever, embedder, paper, chunks)

        await _mark_done(pool, job_id)
        return {"result": "done", "chunks": count}

    except RCError as exc:
        await _handle_failure(
            pool, job_id, stage or "parse", str(exc), new_attempts, max_attempts,
        )
        return {"result": "failed", "error": str(exc)}

    except Exception as exc:
        await _handle_failure(
            pool, job_id, stage or "parse", f"Unexpected error: {exc}", new_attempts,
            max_attempts,
        )
        return {"result": "failed", "error": str(exc)}


async def _fetch_job_or_none(
    pool: Pool, job_id: str,
) -> dict[str, object] | None:
    row = await pool.fetchrow(
        "SELECT * FROM jobs WHERE id = $1::uuid", job_id,
    )
    return dict(row) if row is not None else None


def _extract_pdf_path(job: dict[str, object]) -> str | None:
    payload = job.get("payload") or {}
    pdf_path: str | None = (
        payload.get("pdf_path") if isinstance(payload, dict) else None
    )
    return pdf_path


def _bump_attempts(
    job: dict[str, object],
) -> tuple[int, int]:
    new_attempts = (job.get("attempts") or 0) + 1
    max_attempts = job.get("max_attempts") or 3
    return new_attempts, max_attempts


async def _mark_running(pool: Pool, job_id: str, attempts: int) -> None:
    await pool.execute(
        "UPDATE jobs SET status = 'running', attempts = $1, "
        "updated_at = now() WHERE id = $2::uuid",
        attempts, job_id,
    )


# ---------------------------------------------------------------------------
# Job lifecycle helpers
# ---------------------------------------------------------------------------


async def _set_stage(pool: Pool, job_id: str, stage: str) -> None:
    await pool.execute(
        "UPDATE jobs SET stage = $1, updated_at = now() "
        "WHERE id = $2::uuid",
        stage, job_id,
    )


async def _mark_done(pool: Pool, job_id: str) -> None:
    await pool.execute(
        "UPDATE jobs SET status = 'done', stage = NULL, error = NULL, "
        "updated_at = now() WHERE id = $1::uuid",
        job_id,
    )


async def _mark_dead(pool: Pool, job_id: str, stage: str, error: str) -> None:
    await pool.execute(
        "UPDATE jobs SET status = 'dead', stage = $1, error = $2, "
        "updated_at = now() WHERE id = $3::uuid",
        stage, error, job_id,
    )


async def _handle_failure(
    pool: Pool,
    job_id: str,
    stage: str,
    error: str,
    attempts: int,
    max_attempts: int,
) -> None:
    if attempts >= max_attempts:
        await pool.execute(
            "UPDATE jobs SET status = 'dead', stage = $1, error = $2, "
            "updated_at = now() WHERE id = $3::uuid",
            stage, error, job_id,
        )
    else:
        await pool.execute(
            "UPDATE jobs SET status = 'failed', stage = $1, error = $2, "
            "updated_at = now() WHERE id = $3::uuid",
            stage, error, job_id,
        )


def backoff_delay(attempts: int, base_seconds: float = BASE_DELAY) -> float:
    """Exponential backoff: ``base * 2^{attempts-1}``, capped at 1 hour."""
    return exponential_backoff(
        attempts, base_delay=base_seconds, max_delay=MAX_BACKOFF,
    )
