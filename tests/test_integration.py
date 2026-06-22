"""Integration tests against the real compose stack (pgvector, Redis, GROBID, MinIO).

Every test exercises the real path end-to-end for its scenario.
Mocks are used ONLY where real infrastructure cannot produce the
failure mode deterministically (e.g. partial embedding failure,
LLM producing no citations).

Required external services (see ``docker-compose.test.yml``):
  - PostgreSQL 16 + pgvector + pg_trgm + uuid-ossp
  - Redis 7
  - GROBID 0.8.1
  - MinIO

Usage:
  docker compose -f docker-compose.test.yml up -d --wait
  DATABASE_URL=postgresql://rc:rc@localhost:5432/research_copilot pytest tests/test_integration.py -v
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import asyncpg
import fitz
import pytest
import pytest_asyncio

from core.errors import EmbeddingError, ParseError
from core.models import Paper, UpsertChunk
from ingest.chunking import chunk_document
from ingest.persist import persist_document
from ingest.router import ParserRouter
from ingest.tasks import process_job
from qa.engine import QAEngine
from retrieval.db import Pool
from retrieval.embedder import BgeM3Embedder
from retrieval.pgvector_store import PgVectorStore

pytestmark = pytest.mark.asyncio

DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://rc:rc@localhost:5432/research_copilot",
)
MIGRATIONS_DIR = pathlib.Path(__file__).resolve().parent.parent / "migrations"
VECTOR_DIM = 1024

# ---------------------------------------------------------------------------
# Known test identifiers — avoids UUID entropy in assertions
# ---------------------------------------------------------------------------

PAPER_ID = "10000000-0000-0000-0000-000000000001"
PAPER_ID_2 = "10000000-0000-0000-0000-000000000002"
JOB_ID = "20000000-0000-0000-0000-000000000001"
JOB_ID_2 = "20000000-0000-0000-0000-000000000002"

# ---------------------------------------------------------------------------
# Apply migrations once per module
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module")
async def apply_migrations() -> None:
    """Run all .sql migration files once before any test in this module."""
    try:
        conn = await asyncpg.connect(DSN, timeout=5)
    except Exception:
        pytest.skip(
            f"Database not reachable at {DSN}. "
            "Start services: docker compose -f docker-compose.test.yml up -d --wait",
        )
        return
    try:
        for f in sorted(MIGRATIONS_DIR.glob("*.sql")):
            sql = f.read_text()
            if sql.strip():
                await conn.execute(sql)
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Pool fixture — used by every test
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def pool(apply_migrations: None) -> AsyncGenerator[Pool, None]:
    p = Pool(DSN)
    try:
        await p.open()
    except Exception:
        pytest.skip("Database not available after migration")
        return
    yield p
    await p.close()


@pytest_asyncio.fixture
async def store(pool: Pool) -> AsyncGenerator[PgVectorStore, None]:
    yield PgVectorStore(pool)


# ---------------------------------------------------------------------------
# PDF fixtures — ephemeral documents
# ---------------------------------------------------------------------------


def _make_native_pdf(path: pathlib.Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "The transformer architecture achieves state-of-the-art results "
        "on natural language processing tasks including translation "
        "and summarization. Attention mechanisms allow the model to "
        "focus on relevant parts of the input sequence.",
        fontname="helv",
        fontsize=12,
    )
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "Training data consisted of the WMT 2014 English-German dataset "
        "containing approximately 4.5 million sentence pairs. Models were "
        "trained on 8 NVIDIA P100 GPUs for 12 hours.",
        fontname="helv",
        fontsize=12,
    )
    doc.save(str(path))
    doc.close()


def _make_scanned_pdf(path: pathlib.Path) -> None:
    doc = fitz.open()
    doc.new_page()
    doc.save(str(path))
    doc.close()


@pytest.fixture
def native_pdf(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "test_article.pdf"
    _make_native_pdf(path)
    return path


@pytest.fixture
def scanned_pdf(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "scanned.pdf"
    _make_scanned_pdf(path)
    return path


# ---------------------------------------------------------------------------
# Helpers — insert test data into the database
# ---------------------------------------------------------------------------


async def _insert_paper(
    pool: Pool,
    paper_id: str = PAPER_ID,
    title: str = "Test Paper",
) -> None:
    await pool.execute(
        "INSERT INTO papers (id, title, source, meta) "
        "VALUES ($1::uuid, $2, 'test', '{}'::jsonb) "
        "ON CONFLICT (id) DO NOTHING",
        paper_id,
        title,
    )


async def _insert_job(
    pool: Pool,
    job_id: str,
    pdf_path: str,
    paper_id: str | None = None,
    max_attempts: int = 3,
) -> None:
    await pool.execute(
        "INSERT INTO jobs (id, kind, payload, paper_id, max_attempts) "
        "VALUES ($1::uuid, 'ingest_pdf', $2::jsonb, $3::uuid, $4) "
        "ON CONFLICT (id) DO NOTHING",
        job_id,
        {"pdf_path": pdf_path},
        paper_id,
        max_attempts,
    )


async def _chunk_count(pool: Pool, paper_id: str) -> int:
    row = await pool.fetchrow(
        "SELECT COUNT(*) AS cnt FROM chunks WHERE paper_id = $1::uuid",
        paper_id,
    )
    return row["cnt"] if row else 0


async def _cleanup(pool: Pool, paper_ids: list[str], job_ids: list[str]) -> None:
    for pid in paper_ids:
        await pool.execute(
            "DELETE FROM chunks WHERE paper_id = $1::uuid", pid,
        )
        await pool.execute(
            "DELETE FROM embeddings WHERE chunk_id IN "
            "(SELECT id FROM chunks WHERE paper_id = $1::uuid)", pid,
        )
        await pool.execute(
            "DELETE FROM papers WHERE id = $1::uuid", pid,
        )
    for jid in job_ids:
        await pool.execute(
            "DELETE FROM jobs WHERE id = $1::uuid", jid,
        )


# ---------------------------------------------------------------------------
# Mock infrastructure for injection into process_job / QAEngine
# ---------------------------------------------------------------------------


class _OkEmbedder:
    """Returns a zero vector for every input — no real ML model needed."""

    def __init__(self, dim: int = VECTOR_DIM) -> None:
        self._dim = dim
        self.call_count = 0

    async def embed(self, texts: list[str]) -> list[Sequence[float]]:
        self.call_count += 1
        return [[0.0] * self._dim for _ in texts]


class _FailingEmbedder:
    """Succeeds for N calls, then raises EmbeddingError."""

    def __init__(self, succeed_calls: int = 1, dim: int = VECTOR_DIM) -> None:
        self._succeed_calls = succeed_calls
        self._dim = dim
        self.call_count = 0

    async def embed(self, texts: list[str]) -> list[Sequence[float]]:
        self.call_count += 1
        if self.call_count > self._succeed_calls:
            raise EmbeddingError("Simulated embedding failure mid-batch")
        return [[0.0] * self._dim for _ in texts]


class _OkRouter:
    """Returns a deterministic Paper + text — no real parser needed."""

    def __init__(self, paper_id: str = PAPER_ID) -> None:
        self._paper_id = paper_id

    async def parse(self, pdf_path: str) -> tuple[Paper, str]:
        if not os.path.isfile(pdf_path):
            raise ParseError(f"PDF not found: {pdf_path}")
        text = (
            "The transformer architecture achieves state-of-the-art results "
            "on natural language processing tasks. Attention mechanisms are key."
        )
        paper = Paper(
            id=self._paper_id,
            doi=None,
            title="Test Paper",
            authors=[],
            year=None,
            venue=None,
            abstract=None,
            source="test",
            open_access=None,
            pdf_path=pdf_path,
            grobid_tei=None,
            meta={
                "sections": [
                    {
                        "heading": "Introduction",
                        "text": text,
                        "char_start": 0,
                        "char_end": len(text),
                    },
                ],
            },
        )
        return paper, text


class _FailingRouter:
    """Raises ParseError on every call."""

    async def parse(self, pdf_path: str) -> tuple[Paper, str]:
        raise ParseError(f"Simulated parser failure for {pdf_path}")


class _NoOpRetriever:
    async def upsert(self, items: Sequence[UpsertChunk]) -> int:
        return len(list(items))

    async def delete_by_paper(self, paper_id: str) -> int:
        return 0

    async def count(self) -> int:
        return 0

    async def health(self) -> bool:
        return True

    async def search_dense(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    async def search_lexical(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


# ============================================================================
# 1. GROBID timeout → fallback to PyMuPDF
# ============================================================================


class TestGrobidFallback:
    """When GROBID is unreachable the router must fall back to PyMuPDF
    and still produce a parseable Paper with page-level spans."""

    async def test_timeout_falls_back_to_pymupdf(
        self, pool: Pool, native_pdf: pathlib.Path,
    ) -> None:
        router = ParserRouter(
            grobid_url="http://localhost:1",
            grobid_timeout=0.1,
        )
        paper, full_text = await router.parse(str(native_pdf))

        assert paper.source == "pymupdf"
        assert paper.grobid_tei is None
        assert "pages" in paper.meta
        assert len(paper.meta["pages"]) >= 2
        assert "transformer" in full_text.lower()

    async def test_fallback_persists_correctly(
        self, pool: Pool, store: PgVectorStore, native_pdf: pathlib.Path,
    ) -> None:
        paper_id = PAPER_ID
        router = ParserRouter(
            grobid_url="http://localhost:1",
            grobid_timeout=0.1,
        )

        paper, full_text = await router.parse(str(native_pdf))

        chunks = chunk_document(paper, full_text)

        embedder = BgeM3Embedder()
        count = await persist_document(
            pool, store, embedder, paper, chunks,
        )

        assert count == len(chunks)
        db_count = await _chunk_count(pool, paper.id)
        assert db_count == len(chunks)

        await _cleanup(pool, [paper.id], [])

    async def test_scanned_pdf_still_raises_needs_ocr(
        self, scanned_pdf: pathlib.Path,
    ) -> None:
        router = ParserRouter(
            grobid_url="http://localhost:1",
            grobid_timeout=0.1,
        )
        with pytest.raises(ParseError, match="needs_ocr"):
            await router.parse(str(scanned_pdf))


# ============================================================================
# 2. Partial embedding failure → job NOT done
# ============================================================================


class TestPartialEmbeddingNotDone:
    """If embedding fails partway through the pipeline, the job must be
    recorded as 'failed' and the database must contain zero state
    (no paper row, no chunk rows)."""

    async def test_job_fails_no_db_state(
        self, pool: Pool, native_pdf: pathlib.Path,
    ) -> None:
        await _insert_paper(pool, PAPER_ID)
        await _insert_job(pool, JOB_ID, str(native_pdf), PAPER_ID, max_attempts=3)

        failing_embedder = _FailingEmbedder(succeed_calls=0)
        ok_router = _OkRouter(paper_id=PAPER_ID)

        result = await process_job(
            pool, _NoOpRetriever(), failing_embedder, ok_router, JOB_ID,
        )

        assert result["result"] == "failed"

        row = await pool.fetchrow(
            "SELECT status, stage, error FROM jobs WHERE id = $1::uuid",
            JOB_ID,
        )
        assert row is not None
        assert row["status"] in ("failed",)
        assert "embedding" in (row["error"] or "").lower()

        chunk_rows = await pool.fetch(
            "SELECT id FROM chunks WHERE paper_id = $1::uuid",
            PAPER_ID,
        )
        assert len(chunk_rows) == 0

        await _cleanup(pool, [PAPER_ID], [JOB_ID])

    async def test_mid_batch_failure_no_state(
        self, pool: Pool, native_pdf: pathlib.Path,
    ) -> None:
        """Batch 1 of embedding succeeds, batch 2 fails — still no DB state."""
        await _insert_paper(pool, PAPER_ID_2)
        await _insert_job(pool, JOB_ID_2, str(native_pdf), PAPER_ID_2, max_attempts=3)

        failing_embedder = _FailingEmbedder(succeed_calls=1)
        ok_router = _OkRouter(paper_id=PAPER_ID_2)

        result = await process_job(
            pool, _NoOpRetriever(), failing_embedder, ok_router, JOB_ID_2,
        )

        assert result["result"] == "failed"

        paper_exists = await pool.fetchrow(
            "SELECT id FROM papers WHERE id = $1::uuid",
            PAPER_ID_2,
        )
        assert paper_exists is None

        await _cleanup(pool, [PAPER_ID_2], [JOB_ID_2])


# ============================================================================
# 3. Forced job failure → dead (retries exhausted)
# ============================================================================


class TestForcedFailureDead:
    """When a job exhausts its retries it must transition to 'dead'
    status with the failure stage and error message persisted."""

    async def test_exhausted_retries_goes_dead(
        self, pool: Pool, native_pdf: pathlib.Path,
    ) -> None:
        await _insert_paper(pool, PAPER_ID)
        await _insert_job(
            pool, JOB_ID, str(native_pdf), PAPER_ID, max_attempts=1,
        )

        failing_router = _FailingRouter()

        result = await process_job(
            pool, _NoOpRetriever(), _OkEmbedder(), failing_router, JOB_ID,
        )

        assert result["result"] == "failed"

        row = await pool.fetchrow(
            "SELECT status, stage, error FROM jobs WHERE id = $1::uuid",
            JOB_ID,
        )
        assert row is not None
        assert row["status"] == "dead"
        assert row["stage"] == "parse"
        assert row["error"] is not None
        assert "Simulated parser failure" in row["error"]

        await _cleanup(pool, [PAPER_ID], [JOB_ID])

    async def test_dead_job_stores_max_attempts(
        self, pool: Pool, native_pdf: pathlib.Path,
    ) -> None:
        """A job that starts at max_attempts with already-exhausted retries
        goes dead immediately."""
        existing_id = "20000000-0000-0000-0000-0000000000f0"
        await _insert_paper(pool, PAPER_ID)
        await pool.execute(
            "INSERT INTO jobs (id, kind, payload, paper_id, attempts, max_attempts) "
            "VALUES ($1::uuid, 'ingest_pdf', $2::jsonb, $3::uuid, $4, $5) "
            "ON CONFLICT (id) DO NOTHING",
            existing_id,
            {"pdf_path": str(native_pdf)},
            PAPER_ID,
            3,
            3,
        )

        result = await process_job(
            pool, _NoOpRetriever(), _OkEmbedder(), _FailingRouter(), existing_id,
        )

        assert result["result"] == "failed"

        row = await pool.fetchrow(
            "SELECT status FROM jobs WHERE id = $1::uuid",
            existing_id,
        )
        assert row is not None
        assert row["status"] == "dead"

        await _cleanup(pool, [PAPER_ID], [existing_id])


# ============================================================================
# 4. Re-ingest → no duplicates
# ============================================================================


class TestReIngestNoDups:
    """Ingesting the same content twice must not produce duplicate
    chunk rows — the ON CONFLICT (id) DO UPDATE clause must keep
    the table idempotent."""

    async def test_upsert_same_chunks_idempotent(
        self, pool: Pool, store: PgVectorStore, native_pdf: pathlib.Path,
    ) -> None:
        router = ParserRouter(
            grobid_url="http://localhost:1",
            grobid_timeout=0.1,
        )
        paper, full_text = await router.parse(str(native_pdf))

        chunks = chunk_document(paper, full_text)

        embedder = BgeM3Embedder()

        first_count = await persist_document(
            pool, store, embedder, paper, chunks,
        )
        assert first_count == len(chunks)

        second_count = await persist_document(
            pool, store, embedder, paper, chunks,
        )
        assert second_count == len(chunks)

        db_count = await _chunk_count(pool, paper.id)
        assert db_count == len(chunks), (
            f"Expected {len(chunks)} chunks after re-ingest, "
            f"got {db_count} — duplicate rows detected"
        )

        hash_rows = await pool.fetch(
            "SELECT content_hash, COUNT(*) AS cnt FROM chunks "
            "WHERE paper_id = $1::uuid "
            "GROUP BY content_hash HAVING COUNT(*) > 1",
            paper.id,
        )
        assert len(hash_rows) == 0, (
            f"Found {len(hash_rows)} content_hashes with duplicates"
        )

        await _cleanup(pool, [paper.id], [])

    async def test_same_pdf_parse_repeatable(
        self, native_pdf: pathlib.Path,
    ) -> None:
        """Parsing the same PDF twice produces the same chunk hashes."""
        router = ParserRouter(
            grobid_url="http://localhost:1",
            grobid_timeout=0.1,
        )

        paper_a, text_a = await router.parse(str(native_pdf))
        paper_b, text_b = await router.parse(str(native_pdf))

        chunks_a = chunk_document(paper_a, text_a)
        chunks_b = chunk_document(paper_b, text_b)

        assert len(chunks_a) == len(chunks_b)
        for c1, c2 in zip(chunks_a, chunks_b, strict=True):
            assert c1.content_hash == c2.content_hash
            assert c1.text == c2.text


# ============================================================================
# 5. Unanswerable → abstain
# ============================================================================


class _NoCitationsLLM:
    """Mock LLM that returns an answer without a ## Citations section."""

    async def generate(self, prompt: str, system: str | None = None) -> str:
        return "I cannot answer this question from the provided context."


class _ParametricLLM:
    """Mock LLM that returns a citation referencing a known chunk_id + text.

    Used by TestUnanswerableAbstain to control what the LLM emits while
    keeping chunk_id / text in sync with what was actually persisted.
    """

    def __init__(
        self,
        *,
        chunk_id: str,
        chunk_text: str,
        answer_text: str = "The answer based on the paper.",
        nli_score: str = "0.95",
    ) -> None:
        self.chunk_id = chunk_id
        self.chunk_text = chunk_text
        self.answer_text = answer_text
        self.nli_score = nli_score
        self.call_count = 0

    async def generate(self, prompt: str, system: str | None = None) -> str:
        self.call_count += 1
        if self.call_count == 1:
            return (
                f"{self.answer_text}\n\n"
                "## Citations\n"
                f"[1] chunk: {self.chunk_id} quote: \"{self.chunk_text}\""
            )
        return self.nli_score


class _FixedEmbedder:
    """Returns a non-zero vector so dense search finds actual results."""

    def __init__(self, dim: int = VECTOR_DIM) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[Sequence[float]]:
        v = [1.0 / (i + 1) for i in range(min(self._dim, 5))] + [0.0] * (
            self._dim - min(self._dim, 5)
        )
        return [v for _ in texts]


class TestUnanswerableAbstain:
    """When no claim survives verification the engine must return
    answerable=False with an abstention message."""

    async def test_no_citations_from_llm_abstains(
        self, pool: Pool, store: PgVectorStore, native_pdf: pathlib.Path,
    ) -> None:
        router = ParserRouter(
            grobid_url="http://localhost:1",
            grobid_timeout=0.1,
        )
        paper, full_text = await router.parse(str(native_pdf))
        chunks = chunk_document(paper, full_text)

        real_embedder = BgeM3Embedder()
        await persist_document(pool, store, real_embedder, paper, chunks)

        llm = _NoCitationsLLM()
        qa = QAEngine(
            retriever=store, llm=llm, embedder=_FixedEmbedder(),
        )

        result = await qa.answer("What is the meaning of life?")

        assert result.answerable is False
        assert result.answer == "No hay soporte suficiente"
        assert len(result.claims) == 0
        assert len(result.verdicts) == 0

        await _cleanup(pool, [paper.id], [])

    async def test_low_entailment_abstains(
        self, pool: Pool, store: PgVectorStore, native_pdf: pathlib.Path,
    ) -> None:
        router = ParserRouter(
            grobid_url="http://localhost:1",
            grobid_timeout=0.1,
        )
        paper, full_text = await router.parse(str(native_pdf))
        chunks = chunk_document(paper, full_text)

        real_embedder = BgeM3Embedder()
        await persist_document(pool, store, real_embedder, paper, chunks)

        chunk = chunks[0]
        llm = _ParametricLLM(
            chunk_id=chunk.id,
            chunk_text=chunk.text,
            nli_score="0.1",
        )
        qa = QAEngine(
            retriever=store, llm=llm, embedder=_FixedEmbedder(),
        )

        result = await qa.answer("What does the transformer achieve?")

        assert result.answerable is False
        assert result.answer == "No hay soporte suficiente"

        await _cleanup(pool, [paper.id], [])

    async def test_answerable_returns_correctly(
        self, pool: Pool, store: PgVectorStore, native_pdf: pathlib.Path,
    ) -> None:
        router = ParserRouter(
            grobid_url="http://localhost:1",
            grobid_timeout=0.1,
        )
        paper, full_text = await router.parse(str(native_pdf))
        chunks = chunk_document(paper, full_text)

        real_embedder = BgeM3Embedder()
        await persist_document(pool, store, real_embedder, paper, chunks)

        chunk = chunks[0]
        llm = _ParametricLLM(
            chunk_id=chunk.id,
            chunk_text=chunk.text,
            nli_score="0.95",
        )
        qa = QAEngine(
            retriever=store, llm=llm, embedder=_FixedEmbedder(),
        )

        result = await qa.answer("What does the transformer achieve?")

        assert result.answerable is True
        assert len(result.claims) > 0
        assert any(v.supported for v in result.verdicts)

        await _cleanup(pool, [paper.id], [])
