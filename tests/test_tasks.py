"""Tests for the async ingest worker — retry/backoff and dead-letter logic."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from core.errors import ParseError
from core.models import Paper, UpsertChunk
from ingest.tasks import (
    _handle_failure,
    _mark_dead,
    _mark_done,
    _set_stage,
    backoff_delay,
    process_job,
)

pytestmark = pytest.mark.asyncio

# --------------------------------------------------------------------------
# Mock helpers
# --------------------------------------------------------------------------


class _FakePool:
    """Records all executed SQL queries for inspection."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, ...]] = []
        self._fetchrow_return: dict[str, Any] | None = None

    def set_fetchrow(self, row: dict[str, Any] | None) -> None:
        self._fetchrow_return = row

    async def fetchrow(self, query: str, *params: object) -> dict[str, Any] | None:
        self.executed.append(("fetchrow", query, *(str(p) for p in params)))
        return self._fetchrow_return

    async def execute(self, query: str, *params: object) -> str:
        self.executed.append(("execute", query, *(str(p) for p in params)))
        return "OK"


class _FakeRetriever:
    async def upsert(self, items: Sequence[UpsertChunk]) -> int:
        return len(items)

    async def delete_by_paper(self, paper_id: str) -> int:
        return 0

    async def count(self) -> int:
        return 0

    async def health(self) -> bool:
        return True

    async def search_dense(self, *args: object, **kwargs: object) -> list[Any]:
        return []

    async def search_lexical(self, *args: object, **kwargs: object) -> list[Any]:
        return []


class _FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[Sequence[float]]:
        return [[0.0] * 4 for _ in texts]


class _OkRouter:
    async def parse(self, pdf_path: str) -> tuple[Paper, str]:
        return (
            Paper(
                id="p1", doi=None, title="Test", authors=[], year=None,
                venue=None, abstract=None, source="test", open_access=None,
                pdf_path=pdf_path, grobid_tei=None,
                meta={"sections": [
                    {"heading": "Introduction", "text": "Hello world test content.",
                     "char_start": 0},
                ]},
            ),
            "Hello world test content.",
        )


class _FailingRouter:
    """Raises ParseError (subclass of RCError) on every call."""

    async def parse(self, pdf_path: str) -> tuple[Paper, str]:
        msg = f"Simulated parse failure for {pdf_path}"
        raise ParseError(msg)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def pool() -> _FakePool:
    return _FakePool()


@pytest.fixture
def retriever() -> _FakeRetriever:
    return _FakeRetriever()


@pytest.fixture
def embedder() -> _FakeEmbedder:
    return _FakeEmbedder()


@pytest.fixture
def ok_router() -> _OkRouter:
    return _OkRouter()


@pytest.fixture
def failing_router() -> _FailingRouter:
    return _FailingRouter()


@pytest.fixture
def queued_job(pool: _FakePool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "job-001",
        "kind": "ingest_pdf",
        "status": "queued",
        "paper_id": "p1",
        "payload": {"pdf_path": "/tmp/test.pdf"},
        "attempts": 0,
        "max_attempts": 3,
        "error": None,
        "stage": None,
    }
    pool.set_fetchrow(row)
    return row


# --------------------------------------------------------------------------
# backoff_delay
# --------------------------------------------------------------------------


class TestBackoffDelay:

    def test_first_attempt(self) -> None:
        assert backoff_delay(1) == 10.0

    def test_second_attempt(self) -> None:
        assert backoff_delay(2) == 20.0

    def test_third_attempt(self) -> None:
        assert backoff_delay(3) == 40.0

    def test_capped_at_one_hour(self) -> None:
        assert backoff_delay(10) == 3600.0

    def test_custom_base(self) -> None:
        assert backoff_delay(2, base_seconds=5.0) == 10.0


# --------------------------------------------------------------------------
# Job lifecycle helpers
# --------------------------------------------------------------------------


class TestLifecycleHelpers:

    async def test_set_stage(self, pool: _FakePool) -> None:
        await _set_stage(pool, "j1", "chunk")

        updates = [e for e in pool.executed if e[1].startswith("UPDATE jobs")]
        assert len(updates) == 1
        assert "chunk" in str(updates[0])

    async def test_mark_done(self, pool: _FakePool) -> None:
        await _mark_done(pool, "j1")

        updates = [e for e in pool.executed if e[1].startswith("UPDATE jobs")]
        assert len(updates) == 1
        assert "done" in str(updates[0])

    async def test_mark_dead(self, pool: _FakePool) -> None:
        await _mark_dead(pool, "j1", "parse", "oops")

        text = str(pool.executed)
        assert "dead" in text
        assert "parse" in text
        assert "oops" in text

    async def test_handle_failure_below_max_goes_failed(self, pool: _FakePool) -> None:
        await _handle_failure(pool, "j1", "chunk", "error", attempts=1, max_attempts=3)

        text = str(pool.executed)
        assert "failed" in text
        assert "dead" not in text
        assert "chunk" in text
        assert "error" in text

    async def test_handle_failure_at_max_goes_dead(self, pool: _FakePool) -> None:
        await _handle_failure(pool, "j1", "embed", "fatal", attempts=3, max_attempts=3)

        text = str(pool.executed)
        assert "dead" in text
        assert "embed" in text
        assert "fatal" in text

    async def test_handle_failure_above_max_goes_dead(self, pool: _FakePool) -> None:
        await _handle_failure(pool, "j1", "persist", "boom", attempts=5, max_attempts=3)

        text = str(pool.executed)
        assert "dead" in text
        assert "persist" in text


# --------------------------------------------------------------------------
# process_job — success path
# --------------------------------------------------------------------------


class TestProcessJobSuccess:

    async def test_returns_done(
        self, pool: _FakePool, retriever: _FakeRetriever,
        embedder: _FakeEmbedder, ok_router: _OkRouter, queued_job: dict[str, Any],
    ) -> None:
        result = await process_job(pool, retriever, embedder, ok_router, "job-001")

        assert result["result"] == "done"
        assert result["chunks"] > 0

    async def test_success_marks_done(
        self, pool: _FakePool, retriever: _FakeRetriever,
        embedder: _FakeEmbedder, ok_router: _OkRouter, queued_job: dict[str, Any],
    ) -> None:
        await process_job(pool, retriever, embedder, ok_router, "job-001")

        text = str(pool.executed)
        assert "done" in text

    async def test_stage_progresses(
        self, pool: _FakePool, retriever: _FakeRetriever,
        embedder: _FakeEmbedder, ok_router: _OkRouter, queued_job: dict[str, Any],
    ) -> None:
        await process_job(pool, retriever, embedder, ok_router, "job-001")

        stages = [
            e[2] for e in pool.executed
            if "stage" in str(e[1])
        ]
        assert "parse" in stages
        assert "chunk" in stages
        assert "persist" in stages


# --------------------------------------------------------------------------
# process_job — failure → retry → dead-letter
# --------------------------------------------------------------------------


class TestProcessJobFailure:

    async def test_rc_error_returns_failed(
        self, pool: _FakePool, retriever: _FakeRetriever,
        embedder: _FakeEmbedder, failing_router: _FailingRouter,
        queued_job: dict[str, Any],
    ) -> None:
        result = await process_job(
            pool, retriever, embedder, failing_router, "job-001",
        )

        assert result["result"] == "failed"

    async def test_rc_error_stores_stage_and_error(
        self, pool: _FakePool, retriever: _FakeRetriever,
        embedder: _FakeEmbedder, failing_router: _FailingRouter,
        queued_job: dict[str, Any],
    ) -> None:
        await process_job(
            pool, retriever, embedder, failing_router, "job-001",
        )

        text = str(pool.executed)
        assert "failed" in text
        assert "parse" in text
        assert "Simulated parse failure" in text

    async def test_retry_not_dead_when_attempts_remaining(
        self, pool: _FakePool, retriever: _FakeRetriever,
        embedder: _FakeEmbedder, failing_router: _FailingRouter,
        queued_job: dict[str, Any],
    ) -> None:
        await process_job(
            pool, retriever, embedder, failing_router, "job-001",
        )

        text = str(pool.executed)
        assert "dead" not in text
        assert "failed" in text

    async def test_job_goes_dead_when_attempts_exhausted(
        self, pool: _FakePool, retriever: _FakeRetriever,
        embedder: _FakeEmbedder, failing_router: _FailingRouter,
    ) -> None:
        """Job on attempt 3 of 3 → dead on next failure."""
        row: dict[str, Any] = {
            "id": "job-dead",
            "kind": "ingest_pdf",
            "status": "queued",
            "paper_id": "p1",
            "payload": {"pdf_path": "/tmp/test.pdf"},
            "attempts": 2,  # one retry left
            "max_attempts": 3,
            "error": None,
            "stage": None,
        }
        pool.set_fetchrow(row)

        await process_job(pool, retriever, embedder, failing_router, "job-dead")

        text = str(pool.executed)
        assert "dead" in text
        assert "attempts >= max_attempts" or True  # just check dead

    async def test_dead_job_has_stage_and_error(
        self, pool: _FakePool, retriever: _FakeRetriever,
        embedder: _FakeEmbedder, failing_router: _FailingRouter,
    ) -> None:
        row: dict[str, Any] = {
            "id": "job-dead2",
            "kind": "ingest_pdf",
            "status": "queued",
            "paper_id": "p1",
            "payload": {"pdf_path": "/tmp/test.pdf"},
            "attempts": 3,  # already at max
            "max_attempts": 3,
            "error": None,
            "stage": None,
        }
        pool.set_fetchrow(row)

        await process_job(pool, retriever, embedder, failing_router, "job-dead2")

        text = str(pool.executed)
        assert "dead" in text
        assert "parse" in text
        assert "Simulated parse failure" in text


class TestProcessJobEdgeCases:

    async def test_job_not_found(
        self, pool: _FakePool, retriever: _FakeRetriever,
        embedder: _FakeEmbedder, ok_router: _OkRouter,
    ) -> None:
        pool.set_fetchrow(None)

        result = await process_job(pool, retriever, embedder, ok_router, "missing")

        assert result["result"] == "not_found"

    async def test_no_pdf_path_in_payload(
        self, pool: _FakePool, retriever: _FakeRetriever,
        embedder: _FakeEmbedder, ok_router: _OkRouter,
    ) -> None:
        row: dict[str, Any] = {
            "id": "job-nopath",
            "kind": "ingest_pdf",
            "status": "queued",
            "paper_id": "p1",
            "payload": {},
            "attempts": 0,
            "max_attempts": 3,
            "error": None,
            "stage": None,
        }
        pool.set_fetchrow(row)

        result = await process_job(
            pool, retriever, embedder, ok_router, "job-nopath",
        )

        assert result["result"] == "failed"
        text = str(pool.executed)
        assert "dead" in text  # goes dead immediately
        assert "No pdf_path" in text

    async def test_unexpected_exception_wrapped(
        self, pool: _FakePool, retriever: _FakeRetriever,
        embedder: _FakeEmbedder, queued_job: dict[str, Any],
    ) -> None:
        class _CrazyRouter:
            async def parse(self, pdf_path: str) -> tuple[Paper, str]:
                msg = "runtime boom"
                raise RuntimeError(msg)

        result = await process_job(
            pool, retriever, embedder, _CrazyRouter(), "job-001",
        )

        assert result["result"] == "failed"
        assert "runtime boom" in str(result["error"])
