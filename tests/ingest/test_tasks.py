"""Tests for ingest tasks — all DB calls mocked via fake pool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.errors import IngestError, ParseError
from core.interfaces import Embedder, Parser, Retriever
from core.models import Paper
from ingest.tasks import (
    _handle_failure,
    _mark_dead,
    _mark_done,
    _set_stage,
    backoff_delay,
    process_job,
)
from retrieval.db import Pool


def _make_pool() -> MagicMock:
    pool = MagicMock(spec=Pool)
    pool.execute = AsyncMock(return_value="UPDATE 1")
    pool.fetchrow = AsyncMock()
    pool.fetch = AsyncMock()

    # connection() context manager for persist_document
    conn_cm = MagicMock()
    conn_cm.__aenter__ = AsyncMock(return_value=pool)
    conn_cm.__aexit__ = AsyncMock(return_value=None)
    pool.connection = MagicMock(return_value=conn_cm)

    trans_cm = MagicMock()
    trans_cm.__aenter__ = AsyncMock(return_value=None)
    trans_cm.__aexit__ = AsyncMock(return_value=None)
    pool.transaction = MagicMock(return_value=trans_cm)
    return pool


@pytest.fixture
def mock_pool() -> MagicMock:
    return _make_pool()


@pytest.fixture
def mock_retriever() -> Retriever:
    return MagicMock(spec=Retriever)


@pytest.fixture
def mock_embedder() -> Embedder:
    return MagicMock(spec=Embedder)


@pytest.fixture
def mock_router() -> Parser:
    router = MagicMock(spec=Parser)
    router.parse = AsyncMock()
    return router


class _FakeRecord:
    """Acts like an asyncpg Record retrieved from a query."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def __getitem__(self, k: str):
        return self._data[k]

    def __iter__(self):
        return iter(self._data.items())

    def __len__(self) -> int:
        return len(self._data)

    def get(self, k: str, default=None):
        return self._data.get(k, default)


def _make_job_row(pdf_path: str = "/tmp/test.pdf",
                  attempts: int = 0,
                  max_attempts: int = 3,
                  status: str = "queued") -> _FakeRecord:
    return _FakeRecord({
        "id": "job-1",
        "payload": {"pdf_path": pdf_path},
        "attempts": attempts,
        "max_attempts": max_attempts,
        "status": status,
        "stage": None,
        "error": None,
    })


class TestDoiJob:
    """Kind-based dispatch: process_job routes to DOI handler."""

    @pytest.mark.asyncio
    async def test_doi_job_resolves_and_persists(
        self, mock_pool: MagicMock, mock_retriever: Retriever,
        mock_embedder: Embedder, mock_router: Parser,
    ) -> None:
        mock_pool.fetchrow = AsyncMock(return_value=_FakeRecord({
            "id": "job-doi-1",
            "kind": "ingest_doi",
            "payload": {"doi": "10.1234/test"},
            "attempts": 0,
            "max_attempts": 3,
            "status": "queued",
            "stage": None,
            "error": None,
        }))

        with patch("ingest.tasks.resolve_doi") as mock_resolve:
            mock_resolve.return_value = Paper(
                id="p-doi-1", doi="10.1234/test",
                title="Test Title", authors=[], year=2025,
                venue=None, abstract=None, source="crossref",
                open_access=None, pdf_path=None, grobid_tei=None,
                meta={"crossref": {}},
            )
            result = await process_job(
                mock_pool, mock_retriever, mock_embedder, mock_router, "job-doi-1",
            )

        assert result["result"] == "done"
        assert result["doi"] == "10.1234/test"
        assert result["paper_id"] == "p-doi-1"
        mock_resolve.assert_awaited_once_with("10.1234/test")

    @pytest.mark.asyncio
    async def test_doi_job_missing_doi_in_payload_is_failed(
        self, mock_pool: MagicMock, mock_retriever: Retriever,
        mock_embedder: Embedder, mock_router: Parser,
    ) -> None:
        mock_pool.fetchrow = AsyncMock(return_value=_FakeRecord({
            "id": "job-doi-2",
            "kind": "ingest_doi",
            "payload": {},
            "attempts": 0,
            "max_attempts": 3,
            "status": "queued",
            "stage": None,
            "error": None,
        }))

        result = await process_job(
            mock_pool, mock_retriever, mock_embedder, mock_router, "job-doi-2",
        )

        assert result["result"] == "failed"
        assert "No doi" in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_doi_job_network_error_is_retryable(
        self, mock_pool: MagicMock, mock_retriever: Retriever,
        mock_embedder: Embedder, mock_router: Parser,
    ) -> None:
        mock_pool.fetchrow = AsyncMock(return_value=_FakeRecord({
            "id": "job-doi-3",
            "kind": "ingest_doi",
            "payload": {"doi": "10.1234/slow"},
            "attempts": 0,
            "max_attempts": 3,
            "status": "queued",
            "stage": None,
            "error": None,
        }))

        with patch("ingest.tasks.resolve_doi") as mock_resolve:
            mock_resolve.side_effect = IngestError("Crossref timeout")
            result = await process_job(
                mock_pool, mock_retriever, mock_embedder, mock_router, "job-doi-3",
            )

        assert result["result"] == "failed"
        assert "Crossref timeout" in (result.get("error") or "")
        # M-WRK: 'failed' is retryable with next_attempt_at
        sql = str(mock_pool.execute.call_args_list)
        assert "failed" in sql
        assert "dead" not in sql


class TestProcessJob:
    @pytest.mark.asyncio
    async def test_job_not_found(
        self, mock_pool: MagicMock, mock_retriever: Retriever,
        mock_embedder: Embedder, mock_router: Parser,
    ) -> None:
        mock_pool.fetchrow = AsyncMock(return_value=None)
        result = await process_job(
            mock_pool, mock_retriever, mock_embedder, mock_router, "missing",
        )
        assert result == {"result": "not_found"}

    @pytest.mark.asyncio
    async def test_no_pdf_path(
        self, mock_pool: MagicMock, mock_retriever: Retriever,
        mock_embedder: Embedder, mock_router: Parser,
    ) -> None:
        row = MagicMock()
        row.__getitem__ = lambda self, k: {
            "id": "job-1", "payload": {}, "attempts": 0,
            "max_attempts": 3, "status": "queued", "stage": None,
        }[k]
        mock_pool.fetchrow = AsyncMock(return_value=row)
        result = await process_job(
            mock_pool, mock_retriever, mock_embedder, mock_router, "job-1",
        )
        assert result["result"] == "failed"
        assert "No pdf_path" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_successful_pipeline(
        self, mock_pool: MagicMock, mock_retriever: Retriever,
        mock_embedder: Embedder, mock_router: Parser,
    ) -> None:
        mock_pool.fetchrow = AsyncMock(
            return_value=_make_job_row(),
        )
        paper = Paper(
            id="p1", doi=None, title="T", authors=[], year=None,
            venue=None, abstract=None, source="test", open_access=None,
            pdf_path="/tmp/test.pdf", grobid_tei=None, meta={},
        )
        mock_router.parse = AsyncMock(return_value=(paper, "full text"))
        mock_retriever.upsert = AsyncMock()

        result = await process_job(
            mock_pool, mock_retriever, mock_embedder, mock_router, "job-1",
        )
        assert result["result"] == "done"

    @pytest.mark.asyncio
    async def test_parse_error_marks_failed(
        self, mock_pool: MagicMock, mock_retriever: Retriever,
        mock_embedder: Embedder, mock_router: Parser,
    ) -> None:
        mock_pool.fetchrow = AsyncMock(
            return_value=_make_job_row(attempts=0, max_attempts=3),
        )
        mock_router.parse = AsyncMock(
            side_effect=ParseError("PDF corrupt"),
        )
        result = await process_job(
            mock_pool, mock_retriever, mock_embedder, mock_router, "job-1",
        )
        assert result["result"] == "failed"

    @pytest.mark.asyncio
    async def test_max_attempts_reached_marks_dead(
        self, mock_pool: MagicMock, mock_retriever: Retriever,
        mock_embedder: Embedder, mock_router: Parser,
    ) -> None:
        mock_pool.fetchrow = AsyncMock(
            return_value=_make_job_row(attempts=2, max_attempts=3),
        )
        mock_router.parse = AsyncMock(
            side_effect=ParseError("PDF corrupt"),
        )
        result = await process_job(
            mock_pool, mock_retriever, mock_embedder, mock_router, "job-1",
        )
        assert result["result"] == "failed"

    @pytest.mark.asyncio
    async def test_unexpected_exception_handled(
        self, mock_pool: MagicMock, mock_retriever: Retriever,
        mock_embedder: Embedder, mock_router: Parser,
    ) -> None:
        mock_pool.fetchrow = AsyncMock(
            return_value=_make_job_row(),
        )
        mock_router.parse = AsyncMock(
            side_effect=RuntimeError("unexpected"),
        )
        result = await process_job(
            mock_pool, mock_retriever, mock_embedder, mock_router, "job-1",
        )
        assert result["result"] == "failed"


class TestSetStage:
    @pytest.mark.asyncio
    async def test_sets_stage(self, mock_pool: MagicMock) -> None:
        await _set_stage(mock_pool, "job-1", "parse")
        mock_pool.execute.assert_called_once()


class TestMarkDone:
    @pytest.mark.asyncio
    async def test_marks_done(self, mock_pool: MagicMock) -> None:
        await _mark_done(mock_pool, "job-1")
        sql = mock_pool.execute.call_args[0][0]
        assert "done" in sql
        assert "next_attempt_at = NULL" in sql


class TestMarkDead:
    @pytest.mark.asyncio
    async def test_marks_dead(self, mock_pool: MagicMock) -> None:
        await _mark_dead(mock_pool, "job-1", "parse", "error msg")
        sql = mock_pool.execute.call_args[0][0]
        assert "dead" in sql
        assert "next_attempt_at = NULL" in sql


class TestHandleFailure:
    @pytest.mark.asyncio
    async def test_below_max_marks_failed_with_backoff(self, mock_pool: MagicMock) -> None:
        await _handle_failure(mock_pool, "job-1", "parse",
                              "error", 1, 3)
        sql = mock_pool.execute.call_args[0][0]
        assert "failed" in sql
        assert "next_attempt_at = $" in sql
        assert len(mock_pool.execute.call_args[0]) == 5  # sql, stage, error, id, next_attempt_at

    @pytest.mark.asyncio
    async def test_at_max_marks_dead(self, mock_pool: MagicMock) -> None:
        await _handle_failure(mock_pool, "job-1", "parse",
                              "error", 3, 3)
        sql = mock_pool.execute.call_args[0][0]
        assert "dead" in sql
        assert "next_attempt_at = NULL" in sql

    @pytest.mark.asyncio
    async def test_over_max_marks_dead(self, mock_pool: MagicMock) -> None:
        await _handle_failure(mock_pool, "job-1", "parse",
                              "error", 5, 3)
        sql = mock_pool.execute.call_args[0][0]
        assert "dead" in sql
        assert "next_attempt_at = NULL" in sql


class TestBackoffDelay:
    def test_first_attempt(self) -> None:
        assert backoff_delay(1) == 10.0

    def test_second_attempt(self) -> None:
        assert backoff_delay(2) == 20.0

    def test_third_attempt(self) -> None:
        assert backoff_delay(3) == 40.0

    def test_capped_at_one_hour(self) -> None:
        assert backoff_delay(100) == 3600.0

    def test_custom_base(self) -> None:
        assert backoff_delay(2, base_seconds=5.0) == 10.0


# ---------------------------------------------------------------------------
# Full retry → dead-letter cycle
# ---------------------------------------------------------------------------


class _FakePoolForCycle:
    def __init__(self) -> None:
        self.executed: list[str] = []
        self._fetchrow_return: dict | None = None

    def set_fetchrow(self, row: dict | None) -> None:
        self._fetchrow_return = row

    async def fetchrow(self, query: str, *params: object) -> dict | None:
        self.executed.append(f"fetchrow: {query}")
        return self._fetchrow_return

    async def execute(self, query: str, *params: object) -> str:
        self.executed.append(f"execute: {query}")
        return "OK"

    async def connection(self):
        class _Conn:
            async def __aenter__(s):
                return self
            async def __aexit__(s, *a):
                pass
        return _Conn()

    async def transaction(self):
        class _Tx:
            async def __aenter__(s):
                pass
            async def __aexit__(s, *a):
                pass
        return _Tx()


class _FakeRetrieverForCycle:
    async def upsert(self, items) -> int:
        return len(items)
    async def delete_by_paper(self, paper_id: str) -> int:
        return 0
    async def count(self) -> int:
        return 0
    async def health(self) -> bool:
        return True
    async def search_dense(self, *a, **kw):
        return []
    async def search_lexical(self, *a, **kw):
        return []


class _FakeEmbedderForCycle:
    async def embed(self, texts: list[str]) -> list:
        return [[0.0] * 4 for _ in texts]


class _FailingRouterForCycle:
    async def parse(self, pdf_path: str):
        from core.errors import ParseError
        raise ParseError(f"Simulated parse failure for {pdf_path}")


class TestRetryCycle:
    pytestmark = pytest.mark.asyncio

    async def test_retry_then_dead(self) -> None:
        """Simulate: fail → re-queue → fail → re-queue → fail → dead."""
        pool = _FakePoolForCycle()
        retriever = _FakeRetrieverForCycle()
        embedder = _FakeEmbedderForCycle()
        router = _FailingRouterForCycle()

        # attempt 1: fresh job → fails to 'failed'
        pool.set_fetchrow({
            "id": "job-retry", "kind": "ingest_pdf",
            "status": "queued", "paper_id": "p1",
            "payload": {"pdf_path": "/tmp/test.pdf"},
            "attempts": 0, "max_attempts": 3,
            "error": None, "stage": None,
        })
        result = await process_job(pool, retriever, embedder, router, "job-retry")
        assert result["result"] == "failed"
        text = str(pool.executed)
        assert "dead" not in text
        assert "next_attempt_at" in text

        # attempt 2: status=failed, attempts=1 → fails to 'failed'
        pool.set_fetchrow({
            "id": "job-retry", "kind": "ingest_pdf",
            "status": "failed", "paper_id": "p1",
            "payload": {"pdf_path": "/tmp/test.pdf"},
            "attempts": 1, "max_attempts": 3,
            "error": "previous error", "stage": "parse",
        })
        result = await process_job(pool, retriever, embedder, router, "job-retry")
        assert result["result"] == "failed"
        text2 = str(pool.executed)
        assert "dead" not in text2

        # attempt 3 (last retry): attempts=2 → goes 'dead'
        pool.set_fetchrow({
            "id": "job-retry", "kind": "ingest_pdf",
            "status": "failed", "paper_id": "p1",
            "payload": {"pdf_path": "/tmp/test.pdf"},
            "attempts": 2, "max_attempts": 3,
            "error": "previous error", "stage": "parse",
        })
        result = await process_job(pool, retriever, embedder, router, "job-retry")
        assert result["result"] == "failed"
        text3 = str(pool.executed)
        assert "dead" in text3
        assert "next_attempt_at = NULL" in text3
