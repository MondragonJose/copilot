"""Tests for ingest tasks — all DB calls mocked via fake pool."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.errors import ParseError
from core.interfaces import Embedder, Parser, Retriever
from core.models import Chunk, Paper
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
        mock_pool.execute.assert_called_once()


class TestMarkDead:
    @pytest.mark.asyncio
    async def test_marks_dead(self, mock_pool: MagicMock) -> None:
        await _mark_dead(mock_pool, "job-1", "parse", "error msg")
        mock_pool.execute.assert_called_once()


class TestHandleFailure:
    @pytest.mark.asyncio
    async def test_below_max_marks_failed(self, mock_pool: MagicMock) -> None:
        await _handle_failure(mock_pool, "job-1", "parse",
                              "error", 1, 3)
        call_kwargs = mock_pool.execute.call_args[0][0]
        assert "failed" in call_kwargs

    @pytest.mark.asyncio
    async def test_at_max_marks_dead(self, mock_pool: MagicMock) -> None:
        await _handle_failure(mock_pool, "job-1", "parse",
                              "error", 3, 3)
        call_kwargs = mock_pool.execute.call_args[0][0]
        assert "dead" in call_kwargs

    @pytest.mark.asyncio
    async def test_over_max_marks_dead(self, mock_pool: MagicMock) -> None:
        await _handle_failure(mock_pool, "job-1", "parse",
                              "error", 5, 3)
        call_kwargs = mock_pool.execute.call_args[0][0]
        assert "dead" in call_kwargs


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
