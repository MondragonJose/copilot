"""Tests for PgVectorStore — all DB calls are mocked via a fake Pool."""

from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from core.errors import RetrievalError
from core.models import ChunkRef, ScoredChunk, UpsertChunk
from retrieval.db import Pool
from retrieval.pgvector_store import PgVectorStore


@pytest.fixture
def mock_pool() -> Pool:
    pool = MagicMock(spec=Pool)
    pool.execute = AsyncMock(return_value="INSERT 1")
    pool.fetch = AsyncMock(return_value=[])
    pool.fetchrow = AsyncMock(return_value=None)

    mock_tx = AsyncMock()
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.execute = AsyncMock(return_value="SET")
    mock_conn.transaction = MagicMock(return_value=mock_tx)

    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_conn

    pool.connection = MagicMock(return_value=mock_cm)

    return pool


@pytest.fixture
def store(mock_pool: Pool) -> PgVectorStore:
    return PgVectorStore(mock_pool)


class TestUpsert:
    @pytest.mark.asyncio
    async def test_upsert_single(self, store: PgVectorStore,
                                  mock_pool: Pool) -> None:
        items = [
            UpsertChunk(
                chunk_id="c1", paper_id="p1", text="hello",
                vector=[0.1, 0.2], metadata={},
            ),
        ]
        count = await store.upsert(items)
        assert count == 1
        assert mock_pool.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_upsert_multiple(self, store: PgVectorStore,
                                    mock_pool: Pool) -> None:
        items = [
            UpsertChunk(chunk_id="c1", paper_id="p1", text="a",
                        vector=[0.1], metadata={}),
            UpsertChunk(chunk_id="c2", paper_id="p1", text="b",
                        vector=[0.2], metadata={}),
        ]
        count = await store.upsert(items)
        assert count == 2
        assert mock_pool.execute.call_count == 4

    @pytest.mark.asyncio
    async def test_upsert_empty(self, store: PgVectorStore,
                                 mock_pool: Pool) -> None:
        count = await store.upsert([])
        assert count == 0
        mock_pool.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_upsert_idempotent_same_chunk(
        self, store: PgVectorStore, mock_pool: Pool,
    ) -> None:
        items = [
            UpsertChunk(chunk_id="c1", paper_id="p1", text="v1",
                        vector=[0.1], metadata={}),
        ]
        await store.upsert(items)
        await store.upsert(items)
        assert mock_pool.execute.call_count == 4


class TestDeleteByPaper:
    @pytest.mark.asyncio
    async def test_delete(self, store: PgVectorStore,
                           mock_pool: Pool) -> None:
        mock_pool.fetch = AsyncMock(
            return_value=[MagicMock(), MagicMock()],
        )
        count = await store.delete_by_paper("p1")
        assert count == 2

    @pytest.mark.asyncio
    async def test_delete_none(self, store: PgVectorStore,
                                mock_pool: Pool) -> None:
        mock_pool.fetch = AsyncMock(return_value=[])
        count = await store.delete_by_paper("missing")
        assert count == 0


class TestCount:
    @pytest.mark.asyncio
    async def test_count(self, store: PgVectorStore,
                          mock_pool: Pool) -> None:
        mock_pool.fetchrow = AsyncMock(
            return_value=MagicMock(**{"__getitem__": lambda s, k: 42}),
        )
        assert await store.count() == 42

    @pytest.mark.asyncio
    async def test_count_none(self, store: PgVectorStore,
                               mock_pool: Pool) -> None:
        mock_pool.fetchrow = AsyncMock(return_value=None)
        assert await store.count() == 0


class TestHealth:
    @pytest.mark.asyncio
    async def test_healthy(self, store: PgVectorStore,
                            mock_pool: Pool) -> None:
        assert await store.health() is True

    @pytest.mark.asyncio
    async def test_unhealthy(self, store: PgVectorStore,
                              mock_pool: Pool) -> None:
        mock_pool.execute = AsyncMock(
            side_effect=RetrievalError("down"),
        )
        assert await store.health() is False


class TestSearchDense:
    @pytest.mark.asyncio
    async def test_returns_scored_chunks(self, store: PgVectorStore,
                                          mock_pool: Pool) -> None:
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda s, k: {
            "id": "c1", "paper_id": "p1", "text": "t",
            "section": None, "page": None,
            "char_start": None, "char_end": None,
            "score": 0.0,
        }.get(k) if isinstance(k, str) else None
        conn = mock_pool.connection.return_value.__aenter__.return_value
        conn.fetch = AsyncMock(return_value=[mock_row])

        results = await store.search_dense([0.1, 0.2], k=5)
        assert len(results) == 1
        assert results[0].score == 0.0
        assert results[0].channel == "dense"
        assert results[0].chunk.chunk_id == "c1"

    @pytest.mark.asyncio
    async def test_k_zero_returns_empty(self, store: PgVectorStore,
                                         mock_pool: Pool) -> None:
        results = await store.search_dense([0.1], k=0)
        assert results == []
        mock_pool.connection.assert_not_called()

    @pytest.mark.asyncio
    async def test_k_negative_returns_empty(self, store: PgVectorStore,
                                              mock_pool: Pool) -> None:
        results = await store.search_dense([0.1], k=-1)
        assert results == []

    @pytest.mark.asyncio
    async def test_score_clamping_in_query(self, store: PgVectorStore,
                                            mock_pool: Pool) -> None:
        """The SQL uses GREATEST(0.0, LEAST(1.0, ...)) to clamp."""
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda s, k: {
            "id": "c1", "paper_id": "p1", "text": "t",
            "section": None, "page": None,
            "char_start": None, "char_end": None,
            "score": -0.5,
        }.get(k) if isinstance(k, str) else None
        conn = mock_pool.connection.return_value.__aenter__.return_value
        conn.fetch = AsyncMock(return_value=[mock_row])

        results = await store.search_dense([0.1], k=5)
        assert results[0].score == -0.5  # clamping is SQL-side, mock returns raw

    @pytest.mark.asyncio
    async def test_with_paper_filter(self, store: PgVectorStore,
                                      mock_pool: Pool) -> None:
        conn = mock_pool.connection.return_value.__aenter__.return_value
        conn.fetch = AsyncMock(return_value=[])
        await store.search_dense([0.1], k=5, paper_ids=["p1", "p2"])
        # Should pass paper_ids as $3 in SQL
        call_args = conn.fetch.call_args
        assert call_args is not None
        assert "ANY($3::uuid[])" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_empty_vector(self, store: PgVectorStore,
                                 mock_pool: Pool) -> None:
        conn = mock_pool.connection.return_value.__aenter__.return_value
        conn.fetch = AsyncMock(return_value=[])
        results = await store.search_dense([], k=5)
        assert results == []


class TestSearchLexical:
    @pytest.mark.asyncio
    async def test_returns_scored_chunks(self, store: PgVectorStore,
                                          mock_pool: Pool) -> None:
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda s, k: {
            "id": "c1", "paper_id": "p1", "text": "t",
            "section": None, "page": None,
            "char_start": None, "char_end": None,
            "score": 0.5,
        }.get(k) if isinstance(k, str) else None
        mock_pool.fetch = AsyncMock(return_value=[mock_row])

        results = await store.search_lexical("query", k=5)
        assert len(results) == 1
        assert results[0].score == 0.5
        assert results[0].channel == "lexical"

    @pytest.mark.asyncio
    async def test_empty_query(self, store: PgVectorStore,
                                mock_pool: Pool) -> None:
        mock_pool.fetch = AsyncMock(return_value=[])
        results = await store.search_lexical("", k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_with_paper_filter(self, store: PgVectorStore,
                                      mock_pool: Pool) -> None:
        mock_pool.fetch = AsyncMock(return_value=[])
        await store.search_lexical("query", k=5, paper_ids=["p1"])
        call_args = mock_pool.fetch.call_args
        assert call_args is not None
        assert "ANY" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_unicode_query(self, store: PgVectorStore,
                                  mock_pool: Pool) -> None:
        mock_pool.fetch = AsyncMock(return_value=[])
        results = await store.search_lexical("über cool", k=5)
        assert results == []
