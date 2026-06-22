"""Retrieval regression suite — k clamping, score bounds, idempotent upsert.

Guarded regressions:
  • R2-BUG-002  — k <= 0 crashes retrieval (now returns empty list for search_dense)
  • R2-BUG-001  — Score clamping to [0,1] in pgvector (via GREATEST/LEAST in SQL)
  • ON-CONFLICT — Upsert idempotency pattern
"""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.models import ScoredChunk, UpsertChunk


# ── Issue R2-BUG-002: k <= 0 guard ──────────────────────────────────────────


class TestSearchDenseKClamp:
    """Regression: search_dense with k <= 0 must return [], not crash."""

    @pytest.mark.asyncio
    async def test_k_zero_returns_empty(self) -> None:
        from retrieval.pgvector_store import PgVectorStore

        pool = AsyncMock()
        store = PgVectorStore(pool)
        result = await store.search_dense([0.1, 0.2], k=0)
        assert result == []
        pool.connection.assert_not_called()

    @pytest.mark.asyncio
    async def test_k_negative_returns_empty(self) -> None:
        from retrieval.pgvector_store import PgVectorStore

        pool = AsyncMock()
        store = PgVectorStore(pool)
        result = await store.search_dense([0.1, 0.2], k=-5)
        assert result == []

    @pytest.mark.asyncio
    async def test_k_positive_calls_connection(self) -> None:
        from retrieval.pgvector_store import PgVectorStore

        pool = AsyncMock()
        mock_tx = AsyncMock()
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        conn.execute = AsyncMock(return_value="SET")
        conn.transaction = MagicMock(return_value=mock_tx)
        cm = AsyncMock()
        cm.__aenter__.return_value = conn
        pool.connection = MagicMock(return_value=cm)
        store = PgVectorStore(pool)
        result = await store.search_dense([0.1, 0.2], k=10)
        pool.connection.assert_called_once()
        assert result == []

    @pytest.mark.asyncio
    async def test_k_one_is_valid(self) -> None:
        """Boundary: k=1 should be the minimum valid value."""
        from retrieval.pgvector_store import PgVectorStore

        pool = AsyncMock()
        mock_tx = AsyncMock()
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        conn.execute = AsyncMock(return_value="SET")
        conn.transaction = MagicMock(return_value=mock_tx)
        cm = AsyncMock()
        cm.__aenter__.return_value = conn
        pool.connection = MagicMock(return_value=cm)
        store = PgVectorStore(pool)
        result = await store.search_dense([0.1, 0.2], k=1)
        pool.connection.assert_called_once()
        assert result == []


class TestSearchLexicalKBehavior:
    """search_lexical has no k guard — verify current (un-guarded) behavior."""

    @pytest.mark.asyncio
    async def test_k_zero_passes_k_as_is(self) -> None:
        """No guard exists — k=0 is passed through to SQL."""
        from retrieval.pgvector_store import PgVectorStore

        pool = AsyncMock()
        pool.fetch.return_value = []
        store = PgVectorStore(pool)
        result = await store.search_lexical("query", k=0)
        pool.fetch.assert_called_once()
        assert result == []

    @pytest.mark.asyncio
    async def test_k_positive_returns_scored_chunks(self) -> None:
        from retrieval.pgvector_store import PgVectorStore

        pool = AsyncMock()
        pool.fetch.return_value = [
            {"id": "c1", "paper_id": "p1", "text": "t", "section": None,
             "page": 1, "char_start": 0, "char_end": 1, "score": 0.5},
        ]
        store = PgVectorStore(pool)
        result = await store.search_lexical("query", k=10)
        assert len(result) == 1
        assert isinstance(result[0], ScoredChunk)
        assert result[0].channel == "lexical"
        assert result[0].score == 0.5
        assert result[0].chunk.paper_id == "p1"


# ── Score clamp regression (SQL-level GREATEST/LEAST in search_dense) ────────


class TestScoreClamping:
    """Regression: pgvector SQL clamps scores to [0, 1] via GREATEST/LEAST."""

    def test_dense_select_contains_clamp(self) -> None:
        from retrieval.pgvector_store import _DENSE_SELECT

        assert "GREATEST(0.0, LEAST(1.0," in _DENSE_SELECT
        assert "$1::vector" in _DENSE_SELECT

    def test_lexical_select_no_clamp(self) -> None:
        """Similarity() natively returns [0, 1], so no clamp needed."""
        from retrieval.pgvector_store import _LEXICAL_SELECT

        assert "GREATEST" not in _LEXICAL_SELECT

    @pytest.mark.asyncio
    async def test_dense_score_over_one_clamped_in_sql(self) -> None:
        """If DB returns raw score outside [0,1], _row_to_scored stores it as-is
        (clamping is SQL-level only)."""
        from retrieval.pgvector_store import PgVectorStore

        pool = AsyncMock()
        mock_tx = AsyncMock()
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[
            {"id": "c1", "paper_id": "p1", "text": "t", "section": None,
             "page": 1, "char_start": 0, "char_end": 1, "score": 1.2},
        ])
        conn.execute = AsyncMock(return_value="SET")
        conn.transaction = MagicMock(return_value=mock_tx)
        cm = AsyncMock()
        cm.__aenter__.return_value = conn
        pool.connection = MagicMock(return_value=cm)
        store = PgVectorStore(pool)
        result = await store.search_dense([0.1, 0.2], k=10)
        assert result[0].score == 1.2

    @pytest.mark.asyncio
    async def test_dense_score_negative_passes_through(self) -> None:
        from retrieval.pgvector_store import PgVectorStore

        pool = AsyncMock()
        mock_tx = AsyncMock()
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[
            {"id": "c1", "paper_id": "p1", "text": "t", "section": None,
             "page": 1, "char_start": 0, "char_end": 1, "score": -0.5},
        ])
        conn.execute = AsyncMock(return_value="SET")
        conn.transaction = MagicMock(return_value=mock_tx)
        cm = AsyncMock()
        cm.__aenter__.return_value = conn
        pool.connection = MagicMock(return_value=cm)
        store = PgVectorStore(pool)
        result = await store.search_dense([0.1, 0.2], k=10)
        assert result[0].score == -0.5


# ── Upsert idempotency ────────────────────────────────────────────────────────


class TestUpsertIdempotency:
    """ON CONFLICT pattern must be present in upsert SQL constants."""

    def test_upsert_chunk_sql_has_on_conflict(self) -> None:
        from core._sql import UPSERT_CHUNK_SQL
        assert "ON CONFLICT (id)" in UPSERT_CHUNK_SQL

    def test_upsert_embedding_sql_has_on_conflict(self) -> None:
        from core._sql import UPSERT_EMBEDDING_SQL
        assert "ON CONFLICT (chunk_id)" in UPSERT_EMBEDDING_SQL

    def test_upsert_paper_sql_has_on_conflict(self) -> None:
        from core._sql import UPSERT_PAPER_SQL
        assert "ON CONFLICT (id)" in UPSERT_PAPER_SQL


# ── ChunkRef + ScoredChunk invariants ────────────────────────────────────────


class TestChunkRefIdentity:
    def test_scored_chunk_carries_chunk_id_and_paper_id(self) -> None:
        cr = type("CR", (), {"chunk_id": "cid", "paper_id": "pid", "text": "t",
                             "section": None, "page": 1, "char_start": 0, "char_end": 10})()
        sc = ScoredChunk(chunk=cr, score=0.5, channel="dense")
        assert sc.chunk.chunk_id == "cid"
        assert sc.chunk.paper_id == "pid"

    def test_upsert_chunk_carries_metadata(self) -> None:
        u = UpsertChunk("cid", "pid", "text", [0.1, 0.2], {"page": 1})
        assert u.chunk_id == "cid"
        assert u.paper_id == "pid"
        assert u.metadata["page"] == 1
