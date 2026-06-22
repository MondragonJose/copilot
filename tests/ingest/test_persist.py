"""Tests for ingest persistence — all I/O mocked."""

from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock, patch, ANY

import asyncpg
import pytest

from core.errors import EmbeddingError, IngestError
from core.interfaces import Embedder, Retriever
from core.models import Chunk, Paper, UpsertChunk
from ingest.persist import (
    _build_upsert_items,
    _embed_all,
    _insert_paper,
    _insert_paper_conn,
    _upsert_chunk_conn,
    _delete_paper,
    persist_document,
)
from retrieval.db import Pool


def _make_pool() -> MagicMock:
    pool = MagicMock(spec=Pool)

    # connection() returns an async context manager
    conn_cm = MagicMock()
    conn_cm.__aenter__ = AsyncMock(return_value=pool)
    conn_cm.__aexit__ = AsyncMock(return_value=None)
    pool.connection = MagicMock(return_value=conn_cm)

    # transaction on the connection
    trans_cm = MagicMock()
    trans_cm.__aenter__ = AsyncMock(return_value=None)
    trans_cm.__aexit__ = AsyncMock(return_value=None)
    pool.transaction = MagicMock(return_value=trans_cm)

    pool.execute = AsyncMock(return_value="INSERT 1")
    pool.fetch = AsyncMock(return_value=[])
    pool.fetchrow = AsyncMock(return_value=None)
    return pool


@pytest.fixture
def mock_pool() -> MagicMock:
    return _make_pool()


@pytest.fixture
def mock_retriever() -> Retriever:
    r = MagicMock(spec=Retriever)
    r.upsert = AsyncMock(return_value=1)
    return r


@pytest.fixture
def mock_embedder() -> Embedder:
    e = MagicMock(spec=Embedder)
    e.embed = AsyncMock(return_value=[[0.1], [0.2]])
    return e


def _make_paper() -> Paper:
    return Paper(
        id="p1", doi="10.123/test", title="Test", authors=[],
        year=2024, venue=None, abstract=None, source="test",
        open_access=None, pdf_path=None, grobid_tei=None, meta={},
    )


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(id="c1", paper_id="p1", ordinal=0, section=None,
              text="hello", char_start=0, char_end=5,
              page=None, token_count=2, content_hash="abc"),
        Chunk(id="c2", paper_id="p1", ordinal=1, section="Intro",
              text="world", char_start=6, char_end=11,
              page=1, token_count=2, content_hash="def"),
    ]


class TestPersistDocument:
    @pytest.mark.asyncio
    async def test_successful_persist(
        self, mock_pool: MagicMock, mock_retriever: Retriever,
        mock_embedder: Embedder,
    ) -> None:
        paper = _make_paper()
        chunks = _make_chunks()
        count = await persist_document(
            mock_pool, mock_retriever, mock_embedder, paper, chunks,
        )
        assert count == 2

    @pytest.mark.asyncio
    async def test_embedding_failure_raises_ingest_error(
        self, mock_pool: MagicMock, mock_retriever: Retriever,
    ) -> None:
        bad_embedder = MagicMock(spec=Embedder)
        bad_embedder.embed = AsyncMock(
            side_effect=EmbeddingError("OOM"),
        )
        with pytest.raises(IngestError, match="Embedding failed"):
            await persist_document(
                mock_pool, mock_retriever, bad_embedder,
                _make_paper(), _make_chunks(),
            )

    @pytest.mark.asyncio
    async def test_empty_chunks(
        self, mock_pool: MagicMock, mock_retriever: Retriever,
        mock_embedder: Embedder,
    ) -> None:
        count = await persist_document(
            mock_pool, mock_retriever, mock_embedder,
            _make_paper(), [],
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_db_failure_raises_ingest_error(
        self, mock_pool: MagicMock, mock_retriever: Retriever,
        mock_embedder: Embedder,
    ) -> None:
        mock_pool.execute = AsyncMock(
            side_effect=asyncpg.PostgresError("constraint violation"),
        )
        with pytest.raises(IngestError, match="Failed to persist"):
            await persist_document(
                mock_pool, mock_retriever, mock_embedder,
                _make_paper(), _make_chunks(),
            )


class TestEmbedAll:
    @pytest.mark.asyncio
    async def test_batching(self) -> None:
        embedder = MagicMock(spec=Embedder)
        embedder.embed = AsyncMock(return_value=[[0.1]])
        chunks = _make_chunks()
        vectors = await _embed_all(embedder, chunks, batch_size=1)
        assert len(vectors) == 2
        assert embedder.embed.call_count == 2


class TestBuildUpsertItems:
    def test_builds_correctly(self) -> None:
        chunks = _make_chunks()
        vectors = [[0.1], [0.2]]
        items = _build_upsert_items("p1", chunks, vectors)
        assert len(items) == 2
        assert items[0].chunk_id == "c1"
        assert items[0].paper_id == "p1"
        assert items[0].vector == [0.1]
        assert items[0].metadata["ordinal"] == 0
        assert items[1].metadata["section"] == "Intro"

    def test_mismatched_vectors(self) -> None:
        chunks = _make_chunks()
        vectors = [[0.1]]
        with pytest.raises(IndexError):
            _build_upsert_items("p1", chunks, vectors)


class TestUpsertChunkConn:
    @pytest.mark.asyncio
    async def test_executes_sql(self) -> None:
        conn = AsyncMock()
        item = UpsertChunk(
            chunk_id="c1", paper_id="p1", text="hi",
            vector=[0.1], metadata={},
        )
        await _upsert_chunk_conn(conn, item)
        assert conn.execute.call_count == 2


class TestInsertPaperConn:
    @pytest.mark.asyncio
    async def test_executes_sql(self) -> None:
        conn = AsyncMock()
        paper = _make_paper()
        await _insert_paper_conn(conn, paper)
        conn.execute.assert_called_once()


class TestInsertPaper:
    @pytest.mark.asyncio
    async def test_executes(self, mock_pool: MagicMock) -> None:
        paper = _make_paper()
        await _insert_paper(mock_pool, paper)
        mock_pool.execute.assert_called_once()


class TestDeletePaper:
    @pytest.mark.asyncio
    async def test_executes(self, mock_pool: MagicMock) -> None:
        await _delete_paper(mock_pool, "p1")
        mock_pool.execute.assert_called_once()
