"""Tests for transactional document persistence (embed → paper → upsert).

Key invariant: a partial embedding failure must leave zero database state.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import pytest

from core.errors import EmbeddingError, IngestError
from core.models import Chunk, Paper, UpsertChunk
from ingest.persist import (
    _build_upsert_items,
    _embed_all,
    persist_document,
)

# --------------------------------------------------------------------------
# Mock helpers
# --------------------------------------------------------------------------


class _FakePool:
    """Records executed SQL for inspection."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, ...]] = []

    async def execute(self, query: str, *params: object) -> str:
        self.executed.append((query, *(str(p) for p in params)))
        return "OK"


class _FakeRetriever:
    """Records upserted items; simulates success or failure."""

    def __init__(self, *, fail: bool = False) -> None:
        self.upserted: list[list[UpsertChunk]] = []
        self._fail = fail

    async def upsert(self, items: Sequence[UpsertChunk]) -> int:
        if self._fail:
            msg = "Simulated retriever failure"
            raise RuntimeError(msg)
        self.upserted.append(list(items))
        return len(items)

    async def delete_by_paper(self, paper_id: str) -> int:
        return 0

    async def count(self) -> int:
        return 0

    async def health(self) -> bool:
        return True

    async def search_dense(self, *args: object, **kwargs: object) -> list:
        return []

    async def search_lexical(self, *args: object, **kwargs: object) -> list:
        return []


class _OkEmbedder:
    """Returns a zero-vector for every input text."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[Sequence[float]]:
        self.calls.append(texts)
        return [[0.0] * self._dim for _ in texts]


class _FailingEmbedder:
    """Succeeds for N calls, then raises EmbeddingError."""

    def __init__(self, succeed_calls: int = 1, dim: int = 4) -> None:
        self._succeed_calls = succeed_calls
        self._dim = dim
        self.call_count = 0

    async def embed(self, texts: list[str]) -> list[Sequence[float]]:
        self.call_count += 1
        if self.call_count > self._succeed_calls:
            msg = "Simulated embedding failure"
            raise EmbeddingError(msg)
        return [[float(self.call_count)] * self._dim for _ in texts]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def paper() -> Paper:
    return Paper(
        id=str(uuid.uuid4()),
        doi=None,
        title="Persist Test",
        authors=[],
        year=None,
        venue=None,
        abstract=None,
        source="test",
        open_access=None,
        pdf_path=None,
        grobid_tei=None,
        meta={},
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(
            id=str(uuid.uuid4()),
            paper_id="",
            ordinal=i,
            section="Test",
            text=f"chunk {i} content " * 20,
            char_start=i * 100,
            char_end=(i + 1) * 100,
            page=0,
            token_count=10,
            content_hash=f"hash{i:04x}",
        )
        for i in range(6)
    ]


# --------------------------------------------------------------------------
# _embed_all
# --------------------------------------------------------------------------


class TestEmbedAll:
    pytestmark = pytest.mark.asyncio

    async def test_embeds_in_batches(self) -> None:
        embedder = _OkEmbedder()
        chunks = [
            Chunk(id=str(uuid.uuid4()), paper_id="p", ordinal=i,
                  section=None, text=f"t{i}", char_start=None,
                  char_end=None, page=None, token_count=None,
                  content_hash=f"h{i}")
            for i in range(5)
        ]
        vectors = await _embed_all(embedder, chunks, batch_size=2)

        assert len(vectors) == 5
        assert len(embedder.calls) == 3  # 2 + 2 + 1

    async def test_failure_mid_batch_propagates(self) -> None:
        embedder = _FailingEmbedder(succeed_calls=1)
        chunks = [
            Chunk(id=str(uuid.uuid4()), paper_id="p", ordinal=i,
                  section=None, text=f"t{i}", char_start=None,
                  char_end=None, page=None, token_count=None,
                  content_hash=f"h{i}")
            for i in range(5)
        ]
        with pytest.raises(EmbeddingError):
            await _embed_all(embedder, chunks, batch_size=2)


# --------------------------------------------------------------------------
# _build_upsert_items
# --------------------------------------------------------------------------


class TestBuildUpsertItems:
    """Sync test — no asyncio marker needed."""

    def test_maps_chunk_to_upsert(self) -> None:
        c = Chunk(
            id="c1", paper_id="p1", ordinal=3, section="Methods",
            text="some text", char_start=10, char_end=50,
            page=2, token_count=8, content_hash="abc",
        )
        items = _build_upsert_items("p1", [c], [[0.1, 0.2]])

        assert len(items) == 1
        item = items[0]
        assert item.chunk_id == "c1"
        assert item.paper_id == "p1"
        assert item.text == "some text"
        assert list(item.vector) == [0.1, 0.2]
        assert item.metadata["ordinal"] == 3
        assert item.metadata["section"] == "Methods"
        assert item.metadata["char_start"] == 10
        assert item.metadata["char_end"] == 50
        assert item.metadata["page"] == 2
        assert item.metadata["token_count"] == 8
        assert item.metadata["content_hash"] == "abc"


# --------------------------------------------------------------------------
# persist_document — success path
# --------------------------------------------------------------------------


class TestPersistSuccess:
    pytestmark = pytest.mark.asyncio

    async def test_returns_chunk_count(
        self, paper: Paper, chunks: list[Chunk],
    ) -> None:
        pool = _FakePool()
        retriever = _FakeRetriever()
        embedder = _OkEmbedder()

        count = await persist_document(pool, retriever, embedder, paper, chunks)
        assert count == len(chunks)

    async def test_inserts_paper_row(
        self, paper: Paper, chunks: list[Chunk],
    ) -> None:
        pool = _FakePool()
        retriever = _FakeRetriever()
        embedder = _OkEmbedder()

        await persist_document(pool, retriever, embedder, paper, chunks)

        insert_calls = [
            q for q, *_ in pool.executed if q.startswith("INSERT INTO papers")
        ]
        assert len(insert_calls) == 1

    async def test_calls_retriever_upsert(
        self, paper: Paper, chunks: list[Chunk],
    ) -> None:
        pool = _FakePool()
        retriever = _FakeRetriever()
        embedder = _OkEmbedder()

        await persist_document(pool, retriever, embedder, paper, chunks)

        assert len(retriever.upserted) == 1
        assert len(retriever.upserted[0]) == len(chunks)


# --------------------------------------------------------------------------
# Partial embedding failure — no database state
# --------------------------------------------------------------------------


class TestPartialEmbeddingFailure:
    pytestmark = pytest.mark.asyncio

    async def test_raises_ingest_error(
        self, paper: Paper, chunks: list[Chunk],
    ) -> None:
        pool = _FakePool()
        retriever = _FakeRetriever()
        failing = _FailingEmbedder(succeed_calls=0)

        with pytest.raises(IngestError, match="Embedding failed"):
            await persist_document(pool, retriever, failing, paper, chunks)

    async def test_no_paper_inserted(
        self, paper: Paper, chunks: list[Chunk],
    ) -> None:
        pool = _FakePool()
        retriever = _FakeRetriever()
        failing = _FailingEmbedder(succeed_calls=0)

        with pytest.raises(IngestError):
            await persist_document(pool, retriever, failing, paper, chunks)

        insert_calls = [
            q for q, *_ in pool.executed if q.startswith("INSERT INTO papers")
        ]
        assert len(insert_calls) == 0

    async def test_no_retriever_called(
        self, paper: Paper, chunks: list[Chunk],
    ) -> None:
        pool = _FakePool()
        retriever = _FakeRetriever()
        failing = _FailingEmbedder(succeed_calls=0)

        with pytest.raises(IngestError):
            await persist_document(pool, retriever, failing, paper, chunks)

        assert len(retriever.upserted) == 0

    async def test_embedding_failure_midway(
        self, paper: Paper, chunks: list[Chunk],
    ) -> None:
        """Batch 1 succeeds, batch 2 fails — no DB state."""
        pool = _FakePool()
        retriever = _FakeRetriever()
        # succeed first batch (2 chunks), fail on second
        failing = _FailingEmbedder(succeed_calls=1)

        with pytest.raises(IngestError):
            await persist_document(
                pool, retriever, failing, paper, chunks,
                batch_size=4,
            )

        assert len(pool.executed) == 0  # nothing touched DB
        assert len(retriever.upserted) == 0

    async def test_embedding_failure_on_first_batch(
        self, paper: Paper, chunks: list[Chunk],
    ) -> None:
        """First batch fails — nothing touches DB."""
        pool = _FakePool()
        retriever = _FakeRetriever()
        failing = _FailingEmbedder(succeed_calls=0)  # fail immediately

        with pytest.raises(IngestError):
            await persist_document(
                pool, retriever, failing, paper, chunks,
            )

        assert len(pool.executed) == 0
        assert len(retriever.upserted) == 0


# --------------------------------------------------------------------------
# Retriever failure — paper is cleaned up
# --------------------------------------------------------------------------


class TestRetrieverFailure:
    pytestmark = pytest.mark.asyncio

    async def test_raises_ingest_error(
        self, paper: Paper, chunks: list[Chunk],
    ) -> None:
        pool = _FakePool()
        retriever = _FakeRetriever(fail=True)
        embedder = _OkEmbedder()

        with pytest.raises(IngestError, match="Failed to persist paper"):
            await persist_document(pool, retriever, embedder, paper, chunks)

    async def test_cleans_up_paper(
        self, paper: Paper, chunks: list[Chunk],
    ) -> None:
        pool = _FakePool()
        retriever = _FakeRetriever(fail=True)
        embedder = _OkEmbedder()

        with pytest.raises(IngestError):
            await persist_document(pool, retriever, embedder, paper, chunks)

        delete_calls = [
            q for q, *_ in pool.executed if q.startswith("DELETE FROM papers")
        ]
        assert len(delete_calls) == 1
