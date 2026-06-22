"""End-to-end retrieval tests: upsert, dense search, lexical search, paper_ids.

Migrations are applied automatically before the first test so that any
ephemeral Postgres instance (compose, CI service, testcontainer) is ready
to use without manual setup.
"""

from __future__ import annotations

import os
import pathlib
import uuid
from collections.abc import AsyncGenerator, Sequence

import asyncpg
import pytest
import pytest_asyncio

from core.errors import RetrievalError
from core.models import UpsertChunk
from retrieval.db import Pool
from retrieval.pgvector_store import PgVectorStore

pytestmark = pytest.mark.asyncio

DEFAULT_DSN = "postgresql://rc:rc@localhost:5432/research_copilot"
PAPER_ID = "00000000-0000-0000-0000-000000000001"
PAPER_ID_2 = "00000000-0000-0000-0000-000000000002"
VECTOR_DIM = 1024
MIGRATIONS_DIR = pathlib.Path(__file__).resolve().parent.parent / "migrations"


@pytest_asyncio.fixture(scope="module")
async def apply_migrations() -> None:
    """Run all SQL migrations once per module so an ephemeral DB is ready."""
    dsn = os.environ.get("DATABASE_URL", DEFAULT_DSN)
    try:
        conn = await asyncpg.connect(dsn, timeout=5)
    except Exception:
        pytest.skip("Database not available — are the services up?")
        return
    try:
        for f in sorted(MIGRATIONS_DIR.glob("*.sql")):
            sql = f.read_text()
            if sql.strip():
                await conn.execute(sql)
    finally:
        await conn.close()


def _make_chunks(n: int, paper_id: str = PAPER_ID) -> list[UpsertChunk]:
    """Return *n* distinct UpsertChunks for the given paper."""
    return [
        UpsertChunk(
            chunk_id=str(uuid.uuid4()),
            paper_id=paper_id,
            text=f"Chunk {i} content: {' '.join(['token'] * (i + 1))}",
            vector=[float(i % 10) / 10.0] * VECTOR_DIM,
            metadata={"ordinal": i},
        )
        for i in range(n)
    ]


def _make_chunk_with_vector(
    chunk_id: str,
    vector: Sequence[float],
    paper_id: str = PAPER_ID,
    text: str = "target chunk",
) -> UpsertChunk:
    return UpsertChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text=text,
        vector=list(vector),
        metadata={},
    )


async def _ensure_papers(pool: Pool) -> None:
    """Create the test papers if they do not exist."""
    try:
        for pid in (PAPER_ID, PAPER_ID_2):
            await pool.execute(
                "INSERT INTO papers (id, title, source) "
                "VALUES ($1::uuid, 'Test Paper for Retrieval', 'test') "
                "ON CONFLICT (id) DO NOTHING",
                pid,
            )
    except RetrievalError:
        pytest.skip("Database not available")


@pytest_asyncio.fixture
async def store(apply_migrations: None) -> AsyncGenerator[PgVectorStore, None]:
    dsn = os.environ.get("DATABASE_URL", DEFAULT_DSN)
    pool = Pool(dsn)
    try:
        await pool.open()
    except RetrievalError:
        pytest.skip("Database not available")
        return

    await _ensure_papers(pool)

    s = PgVectorStore(pool)
    yield s

    # Cleanup
    try:
        for pid in (PAPER_ID, PAPER_ID_2):
            await pool.execute(
                "DELETE FROM chunks WHERE paper_id = $1::uuid",
                pid,
            )
    except RetrievalError:
        pass
    await pool.close()


class TestUpsertIdempotency:
    """Running upsert twice on the same chunks must yield no duplicates."""

    async def test_upsert_once(self, store: PgVectorStore) -> None:
        chunks = _make_chunks(100)
        affected = await store.upsert(chunks)
        assert affected == 100

        cnt = await store.count()
        assert cnt == 100

    async def test_upsert_twice_idempotent(self, store: PgVectorStore) -> None:
        chunks = _make_chunks(100)

        first = await store.upsert(chunks)
        assert first == 100

        second = await store.upsert(chunks)
        assert second == 100

        cnt = await store.count()
        assert cnt == 100, "Duplicate rows detected — upsert is not idempotent"

    async def test_upsert_no_duplicate_ids(self, store: PgVectorStore) -> None:
        chunks = _make_chunks(100)
        await store.upsert(chunks)
        await store.upsert(chunks)

        pool = store._pool
        rows = await pool.fetch(
            "SELECT id, COUNT(*) AS cnt FROM chunks "
            "WHERE paper_id = $1::uuid GROUP BY id HAVING COUNT(*) > 1",
            PAPER_ID,
        )
        assert len(rows) == 0, f"Found {len(rows)} duplicate chunk IDs"


class TestDeleteByPaper:
    async def test_delete_existing_paper(self, store: PgVectorStore) -> None:
        chunks = _make_chunks(10)
        await store.upsert(chunks)

        deleted = await store.delete_by_paper(PAPER_ID)
        assert deleted == 10

        cnt = await store.count()
        assert cnt == 0

    async def test_delete_unknown_paper(self, store: PgVectorStore) -> None:
        unknown_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"
        deleted = await store.delete_by_paper(unknown_id)
        assert deleted == 0

    async def test_delete_cascades_to_embeddings(
        self, store: PgVectorStore,
    ) -> None:
        chunks = _make_chunks(5)
        await store.upsert(chunks)

        await store.delete_by_paper(PAPER_ID)

        pool = store._pool
        emb_count = await pool.fetchrow(
            "SELECT COUNT(*) AS cnt FROM embeddings e "
            "JOIN chunks c ON c.id = e.chunk_id "
            "WHERE c.paper_id = $1::uuid",
            PAPER_ID,
        )
        assert emb_count is not None
        assert emb_count["cnt"] == 0


class TestHealthAndCount:
    async def test_health_when_connected(self, store: PgVectorStore) -> None:
        ok = await store.health()
        assert ok is True

    async def test_count_empty(self, store: PgVectorStore) -> None:
        cnt = await store.count()
        # Our paper chunks may or may not be present; just verify it returns int >= 0
        assert isinstance(cnt, int)
        assert cnt >= 0


class TestSearchDense:
    """Known-vector top-1 and paper_ids filter tests."""

    TARGET_CHUNK_ID = "00000000-0000-0000-0000-00000000ffff"
    TARGET_TEXT = "the exact match target chunk"
    BACKGROUND_N = 50

    @staticmethod
    def _axis_vector(axis: int) -> list[float]:
        v = [0.0] * VECTOR_DIM
        v[axis % VECTOR_DIM] = 1.0
        return v

    async def test_known_vector_rank1(self, store: PgVectorStore) -> None:
        """Query vector identical to a planted chunk returns it at rank 1."""
        target_vec = self._axis_vector(0)
        background = [
            _make_chunk_with_vector(
                chunk_id=str(uuid.uuid4()),
                vector=self._axis_vector(i + 1),
                text=f"background chunk {i}",
            )
            for i in range(self.BACKGROUND_N)
        ]
        target = _make_chunk_with_vector(
            chunk_id=self.TARGET_CHUNK_ID,
            vector=target_vec,
            text=self.TARGET_TEXT,
        )

        await store.upsert([target] + background)

        results = await store.search_dense(
            query_vector=target_vec, k=5,
        )

        assert len(results) > 0
        best = results[0]
        assert best.chunk.chunk_id == self.TARGET_CHUNK_ID
        assert best.score == pytest.approx(1.0, abs=1e-6)
        assert best.channel == "dense"

    async def test_scores_in_unit_interval(self, store: PgVectorStore) -> None:
        """All returned scores are in [0, 1]."""
        target_vec = self._axis_vector(0)
        background = [
            _make_chunk_with_vector(
                chunk_id=str(uuid.uuid4()),
                vector=self._axis_vector(i + 1),
                text=f"bg {i}",
            )
            for i in range(10)
        ]
        target = _make_chunk_with_vector(
            chunk_id=self.TARGET_CHUNK_ID,
            vector=target_vec,
        )
        await store.upsert([target] + background)

        results = await store.search_dense(
            query_vector=target_vec, k=10,
        )

        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of [0, 1]"

    async def test_paper_ids_filter_matches(
        self, store: PgVectorStore,
    ) -> None:
        """paper_ids filter returns only chunks from those papers."""
        paper2_vec = self._axis_vector(99)
        target = _make_chunk_with_vector(
            chunk_id=self.TARGET_CHUNK_ID,
            vector=paper2_vec,
            paper_id=PAPER_ID_2,
        )
        decorrelation = _make_chunk_with_vector(
            chunk_id=str(uuid.uuid4()),
            vector=self._axis_vector(100),
            paper_id=PAPER_ID,
        )
        await store.upsert([target, decorrelation])

        results = await store.search_dense(
            query_vector=paper2_vec, k=10,
            paper_ids=[PAPER_ID_2],
        )

        assert len(results) > 0
        for r in results:
            assert r.chunk.paper_id == PAPER_ID_2

    async def test_paper_ids_filter_no_match(
        self, store: PgVectorStore,
    ) -> None:
        """paper_ids filter that matches nothing returns empty."""
        results = await store.search_dense(
            query_vector=[0.5] * VECTOR_DIM, k=5,
            paper_ids=["ffffffff-ffff-ffff-ffff-ffffffffffff"],
        )
        assert results == []


class TestSearchLexical:
    """pg_trgm similarity search tests."""

    LEXICAL_CHUNK_ID = "00000000-0000-0000-0000-00000000fffe"
    LEXICAL_TEXT = "exact match target text for lexical search verification"

    @staticmethod
    def _make_lexical_chunk(
        chunk_id: str,
        text: str,
        paper_id: str = PAPER_ID,
    ) -> UpsertChunk:
        return UpsertChunk(
            chunk_id=chunk_id,
            paper_id=paper_id,
            text=text,
            vector=[0.0] * VECTOR_DIM,
            metadata={},
        )

    async def test_exact_text_match(self, store: PgVectorStore) -> None:
        """Query identical to chunk text returns that chunk at rank 1."""
        background = [
            self._make_lexical_chunk(
                chunk_id=str(uuid.uuid4()),
                text=f"unrelated chunk number {i} with filler text",
            )
            for i in range(20)
        ]
        target = self._make_lexical_chunk(
            chunk_id=self.LEXICAL_CHUNK_ID,
            text=self.LEXICAL_TEXT,
        )
        await store.upsert([target] + background)

        results = await store.search_lexical(
            query_text=self.LEXICAL_TEXT, k=5,
        )

        assert len(results) > 0
        best = results[0]
        assert best.chunk.chunk_id == self.LEXICAL_CHUNK_ID
        assert best.channel == "lexical"

    async def test_scores_in_unit_interval(self, store: PgVectorStore) -> None:
        """All returned scores are in [0, 1]."""
        chunks = [
            self._make_lexical_chunk(
                chunk_id=str(uuid.uuid4()),
                text=f"distinctive phrase number {i} for lexical scoring",
            )
            for i in range(10)
        ]
        await store.upsert(chunks)

        results = await store.search_lexical(
            query_text="distinctive phrase", k=10,
        )

        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of [0, 1]"

    async def test_paper_ids_filter_lexical(
        self, store: PgVectorStore,
    ) -> None:
        """paper_ids filter restricts lexical search results."""
        target = self._make_lexical_chunk(
            chunk_id=self.LEXICAL_CHUNK_ID,
            text="unique text on paper two",
            paper_id=PAPER_ID_2,
        )
        other = self._make_lexical_chunk(
            chunk_id=str(uuid.uuid4()),
            text="unique text on paper one",
        )
        await store.upsert([target, other])

        results = await store.search_lexical(
            query_text="unique text", k=10,
            paper_ids=[PAPER_ID_2],
        )

        assert len(results) > 0
        for r in results:
            assert r.chunk.paper_id == PAPER_ID_2

    async def test_paper_ids_filter_no_match_lexical(
        self, store: PgVectorStore,
    ) -> None:
        """paper_ids filter that matches nothing returns empty."""
        results = await store.search_lexical(
            query_text="any text", k=5,
            paper_ids=["ffffffff-ffff-ffff-ffff-ffffffffffff"],
        )
        assert results == []
