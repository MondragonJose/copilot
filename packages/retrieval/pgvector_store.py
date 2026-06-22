"""PgVectorStore — concrete Retriever implementation backed by asyncpg + pgvector.

Writes to the ``chunks`` and ``embeddings`` tables defined in
``migrations/002_chunks.sql`` and ``migrations/003_embeddings.sql``.
Every upsert is idempotent via ``INSERT … ON CONFLICT (id) DO UPDATE``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from core._sql import UPSERT_CHUNK_SQL, UPSERT_EMBEDDING_SQL
from core.errors import RetrievalError
from core.models import ChunkRef, ScoredChunk, UpsertChunk
from retrieval.db import Pool

_EF_SEARCH_MULTIPLIER = 5
_EF_SEARCH_MIN = 40

_DENSE_SELECT = (
    "SELECT c.id, c.paper_id, c.text, c.section, c.page, "
    "       c.char_start, c.char_end, "
    "       GREATEST(0.0, LEAST(1.0, "
    "           1.0 - (e.vector <=> $1::vector) / 2.0"
    "       )) AS score "
)
_DENSE_FROM = "FROM chunks c JOIN embeddings e ON c.id = e.chunk_id "
_LEXICAL_SELECT = (
    "SELECT id, paper_id, text, section, page, "
    "       char_start, char_end, "
    "       similarity(text, $1::text) AS score "
)
_LEXICAL_FROM = "FROM chunks "


class PgVectorStore:
    """Retriever implementation on top of asyncpg + pgvector."""

    def __init__(
        self,
        pool: Pool,
        model: str = "bge-m3",
        dim: int = 1024,
    ) -> None:
        self._pool = pool
        self._model = model
        self._dim = dim

    async def upsert(self, items: Sequence[UpsertChunk]) -> int:
        """Idempotent upsert of chunks + embedding rows.

        Every item is written in two statements (chunk row then embedding row),
        each guarded by ``ON CONFLICT``.  Returns the total number of items
        processed (every item produces exactly one chunk row).
        """
        affected = 0
        vector: list[float]
        for item in items:
            vector = list(item.vector)
            md = item.metadata

            await self._pool.execute(
                UPSERT_CHUNK_SQL,
                item.chunk_id,
                item.paper_id,
                md.get("ordinal", 0),
                md.get("section"),
                item.text,
                md.get("char_start"),
                md.get("char_end"),
                md.get("page"),
                md.get("token_count"),
                md.get("content_hash", ""),
            )

            await self._pool.execute(
                UPSERT_EMBEDDING_SQL,
                item.chunk_id,
                self._model,
                self._dim,
                vector,
            )

            affected += 1

        return affected

    async def delete_by_paper(self, paper_id: str) -> int:
        """Delete all chunks (and cascade to embeddings) for *paper_id*.

        Returns the number of chunk rows deleted.
        """
        rows = await self._pool.fetch(
            "DELETE FROM chunks WHERE paper_id = $1::uuid RETURNING 1",
            paper_id,
        )
        return len(rows)

    async def count(self) -> int:
        """Total number of indexed chunks."""
        row = await self._pool.fetchrow("SELECT COUNT(*) AS cnt FROM chunks")
        return row["cnt"] if row is not None else 0

    async def health(self) -> bool:
        """Whether the underlying store is reachable."""
        try:
            await self._pool.execute("SELECT 1")
            return True
        except RetrievalError:
            return False

    async def search_dense(
        self,
        query_vector: Sequence[float],
        k: int,
        paper_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        """k-NN cosine similarity search via pgvector.

        Converts cosine distance ``d`` to similarity ``1 - d/2`` and clamps
        to ``[0, 1]``.  Sets ``hnsw.ef_search`` per-connection for the HNSW
        index.  Filters by ``paper_ids`` when provided.
        """
        if k <= 0:
            return []

        vector = list(query_vector)
        ef_search = max(k * _EF_SEARCH_MULTIPLIER, _EF_SEARCH_MIN)

        where = ""
        params: list[object] = [vector, k]
        if paper_ids is not None:
            where = "WHERE c.paper_id = ANY($3::uuid[]) "
            params.append(list(paper_ids))

        sql = (
            _DENSE_SELECT
            + _DENSE_FROM
            + where
            + "ORDER BY e.vector <=> $1::vector "
            + "LIMIT $2"
        )

        async with self._pool.connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    f"SET LOCAL hnsw.ef_search = {ef_search}"
                )
                rows = await conn.fetch(sql, *params)

        return [self._row_to_scored(row, "dense") for row in rows]

    @staticmethod
    def _row_to_scored(row: Any, channel: str) -> ScoredChunk:
        return ScoredChunk(
            chunk=ChunkRef(
                chunk_id=str(row["id"]),
                paper_id=str(row["paper_id"]),
                text=row["text"],
                section=row["section"],
                page=row["page"],
                char_start=row["char_start"],
                char_end=row["char_end"],
            ),
            score=float(row["score"]),
            channel=channel,
        )

    async def search_lexical(
        self,
        query_text: str,
        k: int,
        paper_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        """Trigram similarity search via ``pg_trgm.similarity()``.

        ``similarity()`` natively returns ``[0, 1]``.  Filters by
        ``paper_ids`` when provided.
        """
        where = ""
        params: list[object] = [query_text, k]
        if paper_ids is not None:
            where = "WHERE paper_id = ANY($3::uuid[]) "
            params.append(list(paper_ids))

        sql = (
            _LEXICAL_SELECT
            + _LEXICAL_FROM
            + where
            + "ORDER BY score DESC "
            + "LIMIT $2"
        )
        rows = await self._pool.fetch(sql, *params)

        return [self._row_to_scored(row, "lexical") for row in rows]
