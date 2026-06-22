"""Transactional document persistence — embeds chunks, saves paper + chunks.

Invariant: the paper is never visible unless ALL chunks were successfully
embedded AND persisted.  Partial embedding failures raise ``IngestError``
without touching the database.
"""

from __future__ import annotations

from collections.abc import Sequence

import asyncpg

from core._constants import DEFAULT_BATCH_SIZE
from core._sql import UPSERT_PAPER_SQL
from core.errors import EmbeddingError, IngestError
from core.interfaces import Embedder, Retriever
from core.models import Chunk, Paper, UpsertChunk
from retrieval.db import Pool


async def persist_document(
    pool: Pool,
    retriever: Retriever,
    embedder: Embedder,
    paper: Paper,
    chunks: list[Chunk],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Persist a parsed paper together with its chunk embeddings.

    Steps
    -----
    1. Embed all chunk texts in batches.
    2. Insert the ``paper`` row.
    3. Upsert all chunks + embeddings via ``retriever.upsert()``.

    Invariant
    ---------
    If any step fails the database is left in a clean state — either the
    paper exists with all its chunks or it does not exist at all.

    Returns
    -------
    Number of chunks persisted (always ``len(chunks)`` on success).

    Raises
    ------
    IngestError
        Wraps any embedding or persistence failure.
    """
    # ------------------------------------------------------------------
    # 1. Embed — database is NOT touched until all vectors are ready
    # ------------------------------------------------------------------
    try:
        vectors = await _embed_all(embedder, chunks, batch_size)
    except EmbeddingError as exc:
        raise IngestError(f"Embedding failed — paper not persisted: {exc}") from exc

    # ------------------------------------------------------------------
    # 2. Build UpsertChunk objects
    # ------------------------------------------------------------------
    upsert_items = _build_upsert_items(paper.id, chunks, vectors)

    # ------------------------------------------------------------------
    # 3. Persist paper in a transaction
    # ------------------------------------------------------------------
    async with pool.connection() as conn:
        async with conn.transaction():
            try:
                await _insert_paper_conn(conn, paper)
            except Exception as exc:
                raise IngestError(
                    f"Failed to persist paper {paper.id}: {exc}",
                ) from exc

    # ------------------------------------------------------------------
    # 4. Upsert chunks via retriever (idempotent per chunk)
    # ------------------------------------------------------------------
    try:
        await retriever.upsert(upsert_items)
    except Exception as exc:
        await _delete_paper(pool, paper.id)
        raise IngestError(
            f"Failed to persist paper {paper.id}: {exc}",
        ) from exc

    return len(upsert_items)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _embed_all(
    embedder: Embedder,
    chunks: list[Chunk],
    batch_size: int,
) -> list[Sequence[float]]:
    """Embed all chunk texts; raises EmbeddingError on any failure."""
    all_vectors: list[Sequence[float]] = []
    for i in range(0, len(chunks), batch_size):
        batch = [c.text for c in chunks[i:i + batch_size]]
        vectors = await embedder.embed(batch)
        all_vectors.extend(vectors)
    return all_vectors


def _build_upsert_items(
    paper_id: str,
    chunks: list[Chunk],
    vectors: list[Sequence[float]],
) -> list[UpsertChunk]:
    """Convert ``Chunk`` objects to ``UpsertChunk`` with embedding vectors."""
    return [
        UpsertChunk(
            chunk_id=c.id,
            paper_id=paper_id,
            text=c.text,
            vector=vectors[i],
            metadata={
                "ordinal": c.ordinal,
                "section": c.section,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "page": c.page,
                "token_count": c.token_count,
                "content_hash": c.content_hash,
            },
        )
        for i, c in enumerate(chunks)
    ]


async def _insert_paper_conn(conn: asyncpg.Connection, paper: Paper) -> None:
    await conn.execute(
        UPSERT_PAPER_SQL,
        paper.id,
        paper.doi,
        paper.title,
        paper.authors,
        paper.year,
        paper.venue,
        paper.abstract,
        paper.source,
        paper.open_access,
        paper.pdf_path,
        paper.grobid_tei,
        paper.meta,
    )


async def _insert_paper(pool: Pool, paper: Paper) -> None:
    await pool.execute(
        UPSERT_PAPER_SQL,
        paper.id,
        paper.doi,
        paper.title,
        paper.authors,
        paper.year,
        paper.venue,
        paper.abstract,
        paper.source,
        paper.open_access,
        paper.pdf_path,
        paper.grobid_tei,
        paper.meta,
    )


async def _delete_paper(pool: Pool, paper_id: str) -> None:
    """Delete paper (cascades to chunks + embeddings)."""
    await pool.execute(
        "DELETE FROM papers WHERE id = $1::uuid",
        paper_id,
    )
