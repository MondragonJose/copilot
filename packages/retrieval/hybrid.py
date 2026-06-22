"""Hybrid (dense + lexical) search via Reciprocal Rank Fusion.

Fusion lives here, not in the store.  Callers pass pre-computed dense and
lexical results; when ``use_hybrid`` is ``False`` (the default), the raw
dense list is returned unchanged.  When ``True``, ``rrf_fuse`` combines both
rankings with the RRF formula ``score = Σ 1 / (k + rank)``.

Usage::

    from retrieval.hybrid import rrf_fuse

    dense = await retriever.search_dense(vector, k=10)
    if use_hybrid:
        lexical = await retriever.search_lexical(text, k=10)
        results = rrf_fuse(dense, lexical, top_n=10)
    else:
        results = dense
"""

from __future__ import annotations

from collections.abc import Sequence

from core.models import ChunkRef, ScoredChunk


def rrf_fuse(
    dense: Sequence[ScoredChunk],
    lexical: Sequence[ScoredChunk],
    *,
    k: float = 60.0,
    top_n: int | None = None,
) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion of dense and lexical result lists.

    Parameters
    ----------
    dense, lexical:
        Input ranked lists from ``search_dense`` / ``search_lexical``.
    k:
        RRF constant (default 60, the original metaphone paper value).
    top_n:
        Maximum number of items to return.  ``None`` means no limit.

    Returns
    -------
    list[ScoredChunk]
        Fused results ordered by RRF score descending.  Ties are broken by
        ``chunk_id`` (deterministic).  The ``channel`` of every returned item
        is ``"hybrid"``.

    Notes
    -----
    Items that appear in only one list still receive a partial RRF score.
    Duplicate ``chunk_id`` entries within a single list are ignored (only the
    first occurrence contributes).
    """
    # ── assign rank positions ──────────────────────────────────────
    seen: set[str] = set()
    dense_ranks: dict[str, int] = {}
    for pos, item in enumerate(dense):
        cid = item.chunk.chunk_id
        if cid not in seen:
            dense_ranks[cid] = pos + 1  # 1-indexed
            seen.add(cid)

    seen.clear()
    lexical_ranks: dict[str, int] = {}
    for pos, item in enumerate(lexical):
        cid = item.chunk.chunk_id
        if cid not in seen:
            lexical_ranks[cid] = pos + 1
            seen.add(cid)

    # ── compute RRF scores ────────────────────────────────────────
    all_ids = set(dense_ranks) | set(lexical_ranks)
    rrf_scores: dict[str, float] = {}
    # Store a reference ChunkRef for each chunk_id (first encountered wins)
    chunk_refs: dict[str, ChunkRef] = {}

    for item in dense:
        cid = item.chunk.chunk_id
        if cid not in chunk_refs:
            chunk_refs[cid] = item.chunk
    for item in lexical:
        cid = item.chunk.chunk_id
        if cid not in chunk_refs:
            chunk_refs[cid] = item.chunk

    for cid in all_ids:
        score = 0.0
        if cid in dense_ranks:
            score += 1.0 / (k + dense_ranks[cid])
        if cid in lexical_ranks:
            score += 1.0 / (k + lexical_ranks[cid])
        rrf_scores[cid] = score

    # ── sort by RRF score descending, tie-break by chunk_id ───────
    sorted_ids = sorted(
        rrf_scores,
        key=lambda cid: (-rrf_scores[cid], cid),
    )

    if top_n is not None:
        sorted_ids = sorted_ids[:top_n]

    return [
        ScoredChunk(
            chunk=chunk_refs[cid],
            score=rrf_scores[cid],
            channel="hybrid",
        )
        for cid in sorted_ids
    ]
