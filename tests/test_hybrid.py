"""Unit tests: RRF fusion ordering, dedup, default-off convention."""

from __future__ import annotations

import pytest

from core.models import ChunkRef, ScoredChunk
from retrieval.hybrid import rrf_fuse


def _chunk(chunk_id: str, score: float = 0.0) -> ScoredChunk:
    """Build a minimal ScoredChunk for testing (only *chunk_id* matters in RRF)."""
    ref = ChunkRef(
        chunk_id=chunk_id,
        paper_id="p1",
        text="",
        section=None,
        page=None,
        char_start=None,
        char_end=None,
    )
    return ScoredChunk(chunk=ref, score=score, channel="test")


class TestRrfFuse:
    """RRF ordering and tie-breaking."""

    def test_empty_lists(self) -> None:
        assert rrf_fuse([], []) == []

    def test_only_dense(self) -> None:
        dense = [_chunk("a"), _chunk("b"), _chunk("c")]
        fused = rrf_fuse(dense, [])
        assert len(fused) == 3
        assert [x.chunk.chunk_id for x in fused] == ["a", "b", "c"]
        for x in fused:
            assert x.channel == "hybrid"

    def test_only_lexical(self) -> None:
        lexical = [_chunk("x"), _chunk("y")]
        fused = rrf_fuse([], lexical)
        assert len(fused) == 2
        assert [x.chunk.chunk_id for x in fused] == ["x", "y"]

    def test_identical_lists(self) -> None:
        dense = [_chunk("a"), _chunk("b")]
        lexical = [_chunk("a"), _chunk("b")]
        fused = rrf_fuse(dense, lexical, k=60.0)
        # a: 1/61 + 1/61 ≈ 0.03279
        # b: 1/62 + 1/62 ≈ 0.03226
        assert [x.chunk.chunk_id for x in fused] == ["a", "b"]
        assert fused[0].score > fused[1].score

    def test_different_rankings(self) -> None:
        dense = [_chunk("a"), _chunk("b"), _chunk("c")]
        lexical = [_chunk("c"), _chunk("b"), _chunk("a")]
        fused = rrf_fuse(dense, lexical, k=60.0)

        names = {x.chunk.chunk_id: x.score for x in fused}
        # a: 1/61 + 1/63 ≈ 0.03227
        # b: 1/62 + 1/62 ≈ 0.03226
        # c: 1/63 + 1/61 ≈ 0.03227
        assert names["a"] == pytest.approx(1 / 61 + 1 / 63, abs=1e-10)
        assert names["b"] == pytest.approx(2 / 62, abs=1e-10)
        assert names["c"] == pytest.approx(1 / 63 + 1 / 61, abs=1e-10)
        # a and c tie → tie-break by chunk_id
        assert [x.chunk.chunk_id for x in fused] == ["a", "c", "b"]

    def test_deduplicates_within_single_list(self) -> None:
        dense = [_chunk("a"), _chunk("a"), _chunk("b")]
        assert len(dense) == 3
        fused = rrf_fuse(dense, [], k=60.0)
        # a (rank 1): 1/61 only
        assert len(fused) == 2
        assert fused[0].chunk.chunk_id == "a"
        assert fused[0].score == pytest.approx(1 / 61, abs=1e-10)

    def test_top_n_limits_results(self) -> None:
        dense = [_chunk(f"c{i}") for i in range(10)]
        lexical = [_chunk(f"c{i}") for i in range(5, 15)]
        fused = rrf_fuse(dense, lexical, top_n=3)
        assert len(fused) == 3

    def test_top_n_none_returns_all(self) -> None:
        dense = [_chunk(f"c{i}") for i in range(5)]
        lexical = [_chunk(f"c{i}") for i in range(3)]
        fused = rrf_fuse(dense, lexical, top_n=None)
        assert len(fused) == 5

    def test_deterministic_tie_break(self) -> None:
        dense = [_chunk("b"), _chunk("a")]
        lexical = [_chunk("a"), _chunk("b")]
        # Both have same ranks in each list, so scores are equal
        # a: 1/61 + 1/61 ≈ 0.03279
        # b: 1/62 + 1/62 ≈ 0.03226
        # No tie at top but verify reproducibility
        r1 = rrf_fuse(dense, lexical)
        r2 = rrf_fuse(dense, lexical)
        assert [x.chunk.chunk_id for x in r1] == [x.chunk.chunk_id for x in r2]

    def test_score_range(self) -> None:
        dense = [_chunk("a")]
        lexical = [_chunk("a")]
        fused = rrf_fuse(dense, lexical, k=60.0)
        # a: 1/61 + 1/61 ≈ 0.03279
        assert 0.0 < fused[0].score < 1.0

    def test_custom_k(self) -> None:
        dense = [_chunk("a"), _chunk("b")]
        lexical = [_chunk("c"), _chunk("a")]
        fused = rrf_fuse(dense, lexical, k=1.0)
        # a: dense rank 1 → 1/2 + lexical rank 2 → 1/3 = 0.8333
        # b: dense rank 2 → 1/3 ≈ 0.3333
        # c: lexical rank 1 → 1/2 = 0.5
        assert fused[0].chunk.chunk_id == "a"
        assert fused[0].score == pytest.approx(1 / 2 + 1 / 3, abs=1e-10)
        assert [x.chunk.chunk_id for x in fused] == ["a", "c", "b"]
