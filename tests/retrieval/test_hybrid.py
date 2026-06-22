"""Tests for rrf_fuse — pure logic, no I/O, deterministic."""

from core.models import ChunkRef, ScoredChunk
from retrieval.hybrid import rrf_fuse


def _sc(chunk_id: str, score: float,
        channel: str = "dense",
        text: str = "t",
        paper_id: str = "p1") -> ScoredChunk:
    ref = ChunkRef(
        chunk_id=chunk_id, paper_id=paper_id, text=text,
        section=None, page=None, char_start=None, char_end=None,
    )
    return ScoredChunk(chunk=ref, score=score, channel=channel)


# ---------------------------------------------------------------------------
# Basic fusion
# ---------------------------------------------------------------------------

class TestRRFFuse:
    def test_both_lists_identical(self) -> None:
        dense = [_sc("c1", 0.9), _sc("c2", 0.8)]
        lexical = [_sc("c1", 0.7), _sc("c2", 0.6)]
        fused = rrf_fuse(dense, lexical)
        assert len(fused) == 2
        assert fused[0].channel == "hybrid"
        assert fused[0].score > fused[1].score

    def test_only_in_dense(self) -> None:
        dense = [_sc("c1", 0.9), _sc("c2", 0.8)]
        fused = rrf_fuse(dense, [])
        assert len(fused) == 2
        assert all(f.channel == "hybrid" for f in fused)

    def test_only_in_lexical(self) -> None:
        lexical = [_sc("c1", 0.9)]
        fused = rrf_fuse([], lexical)
        assert len(fused) == 1
        assert fused[0].chunk.chunk_id == "c1"

    def test_both_empty(self) -> None:
        assert rrf_fuse([], []) == []

    def test_no_overlap(self) -> None:
        dense = [_sc("c1", 0.9)]
        lexical = [_sc("c2", 0.8)]
        fused = rrf_fuse(dense, lexical)
        assert len(fused) == 2
        assert {f.chunk.chunk_id for f in fused} == {"c1", "c2"}

    def test_deduplicates_within_same_list(self) -> None:
        dense = [_sc("c1", 0.9), _sc("c1", 0.8)]
        fused = rrf_fuse(dense, [])
        assert len(fused) == 1  # c1 should only appear once

    def test_deterministic_tie_break(self) -> None:
        """Items with same RRF score tie-break by chunk_id."""
        dense = [_sc("b", 0.9)]
        lexical = [_sc("a", 0.9)]
        fused = rrf_fuse(dense, lexical)
        assert len(fused) == 2
        # "a" comes before "b" alphabetically
        assert fused[0].chunk.chunk_id == "a"
        assert fused[1].chunk.chunk_id == "b"

    def test_top_n_limit(self) -> None:
        dense = [_sc("c1", 0.9), _sc("c2", 0.8)]
        fused = rrf_fuse(dense, [], top_n=1)
        assert len(fused) == 1

    def test_custom_k(self) -> None:
        dense = [_sc("c1", 0.9)]
        lexical = [_sc("c1", 0.9)]
        fused_default = rrf_fuse(dense, lexical)
        fused_custom = rrf_fuse(dense, lexical, k=1.0)
        # k=1 gives higher RRF scores
        assert fused_custom[0].score > fused_default[0].score

    def test_first_item_gets_highest_score(self) -> None:
        dense = [_sc("c1", 0.9, channel="dense")]
        lexical = [_sc("c1", 0.9, channel="lexical")]
        fused = rrf_fuse(dense, lexical)
        expected = 1.0 / (60 + 1) + 1.0 / (60 + 1)
        assert fused[0].score == expected

    def test_chunk_refs_from_both_sources(self) -> None:
        dense = [_sc("c1", 0.9, text="from dense")]
        lexical = [_sc("c2", 0.8, text="from lexical")]
        fused = rrf_fuse(dense, lexical)
        texts = {f.chunk.chunk_id: f.chunk.text for f in fused}
        assert texts["c1"] == "from dense"
        assert texts["c2"] == "from lexical"

    def test_score_range(self) -> None:
        dense = [_sc("a", score=0.5)]
        lexical = [_sc("a", score=0.5)]
        fused = rrf_fuse(dense, lexical, k=60.0)
        assert 0.0 < fused[0].score < 1.0

    def test_top_n_none_returns_all(self) -> None:
        dense = [_sc(f"c{i}", score=1.0) for i in range(5)]
        lexical = [_sc(f"c{i}", score=1.0) for i in range(3)]
        fused = rrf_fuse(dense, lexical, top_n=None)
        assert len(fused) == 5
