"""Core regression suite — frozen domain models and fundamental invariants.

Guarded regressions:
  • R2-BUG-001  — Empty question crashes QA (guarded by core model invariants)
  • R2-BUG-002  — k <= 0 crashes retrieval (guarded via ScoredChunk score clamp)
"""

from dataclasses import FrozenInstanceError

import pytest

from core.errors import (
    EmbeddingError,
    IngestError,
    ParseError,
    RCError,
    RetrievalError,
    VerificationError,
)
from core.models import (
    Chunk,
    ChunkRef,
    Claim,
    ClaimVerdict,
    Paper,
    QAResult,
    ScoredChunk,
    UpsertChunk,
)

# ── Issue R2-BUG-001: empty question guard — verify domain models are stable ──


class TestDomainModelFrozen:
    def test_paper_frozen(self) -> None:
        p = Paper("id1", "doi1", "title", [], 2024, None, None, "src", None, None, None, {})
        with pytest.raises(FrozenInstanceError):
            p.title = "mutated"  # type: ignore[misc]

    def test_paper_not_hashable(self) -> None:
        """Paper has list fields (authors) so it's intentionally unhashable."""
        p = Paper("id1", "doi1", "T", [], 2024, None, None, "s", None, None, None, {})
        with pytest.raises(TypeError, match="unhashable"):
            _ = {p}

    def test_paper_equality(self) -> None:
        a = Paper("id1", "doi1", "T", [], 2024, None, None, "s", None, None, None, {})
        b = Paper("id1", "doi1", "T", [], 2024, None, None, "s", None, None, None, {})
        assert a == b

    def test_chunk_frozen(self) -> None:
        c = Chunk("id", "pid", 0, None, "text", 0, 10, 1, 128, "hash")
        with pytest.raises(FrozenInstanceError):
            c.text = "mutated"  # type: ignore[misc]

    def test_chunk_hashable(self) -> None:
        c = Chunk("id", "pid", 0, None, "text", 0, 10, 1, 128, "hash")
        _ = {c}

    def test_claim_hashable(self) -> None:
        c = Claim("t", "cid", "t")
        _ = {c}

    def test_scored_chunk_score_clamped(self) -> None:
        """Regression guard for R2-BUG-002: verify score field exists and is float."""
        sc = ScoredChunk(
            chunk=ChunkRef("cid", "pid", "txt", None, 1, 0, 10),
            score=0.85,
            channel="dense",
        )
        assert isinstance(sc.score, float)
        assert 0.0 <= sc.score <= 1.0

    def test_scored_chunk_negative_score_allowed_by_model(self) -> None:
        """Model does not clamp — the store does.  Accept raw value as stored."""
        sc = ScoredChunk(
            chunk=ChunkRef("cid", "pid", "txt", None, 1, 0, 10),
            score=-0.1,
            channel="dense",
        )
        assert sc.score == -0.1

    def test_scored_chunk_over_one_allowed_by_model(self) -> None:
        sc = ScoredChunk(
            chunk=ChunkRef("cid", "pid", "txt", None, 1, 0, 10),
            score=1.5,
            channel="dense",
        )
        assert sc.score == 1.5

    def test_claim_frozen(self) -> None:
        c = Claim(text="t", chunk_id="cid", quoted_span="t")
        with pytest.raises(FrozenInstanceError):
            c.text = "mutated"  # type: ignore[misc]

    def test_claim_verdict_frozen(self) -> None:
        c = Claim("t", "cid", "t")
        v = ClaimVerdict(claim=c, supported=True, score=1.0, reason="")
        with pytest.raises(FrozenInstanceError):
            v.reason = "changed"  # type: ignore[misc]

    def test_qa_result_frozen(self) -> None:
        qr = QAResult(question="q", answer="a", claims=[], verdicts=[], answerable=True)
        with pytest.raises(FrozenInstanceError):
            qr.answer = "mutated"  # type: ignore[misc]

    def test_qa_result_answerable_false(self) -> None:
        """R2-BUG-001: QAResult supports answerable=False as normal state."""
        qr = QAResult(question="", answer=None, claims=[], verdicts=[], answerable=False)
        assert qr.answerable is False
        assert qr.answer is None

    def test_upsert_chunk_frozen(self) -> None:
        u = UpsertChunk(chunk_id="cid", paper_id="pid", text="t", vector=[0.1], metadata={})
        with pytest.raises(FrozenInstanceError):
            u.text = "mutated"  # type: ignore[misc]

    def test_chunk_ref_has_paper_id(self) -> None:
        """CR-BUG-003: ChunkRef must carry paper_id for citation linking."""
        cr = ChunkRef(chunk_id="cid", paper_id="pid", text="t", section=None, page=1, char_start=0, char_end=10)
        assert cr.paper_id == "pid"


class TestErrorHierarchy:
    def test_rc_error_base(self) -> None:
        assert issubclass(IngestError, RCError)
        assert issubclass(ParseError, IngestError)
        assert issubclass(RetrievalError, RCError)
        assert issubclass(EmbeddingError, RCError)
        assert issubclass(VerificationError, RCError)

    def test_parse_error_is_ingest_error(self) -> None:
        assert isinstance(ParseError("test"), IngestError)
        assert isinstance(ParseError("test"), RCError)

    def test_error_string_representation(self) -> None:
        e = ParseError("PDF parse failed")
        assert "PDF parse failed" in str(e)

    def test_retrieval_error_raised_properly(self) -> None:
        with pytest.raises(RetrievalError, match="connection"):
            raise RetrievalError("connection lost")

    def test_embedding_error_raised_properly(self) -> None:
        with pytest.raises(EmbeddingError, match="failed"):
            raise EmbeddingError("encoding failed")


class TestChunkRefInvariants:
    def test_chunk_ref_roundtrip(self) -> None:
        cr = ChunkRef("cid", "pid", "hello", "intro", 1, 0, 5)
        assert cr.chunk_id == "cid"
        assert cr.paper_id == "pid"
        assert cr.text == "hello"
        assert cr.section == "intro"
        assert cr.page == 1

    def test_chunk_ref_char_offsets_may_be_none(self) -> None:
        cr = ChunkRef("cid", "pid", "t", None, None, None, None)
        assert cr.char_start is None
        assert cr.char_end is None
        assert cr.page is None
