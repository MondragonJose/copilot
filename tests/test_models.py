"""Verify every frozen dataclass can be constructed and is immutable."""

import pytest

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


class TestPaper:
    def test_construct(self) -> None:
        p = Paper(
            id="p1",
            doi="10.1234/test",
            title="Test Paper",
            authors=[{"name": "Alice"}],
            year=2024,
            venue="Test Journal",
            abstract="An abstract.",
            source="upload",
            open_access=True,
            pdf_path="/tmp/test.pdf",
            grobid_tei=None,
            meta={"key": "val"},
        )
        assert p.id == "p1"
        assert p.authors == [{"name": "Alice"}]
        assert p.meta == {"key": "val"}

    def test_is_frozen(self) -> None:
        p = Paper(
            id="p1",
            doi=None,
            title="Test",
            authors=[],
            year=None,
            venue=None,
            abstract=None,
            source="upload",
            open_access=None,
            pdf_path=None,
            grobid_tei=None,
            meta={},
        )
        with pytest.raises(AttributeError):
            p.title = "Mutated"  # type: ignore[misc]


class TestChunk:
    def test_construct(self) -> None:
        c = Chunk(
            id="c1",
            paper_id="p1",
            ordinal=1,
            section="Methods",
            text="We used a model.",
            char_start=10,
            char_end=30,
            page=3,
            token_count=50,
            content_hash="abc123",
        )
        assert c.id == "c1"
        assert c.token_count == 50

    def test_is_frozen(self) -> None:
        c = Chunk(
            id="c1", paper_id="p1", ordinal=1, section=None,
            text="text", char_start=None, char_end=None, page=None,
            token_count=None, content_hash="abc",
        )
        with pytest.raises(AttributeError):
            c.text = "mutated"  # type: ignore[misc]


class TestChunkRef:
    def test_construct(self) -> None:
        r = ChunkRef(
            chunk_id="c1",
            paper_id="p1",
            text="Evidence text.",
            section="Results",
            page=5,
            char_start=100,
            char_end=150,
        )
        assert r.chunk_id == "c1"
        assert r.char_start == 100


class TestScoredChunk:
    def test_construct(self) -> None:
        ref = ChunkRef(
            chunk_id="c1", paper_id="p1", text="text",
            section=None, page=None, char_start=None, char_end=None,
        )
        s = ScoredChunk(chunk=ref, score=0.95, channel="dense")
        assert s.score == 0.95
        assert s.channel == "dense"


class TestUpsertChunk:
    def test_construct(self) -> None:
        u = UpsertChunk(
            chunk_id="c1",
            paper_id="p1",
            text="text",
            vector=[0.1, 0.2, 0.3],
            metadata={"source": "pdf"},
        )
        assert u.chunk_id == "c1"
        assert list(u.vector) == [0.1, 0.2, 0.3]

    def test_vector_as_tuple(self) -> None:
        u = UpsertChunk(
            chunk_id="c1", paper_id="p1", text="text",
            vector=(0.1, 0.2), metadata={},
        )
        assert len(u.vector) == 2


class TestClaim:
    def test_construct(self) -> None:
        cl = Claim(
            text="The model achieved 95% accuracy.",
            chunk_id="c1",
            quoted_span="95% accuracy",
        )
        assert cl.text == "The model achieved 95% accuracy."
        assert cl.quoted_span == "95% accuracy"


class TestClaimVerdict:
    def test_construct(self) -> None:
        cl = Claim(text="Claim text", chunk_id="c1", quoted_span="span")
        v = ClaimVerdict(claim=cl, supported=True, score=0.98, reason="Entailment confirmed.")
        assert v.supported is True
        assert v.reason == "Entailment confirmed."


class TestQAResult:
    def test_construct(self) -> None:
        cl = Claim(text="Claim", chunk_id="c1", quoted_span="span")
        v = ClaimVerdict(claim=cl, supported=True, score=0.99, reason="OK")
        q = QAResult(
            question="What was the accuracy?",
            answer="95%",
            claims=[cl],
            verdicts=[v],
            answerable=True,
        )
        assert q.question == "What was the accuracy?"
        assert q.answerable is True
        assert len(q.claims) == 1

    def test_unanswerable(self) -> None:
        q = QAResult(
            question="Unknown?",
            answer=None,
            claims=[],
            verdicts=[],
            answerable=False,
        )
        assert q.answer is None
        assert q.answerable is False
