from collections.abc import Sequence

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


def test_paper_with_all_fields() -> None:
    p = Paper(
        id="p1",
        doi="10.1234/zenodo.123",
        title="Test Paper",
        authors=[{"given": "Alice", "family": "Smith"}],
        year=2024,
        venue="NeurIPS",
        abstract="An abstract.",
        source="arxiv",
        open_access=True,
        pdf_path="/tmp/paper.pdf",
        grobid_tei="<TEI/>",
        meta={"key": "val"},
    )
    assert p.id == "p1"
    assert p.title == "Test Paper"
    assert p.authors == [{"given": "Alice", "family": "Smith"}]


def test_paper_frozen() -> None:
    p = Paper(
        id="p1", doi=None, title="T", authors=[], year=None,
        venue=None, abstract=None, source="src", open_access=None,
        pdf_path=None, grobid_tei=None, meta={},
    )
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.title = "changed"


def test_paper_minimal() -> None:
    p = Paper(
        id="p1", doi=None, title="", authors=[], year=None,
        venue=None, abstract=None, source="src", open_access=None,
        pdf_path=None, grobid_tei=None, meta={},
    )
    assert p.title == ""


def test_chunk_all_fields() -> None:
    c = Chunk(
        id="c1", paper_id="p1", ordinal=0, section="Intro",
        text="Hello world.", char_start=0, char_end=12,
        page=1, token_count=3, content_hash="abc",
    )
    assert c.section == "Intro"
    assert c.char_start == 0
    assert c.char_end == 12


def test_chunk_nullable_fields() -> None:
    c = Chunk(
        id="c1", paper_id="p1", ordinal=0, section=None,
        text="Hello.", char_start=None, char_end=None,
        page=None, token_count=None, content_hash="def",
    )
    assert c.section is None
    assert c.char_start is None
    assert c.page is None


def test_chunk_ref_all_fields() -> None:
    r = ChunkRef(
        chunk_id="c1", paper_id="p1", text="t",
        section="Results", page=2, char_start=10, char_end=20,
    )
    assert r.chunk_id == "c1"
    assert r.page == 2


def test_chunk_ref_nullable() -> None:
    r = ChunkRef(
        chunk_id="c1", paper_id="p1", text="t",
        section=None, page=None, char_start=None, char_end=None,
    )
    assert r.section is None


def test_scored_chunk() -> None:
    ref = ChunkRef(
        chunk_id="c1", paper_id="p1", text="t",
        section=None, page=None, char_start=None, char_end=None,
    )
    sc = ScoredChunk(chunk=ref, score=0.95, channel="dense")
    assert sc.score == 0.95
    assert sc.channel == "dense"


def test_scored_chunk_boundary_scores() -> None:
    ref = ChunkRef(
        chunk_id="c1", paper_id="p1", text="t",
        section=None, page=None, char_start=None, char_end=None,
    )
    sc0 = ScoredChunk(chunk=ref, score=0.0, channel="dense")
    sc1 = ScoredChunk(chunk=ref, score=1.0, channel="hybrid")
    assert sc0.score == 0.0
    assert sc1.score == 1.0


def test_upsert_chunk() -> None:
    uc = UpsertChunk(
        chunk_id="c1", paper_id="p1", text="t",
        vector=[0.1, 0.2, 0.3],
        metadata={"ordinal": 0},
    )
    assert list(uc.vector) == [0.1, 0.2, 0.3]
    assert uc.metadata == {"ordinal": 0}


def test_upsert_chunk_empty_vector() -> None:
    uc = UpsertChunk(
        chunk_id="c1", paper_id="p1", text="t",
        vector=[], metadata={},
    )
    assert uc.vector == []


def test_claim() -> None:
    cl = Claim(text="The sky is blue", chunk_id="c1", quoted_span="sky is blue")
    assert cl.text == "The sky is blue"
    assert cl.quoted_span == "sky is blue"


def test_claim_unicode() -> None:
    cl = Claim(text="über cool", chunk_id="c1", quoted_span="über")
    assert cl.text == "über cool"


def test_claim_empty_fields() -> None:
    cl = Claim(text="", chunk_id="c1", quoted_span="")
    assert cl.text == ""


def test_claim_verdict_supported() -> None:
    cl = Claim(text="x", chunk_id="c1", quoted_span="x")
    cv = ClaimVerdict(claim=cl, supported=True, score=0.9, reason="")
    assert cv.supported is True
    assert cv.score == 0.9


def test_claim_verdict_not_supported() -> None:
    cl = Claim(text="x", chunk_id="c1", quoted_span="x")
    cv = ClaimVerdict(claim=cl, supported=False, score=0.0, reason="span_not_found")
    assert cv.supported is False
    assert cv.reason == "span_not_found"


def test_qa_result_answerable() -> None:
    cl = Claim(text="x", chunk_id="c1", quoted_span="x")
    cv = ClaimVerdict(claim=cl, supported=True, score=0.9, reason="")
    qr = QAResult(
        question="What is x?", answer="x is y",
        claims=[cl], verdicts=[cv], answerable=True,
    )
    assert qr.answerable is True
    assert qr.answer == "x is y"


def test_qa_result_unanswerable() -> None:
    qr = QAResult(
        question="?", answer=None,
        claims=[], verdicts=[], answerable=False,
    )
    assert qr.answerable is False
    assert qr.answer is None


def test_qa_result_empty_question() -> None:
    qr = QAResult(
        question="", answer=None,
        claims=[], verdicts=[], answerable=False,
    )
    assert qr.question == ""
