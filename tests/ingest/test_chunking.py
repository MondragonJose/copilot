"""Tests for chunking — pure logic, no I/O."""

from core.models import Paper, Chunk
from ingest.chunking import (
    chunk_document,
    _estimate_tokens,
    _split_text,
    _get_span_section,
    _get_span_page_grobid,
    _get_span_section_pymupdf,
    _get_span_page_pymupdf,
)


def _make_paper(sections: list[dict] | None = None,
                pages: list[dict] | None = None) -> Paper:
    meta: dict = {}
    if sections is not None:
        meta["sections"] = sections
    if pages is not None:
        meta["pages"] = pages
    return Paper(
        id="p1", doi=None, title="T", authors=[], year=None,
        venue=None, abstract=None, source="src", open_access=None,
        pdf_path=None, grobid_tei=None, meta=meta,
    )


class TestChunkDocument:
    def test_no_meta_returns_empty(self) -> None:
        paper = _make_paper()
        assert chunk_document(paper, "text") == []

    def test_single_section_single_chunk(self) -> None:
        paper = _make_paper(sections=[
            {"heading": "Intro", "text": "Hello world.",
             "char_start": 0, "char_end": 12},
        ])
        chunks = chunk_document(
            paper, "Hello world.",
            token_window=512, token_overlap=64,
        )
        assert len(chunks) == 1
        assert chunks[0].section == "Intro"
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == 12
        assert chunks[0].paper_id == "p1"

    def test_section_without_heading(self) -> None:
        paper = _make_paper(sections=[
            {"heading": "", "text": "Content without heading.",
             "char_start": 0, "char_end": 25},
        ])
        chunks = chunk_document(paper, "Content without heading.")
        assert len(chunks) == 1
        assert chunks[0].section is None  # empty heading becomes None

    def test_section_heading_none(self) -> None:
        paper = _make_paper(sections=[
            {"heading": None, "text": "Content.",
             "char_start": 0, "char_end": 8},
        ])
        chunks = chunk_document(paper, "Content.")
        assert len(chunks) == 1
        assert chunks[0].section is None

    def test_empty_text_span_skipped(self) -> None:
        paper = _make_paper(sections=[
            {"heading": "Intro", "text": "", "char_start": 0, "char_end": 0},
            {"heading": "Body", "text": "Real content.",
             "char_start": 0, "char_end": 13},
        ])
        chunks = chunk_document(paper, "Real content.")
        assert len(chunks) == 1

    def test_long_text_split_into_windows(self) -> None:
        long_text = "word " * 500
        paper = _make_paper(sections=[
            {"heading": "Long", "text": long_text,
             "char_start": 0, "char_end": len(long_text)},
        ])
        chunks = chunk_document(
            paper, long_text,
            token_window=100, token_overlap=10,
        )
        assert len(chunks) >= 2

    def test_unicode_content(self) -> None:
        paper = _make_paper(sections=[
            {"heading": "Intro", "text": "über cool résumé",
             "char_start": 0, "char_end": 16},
        ])
        chunks = chunk_document(paper, "über cool résumé")
        assert len(chunks) == 1
        assert "über" in chunks[0].text

    def test_pages_meta_uses_page_span(self) -> None:
        paper = _make_paper(pages=[
            {"page": 1, "text": "Page one content.",
             "char_start": 0, "char_end": 18},
            {"page": 2, "text": "Page two content.",
             "char_start": 19, "char_end": 36},
        ])
        chunks = chunk_document(paper, "Page one content.\n\nPage two content.")
        assert len(chunks) == 2
        assert chunks[0].page == 1
        assert chunks[1].page == 2

    def test_pages_section_none(self) -> None:
        paper = _make_paper(pages=[
            {"page": 1, "text": "Content.", "char_start": 0, "char_end": 8},
        ])
        chunks = chunk_document(paper, "Content.")
        assert chunks[0].section is None

    def test_content_hash_present(self) -> None:
        paper = _make_paper(sections=[
            {"heading": "", "text": "Test.", "char_start": 0, "char_end": 5},
        ])
        chunks = chunk_document(paper, "Test.")
        assert len(chunks[0].content_hash) == 64  # SHA-256 hex

    def test_token_count(self) -> None:
        paper = _make_paper(sections=[
            {"heading": "", "text": "Hello world.",
             "char_start": 0, "char_end": 12},
        ])
        chunks = chunk_document(paper, "Hello world.")
        assert chunks[0].token_count is not None
        assert chunks[0].token_count >= 1

    def test_char_start_end_match(self) -> None:
        paper = _make_paper(sections=[
            {"heading": "", "text": "Hello world.",
             "char_start": 0, "char_end": 12},
        ])
        chunks = chunk_document(paper, "Hello world.")
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == 12


class TestSplitText:
    def test_no_overlap(self) -> None:
        result = _split_text("abcdefghij", 0, 5, 0)
        assert len(result) == 2
        assert result[0] == ("abcde", 0, 5)
        assert result[1] == ("fghij", 5, 10)

    def test_with_overlap(self) -> None:
        result = _split_text("abcdefghij", 0, 5, 2)
        assert len(result) >= 2

    def test_shorter_than_window(self) -> None:
        result = _split_text("abc", 10, 5, 0)
        assert result == [("abc", 10, 13)]

    def test_empty_text(self) -> None:
        assert _split_text("", 0, 5, 0) == []

    def test_overlap_greater_than_max(self) -> None:
        """When overlap >= max_chars, step = max_chars (no overlap)."""
        result = _split_text("abcdefghij", 0, 5, 10)
        assert result[0] == ("abcde", 0, 5)

    def test_unicode(self) -> None:
        result = _split_text("übercool", 0, 10, 0)
        assert len(result) == 1
        assert result[0][0] == "übercool"


class TestEstimateTokens:
    def test_normal(self) -> None:
        assert _estimate_tokens("Hello world") == 3  # 11/4 ≈ 2.75 → 3

    def test_empty(self) -> None:
        assert _estimate_tokens("") == 1

    def test_unicode(self) -> None:
        assert _estimate_tokens("über") >= 1


class TestSpanHelpers:
    def test_get_span_section_with_heading(self) -> None:
        assert _get_span_section({"heading": "Intro"}) == "Intro"

    def test_get_span_section_empty_heading(self) -> None:
        assert _get_span_section({"heading": ""}) is None

    def test_get_span_section_missing_heading(self) -> None:
        assert _get_span_section({"text": "x"}) is None

    def test_get_span_page_grobid(self) -> None:
        assert _get_span_page_grobid({}) is None

    def test_get_span_section_pymupdf(self) -> None:
        assert _get_span_section_pymupdf({}) is None

    def test_get_span_page_pymupdf_present(self) -> None:
        assert _get_span_page_pymupdf({"page": 3}) == 3

    def test_get_span_page_pymupdf_missing(self) -> None:
        assert _get_span_page_pymupdf({"text": "x"}) is None
