"""Tests for structure-aware document chunking (GROBID sections / PyMuPDF pages)."""

from __future__ import annotations

import uuid

from core.models import Paper
from ingest.chunking import (
    _estimate_tokens,
    _split_text,
    chunk_document,
)

# --------------------------------------------------------------------------
# Helpers — build Paper fixtures with controlled meta
# --------------------------------------------------------------------------


def _paper(
    *,
    meta: dict | None = None,
    title: str = "Chunking Test",
) -> Paper:
    return Paper(
        id=str(uuid.uuid4()),
        doi=None,
        title=title,
        authors=[],
        year=None,
        venue=None,
        abstract=None,
        source="test",
        open_access=None,
        pdf_path="/dev/null",
        grobid_tei=None,
        meta=meta or {},
    )


def _sec(heading: str, text: str, offset: int) -> dict[str, object]:
    return {
        "heading": heading,
        "text": text,
        "char_start": offset,
        "char_end": offset + len(text),
    }


def _page(page_num: int, text: str, offset: int) -> dict[str, object]:
    return {
        "page": page_num,
        "text": text,
        "char_start": offset,
        "char_end": offset + len(text),
    }


# --------------------------------------------------------------------------
# Token estimation
# --------------------------------------------------------------------------


class TestEstimateTokens:

    def test_empty_text(self) -> None:
        assert _estimate_tokens("") == 1

    def test_short_text(self) -> None:
        n = _estimate_tokens("hello world")
        assert isinstance(n, int)
        assert n >= 1

    def test_proportional(self) -> None:
        n4 = _estimate_tokens("a" * 40)
        n8 = _estimate_tokens("a" * 80)
        assert n8 == n4 * 2


# --------------------------------------------------------------------------
# _split_text — low-level windowing
# --------------------------------------------------------------------------


class TestSplitText:

    def test_empty_returns_empty(self) -> None:
        assert _split_text("", 0, 100, 20) == []

    def test_short_text_single_window(self) -> None:
        result = _split_text("hello", 10, 100, 20)
        assert len(result) == 1
        text, start, end = result[0]
        assert text == "hello"
        assert start == 10
        assert end == 15

    def test_exact_fit(self) -> None:
        text = "a" * 100
        result = _split_text(text, 0, 100, 20)
        assert len(result) == 1

    def test_multiple_windows(self) -> None:
        text = "x" * 500
        result = _split_text(text, 0, 200, 40)
        assert len(result) > 1

    def test_monotonic_offsets(self) -> None:
        text = "x" * 500
        result = _split_text(text, 100, 200, 40)
        for i in range(1, len(result)):
            assert result[i][1] >= result[i - 1][2] - 40  # overlap accounted

    def test_no_window_larger_than_max_chars(self) -> None:
        text = "x" * 1000
        result = _split_text(text, 0, 200, 40)
        for _, start, end in result:
            assert end - start <= 200

    def test_overlap_smaller_than_window(self) -> None:
        text = "x" * 300
        result = _split_text(text, 0, 100, 20)
        if len(result) >= 2:
            gap = result[0][2] - result[1][1]
            assert gap == 0 or gap <= 20

    def test_step_never_zero(self) -> None:
        text = "x" * 100
        result = _split_text(text, 0, 50, 50)  # overlap == window
        assert len(result) == 2  # should still advance


# --------------------------------------------------------------------------
# GROBID: sections path
# --------------------------------------------------------------------------


class TestChunkSections:

    def test_single_short_section(self) -> None:
        s1 = _sec("Intro", "Short intro text.", 0)
        full = "Short intro text."
        paper = _paper(meta={"sections": [s1]})
        chunks = chunk_document(paper, full)

        assert len(chunks) == 1
        assert chunks[0].section == "Intro"
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == len(full)

    def test_no_chunk_crosses_section_boundary(self) -> None:
        """Large sections each produce windows; no window spans two sections."""
        s1 = _sec("S1", "A" * 5000, 0)
        s2 = _sec("S2", "B" * 5000, 5002)
        full = "A" * 5000 + "\n\n" + "B" * 5000
        paper = _paper(meta={"sections": [s1, s2]})
        chunks = chunk_document(paper, full)

        s1_chunks = [c for c in chunks if c.section == "S1"]
        s2_chunks = [c for c in chunks if c.section == "S2"]

        assert len(s1_chunks) >= 2
        assert len(s2_chunks) >= 2

        for c in s1_chunks:
            assert c.char_start is not None and c.char_end is not None
            assert c.char_end <= 5000, "S1 chunk overflowed into S2"

        for c in s2_chunks:
            assert c.char_start is not None and c.char_end is not None
            assert c.char_start >= 5002, "S2 chunk underflowed into S1"

    def test_offsets_monotonic(self) -> None:
        s1 = _sec("A", "x" * 5000, 0)
        full = "x" * 5000
        paper = _paper(meta={"sections": [s1]})
        chunks = chunk_document(paper, full)

        # Each chunk's own offsets must be valid
        for c in chunks:
            assert c.char_start is not None
            assert c.char_end is not None
            assert c.char_start < c.char_end

        # Ordinals must be strictly increasing without gaps
        for i in range(1, len(chunks)):
            assert chunks[i].ordinal == chunks[i - 1].ordinal + 1

    def test_all_chunks_have_required_fields(self) -> None:
        s1 = _sec("Method", "Detailed method description. " * 100, 0)
        full = "Detailed method description. " * 100
        paper = _paper(meta={"sections": [s1]})
        chunks = chunk_document(paper, full)

        for c in chunks:
            assert c.id
            assert c.paper_id == paper.id
            assert isinstance(c.ordinal, int)
            assert c.text
            assert c.char_start is not None
            assert c.char_end is not None
            assert c.char_start < c.char_end
            assert c.token_count is not None and c.token_count > 0
            assert c.content_hash

    def test_empty_section_skipped(self) -> None:
        s1 = _sec("Empty", "", 0)
        s2 = _sec("Real", "content", 2)
        full = "\n\ncontent"
        paper = _paper(meta={"sections": [s1, s2]})
        chunks = chunk_document(paper, full)

        assert len(chunks) == 1
        assert chunks[0].section == "Real"

    def test_section_without_heading(self) -> None:
        """A section with an empty heading stores section=None."""
        span = {"text": "content", "char_start": 0, "char_end": 7}
        paper = _paper(meta={"sections": [span]})
        chunks = chunk_document(paper, "content")

        assert len(chunks) >= 1
        assert chunks[0].section is None


# --------------------------------------------------------------------------
# PyMuPDF: pages path
# --------------------------------------------------------------------------


class TestChunkPages:

    def test_pages_have_correct_page_numbers(self) -> None:
        p0 = _page(0, "P" * 5000, 0)
        p1 = _page(1, "Q" * 3000, 5002)
        full = "P" * 5000 + "\n\n" + "Q" * 3000
        paper = _paper(meta={"pages": [p0, p1]})
        chunks = chunk_document(paper, full)

        p0_chunks = [c for c in chunks if c.page == 0]
        p1_chunks = [c for c in chunks if c.page == 1]

        assert len(p0_chunks) >= 2
        assert len(p1_chunks) >= 1

        for c in p0_chunks:
            assert c.char_end is not None and c.char_end <= 5000

        for c in p1_chunks:
            assert c.char_start is not None and c.char_start >= 5002

    def test_no_chunk_crosses_page_boundary(self) -> None:
        p0 = _page(0, "X" * 5000, 0)
        p1 = _page(1, "Y" * 5000, 5002)
        full = "X" * 5000 + "\n\n" + "Y" * 5000
        paper = _paper(meta={"pages": [p0, p1]})
        chunks = chunk_document(paper, full)

        for c in chunks:
            if c.page == 0:
                assert c.char_end is not None and c.char_end <= 5000
            elif c.page == 1:
                assert c.char_start is not None and c.char_start >= 5002

    def test_page_without_text_skipped(self) -> None:
        p0 = _page(0, "", 0)
        p1 = _page(1, "real page content", 2)
        full = "\n\nreal page content"
        paper = _paper(meta={"pages": [p0, p1]})
        chunks = chunk_document(paper, full)

        for c in chunks:
            assert c.page == 1


# --------------------------------------------------------------------------
# Hash uniqueness
# --------------------------------------------------------------------------


class TestContentHash:

    def test_hash_is_sha256(self) -> None:
        chunk = chunk_document(
            _paper(meta={"sections": [_sec("A", "hello", 0)]}),
            "hello",
        )[0]
        assert len(chunk.content_hash) == 64  # sha256 hex

    def test_same_text_same_hash(self) -> None:
        text = "identical text"
        paper = _paper(meta={"sections": [_sec("A", text, 0)]})
        c1 = chunk_document(paper, text)[0]
        c2 = chunk_document(paper, text)[0]
        assert c1.content_hash == c2.content_hash

    def test_different_text_different_hash(self) -> None:
        paper = _paper(meta={"sections": [_sec("A", "aaa", 0), _sec("B", "bbb", 4)]})
        chunks = chunk_document(paper, "aaa\n\nbbb")
        assert len(chunks) >= 2
        assert chunks[0].content_hash != chunks[1].content_hash


# --------------------------------------------------------------------------
# No known source in meta
# --------------------------------------------------------------------------


class TestNoKnownMeta:

    def test_empty_meta_returns_empty(self) -> None:
        paper = _paper(meta={})
        chunks = chunk_document(paper, "")
        assert chunks == []

    def test_unknown_meta_keys_ignored(self) -> None:
        paper = _paper(meta={"unknown": [1, 2, 3]})
        chunks = chunk_document(paper, "text")
        assert chunks == []
