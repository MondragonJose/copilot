"""Tests for the PyMuPDF parser — uses inline fixture PDFs created with fitz.

Native PDF fixture has 2 pages of extractable text.
Scanned PDF fixture has pages with no text layer.
"""

from __future__ import annotations

import pathlib

import fitz
import pytest

from core.errors import ParseError
from ingest.parsers.pymupdf import PyMuPDFParser, _title_from_path

# --------------------------------------------------------------------------
# Fixture helpers — create ephemeral PDFs with fitz
# --------------------------------------------------------------------------


def _make_native_pdf(path: pathlib.Path) -> None:
    """Create a 2-page PDF with extractable text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Page 1 content begins here.", fontname="helv", fontsize=12)
    page.insert_text((50, 80), "Second line on page one.", fontname="helv", fontsize=12)

    page = doc.new_page()
    page.insert_text((50, 50), "Page 2 has different text.", fontname="helv", fontsize=12)

    doc.save(str(path))
    doc.close()


def _make_scanned_pdf(path: pathlib.Path) -> None:
    """Create a PDF with no extractable text layer (mimics a scanned doc)."""
    doc = fitz.open()
    doc.new_page()  # blank page — get_text() returns ""
    doc.new_page()
    doc.save(str(path))
    doc.close()


def _make_single_page_pdf(path: pathlib.Path) -> None:
    """Create a 1-page PDF with text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Single page document.", fontname="helv", fontsize=12)
    doc.save(str(path))
    doc.close()


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def native_pdf(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "test_article.pdf"
    _make_native_pdf(path)
    return path


@pytest.fixture
def scanned_pdf(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "scanned_doc.pdf"
    _make_scanned_pdf(path)
    return path


@pytest.fixture
def single_page_pdf(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "single_page.pdf"
    _make_single_page_pdf(path)
    return path


@pytest.fixture
def parser() -> PyMuPDFParser:
    return PyMuPDFParser()


# --------------------------------------------------------------------------
# End-to-end: native PDF
# --------------------------------------------------------------------------


class TestNativePDF:
    pytestmark = pytest.mark.asyncio

    async def test_returns_paper_and_full_text(
        self, parser: PyMuPDFParser, native_pdf: pathlib.Path,
    ) -> None:
        paper, full_text = await parser.parse(str(native_pdf))

        assert isinstance(paper, object)
        assert len(full_text) > 0
        assert "Page 1 content begins here." in full_text
        assert "Page 2" in full_text

    async def test_metadata(
        self, parser: PyMuPDFParser, native_pdf: pathlib.Path,
    ) -> None:
        paper, _ = await parser.parse(str(native_pdf))

        assert paper.title == "test article"
        assert paper.authors == []
        assert paper.year is None
        assert paper.doi is None
        assert paper.abstract is None
        assert paper.source == "pymupdf"
        assert paper.grobid_tei is None
        assert paper.pdf_path == str(native_pdf)

    async def test_page_spans_in_meta(
        self, parser: PyMuPDFParser, native_pdf: pathlib.Path,
    ) -> None:
        paper, _ = await parser.parse(str(native_pdf))
        spans = paper.meta["pages"]

        assert len(spans) >= 2
        for span in spans:
            assert "page" in span
            assert isinstance(span["page"], int)
            assert "text" in span
            assert isinstance(span["text"], str)
            assert len(span["text"]) > 0
            assert "char_start" in span
            assert isinstance(span["char_start"], int)
            assert "char_end" in span
            assert isinstance(span["char_end"], int)
            assert span["char_start"] >= 0
            assert span["char_end"] > span["char_start"]

    async def test_page_spans_monotonic(
        self, parser: PyMuPDFParser, native_pdf: pathlib.Path,
    ) -> None:
        paper, _ = await parser.parse(str(native_pdf))
        spans = paper.meta["pages"]

        for i in range(1, len(spans)):
            assert spans[i]["char_start"] > spans[i - 1]["char_end"]

    async def test_page_numbers_match(
        self, parser: PyMuPDFParser, native_pdf: pathlib.Path,
    ) -> None:
        paper, _ = await parser.parse(str(native_pdf))
        spans = paper.meta["pages"]

        for i, span in enumerate(spans):
            assert span["page"] == i

    async def test_full_text_concatenates_pages(
        self, parser: PyMuPDFParser, native_pdf: pathlib.Path,
    ) -> None:
        paper, full_text = await parser.parse(str(native_pdf))
        spans = paper.meta["pages"]

        reconstructed = "\n\n".join(str(s["text"]) for s in spans)
        assert full_text == reconstructed


# --------------------------------------------------------------------------
# Single-page PDF
# --------------------------------------------------------------------------


class TestSinglePage:
    pytestmark = pytest.mark.asyncio

    async def test_single_page_extracts_text(
        self, parser: PyMuPDFParser, single_page_pdf: pathlib.Path,
    ) -> None:
        paper, full_text = await parser.parse(str(single_page_pdf))

        assert "Single page document." in full_text
        assert len(paper.meta["pages"]) == 1

    async def test_single_page_offsets(
        self, parser: PyMuPDFParser, single_page_pdf: pathlib.Path,
    ) -> None:
        paper, full_text = await parser.parse(str(single_page_pdf))
        span = paper.meta["pages"][0]

        assert span["char_start"] == 0
        assert span["char_end"] == len(full_text)


# --------------------------------------------------------------------------
# Scanned PDF (no text layer)
# --------------------------------------------------------------------------


class TestScannedPDF:
    pytestmark = pytest.mark.asyncio

    async def test_raises_parse_error_with_needs_ocr(
        self, parser: PyMuPDFParser, scanned_pdf: pathlib.Path,
    ) -> None:
        with pytest.raises(ParseError) as exc_info:
            await parser.parse(str(scanned_pdf))

        assert "needs_ocr" in str(exc_info.value)

    async def test_error_message_informs_router(
        self, parser: PyMuPDFParser, scanned_pdf: pathlib.Path,
    ) -> None:
        with pytest.raises(ParseError, match="needs_ocr"):
            await parser.parse(str(scanned_pdf))


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------


class TestErrorHandling:
    pytestmark = pytest.mark.asyncio

    async def test_nonexistent_path_raises_parse_error(
        self, parser: PyMuPDFParser,
    ) -> None:
        with pytest.raises(ParseError, match="PDF not found"):
            await parser.parse("/nonexistent/file.pdf")

    async def test_corrupted_pdf_raises_parse_error(
        self, parser: PyMuPDFParser, tmp_path: pathlib.Path,
    ) -> None:
        bad = tmp_path / "corrupted.pdf"
        bad.write_bytes(b"%PDF-1.4 trash\x00\x00")

        with pytest.raises(ParseError, match="Failed to open PDF"):
            await parser.parse(str(bad))


# --------------------------------------------------------------------------
# Unit: _title_from_path
# --------------------------------------------------------------------------


class TestTitleFromPath:

    def test_underscores_to_spaces(self) -> None:
        assert _title_from_path("/path/my_paper.pdf") == "my paper"

    def test_hyphens_to_spaces(self) -> None:
        assert _title_from_path("/path/my-paper.pdf") == "my paper"

    def test_simple_stem(self) -> None:
        assert _title_from_path("article.pdf") == "article"

    def test_no_extension(self) -> None:
        assert _title_from_path("/path/MyPaper") == "MyPaper"

    def test_dotfile_name(self) -> None:
        assert _title_from_path("/path/.pdf") == "pdf"
