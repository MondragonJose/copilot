"""Tests for PyMuPDFParser — fitz is mocked; no real PDFs opened.

The first half of this file mocks fitz for pure unit tests.
The second half creates ephemeral real PDFs with fitz for integration-level tests.
"""

import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import fitz
import pytest

from core.errors import ParseError
from core.models import Paper
from ingest.parsers.pymupdf import PyMuPDFParser, _title_from_path


# ===================================================================
# Real-PDF integration helpers (ephemeral fixture PDFs)
# ===================================================================


def _make_native_pdf(path: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Page 1 content begins here.", fontname="helv", fontsize=12)
    page.insert_text((50, 80), "Second line on page one.", fontname="helv", fontsize=12)
    page = doc.new_page()
    page.insert_text((50, 50), "Page 2 has different text.", fontname="helv", fontsize=12)
    doc.save(str(path))
    doc.close()


def _make_scanned_pdf(path: str) -> None:
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    doc.save(str(path))
    doc.close()


def _make_single_page_pdf(path: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Single page document.", fontname="helv", fontsize=12)
    doc.save(str(path))
    doc.close()


# ===================================================================
# Mock-based unit tests
# ===================================================================


@pytest.fixture
def mock_page() -> MagicMock:
    page = MagicMock()
    page.get_text.return_value = "Page content here."
    return page


@pytest.fixture
def mock_doc(mock_page: MagicMock) -> MagicMock:
    doc = MagicMock()
    doc.page_count = 2
    doc.load_page = MagicMock(return_value=mock_page)
    doc.close = MagicMock()
    return doc


class TestPyMuPDFParser:
    @pytest.mark.asyncio
    async def test_parse_success(self, mock_doc: MagicMock) -> None:
        with patch("ingest.parsers.pymupdf.os.path.isfile",
                   return_value=True):
            with patch("ingest.parsers.pymupdf.fitz.open",
                       return_value=mock_doc):
                parser = PyMuPDFParser()
                paper, full_text = await parser.parse("/path/to.pdf")

        assert isinstance(paper, Paper)
        assert paper.source == "pymupdf"
        assert paper.authors == []
        assert paper.doi is None
        assert "pages" in paper.meta
        assert len(paper.meta["pages"]) == 2
        assert "Page content" in full_text

    @pytest.mark.asyncio
    async def test_file_not_found(self) -> None:
        with patch("ingest.parsers.pymupdf.os.path.isfile",
                   return_value=False):
            parser = PyMuPDFParser()
            with pytest.raises(ParseError, match="PDF not found"):
                await parser.parse("/nonexistent.pdf")

    @pytest.mark.asyncio
    async def test_cannot_open(self) -> None:
        with patch("ingest.parsers.pymupdf.os.path.isfile",
                   return_value=True):
            with patch("ingest.parsers.pymupdf.fitz.open",
                       side_effect=RuntimeError("corrupt")):
                parser = PyMuPDFParser()
                with pytest.raises(ParseError, match="Failed to open PDF"):
                    await parser.parse("/path/to.pdf")

    @pytest.mark.asyncio
    async def test_empty_document(self) -> None:
        doc = MagicMock()
        doc.page_count = 0
        doc.close = MagicMock()
        with patch("ingest.parsers.pymupdf.os.path.isfile",
                   return_value=True):
            with patch("ingest.parsers.pymupdf.fitz.open",
                       return_value=doc):
                parser = PyMuPDFParser()
                with pytest.raises(ParseError, match="empty"):
                    await parser.parse("/path/to.pdf")

    @pytest.mark.asyncio
    async def test_no_text_layer(self) -> None:
        page = MagicMock()
        page.get_text.return_value = ""
        doc = MagicMock()
        doc.page_count = 1
        doc.load_page = MagicMock(return_value=page)
        doc.close = MagicMock()
        with patch("ingest.parsers.pymupdf.os.path.isfile",
                   return_value=True):
            with patch("ingest.parsers.pymupdf.fitz.open",
                       return_value=doc):
                parser = PyMuPDFParser()
                with pytest.raises(ParseError, match="needs_ocr"):
                    await parser.parse("/path/to.pdf")

    @pytest.mark.asyncio
    async def test_single_page(self) -> None:
        page = MagicMock()
        page.get_text.return_value = "Single page text."
        doc = MagicMock()
        doc.page_count = 1
        doc.load_page = MagicMock(return_value=page)
        doc.close = MagicMock()
        with patch("ingest.parsers.pymupdf.os.path.isfile",
                   return_value=True):
            with patch("ingest.parsers.pymupdf.fitz.open",
                       return_value=doc):
                parser = PyMuPDFParser()
                paper, full_text = await parser.parse("/path/to.pdf")
                assert full_text == "Single page text."
                assert len(paper.meta["pages"]) == 1

    @pytest.mark.asyncio
    async def test_unicode_content(self) -> None:
        page = MagicMock()
        page.get_text.return_value = "über cool résumé"
        doc = MagicMock()
        doc.page_count = 1
        doc.load_page = MagicMock(return_value=page)
        doc.close = MagicMock()
        with patch("ingest.parsers.pymupdf.os.path.isfile",
                   return_value=True):
            with patch("ingest.parsers.pymupdf.fitz.open",
                       return_value=doc):
                parser = PyMuPDFParser()
                paper, full_text = await parser.parse("/path/to.pdf")
                assert "über" in full_text

    @pytest.mark.asyncio
    async def test_doc_closed_on_success(self, mock_doc: MagicMock) -> None:
        with patch("ingest.parsers.pymupdf.os.path.isfile",
                   return_value=True):
            with patch("ingest.parsers.pymupdf.fitz.open",
                       return_value=mock_doc):
                parser = PyMuPDFParser()
                await parser.parse("/path/to.pdf")
        mock_doc.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_doc_closed_on_failure(self) -> None:
        doc = MagicMock()
        doc.page_count = 1
        doc.load_page = MagicMock(side_effect=RuntimeError("bad page"))
        doc.close = MagicMock()
        with patch("ingest.parsers.pymupdf.os.path.isfile",
                   return_value=True):
            with patch("ingest.parsers.pymupdf.fitz.open",
                       return_value=doc):
                parser = PyMuPDFParser()
                with pytest.raises(Exception):
                    await parser.parse("/path/to.pdf")
        doc.close.assert_called_once()


class TestTitleFromPath:
    def test_basic(self) -> None:
        assert _title_from_path("/path/to/my_paper.pdf") == "my paper"

    def test_with_hyphens(self) -> None:
        assert _title_from_path("/path/to/paper-v2.pdf") == "paper v2"

    def test_stem_only(self) -> None:
        assert _title_from_path("paper.pdf") == "paper"

    def test_empty_stem(self) -> None:
        result = _title_from_path("/path/to/.pdf")
        assert result == "pdf"  # just the extension stem


# ===================================================================
# Real-PDF integration tests (ephemeral fixture PDFs via fitz)
# ===================================================================


@pytest.fixture
def native_pdf(tmp_path) -> str:
    p = str(tmp_path / "test_article.pdf")
    _make_native_pdf(p)
    return p


@pytest.fixture
def scanned_pdf(tmp_path) -> str:
    p = str(tmp_path / "scanned_doc.pdf")
    _make_scanned_pdf(p)
    return p


@pytest.fixture
def single_page_pdf(tmp_path) -> str:
    p = str(tmp_path / "single_page.pdf")
    _make_single_page_pdf(p)
    return p


class TestNativePDFIntegration:
    pytestmark = pytest.mark.asyncio

    async def test_returns_paper_and_full_text(self, native_pdf: str) -> None:
        parser = PyMuPDFParser()
        paper, full_text = await parser.parse(native_pdf)

        assert isinstance(paper, object)
        assert len(full_text) > 0
        assert "Page 1 content begins here." in full_text
        assert "Page 2" in full_text

    async def test_metadata(self, native_pdf: str) -> None:
        parser = PyMuPDFParser()
        paper, _ = await parser.parse(native_pdf)

        assert paper.title == "test article"
        assert paper.authors == []
        assert paper.year is None
        assert paper.doi is None
        assert paper.abstract is None
        assert paper.source == "pymupdf"
        assert paper.grobid_tei is None
        assert paper.pdf_path == native_pdf

    async def test_page_spans_in_meta(self, native_pdf: str) -> None:
        parser = PyMuPDFParser()
        paper, _ = await parser.parse(native_pdf)
        spans = paper.meta["pages"]

        assert len(spans) >= 2
        for span in spans:
            assert isinstance(span["page"], int)
            assert isinstance(span["text"], str)
            assert len(span["text"]) > 0
            assert isinstance(span["char_start"], int)
            assert isinstance(span["char_end"], int)
            assert span["char_start"] >= 0
            assert span["char_end"] > span["char_start"]

    async def test_page_spans_monotonic(self, native_pdf: str) -> None:
        parser = PyMuPDFParser()
        paper, _ = await parser.parse(native_pdf)
        spans = paper.meta["pages"]

        for i in range(1, len(spans)):
            assert spans[i]["char_start"] > spans[i - 1]["char_end"]

    async def test_page_numbers_match(self, native_pdf: str) -> None:
        parser = PyMuPDFParser()
        paper, _ = await parser.parse(native_pdf)
        spans = paper.meta["pages"]

        for i, span in enumerate(spans):
            assert span["page"] == i

    async def test_full_text_concatenates_pages(self, native_pdf: str) -> None:
        parser = PyMuPDFParser()
        paper, full_text = await parser.parse(native_pdf)
        spans = paper.meta["pages"]

        reconstructed = "\n\n".join(str(s["text"]) for s in spans)
        assert full_text == reconstructed


class TestSinglePageIntegration:
    pytestmark = pytest.mark.asyncio

    async def test_single_page_extracts_text(self, single_page_pdf: str) -> None:
        parser = PyMuPDFParser()
        paper, full_text = await parser.parse(single_page_pdf)
        assert "Single page document." in full_text
        assert len(paper.meta["pages"]) == 1

    async def test_single_page_offsets(self, single_page_pdf: str) -> None:
        parser = PyMuPDFParser()
        paper, full_text = await parser.parse(single_page_pdf)
        span = paper.meta["pages"][0]
        assert span["char_start"] == 0
        assert span["char_end"] == len(full_text)


class TestScannedPDFIntegration:
    pytestmark = pytest.mark.asyncio

    async def test_raises_parse_error_with_needs_ocr(self, scanned_pdf: str) -> None:
        parser = PyMuPDFParser()
        with pytest.raises(ParseError) as exc_info:
            await parser.parse(scanned_pdf)
        assert "needs_ocr" in str(exc_info.value)


class TestErrorHandlingIntegration:
    pytestmark = pytest.mark.asyncio

    async def test_nonexistent_path_raises_parse_error(self) -> None:
        parser = PyMuPDFParser()
        with pytest.raises(ParseError, match="PDF not found"):
            await parser.parse("/nonexistent/file.pdf")

    async def test_corrupted_pdf_raises_parse_error(self, tmp_path) -> None:
        bad = str(tmp_path / "corrupted.pdf")
        pathlib.Path(bad).write_bytes(b"%PDF-1.4 trash\x00\x00")
        parser = PyMuPDFParser()
        with pytest.raises(ParseError, match="Failed to open PDF"):
            await parser.parse(bad)
