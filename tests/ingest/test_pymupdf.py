"""Tests for PyMuPDFParser — fitz is mocked; no real PDFs opened."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.errors import ParseError
from core.models import Paper
from ingest.parsers.pymupdf import PyMuPDFParser, _title_from_path


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
