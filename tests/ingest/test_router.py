"""Tests for ParserRouter — GROBID and PyMuPDF are mocked.

First half: pure unit tests with MagicMock parsers.
Second half: integration-style tests with real PyMuPDF + mock GROBID HTTP clients.
"""

import pathlib
from unittest.mock import AsyncMock, MagicMock

import fitz
import httpx
import pytest

from core.errors import ParseError
from ingest.router import ParserRouter


# ===================================================================
# Integration-style helpers (real PDFs + mock GROBID HTTP)
# ===================================================================


def _make_native_pdf(path: pathlib.Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Page one content.", fontname="helv", fontsize=12)
    page = doc.new_page()
    page.insert_text((50, 50), "Page two content.", fontname="helv", fontsize=12)
    doc.save(str(path))
    doc.close()


def _make_scanned_pdf(path: pathlib.Path) -> None:
    doc = fitz.open()
    doc.new_page()
    doc.save(str(path))
    doc.close()


@pytest.fixture
def native_pdf(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "test_article.pdf"
    _make_native_pdf(path)
    return path


@pytest.fixture
def scanned_pdf(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "scanned.pdf"
    _make_scanned_pdf(path)
    return path


GROBID_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>GROBID Title</title>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <monogr>
            <imprint><date type="published" when="2024"/></imprint>
          </monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>Intro</head>
        <p>GROBID extracted text.</p>
      </div>
    </body>
  </text>
</TEI>"""


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("POST", "http://grobid/"),
                response=httpx.Response(self.status_code),
            )


class _GrobidSuccessClient:
    async def post(self, url: str, **kwargs: object) -> _FakeResponse:
        return _FakeResponse(GROBID_TEI)


class _GrobidTimeoutClient:
    async def post(self, url: str, **kwargs: object) -> _FakeResponse:
        raise httpx.TimeoutException("Connection timeout after 0.001s")


class _GrobidErrorClient:
    async def post(self, url: str, **kwargs: object) -> _FakeResponse:
        return _FakeResponse("Internal Server Error", status_code=500)


# ===================================================================
# Unit tests (mocked parsers)
# ===================================================================
def mock_grobid() -> MagicMock:
    g = MagicMock()
    g.parse = AsyncMock()
    return g


@pytest.fixture
def mock_pymupdf() -> MagicMock:
    p = MagicMock()
    p.parse = AsyncMock()
    return p


class TestParserRouter:
    @pytest.mark.asyncio
    async def test_grobid_succeeds(self) -> None:
        router = ParserRouter(grobid_client=MagicMock(spec=httpx.AsyncClient))
        router._grobid = MagicMock()
        router._grobid.parse = AsyncMock(return_value=("paper", "text"))
        result = await router.parse("/path/to.pdf")
        assert result == ("paper", "text")

    @pytest.mark.asyncio
    async def test_grobid_fails_pymupdf_succeeds(self) -> None:
        router = ParserRouter(grobid_client=MagicMock(spec=httpx.AsyncClient))
        router._grobid.parse = AsyncMock(side_effect=ParseError("GROBID down"))
        router._pymupdf.parse = AsyncMock(return_value=("paper2", "text2"))
        result = await router.parse("/path/to.pdf")
        assert result == ("paper2", "text2")

    @pytest.mark.asyncio
    async def test_both_fail_without_ocr(self) -> None:
        router = ParserRouter(grobid_client=MagicMock(spec=httpx.AsyncClient))
        router._grobid.parse = AsyncMock(side_effect=ParseError("GROBID down"))
        router._pymupdf.parse = AsyncMock(
            side_effect=ParseError("no text layer"),
        )
        with pytest.raises(ParseError, match="GROBID unavailable"):
            await router.parse("/path/to.pdf")

    @pytest.mark.asyncio
    async def test_pymupdf_needs_ocr_propagates(self) -> None:
        router = ParserRouter(grobid_client=MagicMock(spec=httpx.AsyncClient))
        router._grobid.parse = AsyncMock(side_effect=ParseError("GROBID down"))
        router._pymupdf.parse = AsyncMock(
            side_effect=ParseError("needs_ocr: scanned PDF"),
        )
        with pytest.raises(ParseError, match="needs_ocr"):
            await router.parse("/path/to.pdf")

    @pytest.mark.asyncio
    async def test_default_constructor(self) -> None:
        """Verify default constructor does not crash (no real HTTP calls)."""
        router = ParserRouter()
        assert router._grobid is not None
        assert router._pymupdf is not None


# ===================================================================
# Integration-style tests (real PyMuPDF + mock GROBID HTTP)
# ===================================================================


class TestGrobidSucceedsIntegration:
    pytestmark = pytest.mark.asyncio

    async def test_uses_grobid_result(self, native_pdf: pathlib.Path) -> None:
        router = ParserRouter(grobid_client=_GrobidSuccessClient())
        paper, full_text = await router.parse(str(native_pdf))

        assert paper.title == "GROBID Title"
        assert paper.year == 2024
        assert paper.source == "grobid"
        assert "GROBID extracted text" in full_text
        assert len(paper.meta["sections"]) >= 1

    async def test_grobid_returns_paper_source_grobid(self, native_pdf: pathlib.Path) -> None:
        router = ParserRouter(grobid_client=_GrobidSuccessClient())
        paper, _ = await router.parse(str(native_pdf))
        assert paper.source == "grobid"


class TestFallbackOnTimeoutIntegration:
    pytestmark = pytest.mark.asyncio

    async def test_fallback_still_returns_text(self, native_pdf: pathlib.Path) -> None:
        router = ParserRouter(grobid_client=_GrobidTimeoutClient())
        paper, full_text = await router.parse(str(native_pdf))

        assert len(full_text) > 0
        assert "Page one content." in full_text
        assert "Page two content." in full_text

    async def test_fallback_uses_pymupdf_source(self, native_pdf: pathlib.Path) -> None:
        router = ParserRouter(grobid_client=_GrobidTimeoutClient())
        paper, _ = await router.parse(str(native_pdf))
        assert paper.source == "pymupdf"
        assert paper.grobid_tei is None

    async def test_fallback_returns_page_spans(self, native_pdf: pathlib.Path) -> None:
        router = ParserRouter(grobid_client=_GrobidTimeoutClient())
        paper, _ = await router.parse(str(native_pdf))
        spans = paper.meta["pages"]
        assert len(spans) >= 2
        assert spans[0]["page"] == 0
        assert spans[1]["page"] == 1


class TestFallbackOnHttpErrorIntegration:
    pytestmark = pytest.mark.asyncio

    async def test_fallback_on_500(self, native_pdf: pathlib.Path) -> None:
        router = ParserRouter(grobid_client=_GrobidErrorClient())
        paper, full_text = await router.parse(str(native_pdf))
        assert "Page one content." in full_text
        assert paper.source == "pymupdf"


class TestScannedPDFIntegration:
    pytestmark = pytest.mark.asyncio

    async def test_needs_ocr_propagates_through_router(self, scanned_pdf: pathlib.Path) -> None:
        router = ParserRouter(grobid_client=_GrobidTimeoutClient())
        with pytest.raises(ParseError, match="needs_ocr"):
            await router.parse(str(scanned_pdf))

    async def test_needs_ocr_even_when_grobid_fails(self, scanned_pdf: pathlib.Path) -> None:
        router = ParserRouter(grobid_client=_GrobidErrorClient())
        with pytest.raises(ParseError, match="needs_ocr"):
            await router.parse(str(scanned_pdf))


class TestBothParsersFailIntegration:
    pytestmark = pytest.mark.asyncio

    async def test_combined_error_message(self, tmp_path: pathlib.Path) -> None:
        bad = tmp_path / "corrupted.pdf"
        bad.write_bytes(b"%PDF-1.4 trash\x00\x00")
        router = ParserRouter(grobid_client=_GrobidTimeoutClient())
        with pytest.raises(ParseError) as exc_info:
            await router.parse(str(bad))
        assert "GROBID unavailable" in str(exc_info.value)
