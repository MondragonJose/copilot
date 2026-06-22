"""Tests for the rule-based parser router (GROBID → PyMuPDF → needs_ocr)."""

from __future__ import annotations

import pathlib

import fitz
import httpx
import pytest

from core.errors import ParseError
from ingest.router import ParserRouter

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------
# Fixture PDFs (ephemeral, created with fitz)
# --------------------------------------------------------------------------


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
    doc.new_page()  # blank — no text layer
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


# --------------------------------------------------------------------------
# Mock GROBID HTTP clients for the three scenarios
# --------------------------------------------------------------------------

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
    """GROBID returns valid TEI XML."""

    async def post(self, url: str, **kwargs: object) -> _FakeResponse:
        return _FakeResponse(GROBID_TEI)


class _GrobidTimeoutClient:
    """GROBID is unreachable — simulates a timeout."""

    async def post(self, url: str, **kwargs: object) -> _FakeResponse:
        raise httpx.TimeoutException("Connection timeout after 0.001s")


class _GrobidErrorClient:
    """GROBID returns a server error."""

    async def post(self, url: str, **kwargs: object) -> _FakeResponse:
        return _FakeResponse("Internal Server Error", status_code=500)


# --------------------------------------------------------------------------
# GROBID succeeds
# --------------------------------------------------------------------------


class TestGrobidSucceeds:

    async def test_uses_grobid_result(self, native_pdf: pathlib.Path) -> None:
        router = ParserRouter(grobid_client=_GrobidSuccessClient())
        paper, full_text = await router.parse(str(native_pdf))

        assert paper.title == "GROBID Title"
        assert paper.year == 2024
        assert paper.source == "grobid"
        assert "GROBID extracted text" in full_text
        assert len(paper.meta["sections"]) >= 1

    async def test_grobid_returns_paper_source_grobid(
        self, native_pdf: pathlib.Path,
    ) -> None:
        router = ParserRouter(grobid_client=_GrobidSuccessClient())
        paper, _ = await router.parse(str(native_pdf))
        assert paper.source == "grobid"


# --------------------------------------------------------------------------
# GROBID times out → fallback to PyMuPDF
# --------------------------------------------------------------------------


class TestFallbackOnTimeout:

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


# --------------------------------------------------------------------------
# GROBID HTTP error → fallback to PyMuPDF
# --------------------------------------------------------------------------


class TestFallbackOnHttpError:

    async def test_fallback_on_500(self, native_pdf: pathlib.Path) -> None:
        router = ParserRouter(grobid_client=_GrobidErrorClient())
        paper, full_text = await router.parse(str(native_pdf))

        assert "Page one content." in full_text
        assert paper.source == "pymupdf"


# --------------------------------------------------------------------------
# Scanned PDF (no text layer) — needs_ocr propagates
# --------------------------------------------------------------------------


class TestScannedPDF:

    async def test_needs_ocr_propagates_through_router(
        self, scanned_pdf: pathlib.Path,
    ) -> None:
        router = ParserRouter(grobid_client=_GrobidTimeoutClient())
        with pytest.raises(ParseError, match="needs_ocr"):
            await router.parse(str(scanned_pdf))

    async def test_needs_ocr_even_when_grobid_fails(
        self, scanned_pdf: pathlib.Path,
    ) -> None:
        router = ParserRouter(grobid_client=_GrobidErrorClient())
        with pytest.raises(ParseError, match="needs_ocr"):
            await router.parse(str(scanned_pdf))


# --------------------------------------------------------------------------
# Both parsers fail (non-OCR)
# --------------------------------------------------------------------------


class TestBothParsersFail:

    async def test_combined_error_message(self, tmp_path: pathlib.Path) -> None:
        """Neither GROBID nor PyMuPDF can open a corrupted file."""
        bad = tmp_path / "corrupted.pdf"
        bad.write_bytes(b"%PDF-1.4 trash\x00\x00")

        router = ParserRouter(grobid_client=_GrobidTimeoutClient())
        with pytest.raises(ParseError) as exc_info:
            await router.parse(str(bad))

        msg = str(exc_info.value)
        assert "GROBID unavailable" in msg
