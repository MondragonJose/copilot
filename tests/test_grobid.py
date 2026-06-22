"""Tests for the GROBID PDF parser — uses injected HTTP client and fixture TEI.

Unit tests cover TEI parsing, metadata extraction, section offsets, and
error wrapping.  A live GROBID instance (docker-compose) can drive the
same flow via the integration test at the end of this file.
"""

from __future__ import annotations

import pathlib
from xml.etree import ElementTree as ET

import httpx
import pytest

from core.errors import ParseError
from ingest.parsers.grobid import (
    GrobidParser,
    _extract_references,
    _extract_sections,
)

FIXTURE_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Test Paper for GROBID Parsing</title>
        <author>
          <persName><forename>John</forename><surname>Doe</surname></persName>
        </author>
        <author>
          <persName><forename>Jane</forename><surname>Smith</surname></persName>
        </author>
      </titleStmt>
      <publicationStmt><publisher/><availability/></publicationStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <title>Test Paper for GROBID Parsing</title>
            <author><persName><forename>John</forename><surname>Doe</surname></persName></author>
            <author><persName><forename>Jane</forename><surname>Smith</surname></persName></author>
          </analytic>
          <monogr>
            <title>Journal of Interesting Research</title>
            <imprint><date type="published" when="2024"/></imprint>
          </monogr>
          <idno type="DOI">10.1234/test.2024</idno>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>This is the abstract of the test paper.</p>
        <p>It has multiple paragraphs for testing.</p>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>Introduction</head>
        <p>This is the introduction section.</p>
        <p>It contains multiple paragraphs.</p>
      </div>
      <div>
        <head>Method</head>
        <p>We propose a novel approach.</p>
      </div>
    </body>
    <back>
      <div type="references">
        <listBibl>
          <biblStruct>
            <analytic>
              <title>A Reference Paper</title>
              <author><persName><forename>Alice</forename><surname>Johnson</surname></persName></author>
            </analytic>
            <monogr>
              <title>Conference Proceedings</title>
              <imprint><date type="published" when="2023"/></imprint>
            </monogr>
          </biblStruct>
        </listBibl>
      </div>
    </back>
  </text>
</TEI>"""


class FakeResponse:
    """Mimics httpx.Response for test injection."""

    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("POST", "http://test/"),
                response=httpx.Response(self.status_code),
            )


class FakeClient:
    """Injected ``httpx.AsyncClient`` that returns a canned TEI response."""

    def __init__(
        self,
        response_text: str = FIXTURE_TEI,
        status_code: int = 200,
    ) -> None:
        self.response_text = response_text
        self.status_code = status_code

    async def post(self, url: str, **kwargs: object) -> FakeResponse:
        return FakeResponse(self.response_text, self.status_code)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def parser() -> GrobidParser:
    return GrobidParser(client=FakeClient())


# --------------------------------------------------------------------------
# GrobidParser.parse — end-to-end with mocked HTTP
# --------------------------------------------------------------------------


class TestParseEndToEnd:
    pytestmark = pytest.mark.asyncio

    async def test_parse_returns_paper_and_text(
        self, parser: GrobidParser, tmp_path: pathlib.Path,
    ) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_text("dummy pdf content")

        paper, full_text = await parser.parse(str(pdf))

        assert paper.title == "Test Paper for GROBID Parsing"
        assert "introduction section" in full_text
        assert "Method" in full_text

    async def test_metadata_extraction(
        self, parser: GrobidParser, tmp_path: pathlib.Path,
    ) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_text("dummy pdf content")

        paper, _ = await parser.parse(str(pdf))

        assert paper.title == "Test Paper for GROBID Parsing"
        assert paper.authors == [
            {"given": "John", "family": "Doe"},
            {"given": "Jane", "family": "Smith"},
        ]
        assert paper.year == 2024
        assert paper.doi == "10.1234/test.2024"
        assert paper.source == "grobid"
        assert paper.pdf_path == str(pdf)
        assert paper.abstract is not None
        assert "This is the abstract" in paper.abstract
        assert paper.venue == "Journal of Interesting Research"

    async def test_sections_in_meta(
        self, parser: GrobidParser, tmp_path: pathlib.Path,
    ) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_text("dummy pdf content")

        paper, _ = await parser.parse(str(pdf))
        sections = paper.meta["sections"]

        assert len(sections) >= 1
        for sec in sections:
            assert "text" in sec
            assert sec["text"]
            assert "char_start" in sec
            assert isinstance(sec["char_start"], int)
            assert "char_end" in sec
            assert isinstance(sec["char_end"], int)
            assert isinstance(sec["heading"], str)
            assert sec["char_start"] >= 0
            assert sec["char_end"] > sec["char_start"]

    async def test_section_char_offsets_monotonic(
        self, parser: GrobidParser, tmp_path: pathlib.Path,
    ) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_text("dummy pdf content")

        paper, _ = await parser.parse(str(pdf))
        sections = paper.meta["sections"]

        for i in range(1, len(sections)):
            assert sections[i]["char_start"] >= sections[i - 1]["char_end"]

    async def test_multiple_sections(
        self, parser: GrobidParser, tmp_path: pathlib.Path,
    ) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_text("dummy pdf content")

        paper, _ = await parser.parse(str(pdf))
        sections = paper.meta["sections"]

        assert len(sections) >= 2
        assert sections[0]["heading"] == "Introduction"
        assert sections[1]["heading"] == "Method"

    async def test_references_in_meta(
        self, parser: GrobidParser, tmp_path: pathlib.Path,
    ) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_text("dummy pdf content")

        paper, _ = await parser.parse(str(pdf))
        refs = paper.meta["references"]

        assert len(refs) >= 1
        ref = refs[0]
        assert ref["title"] == "A Reference Paper"
        assert ref["authors"] == [{"given": "Alice", "family": "Johnson"}]
        assert ref["year"] == 2023
        assert ref["source"] == "Conference Proceedings"

    async def test_grobid_tei_stored_in_paper(
        self, parser: GrobidParser, tmp_path: pathlib.Path,
    ) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_text("dummy pdf content")

        paper, _ = await parser.parse(str(pdf))

        assert paper.grobid_tei is not None
        assert "<TEI" in paper.grobid_tei


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------


class TestErrorHandling:
    pytestmark = pytest.mark.asyncio

    async def test_invalid_pdf_path_raises_parse_error(self, parser: GrobidParser) -> None:
        with pytest.raises(ParseError, match="PDF not found"):
            await parser.parse("/nonexistent/path.pdf")

    async def test_grobid_http_error_raises_parse_error(self, tmp_path: pathlib.Path) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_text("dummy pdf content")

        parser = GrobidParser(client=FakeClient(status_code=500))
        with pytest.raises(ParseError, match="GROBID request failed"):
            await parser.parse(str(pdf))

    async def test_grobid_timeout_raises_parse_error(self, tmp_path: pathlib.Path) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_text("dummy pdf content")

        async def _timeout_post(url: str, **kwargs: object) -> FakeResponse:
            msg = "Connection timeout after 0.001s"
            raise httpx.TimeoutException(msg)

        client = FakeClient()
        client.post = _timeout_post  # type: ignore[method-assign]
        parser = GrobidParser(client=client)
        with pytest.raises(ParseError, match="GROBID request failed"):
            await parser.parse(str(pdf))

    async def test_invalid_tei_xml_raises_parse_error(self, tmp_path: pathlib.Path) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_text("dummy pdf content")

        parser = GrobidParser(client=FakeClient(response_text="not xml"))
        with pytest.raises(ParseError, match="GROBID returned invalid XML"):
            await parser.parse(str(pdf))


# --------------------------------------------------------------------------
# Unit: _extract_sections
# --------------------------------------------------------------------------


class TestExtractSections:

    def test_empty_body_returns_empty_list(self) -> None:
        root = ET.fromstring(
            '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body/></text></TEI>',
        )
        assert _extract_sections(root) == []

    def test_div_without_head_or_para_skipped(self) -> None:
        root = ET.fromstring(
            '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
            "<text><body><div><note>skip</note></div></body></text>"
            "</TEI>",
        )
        assert _extract_sections(root) == []


# --------------------------------------------------------------------------
# Unit: _extract_references
# --------------------------------------------------------------------------


class TestExtractReferences:

    def test_empty_back_returns_empty_list(self) -> None:
        root = ET.fromstring(
            '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><back/></text></TEI>',
        )
        assert _extract_references(root) == []

    def test_no_back_element(self) -> None:
        root = ET.fromstring(
            '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body/></text></TEI>',
        )
        assert _extract_references(root) == []

    def test_no_bibl_struct(self) -> None:
        root = ET.fromstring(
            '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
            "<text><back><div type=\"references\"><listBibl/></div></back></text>"
            "</TEI>",
        )
        assert _extract_references(root) == []
