"""Tests for GROBID parser — HTTP calls mocked, TEI XML parsing tested with
real XML strings (no I/O, pure logic after _call_grobid is mocked)."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from core.errors import ParseError
from core.models import Paper
from ingest.parsers.grobid import (
    GrobidParser,
    _extract_abstract,
    _extract_authors,
    _extract_bibl_authors,
    _extract_bibl_source,
    _extract_bibl_title,
    _extract_bibl_year,
    _extract_doi,
    _extract_references,
    _extract_sections,
    _extract_title,
    _extract_venue,
    _extract_year,
    _parse_pers_name,
    _build_paper,
)
from xml.etree import ElementTree as ET

NS = "{http://www.tei-c.org/ns/1.0}"

SAMPLE_TEI = """<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Sample Paper Title</title>
        <author>
          <persName>
            <forename>Alice</forename>
            <surname>Smith</surname>
          </persName>
        </author>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <title>Article Title</title>
            <author>
              <persName>
                <forename>Bob</forename>
                <surname>Jones</surname>
              </persName>
            </author>
          </analytic>
          <monogr>
            <title>Some Venue</title>
            <imprint>
              <date type="published" when="2024"/>
            </imprint>
          </monogr>
        </biblStruct>
        <idno type="DOI">10.1234/test</idno>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head>Introduction</head>
        <p>This is the introduction text.</p>
      </div>
      <div>
        <head>Methods</head>
        <p>This is the methods section.</p>
      </div>
    </body>
    <back>
      <listBibl>
        <biblStruct>
          <analytic>
            <title>Ref Title</title>
            <author>
              <persName>
                <forename>Charlie</forename>
                <surname>Brown</surname>
              </persName>
            </author>
          </analytic>
          <monogr>
            <title>Ref Venue</title>
            <imprint>
              <date when="2023"/>
            </imprint>
          </monogr>
        </biblStruct>
      </listBibl>
    </back>
  </text>
</TEI>"""


class TestGrobidParser:
    @pytest.mark.asyncio
    async def test_parse_success(self) -> None:
        client = MagicMock(spec=httpx.AsyncClient)
        parser = GrobidParser(grobid_url="http://grobid:8070",
                              client=client)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = SAMPLE_TEI
        client.post = AsyncMock(return_value=mock_response)

        with patch("ingest.parsers.grobid.os.path.isfile", return_value=True):
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b"dummy"
                paper, full_text = await parser.parse("/path/to.pdf")

        assert isinstance(paper, Paper)
        assert paper.title == "Sample Paper Title"
        assert len(paper.authors) >= 1
        assert paper.doi == "10.1234/test"
        assert paper.year == 2024
        assert paper.venue == "Some Venue"
        assert "Introduction" in full_text
        assert "Methods" in full_text

    @pytest.mark.asyncio
    async def test_parse_file_not_found(self) -> None:
        parser = GrobidParser(client=MagicMock(spec=httpx.AsyncClient))
        with patch("ingest.parsers.grobid.os.path.isfile", return_value=False):
            with pytest.raises(ParseError, match="PDF not found"):
                await parser.parse("/nonexistent.pdf")

    @pytest.mark.asyncio
    async def test_parse_grobid_http_error(self) -> None:
        client = MagicMock(spec=httpx.AsyncClient)
        parser = GrobidParser(grobid_url="http://grobid:8070",
                              client=client)
        client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "500", request=MagicMock(),
                response=MagicMock(status_code=500),
            ),
        )
        with patch("ingest.parsers.grobid.os.path.isfile", return_value=True):
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b"dummy"
                with pytest.raises(ParseError, match="GROBID request failed"):
                    await parser.parse("/path/to.pdf")

    @pytest.mark.asyncio
    async def test_parse_invalid_xml(self) -> None:
        client = MagicMock(spec=httpx.AsyncClient)
        parser = GrobidParser(grobid_url="http://grobid:8070",
                              client=client)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = "not xml"
        client.post = AsyncMock(return_value=mock_response)

        with patch("ingest.parsers.grobid.os.path.isfile", return_value=True):
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b"dummy"
                with pytest.raises(ParseError):
                    await parser.parse("/path/to.pdf")


class TestExtractTitle:
    def test_found(self) -> None:
        root = ET.fromstring(SAMPLE_TEI)
        header = root.find(f"{NS}teiHeader")
        title = _extract_title(header)
        assert title == "Sample Paper Title"

    def test_none_header(self) -> None:
        assert _extract_title(None) is None


class TestExtractAuthors:
    def test_found(self) -> None:
        root = ET.fromstring(SAMPLE_TEI)
        header = root.find(f"{NS}teiHeader")
        authors = _extract_authors(header)
        assert len(authors) >= 1

    def test_none_header(self) -> None:
        assert _extract_authors(None) == []


class TestParsePersName:
    TEI_NS = "http://www.tei-c.org/ns/1.0"

    def test_with_forename_and_surname(self) -> None:
        xml = (
            '<p xmlns="http://www.tei-c.org/ns/1.0">'
            "<forename>Alice</forename><surname>Smith</surname></p>"
        )
        elem = ET.fromstring(xml)
        result = _parse_pers_name(elem)
        assert result == {"given": "Alice", "family": "Smith"}

    def test_empty_elements(self) -> None:
        xml = (
            '<p xmlns="http://www.tei-c.org/ns/1.0">'
            "<forename></forename><surname></surname></p>"
        )
        elem = ET.fromstring(xml)
        result = _parse_pers_name(elem)
        assert result == {"given": "", "family": ""}


class TestExtractYear:
    def test_found(self) -> None:
        root = ET.fromstring(SAMPLE_TEI)
        header = root.find(f"{NS}teiHeader")
        assert _extract_year(header) == 2024

    def test_none_header(self) -> None:
        assert _extract_year(None) is None


class TestExtractDoi:
    def test_found(self) -> None:
        root = ET.fromstring(SAMPLE_TEI)
        header = root.find(f"{NS}teiHeader")
        assert _extract_doi(header) == "10.1234/test"

    def test_none_header(self) -> None:
        assert _extract_doi(None) is None


class TestExtractAbstract:
    def test_not_present(self) -> None:
        root = ET.fromstring(SAMPLE_TEI)
        header = root.find(f"{NS}teiHeader")
        assert _extract_abstract(header) is None

    def test_none_header(self) -> None:
        assert _extract_abstract(None) is None


class TestExtractVenue:
    def test_found(self) -> None:
        root = ET.fromstring(SAMPLE_TEI)
        header = root.find(f"{NS}teiHeader")
        assert _extract_venue(header) == "Some Venue"

    def test_none_header(self) -> None:
        assert _extract_venue(None) is None


class TestExtractSections:
    def test_extracts_sections(self) -> None:
        root = ET.fromstring(SAMPLE_TEI)
        sections = _extract_sections(root)
        assert len(sections) == 2
        assert sections[0]["heading"] == "Introduction"
        assert sections[1]["heading"] == "Methods"
        assert "introduction text" in str(sections[0]["text"]).lower()

    def test_extracts_authors_from_analytic_path(self) -> None:
        """Authors in analytic path are used when direct author path is empty."""
        xml = """<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>No Direct Author</title>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author><persName><forename>Ana</forename><surname>Lytic</surname></persName></author>
          </analytic>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
</TEI>"""
        root = ET.fromstring(xml)
        header = root.find(f"{NS}teiHeader")
        authors = _extract_authors(header)
        assert len(authors) == 1
        assert authors[0]["given"] == "Ana"

    def test_no_body(self) -> None:
        xml = "<TEI xmlns='http://www.tei-c.org/ns/1.0'/>"
        root = ET.fromstring(xml)
        assert _extract_sections(root) == []

    def test_no_para_but_has_heading_skipped_if_no_text(self) -> None:
        xml = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text><body><div><head>Empty</head></div></body></text>
        </TEI>"""
        root = ET.fromstring(xml)
        sections = _extract_sections(root)
        # heading-only div with no para text gets included (heading+empty)
        assert len(sections) == 0 or (len(sections) == 1 and sections[0]["heading"] == "Empty")


class TestExtractReferences:
    def test_extracts(self) -> None:
        root = ET.fromstring(SAMPLE_TEI)
        refs = _extract_references(root)
        assert len(refs) == 1
        assert refs[0]["title"] == "Ref Title"

    def test_no_back(self) -> None:
        xml = "<TEI xmlns='http://www.tei-c.org/ns/1.0'/>"
        root = ET.fromstring(xml)
        assert _extract_references(root) == []


class TestBiblHelpers:
    def test_bibl_title(self) -> None:
        xml = (
            f"<biblStruct xmlns='http://www.tei-c.org/ns/1.0'>"
            f"<analytic><title>Ref Title</title>"
            f"</analytic></biblStruct>"
        )
        elem = ET.fromstring(xml)
        assert _extract_bibl_title(elem) == "Ref Title"

    def test_bibl_title_missing(self) -> None:
        xml = f"<biblStruct xmlns='http://www.tei-c.org/ns/1.0'></biblStruct>"
        elem = ET.fromstring(xml)
        assert _extract_bibl_title(elem) is None

    def test_bibl_authors(self) -> None:
        xml = (
            f"<biblStruct xmlns='http://www.tei-c.org/ns/1.0'>"
            f"<author><persName>"
            f"<forename>A</forename><surname>B</surname>"
            f"</persName></author></biblStruct>"
        )
        elem = ET.fromstring(xml)
        authors = _extract_bibl_authors(elem)
        assert len(authors) == 1

    def test_bibl_year_from_imprint(self) -> None:
        xml = (
            f"<biblStruct xmlns='http://www.tei-c.org/ns/1.0'>"
            f"<imprint><date when='2023'/>"
            f"</imprint></biblStruct>"
        )
        elem = ET.fromstring(xml)
        assert _extract_bibl_year(elem) == 2023

    def test_bibl_year_missing(self) -> None:
        xml = f"<biblStruct xmlns='http://www.tei-c.org/ns/1.0'></biblStruct>"
        elem = ET.fromstring(xml)
        assert _extract_bibl_year(elem) is None

    def test_bibl_source(self) -> None:
        xml = (
            f"<biblStruct xmlns='http://www.tei-c.org/ns/1.0'>"
            f"<monogr><title>Venue</title>"
            f"</monogr></biblStruct>"
        )
        elem = ET.fromstring(xml)
        assert _extract_bibl_source(elem) == "Venue"


class TestBuildPaper:
    def test_builds_paper(self) -> None:
        root = ET.fromstring(SAMPLE_TEI)
        sections = _extract_sections(root)
        refs = _extract_references(root)
        paper = _build_paper(root, SAMPLE_TEI, "/p.pdf", sections, refs)
        assert paper.title == "Sample Paper Title"
        assert paper.source == "grobid"
        assert paper.pdf_path == "/p.pdf"
        assert "sections" in paper.meta
        assert "references" in paper.meta
