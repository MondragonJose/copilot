"""GROBID-based PDF parser — sends PDF to GROBID HTTP API, parses TEI XML,
returns Paper metadata + full text with structured sections in meta["sections"].

Usage:
    parser = GrobidParser()
    paper, full_text = await parser.parse("/path/to/paper.pdf")
    for sec in paper.meta["sections"]:
        print(sec["heading"], sec["char_start"], sec["char_end"])
"""

from __future__ import annotations

import os
import uuid
from xml.etree import ElementTree as ET

import httpx

from core.errors import ParseError
from core.models import Paper

NS = "{http://www.tei-c.org/ns/1.0}"
DEFAULT_GROBID_URL = "http://localhost:8070"
DEFAULT_TIMEOUT = 30.0


class GrobidParser:
    """Parse PDFs via the GROBID /api/processFulltextDocument endpoint."""

    def __init__(
        self,
        grobid_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.grobid_url = (
            grobid_url or os.environ.get("GROBID_URL", DEFAULT_GROBID_URL)
        ).rstrip("/")
        self.timeout = timeout
        self._client = client

    async def parse(self, pdf_path: str) -> tuple[Paper, str]:
        """Parse a PDF and return (Paper metadata, full text).

        Sections with char offsets are stored in ``Paper.meta["sections"]``.
        References are stored in ``Paper.meta["references"]``.
        """
        if not os.path.isfile(pdf_path):
            raise ParseError(f"PDF not found: {pdf_path}")

        try:
            tei_xml = await self._call_grobid(pdf_path)
        except Exception as exc:
            raise ParseError(f"GROBID request failed for {pdf_path}: {exc}") from exc

        try:
            root = ET.fromstring(tei_xml)
        except ET.ParseError as exc:
            raise ParseError(f"GROBID returned invalid XML: {exc}") from exc

        sections = _extract_sections(root)
        full_text = "\n\n".join(str(s["text"]) for s in sections)
        refs = _extract_references(root)
        paper = _build_paper(root, tei_xml, pdf_path, sections, refs)

        return paper, full_text

    async def _call_grobid(self, pdf_path: str) -> str:
        """POST PDF to GROBID, return TEI XML string."""
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        owned = self._client is None
        try:
            with open(pdf_path, "rb") as f:
                files = {"input": (os.path.basename(pdf_path), f, "application/pdf")}
                resp = await client.post(
                    f"{self.grobid_url}/api/processFulltextDocument",
                    files=files,
                )
            resp.raise_for_status()
            return resp.text
        finally:
            if owned:
                await client.aclose()


# ---------------------------------------------------------------------------
# Pure helpers — typed, no I/O, no self
# ---------------------------------------------------------------------------


def _extract_title(header: ET.Element | None) -> str | None:
    if header is None:
        return None
    paths = (
        f"{NS}fileDesc/{NS}titleStmt/{NS}title",
        f"{NS}fileDesc/{NS}sourceDesc/{NS}biblStruct/{NS}analytic/{NS}title",
    )
    for path in paths:
        elem = header.find(f".//{path}")
        if elem is not None and elem.text:
            return elem.text.strip()
    return None


def _extract_authors(header: ET.Element | None) -> list[dict[str, str]]:
    if header is None:
        return []
    authors: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pers in header.findall(f".//{NS}author/{NS}persName"):
        author = _parse_pers_name(pers)
        key = (author["given"], author["family"])
        if key not in seen and (author["given"] or author["family"]):
            seen.add(key)
            authors.append(author)
    if not authors:
        for pers in header.findall(f".//{NS}analytic/{NS}author/{NS}persName"):
            author = _parse_pers_name(pers)
            key = (author["given"], author["family"])
            if key not in seen and (author["given"] or author["family"]):
                seen.add(key)
                authors.append(author)
    return authors


def _parse_pers_name(pers: ET.Element) -> dict[str, str]:
    fore = pers.find(f"{NS}forename")
    sur = pers.find(f"{NS}surname")
    given = fore.text.strip() if fore is not None and fore.text else ""
    family = sur.text.strip() if sur is not None and sur.text else ""
    return {"given": given, "family": family}


def _extract_year(header: ET.Element | None) -> int | None:
    if header is None:
        return None
    for path in (f".//{NS}date[@type='published']", f".//{NS}imprint/{NS}date"):
        date = header.find(path)
        if date is not None:
            when = date.get("when", "")
            if when and when.isdigit():
                return int(when)
    return None


def _extract_doi(header: ET.Element | None) -> str | None:
    if header is None:
        return None
    idno = header.find(f".//{NS}idno[@type='DOI']")
    if idno is not None and idno.text:
        return idno.text.strip()
    return None


def _extract_abstract(header: ET.Element | None) -> str | None:
    if header is None:
        return None
    abstract = header.find(f".//{NS}abstract")
    if abstract is None:
        return None
    paras = abstract.findall(f".//{NS}p")
    if paras:
        texts = ["".join(p.itertext()).strip() for p in paras]
        texts = [t for t in texts if t]
        return "\n\n".join(texts) if texts else None
    text = "".join(abstract.itertext()).strip()
    return text or None


def _extract_venue(header: ET.Element | None) -> str | None:
    if header is None:
        return None
    title = header.find(f".//{NS}monogr/{NS}title")
    if title is not None and title.text:
        return title.text.strip()
    return None


def _build_paper(
    root: ET.Element,
    tei_xml: str,
    pdf_path: str,
    sections: list[dict[str, object]],
    refs: list[dict[str, object]],
) -> Paper:
    header = root.find(f"{NS}teiHeader")
    title = _extract_title(header) or "Untitled"
    authors = _extract_authors(header)
    year = _extract_year(header)
    doi = _extract_doi(header)
    abstract = _extract_abstract(header)
    venue = _extract_venue(header)

    return Paper(
        id=str(uuid.uuid4()),
        doi=doi,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        abstract=abstract,
        source="grobid",
        open_access=None,
        pdf_path=pdf_path,
        grobid_tei=tei_xml,
        meta={"sections": sections, "references": refs},
    )


def _extract_sections(root: ET.Element) -> list[dict[str, object]]:
    """Extract structured sections from the TEI body.

    Returns a list of dicts with keys ``heading``, ``text``, ``char_start``,
    ``char_end``.
    """
    text_wrapper = root.find(f"{NS}text")
    body: ET.Element | None
    if text_wrapper is not None:
        body = text_wrapper.find(f"{NS}body")
    else:
        body = root.find(f"{NS}body")
    if body is None:
        return []

    sections: list[dict[str, object]] = []
    cursor = 0

    for child in body:
        if child.tag != f"{NS}div":
            continue

        heading = ""
        heading_elem = child.find(f"{NS}head")
        if heading_elem is not None and heading_elem.text:
            heading = heading_elem.text.strip()

        para_texts: list[str] = []
        for para in child:
            if para.tag == f"{NS}p":
                text = "".join(para.itertext()).strip()
                if text:
                    para_texts.append(text)

        if not para_texts and not heading:
            continue

        sep = "\n\n"
        combined = (heading + sep + sep.join(para_texts)) if heading else sep.join(para_texts)
        combined = combined.strip()
        if not combined:
            continue

        sections.append({
            "heading": heading,
            "text": combined,
            "char_start": cursor,
            "char_end": cursor + len(combined),
        })
        cursor += len(combined) + 2  # account for the "\n\n" separator

    return sections


def _extract_references(root: ET.Element) -> list[dict[str, object]]:
    """Extract bibliographic references from the TEI back matter."""
    text_wrapper = root.find(f"{NS}text")
    back: ET.Element | None
    if text_wrapper is not None:
        back = text_wrapper.find(f"{NS}back")
    else:
        back = root.find(f"{NS}back")
    if back is None:
        return []

    refs: list[dict[str, object]] = []
    for bibl in back.findall(f".//{NS}biblStruct"):
        title = _extract_bibl_title(bibl)
        authors = _extract_bibl_authors(bibl)
        year = _extract_bibl_year(bibl)
        source = _extract_bibl_source(bibl)

        refs.append({
            "title": title,
            "authors": authors,
            "year": year,
            "source": source,
        })

    return refs


def _extract_bibl_title(bibl: ET.Element) -> str | None:
    title_elem = bibl.find(f"{NS}analytic/{NS}title")
    if title_elem is not None and title_elem.text:
        return title_elem.text.strip()
    return None


def _extract_bibl_authors(bibl: ET.Element) -> list[dict[str, str]]:
    authors: list[dict[str, str]] = []
    for pers in bibl.findall(f".//{NS}author/{NS}persName"):
        author = _parse_pers_name(pers)
        if author["given"] or author["family"]:
            authors.append(author)
    return authors


def _extract_bibl_year(bibl: ET.Element) -> int | None:
    for path in (f"{NS}imprint/{NS}date", f"{NS}date[@type='published']"):
        date = bibl.find(f".//{path}")
        if date is not None:
            when = date.get("when", "")
            if when and when.isdigit():
                return int(when)
    return None


def _extract_bibl_source(bibl: ET.Element) -> str | None:
    title = bibl.find(f"{NS}monogr/{NS}title")
    if title is not None and title.text:
        return title.text.strip()
    return None
