"""PyMuPDF-based PDF parser — extracts plain text with per-page char offsets.

This is the fallback parser used when GROBID is unavailable or times out.
For PDFs without a text layer (scanned documents) raises ``ParseError``
with ``"needs_ocr"`` in the message so the router can mark the job dead.

Usage:
    parser = PyMuPDFParser()
    paper, full_text = await parser.parse("/path/to/paper.pdf")
    for span in paper.meta["pages"]:
        print(span["page"], span["char_start"], span["char_end"])
"""

from __future__ import annotations

import os
import uuid

import fitz

from core.errors import ParseError
from core.models import Paper


class PyMuPDFParser:
    """Parse PDFs via PyMuPDF (fitz), extracting plain text with page spans."""

    async def parse(self, pdf_path: str) -> tuple[Paper, str]:
        if not os.path.isfile(pdf_path):
            raise ParseError(f"PDF not found: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            raise ParseError(f"Failed to open PDF {pdf_path}: {exc}") from exc

        try:
            if doc.page_count == 0:
                raise ParseError(f"PDF is empty (0 pages): {pdf_path}")

            pages_text: list[str] = []
            spans: list[dict[str, int | str]] = []
            cursor = 0

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = page.get_text().strip()
                if text:
                    sep = "\n\n" if pages_text else ""
                    pages_text.append(text)
                    char_start = cursor + len(sep)
                    char_end = char_start + len(text)
                    spans.append({
                        "page": page_num,
                        "text": text,
                        "char_start": char_start,
                        "char_end": char_end,
                    })
                    cursor = char_end

            full_text = "\n\n".join(pages_text)

            if not full_text.strip():
                raise ParseError(
                    "needs_ocr: no extractable text layer — scanned PDF or image-only",
                )

            paper = Paper(
                id=str(uuid.uuid4()),
                doi=None,
                title=_title_from_path(pdf_path),
                authors=[],
                year=None,
                venue=None,
                abstract=None,
                source="pymupdf",
                open_access=None,
                pdf_path=pdf_path,
                grobid_tei=None,
                meta={"pages": spans},
            )

            return paper, full_text

        finally:
            doc.close()


def _title_from_path(pdf_path: str) -> str:
    """Derive a readable title from the PDF filename."""
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    stem = stem.replace("_", " ").replace("-", " ").strip().strip(".")
    return stem or "Untitled"
