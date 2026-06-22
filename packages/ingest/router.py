"""Rule-based parser router per Blueprint §4.

Selection logic:
  1. Try GROBID → returns (Paper, full_text) with sections + refs + metadata.
  2. If GROBID raises ParseError → fall back to PyMuPDF.
  3. If PyMuPDF raises ParseError with ``"needs_ocr"`` → propagate (caller
     marks the job dead).
  4. If both fail → raise ParseError with combined message.
"""

from __future__ import annotations

import httpx

from core.errors import ParseError
from core.models import Paper
from ingest.parsers.grobid import GrobidParser
from ingest.parsers.pymupdf import PyMuPDFParser


class ParserRouter:
    """Rule-based parser selection: GROBID → PyMuPDF → needs_ocr.

    Parameters forwarded to ``GrobidParser``:
        grobid_url     — GROBID HTTP endpoint (default $GROBID_URL / localhost:8070)
        grobid_timeout — request timeout in seconds (default 30.0)
        grobid_client  — injectable ``httpx.AsyncClient`` for testing

    ``PyMuPDFParser`` uses its own defaults (no config needed).
    """

    def __init__(
        self,
        grobid_url: str | None = None,
        grobid_timeout: float = 30.0,
        grobid_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._grobid = GrobidParser(
            grobid_url=grobid_url,
            timeout=grobid_timeout,
            client=grobid_client,
        )
        self._pymupdf = PyMuPDFParser()

    async def parse(self, pdf_path: str) -> tuple[Paper, str]:
        """Parse *pdf_path* by trying GROBID first, then falling back to PyMuPDF.

        Returns:
            ``(Paper, full_text)`` from whichever parser succeeded.
        Raises:
            ParseError — if both parsers fail, or the PDF has no text layer.
        """
        try:
            return await self._grobid.parse(pdf_path)
        except ParseError:
            pass  # fall through to PyMuPDF

        try:
            return await self._pymupdf.parse(pdf_path)
        except ParseError as exc:
            # Propagate needs_ocr so the caller can mark the job dead
            err_msg = str(exc)
            if "needs_ocr" in err_msg:
                raise
            raise ParseError(
                f"GROBID unavailable and PyMuPDF failed: {err_msg}",
            ) from exc
