"""Tests for DOI metadata resolution — external API calls mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from core.errors import IngestError
from ingest.doi import _build_paper, resolve_doi

# ---------------------------------------------------------------------------
# _build_paper — unit tests (no network)
# ---------------------------------------------------------------------------


class TestBuildPaper:
    DOI = "10.1234/example"

    def test_full_metadata(self) -> None:
        msg = {
            "title": ["A Full Paper Title"],
            "author": [
                {"given": "John", "family": "Doe"},
                {"given": "Jane", "family": "Smith"},
            ],
            "published-print": {"date-parts": [[2024, 6]]},
            "container-title": ["Journal of Examples"],
            "abstract": "<jats:p>An important abstract.</jats:p>",
        }
        paper = _build_paper(self.DOI, msg)

        assert paper.doi == self.DOI
        assert paper.title == "A Full Paper Title"
        assert paper.authors == [
            {"given": "John", "family": "Doe"},
            {"given": "Jane", "family": "Smith"},
        ]
        assert paper.year == 2024
        assert paper.venue == "Journal of Examples"
        assert paper.abstract == "<jats:p>An important abstract.</jats:p>"
        assert paper.source == "crossref"
        assert paper.open_access is None
        assert paper.pdf_path is None
        assert paper.grobid_tei is None
        assert "crossref" in paper.meta

    def test_minimal_metadata(self) -> None:
        msg: dict = {}
        paper = _build_paper(self.DOI, msg)

        assert paper.doi == self.DOI
        assert paper.title == "Untitled"
        assert paper.authors == []
        assert paper.year is None
        assert paper.venue is None
        assert paper.abstract is None

    def test_picks_online_date_when_no_print(self) -> None:
        msg = {
            "published-online": {"date-parts": [[2023]]},
        }
        paper = _build_paper(self.DOI, msg)
        assert paper.year == 2023

    def test_falls_back_to_issued(self) -> None:
        msg = {
            "issued": {"date-parts": [[2022]]},
        }
        paper = _build_paper(self.DOI, msg)
        assert paper.year == 2022

    def test_handles_missing_author_fields(self) -> None:
        msg = {
            "author": [
                {"given": "John"},  # no family
                {"family": "Doe"},   # no given
                {},                   # empty
            ],
        }
        paper = _build_paper(self.DOI, msg)
        assert paper.authors == [
            {"given": "John", "family": ""},
            {"given": "", "family": "Doe"},
            {"given": "", "family": ""},
        ]

    def test_title_list_empty(self) -> None:
        msg = {"title": []}
        paper = _build_paper(self.DOI, msg)
        assert paper.title == "Untitled"

    def test_venue_list_empty(self) -> None:
        msg = {"container-title": []}
        paper = _build_paper(self.DOI, msg)
        assert paper.venue is None


# ---------------------------------------------------------------------------
# resolve_doi — integration with mocked httpx client
# ---------------------------------------------------------------------------


def _mock_client(status: int = 200, body: object = None) -> httpx.AsyncClient:
    """Build an ``AsyncClient`` whose ``get`` returns a canned response."""
    response = httpx.Response(status_code=status, json=body or {})
    client = httpx.AsyncClient()
    client.get = AsyncMock(return_value=response)  # type: ignore[method-assign]
    return client


class TestResolveDoi:

    @pytest.mark.asyncio
    async def test_valid_doi_returns_paper(self) -> None:
        body = {
            "message": {
                "title": ["Resolved Title"],
                "author": [{"given": "Alice", "family": "Jones"}],
                "published-print": {"date-parts": [[2025]]},
            },
        }
        client = _mock_client(body=body)
        paper = await resolve_doi("10.1234/test", client=client)

        assert paper.doi == "10.1234/test"
        assert paper.title == "Resolved Title"
        assert paper.authors == [{"given": "Alice", "family": "Jones"}]

    @pytest.mark.asyncio
    async def test_http_404_raises_ingest_error(self) -> None:
        client = _mock_client(status=404, body={"error": "not found"})
        with pytest.raises(IngestError, match="HTTP 404"):
            await resolve_doi("10.1234/missing", client=client)

    @pytest.mark.asyncio
    async def test_http_500_raises_ingest_error(self) -> None:
        client = _mock_client(status=500, body={"error": "server error"})
        with pytest.raises(IngestError, match="HTTP 500"):
            await resolve_doi("10.1234/broken", client=client)

    @pytest.mark.asyncio
    async def test_timeout_raises_ingest_error(self) -> None:
        client = httpx.AsyncClient()
        client.get = AsyncMock(  # type: ignore[method-assign]
            side_effect=httpx.TimeoutException("timed out"),
        )
        with pytest.raises(IngestError, match="timeout"):
            await resolve_doi("10.1234/slow", client=client)

    @pytest.mark.asyncio
    async def test_missing_message_field_raises_ingest_error(self) -> None:
        client = _mock_client(body={"status": "ok"})  # no "message" key
        with pytest.raises(IngestError, match="no 'message' field"):
            await resolve_doi("10.1234/weird", client=client)
