"""Tests for ParserRouter — GROBID and PyMuPDF are mocked."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from core.errors import ParseError
from ingest.router import ParserRouter


@pytest.fixture
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
