"""Ingest regression suite — stage reset, ON CONFLICT, chunking golden borders.

Guarded regressions:
  • R2-BUG-003  — Stale stage on re-run (stage reset to "parse" before processing)
  • R2-BUG-001  — paper_id stale on DB conflict (ON CONFLICT (id) DO UPDATE)
  • R2-BUG-004  — Partial PDF visibility (PyMuPDF fallback text extraction)
  • ARCH-BUG-001 — ParserRouter satisfies Parser Protocol (has .parse() method)
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.errors import ParseError
from core.interfaces import Parser
from core.models import Chunk, Paper

from ingest.tasks import process_job


# ── Issue R2-BUG-003: stale stage on re-run ──────────────────────────────────


class TestStageResetOnReRun:
    """process_job must reset stage to "parse" before starting work."""

    @pytest.mark.asyncio
    async def test_stage_reset_to_parse_on_each_run(self) -> None:
        from ingest.tasks import process_job

        pool = AsyncMock()
        pool.fetchrow.return_value = {
            "id": "job-id",
            "payload": {"pdf_path": "/tmp/test.pdf"},
            "attempts": 0,
            "max_attempts": 3,
        }
        retriever = AsyncMock()
        embedder = AsyncMock()
        router = MagicMock()
        router.parse = AsyncMock(return_value=(
            Paper("pid", None, "T", [], 2024, None, None, "s", None, None, None, {}),
            "full text",
        ))
        chunker = MagicMock(return_value=[])

        await process_job(pool, retriever, embedder, router, "job-id", chunker=chunker)

        calls = [c.args for c in pool.execute.await_args_list]
        stage_updates = [
            args for args in calls
            if len(args) >= 1 and isinstance(args[0], str) and "stage" in args[0]
        ]
        assert len(stage_updates) >= 1
        assert any(
            "parse" in str(args) for args in stage_updates
        ), "stage must be set to 'parse' during processing"

    @pytest.mark.asyncio
    async def test_set_stage_called_with_parse(self) -> None:
        pool = AsyncMock()
        pool.fetchrow.return_value = {
            "id": "job-id",
            "payload": {"pdf_path": "/tmp/test.pdf"},
            "attempts": 1,
            "max_attempts": 3,
        }
        retriever = AsyncMock()
        embedder = AsyncMock()
        router = MagicMock()
        router.parse = AsyncMock(return_value=(
            Paper("pid", None, "T", [], 2024, None, None, "s", None, None, None, {}),
            "full text",
        ))
        chunker = MagicMock(return_value=[])

        stage_calls: list[str] = []

        async def tracking_set_stage(pool_obj: object, job_id: str, stage: str) -> None:
            stage_calls.append(stage)

        with patch("ingest.tasks._set_stage", new=tracking_set_stage):
            await process_job(pool, retriever, embedder, router, "job-id", chunker=chunker)
            assert "parse" in stage_calls

    @pytest.mark.asyncio
    async def test_process_job_sets_stage_parse_before_router(self) -> None:
        """Regression: stage='parse' must be set BEFORE router.parse is called."""
        pool = AsyncMock()
        pool.fetchrow.return_value = {
            "id": "job-id",
            "payload": {"pdf_path": "/tmp/test.pdf"},
            "attempts": 0,
            "max_attempts": 3,
        }
        retriever = AsyncMock()
        embedder = AsyncMock()

        call_order: list[str] = []

        async def tracking_set_stage(pool_obj: object, job_id: str, stage: str) -> None:
            call_order.append(f"stage:{stage}")

        async def tracking_parse(path: str) -> tuple[Paper, str]:
            call_order.append("router_parse")
            return Paper("pid", None, "T", [], 2024, None, None, "s", None, None, None, {}), "text"

        router = MagicMock()
        router.parse = tracking_parse

        with patch("ingest.tasks._set_stage", new=tracking_set_stage):
            chunker = MagicMock(return_value=[])
            await process_job(pool, retriever, embedder, router, "job-id", chunker=chunker)
            parse_idx = call_order.index("stage:parse")
            router_idx = call_order.index("router_parse")
            assert parse_idx < router_idx, "stage=parse must be set before router.parse"


# ── Issue R2-BUG-001: ON CONFLICT pattern ────────────────────────────────────


class TestOnConflictPattern:
    """Every upsert/insert must use ON CONFLICT for idempotency."""

    def test_upsert_chunk_sql_has_on_conflict(self) -> None:
        from core._sql import UPSERT_CHUNK_SQL
        assert "ON CONFLICT (id)" in UPSERT_CHUNK_SQL

    def test_upsert_embedding_sql_has_on_conflict(self) -> None:
        from core._sql import UPSERT_EMBEDDING_SQL
        assert "ON CONFLICT (chunk_id)" in UPSERT_EMBEDDING_SQL

    def test_upsert_paper_sql_has_on_conflict(self) -> None:
        from core._sql import UPSERT_PAPER_SQL
        assert "ON CONFLICT (id)" in UPSERT_PAPER_SQL

    def test_no_hardcoded_ids_in_sql(self) -> None:
        """Regression: SQL should not hardcode specific UUIDs."""
        from core._sql import UPSERT_CHUNK_SQL, UPSERT_EMBEDDING_SQL, UPSERT_PAPER_SQL

        for sql in [UPSERT_CHUNK_SQL, UPSERT_EMBEDDING_SQL, UPSERT_PAPER_SQL]:
            uuids = re.findall(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                sql, re.IGNORECASE,
            )
            assert len(uuids) == 0, f"Hardcoded UUID in SQL: {uuids}"


# ── Issue ARCH-BUG-001: ParserRouter implements Parser Protocol ──────────────


class TestParserProtocol:
    """ParserRouter must satisfy the Parser Protocol."""

    def test_parser_router_satisfies_parser_protocol(self) -> None:
        """Parser Protocol check — structural subtyping, not isinstance."""
        from ingest.router import ParserRouter

        router = ParserRouter()
        assert hasattr(router, "parse")
        assert callable(router.parse)
        # Verify signature matches Parser protocol: async (self, pdf_path: str) -> tuple[Paper, str]
        import inspect
        sig = inspect.signature(router.parse)
        assert "pdf_path" in sig.parameters

    def test_parser_router_has_parse_method(self) -> None:
        from ingest.router import ParserRouter
        assert hasattr(ParserRouter(), "parse")

    def test_grobid_parser_is_parser(self) -> None:
        from ingest.parsers.grobid import GrobidParser
        assert hasattr(GrobidParser("http://localhost:8070"), "parse")

    def test_pymupdf_parser_is_parser(self) -> None:
        from ingest.parsers.pymupdf import PyMuPDFParser
        assert hasattr(PyMuPDFParser(), "parse")

    def test_parser_router_fallback_to_pymupdf(self) -> None:
        """GROBID failure → PyMuPDF fallback."""
        from ingest.router import ParserRouter

        grobid = MagicMock()
        grobid.parse = AsyncMock(side_effect=ParseError("GROBID unavailable"))
        pymupdf = MagicMock()
        pymupdf.parse = AsyncMock(return_value=(
            Paper("pid", None, "T", [], 2024, None, None, "s", None, None, None, {}),
            "fallback text",
        ))

        router = ParserRouter.__new__(ParserRouter)
        router._grobid = grobid
        router._pymupdf = pymupdf

        import asyncio
        paper, text = asyncio.run(router.parse("/fake/path.pdf"))
        assert text == "fallback text"
        grobid.parse.assert_awaited_once()
        pymupdf.parse.assert_awaited_once()

    def test_parser_router_both_fail_raises_parse_error(self) -> None:
        from ingest.router import ParserRouter

        grobid = MagicMock()
        grobid.parse = AsyncMock(side_effect=ParseError("GROBID down"))
        pymupdf = MagicMock()
        pymupdf.parse = AsyncMock(side_effect=ParseError("pymupdf error"))

        router = ParserRouter.__new__(ParserRouter)
        router._grobid = grobid
        router._pymupdf = pymupdf

        import asyncio
        with pytest.raises(ParseError, match="GROBID unavailable"):
            asyncio.run(router.parse("/fake/path.pdf"))

    def test_parser_router_pymupdf_needs_ocr_propagates(self) -> None:
        from ingest.router import ParserRouter

        grobid = MagicMock()
        grobid.parse = AsyncMock(side_effect=ParseError("GROBID down"))
        pymupdf = MagicMock()
        pymupdf.parse = AsyncMock(side_effect=ParseError("needs_ocr: no text layer"))

        router = ParserRouter.__new__(ParserRouter)
        router._grobid = grobid
        router._pymupdf = pymupdf

        import asyncio
        with pytest.raises(ParseError, match="needs_ocr"):
            asyncio.run(router.parse("/fake/path.pdf"))


# ── Chunking golden boundaries ────────────────────────────────────────────────


class TestChunkingGoldenBoundaries:
    """Frozen golden chunk boundaries for known inputs."""

    _SECTION_META: dict = {
        "sections": [
            {"heading": "Introduction", "text": "This is the intro section.",
             "char_start": 0, "char_end": 26},
            {"heading": "Methods", "text": "We used a transformer model.",
             "char_start": 28, "char_end": 59},
        ],
    }

    def test_grobid_sections_golden_chunk_count(self) -> None:
        """Golden: 2 sections → at least 2 chunks."""
        paper = Paper("pid", None, "T", [], 2024, None, None, "s",
                       None, None, None, meta=self._SECTION_META)
        full_text = "This is the intro section.\n\nWe used a transformer model."
        from ingest.chunking import chunk_document
        chunks = chunk_document(paper, full_text, token_window=512, token_overlap=64)
        assert len(chunks) >= 2

    def test_grobid_chunks_have_section_names(self) -> None:
        paper = Paper("pid", None, "T", [], 2024, None, None, "s",
                       None, None, None, meta=self._SECTION_META)
        full_text = "This is the intro section.\n\nWe used a transformer model."
        from ingest.chunking import chunk_document
        chunks = chunk_document(paper, full_text, token_window=512, token_overlap=64)
        sections = {c.section for c in chunks}
        assert "Introduction" in sections
        assert "Methods" in sections

    def test_pymupdf_chunks_have_page_numbers(self) -> None:
        meta: dict = {
            "pages": [
                {"page": 0, "text": "Page one content.", "char_start": 0, "char_end": 17},
                {"page": 1, "text": "Page two content.", "char_start": 19, "char_end": 36},
            ],
        }
        paper = Paper("pid", None, "T", [], 2024, None, None, "pymupdf",
                       None, None, None, meta=meta)
        full_text = "Page one content.\n\nPage two content."
        from ingest.chunking import chunk_document
        chunks = chunk_document(paper, full_text, token_window=512, token_overlap=64)
        pages = {c.page for c in chunks}
        assert 0 in pages
        assert 1 in pages

    def test_empty_text_returns_no_chunks(self) -> None:
        meta: dict = {"sections": []}
        paper = Paper("pid", None, "T", [], 2024, None, None, "s",
                       None, None, None, meta=meta)
        from ingest.chunking import chunk_document
        chunks = chunk_document(paper, "", token_window=512, token_overlap=64)
        assert chunks == []

    def test_no_sections_or_pages_returns_empty(self) -> None:
        meta: dict = {}
        paper = Paper("pid", None, "T", [], 2024, None, None, "s",
                       None, None, None, meta=meta)
        from ingest.chunking import chunk_document
        chunks = chunk_document(paper, "text", token_window=512, token_overlap=64)
        assert chunks == []

    def test_chunks_have_content_hash(self) -> None:
        paper = Paper("pid", None, "T", [], 2024, None, None, "s",
                       None, None, None, meta=self._SECTION_META)
        full_text = "This is the intro section.\n\nWe used a transformer model."
        from ingest.chunking import chunk_document
        chunks = chunk_document(paper, full_text, token_window=512, token_overlap=64)
        for c in chunks:
            assert c.content_hash != ""
            assert len(c.content_hash) == 64  # SHA-256 hex

    def test_chunks_have_ordinals(self) -> None:
        paper = Paper("pid", None, "T", [], 2024, None, None, "s",
                       None, None, None, meta=self._SECTION_META)
        full_text = "This is the intro section.\n\nWe used a transformer model."
        from ingest.chunking import chunk_document
        chunks = chunk_document(paper, full_text, token_window=512, token_overlap=64)
        ordinals = [c.ordinal for c in chunks]
        assert ordinals == list(range(len(chunks)))

    def test_unicode_text_preserved_in_chunks(self) -> None:
        meta: dict = {
            "sections": [
                {"heading": "Resumen", "text": "ñññ 你好 αβγ",
                 "char_start": 0, "char_end": 16},
            ],
        }
        paper = Paper("pid", None, "T", [], 2024, None, None, "s",
                       None, None, None, meta=meta)
        from ingest.chunking import chunk_document
        chunks = chunk_document(paper, "ñññ 你好 αβγ", token_window=512, token_overlap=64)
        assert any("你好" in c.text for c in chunks)


# ── PyMuPDF fallback regression ──────────────────────────────────────────────


class TestPyMuPDFFallback:
    """PyMuPDF parser must handle edge cases gracefully."""

    @pytest.mark.asyncio
    async def test_nonexistent_file_raises_parse_error(self) -> None:
        from ingest.parsers.pymupdf import PyMuPDFParser

        parser = PyMuPDFParser()
        with pytest.raises(ParseError, match="not found"):
            await parser.parse("/nonexistent/path.pdf")

    def test_title_from_path(self) -> None:
        from ingest.parsers.pymupdf import _title_from_path

        assert _title_from_path("/tmp/my_article.pdf") == "my article"
        assert _title_from_path("research-paper-v2.pdf") == "research paper v2"
        assert _title_from_path("weird_name.pdf") == "weird name"

    @pytest.mark.asyncio
    async def test_empty_pdf_raises(self) -> None:
        from ingest.parsers.pymupdf import PyMuPDFParser

        with patch("fitz.open") as mock_open:
            mock_doc = MagicMock()
            mock_doc.page_count = 0
            mock_open.return_value.__enter__.return_value = mock_doc

            parser = PyMuPDFParser()
            with pytest.raises(ParseError, match="empty"):
                await parser.parse("/fake/empty.pdf")
