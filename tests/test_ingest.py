"""Route tests for POST /ingest and GET /jobs/{id} with Pool mocked."""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.errors import ParseError
from core.models import Chunk, Paper, UpsertChunk
from ingest.chunking import chunk_document
from ingest.router import ParserRouter
from ingest.tasks import process_job

pytestmark = pytest.mark.asyncio

client = TestClient(app)


# --------------------------------------------------------------------------
# Mock Pool — simulates asyncpg interactions without a real database
# --------------------------------------------------------------------------


class _MockPool:
    """Stand-in for ``retrieval.db.Pool`` used by the route handlers.

    Records ``execute`` and ``fetchrow`` calls for inspection.
    """

    def __init__(self) -> None:
        self.executed: list[tuple[str, ...]] = []
        self._fetchrow_result: dict[str, Any] | None = None
        self._open_called = False
        self._close_called = False

    async def open(self) -> None:
        self._open_called = True

    async def close(self) -> None:
        self._close_called = True

    def set_fetchrow(self, row: dict[str, Any] | None) -> None:
        self._fetchrow_result = row

    async def fetchrow(self, query: str, *params: object) -> dict[str, Any] | None:
        self.executed.append(("fetchrow", query, *(str(p) for p in params)))
        return self._fetchrow_result

    async def execute(self, query: str, *params: object) -> str:
        self.executed.append(("execute", query, *(str(p) for p in params)))
        return "OK"

    @property
    def size(self) -> int | None:
        return 2


@pytest.fixture
def mock_pool() -> _MockPool:
    return _MockPool()


# --------------------------------------------------------------------------
# POST /ingest — DOI
# --------------------------------------------------------------------------


class TestIngestDOI:

    @patch("api.deps.pool.execute", new_callable=AsyncMock)
    def test_creates_job_and_returns_id(self, mock_exec: AsyncMock) -> None:
        mock_exec.return_value = "OK"

        resp = client.post("/ingest", json={"doi": "10.1234/example"})

        assert resp.status_code == 202
        body = resp.json()
        assert "job_id" in body
        assert isinstance(body["job_id"], str)

    @patch("api.deps.pool.execute", new_callable=AsyncMock)
    def test_inserts_ingest_doi_kind(self, mock_exec: AsyncMock) -> None:
        mock_exec.return_value = "OK"

        client.post("/ingest", json={"doi": "10.1234/example"})

        call_args = mock_exec.call_args
        assert call_args is not None
        sql = call_args[0][0]
        assert "ingest_doi" in sql

    @patch("api.deps.pool.execute", new_callable=AsyncMock)
    def test_returns_422_when_doi_missing(self, mock_exec: AsyncMock) -> None:
        resp = client.post("/ingest", json={})
        assert resp.status_code == 422

    @patch("api.deps.pool.execute", new_callable=AsyncMock)
    def test_returns_422_when_doi_empty_string(
        self, mock_exec: AsyncMock,
    ) -> None:
        resp = client.post("/ingest", json={"doi": ""})
        assert resp.status_code == 422


# --------------------------------------------------------------------------
# POST /ingest — PDF upload
# --------------------------------------------------------------------------


class TestIngestFile:

    @patch("api.deps.pool.execute", new_callable=AsyncMock)
    def test_creates_job_for_uploaded_pdf(self, mock_exec: AsyncMock) -> None:
        mock_exec.return_value = "OK"

        resp = client.post(
            "/ingest",
            files={"file": ("test.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )

        assert resp.status_code == 202
        body = resp.json()
        assert "job_id" in body

    @patch("api.deps.pool.execute", new_callable=AsyncMock)
    def test_inserts_ingest_pdf_kind(self, mock_exec: AsyncMock) -> None:
        mock_exec.return_value = "OK"

        client.post(
            "/ingest",
            files={"file": ("paper.pdf", b"%PDF-1.4 data", "application/pdf")},
        )

        call_args = mock_exec.call_args
        assert call_args is not None
        sql = call_args[0][0]
        assert "ingest_pdf" in sql

    @patch("api.deps.pool.execute", new_callable=AsyncMock)
    def test_returns_error_when_file_missing(self, mock_exec: AsyncMock) -> None:
        """Empty files dict does not send multipart — content-type check
        returns 415. If it did reach the handler a 422 would be returned."""
        resp = client.post("/ingest", files={})
        assert resp.status_code in (415, 422)


# --------------------------------------------------------------------------
# POST /ingest — content-type edge cases
# --------------------------------------------------------------------------


class TestIngestContentType:

    @patch("api.deps.pool.execute", new_callable=AsyncMock)
    def test_returns_415_for_unknown_content_type(
        self, mock_exec: AsyncMock,
    ) -> None:
        resp = client.post(
            "/ingest",
            content=b"some data",
            headers={"content-type": "text/plain"},
        )
        assert resp.status_code == 415

    @patch("api.deps.pool.execute", new_callable=AsyncMock)
    def test_multipart_without_file_field_returns_422(
        self, mock_exec: AsyncMock,
    ) -> None:
        """Multipart body that lacks a file field → 422."""
        resp = client.post(
            "/ingest",
            files={"not_a_file": ("hello.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 422


# --------------------------------------------------------------------------
# GET /jobs/{id}
# --------------------------------------------------------------------------


class TestGetJob:

    @patch("api.deps.pool.fetchrow", new_callable=AsyncMock)
    def test_returns_job_when_found(self, mock_fetchrow: AsyncMock) -> None:
        mock_fetchrow.return_value = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "kind": "ingest_pdf",
            "status": "queued",
            "stage": None,
            "error": None,
            "attempts": 0,
            "max_attempts": 3,
            "created_at": __import__("datetime").datetime(2025, 1, 1, 0, 0, 0),
            "updated_at": __import__("datetime").datetime(2025, 1, 1, 0, 0, 0),
        }

        resp = client.get("/jobs/550e8400-e29b-41d4-a716-446655440000")

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert body["kind"] == "ingest_pdf"
        assert body["status"] == "queued"
        assert body["stage"] is None
        assert body["error"] is None
        assert body["attempts"] == 0
        assert body["max_attempts"] == 3

    @patch("api.deps.pool.fetchrow", new_callable=AsyncMock)
    def test_returns_job_with_stage_and_error(
        self, mock_fetchrow: AsyncMock,
    ) -> None:
        mock_fetchrow.return_value = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "kind": "ingest_pdf",
            "status": "failed",
            "stage": "chunk",
            "error": "Section boundary error",
            "attempts": 2,
            "max_attempts": 3,
            "created_at": __import__("datetime").datetime(2025, 1, 1, 0, 0, 0),
            "updated_at": __import__("datetime").datetime(2025, 1, 1, 0, 0, 0),
        }

        resp = client.get("/jobs/550e8400-e29b-41d4-a716-446655440000")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"
        assert body["stage"] == "chunk"
        assert body["error"] == "Section boundary error"
        assert body["attempts"] == 2

    @patch("api.deps.pool.fetchrow", new_callable=AsyncMock)
    def test_returns_404_when_not_found(self, mock_fetchrow: AsyncMock) -> None:
        mock_fetchrow.return_value = None

        resp = client.get("/jobs/00000000-0000-0000-0000-000000000000")

        assert resp.status_code == 404
        assert resp.json()["error"] == "Job not found"

    @patch("api.deps.pool.fetchrow", new_callable=AsyncMock)
    def test_returns_404_for_invalid_uuid(
        self, mock_fetchrow: AsyncMock,
    ) -> None:
        """A non-UUID string is still passed through; the DB layer will fail.
        For test purposes we just ensure it doesn't crash."""
        mock_fetchrow.return_value = None

        resp = client.get("/jobs/not-a-uuid")

        assert resp.status_code == 404


# --------------------------------------------------------------------------
# Full pipeline — fallback: GROBID → PyMuPDF
# --------------------------------------------------------------------------


class _GrobidFailingClient:
    """Simulates a GROBID timeout or HTTP error."""

    async def post(self, url: str, **kwargs: object) -> None:
        msg = "GROBID not reachable (simulated)"
        raise httpx.TimeoutException(msg)


class _GrobidSuccessClient:
    """Returns valid TEI XML so GROBID wins the fallback race."""

    GROBID_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>GROBID Title</title></titleStmt>
      <sourceDesc>
        <biblStruct>
          <monogr>
            <imprint><date type="published" when="2024"/></imprint>
          </monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
  <text><body>
    <div><head>Intro</head><p>GROBID extracted text.</p></div>
  </body></text>
</TEI>"""

    async def post(self, url: str, **kwargs: object) -> _FakeResponse:
        return _FakeResponse(self.GROBID_TEI)


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


class TestFallback:

    async def test_grobid_failure_falls_back_to_pymupdf(
        self, native_pdf: pathlib.Path,
    ) -> None:
        router = ParserRouter(grobid_client=_GrobidFailingClient())
        paper, full_text = await router.parse(str(native_pdf))

        assert paper.source == "pymupdf"
        assert "pages" in paper.meta
        assert len(paper.meta["pages"]) == 2

    async def test_grobid_timeout_falls_back_to_pymupdf(
        self, native_pdf: pathlib.Path,
    ) -> None:
        router = ParserRouter(grobid_client=_GrobidFailingClient())
        paper, full_text = await router.parse(str(native_pdf))

        assert paper.source == "pymupdf"
        assert full_text.strip()

    async def test_grobid_wins_when_available(
        self, native_pdf: pathlib.Path,
    ) -> None:
        router = ParserRouter(grobid_client=_GrobidSuccessClient())
        paper, full_text = await router.parse(str(native_pdf))

        assert paper.source == "grobid"
        assert "sections" in paper.meta

    async def test_scanned_pdf_still_fails_via_router(
        self, scanned_pdf: pathlib.Path,
    ) -> None:
        router = ParserRouter(grobid_client=_GrobidFailingClient())
        with pytest.raises(ParseError, match="needs_ocr"):
            await router.parse(str(scanned_pdf))


# --------------------------------------------------------------------------
# Full pipeline — re-ingest no-dup (same content → same content_hashes)
# --------------------------------------------------------------------------


class TestReIngestNoDup:

    async def test_chunking_same_paper_produces_same_hashes(self) -> None:
        paper = Paper(
            id="p-reingest",
            doi=None, title="Re-ingest Test", authors=[], year=None,
            venue=None, abstract=None, source="test", open_access=None,
            pdf_path="/dev/null", grobid_tei=None,
            meta={"sections": [
                {"heading": "Section A", "text": "Content of section A. " * 30,
                 "char_start": 0},
                {"heading": "Section B", "text": "Content of section B. " * 30,
                 "char_start": 400},
            ]},
        )
        full_text = "Content of section A. " * 30 + "Content of section B. " * 30

        chunks1 = chunk_document(paper, full_text)
        chunks2 = chunk_document(paper, full_text)

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2, strict=True):
            assert c1.content_hash == c2.content_hash
            assert c1.ordinal == c2.ordinal
            assert c1.section == c2.section

    async def test_parse_same_pdf_same_chunk_count(
        self, native_pdf: pathlib.Path,
    ) -> None:
        router = ParserRouter(grobid_client=_GrobidFailingClient())

        paper1, text1 = await router.parse(str(native_pdf))
        paper2, text2 = await router.parse(str(native_pdf))

        chunks1 = chunk_document(paper1, text1)
        chunks2 = chunk_document(paper2, text2)

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2, strict=True):
            assert c1.content_hash == c2.content_hash


# --------------------------------------------------------------------------
# Full pipeline — forced-fail → dead
# --------------------------------------------------------------------------


class _RecordingPool:
    """Pool that records every ``execute`` call for verification."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, ...]] = []

    async def open(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def set_fetchrow(self, row: dict[str, Any] | None) -> None:
        self._fetchrow = row

    async def fetchrow(
        self, query: str, *params: object,
    ) -> dict[str, Any] | None:
        self.executed.append(("fetchrow", query, *(str(p) for p in params)))
        return getattr(self, "_fetchrow", None)

    async def execute(self, query: str, *params: object) -> str:
        self.executed.append(("execute", query, *(str(p) for p in params)))
        return "OK"

    @property
    def size(self) -> int | None:
        return 2


class _NoOpEmbedder:
    async def embed(self, texts: list[str]) -> list[Sequence[float]]:
        return [[0.0] * 4 for _ in texts]


class _NoOpRetriever:
    async def upsert(self, items: Sequence[UpsertChunk]) -> int:
        return len(list(items))

    async def delete_by_paper(self, paper_id: str) -> int:
        return 0

    async def count(self) -> int:
        return 0

    async def health(self) -> bool:
        return True

    async def search_dense(
        self, *args: object, **kwargs: object,
    ) -> list[Any]:
        return []

    async def search_lexical(
        self, *args: object, **kwargs: object,
    ) -> list[Any]:
        return []


class TestForcedFailDead:

    async def test_nonexistent_pdf_goes_dead_with_stage_and_error(
        self,
    ) -> None:
        pool = _RecordingPool()
        pool.set_fetchrow({
            "id": "dead-job-001",
            "kind": "ingest_pdf",
            "status": "queued",
            "paper_id": "p-dead",
            "payload": {"pdf_path": "/tmp/nonexistent-12345.pdf"},
            "attempts": 0,
            "max_attempts": 1,
            "error": None,
            "stage": None,
        })

        router = ParserRouter(grobid_client=_GrobidFailingClient())
        result = await process_job(
            pool, _NoOpRetriever(), _NoOpEmbedder(), router, "dead-job-001",
        )

        assert result["result"] == "failed"

        dead_updates = [
            e for e in pool.executed
            if "dead" in str(e[1])
        ]
        assert len(dead_updates) >= 1
        dead_sql = str(dead_updates[-1])
        assert "dead" in dead_sql
        assert "parse" in dead_sql
        assert "PDF not found" in dead_sql or "nonexistent" in dead_sql

    async def test_exhausted_retries_end_in_dead(self) -> None:
        pool = _RecordingPool()
        pool.set_fetchrow({
            "id": "dead-job-002",
            "kind": "ingest_pdf",
            "status": "queued",
            "paper_id": "p-dead2",
            "payload": {"pdf_path": "/tmp/nonexistent-99999.pdf"},
            "attempts": 2,
            "max_attempts": 3,
            "error": None,
            "stage": "parse",
        })

        router = ParserRouter(grobid_client=_GrobidFailingClient())
        result = await process_job(
            pool, _NoOpRetriever(), _NoOpEmbedder(), router, "dead-job-002",
        )

        assert result["result"] == "failed"

        dead_updates = [
            e for e in pool.executed
            if "dead" in str(e[1])
        ]
        assert len(dead_updates) >= 1


# --------------------------------------------------------------------------
# Full pipeline — no cross-section chunk
# --------------------------------------------------------------------------


class TestNoCrossSectionChunk:

    async def test_sections_never_mixed_in_chunks(self) -> None:
        texts = {
            "Introduction": "This is the introduction paragraph. " * 10,
            "Methodology": "The methodology section describes the approach. " * 10,
            "Results": "The results show significant improvement. " * 10,
        }

        sections: list[dict[str, object]] = []
        cursor = 0
        for heading, body in texts.items():
            sections.append({
                "heading": heading,
                "text": body,
                "char_start": cursor,
                "char_end": cursor + len(body),
            })
            cursor += len(body)

        paper = Paper(
            id="p-no-cross",
            doi=None, title="No Cross Test", authors=[], year=None,
            venue=None, abstract=None, source="test", open_access=None,
            pdf_path="/dev/null", grobid_tei=None,
            meta={"sections": sections},
        )
        full_text = "".join(texts.values())

        chunks = chunk_document(paper, full_text)

        assert len(chunks) > 0
        for chunk in chunks:
            sec_name = chunk.section
            assert sec_name is not None
            expected_text = texts[sec_name]
            assert chunk.text in expected_text, (
                f"Chunk text from section '{sec_name}' contains content "
                f"outside that section"
            )

    async def test_pages_never_mixed_in_chunks(
        self, native_pdf: pathlib.Path,
    ) -> None:
        router = ParserRouter(grobid_client=_GrobidFailingClient())
        paper, full_text = await router.parse(str(native_pdf))

        chunks = chunk_document(paper, full_text)

        for chunk in chunks:
            assert chunk.page is not None

        page_chunks: dict[int, list[Chunk]] = {}
        for c in chunks:
            page_chunks.setdefault(c.page or 0, []).append(c)

        # Each page's chunks should only reference text from that page
        for page_num, page_chunks_list in page_chunks.items():
            page_text = ""
            for span in paper.meta.get("pages", []):
                if span.get("page") == page_num:
                    page_text = str(span.get("text", ""))
            for c in page_chunks_list:
                assert c.text in page_text, (
                    f"Chunk on page {page_num} contains text from another page"
                )
