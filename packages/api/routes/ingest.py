"""POST /ingest — queue a PDF or DOI for ingestion."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.deps import pool, settings

router = APIRouter()


@router.post("/ingest", status_code=202)
async def ingest(request: Request) -> JSONResponse:
    """Queue a new ingest job from a PDF file upload or a DOI.

    **PDF upload** — ``multipart/form-data`` with a ``file`` field.
    **DOI** — ``application/json`` with ``{"doi": "10.xxx/..."}``.
    """
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("multipart/form-data"):
        return await _handle_file_upload(request)
    if content_type.startswith("application/json"):
        return await _handle_doi(request)

    return JSONResponse(
        {"error": "Send multipart/form-data (file) or application/json (doi)"},
        status_code=415,
    )


async def _handle_file_upload(request: Request) -> JSONResponse:
    form = await request.form()
    raw = form.get("file")
    if raw is None or not hasattr(raw, "read"):
        return JSONResponse({"error": "Missing file field"}, status_code=422)

    os.makedirs(settings.upload_dir, exist_ok=True)
    ext = Path(raw.filename).suffix if raw.filename else ".pdf"  # type: ignore[union-attr]
    dest = Path(settings.upload_dir) / f"{uuid.uuid4()}{ext}"
    content = await raw.read()
    dest.write_bytes(content)

    job_id = str(uuid.uuid4())
    await pool.execute(
        "INSERT INTO jobs (id, kind, payload) VALUES ($1::uuid, 'ingest_pdf', $2::jsonb)",
        job_id, {"pdf_path": str(dest)},
    )

    return JSONResponse({"job_id": job_id}, status_code=202)


async def _handle_doi(request: Request) -> JSONResponse:
    body = await request.json()
    doi: str | None = body.get("doi")
    if not doi or not isinstance(doi, str):
        return JSONResponse({"error": "Missing or invalid doi field"}, status_code=422)

    job_id = str(uuid.uuid4())
    await pool.execute(
        "INSERT INTO jobs (id, kind, payload) VALUES ($1::uuid, 'ingest_doi', $2::jsonb)",
        job_id, {"doi": doi},
    )

    return JSONResponse({"job_id": job_id}, status_code=202)
