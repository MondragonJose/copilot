"""GET /jobs/{id} — query job status."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.deps import pool

router = APIRouter()


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> JSONResponse:
    """Return the current status of an ingest job."""
    row = await pool.fetchrow(
        "SELECT id, kind, status, stage, error, attempts, max_attempts, "
        "created_at, updated_at FROM jobs WHERE id = $1::uuid",
        job_id,
    )
    if row is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    record = dict(row)
    return JSONResponse(
        {
            "id": str(record["id"]),
            "kind": record["kind"],
            "status": record["status"],
            "stage": record.get("stage"),
            "error": record.get("error"),
            "attempts": record.get("attempts", 0),
            "max_attempts": record.get("max_attempts", 3),
            "created_at": record["created_at"].isoformat()
            if record.get("created_at") else None,
            "updated_at": record["updated_at"].isoformat()
            if record.get("updated_at") else None,
        },
        status_code=200,
    )
