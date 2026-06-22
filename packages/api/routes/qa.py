"""POST /qa — answer a research question.

Thin route handler — delegates all business logic to ``QAEngine``
(from :mod:`qa.engine`) and enriches the response with chunk-level
metadata (page, char_start, char_end) from the database.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from api.deps import get_qa_engine, pool

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class QARequest(BaseModel):
    question: str
    paper_ids: list[str] | None = None


class ClaimOut(BaseModel):
    text: str
    chunk_id: str
    quoted_span: str
    supported: bool
    score: float
    reason: str
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    paper_id: str | None = None


class QAResponse(BaseModel):
    question: str
    answer: str | None
    answerable: bool
    claims: list[ClaimOut]


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/qa")
async def answer_question(body: QARequest) -> QAResponse:
    """Answer a research question against the ingested corpus."""
    engine = get_qa_engine(paper_ids=body.paper_ids)
    result = await engine.answer(body.question)

    chunk_meta = await _fetch_chunk_meta(
        [c.chunk_id for c in result.claims],
    )

    claims_out: list[ClaimOut] = []
    for claim, verdict in zip(result.claims, result.verdicts):
        meta = chunk_meta.get(claim.chunk_id, {})
        claims_out.append(ClaimOut(
            text=claim.text,
            chunk_id=claim.chunk_id,
            quoted_span=claim.quoted_span,
            supported=verdict.supported,
            score=verdict.score,
            reason=verdict.reason,
            page=meta.get("page"),
            char_start=meta.get("char_start"),
            char_end=meta.get("char_end"),
            paper_id=meta.get("paper_id"),
        ))

    return QAResponse(
        question=result.question,
        answer=result.answer,
        answerable=result.answerable,
        claims=claims_out,
    )


async def _fetch_chunk_meta(
    chunk_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Batch-fetch page / offset from the ``chunks`` table.

    Returns a ``{chunk_id: {"page": ..., "char_start": ..., "char_end": ...}}``
    map.  Missing IDs are silently omitted from the map.
    """
    if not chunk_ids:
        return {}

    unique = list(set(chunk_ids))
    rows = await pool.fetch(
        "SELECT id::text, paper_id::text, page, char_start, char_end "
        "FROM chunks WHERE id = ANY($1::uuid[])",
        unique,
    )
    return {
        str(r["id"]): {
            "paper_id": r["paper_id"],
            "page": r["page"],
            "char_start": r["char_start"],
            "char_end": r["char_end"],
        }
        for r in rows
    }
