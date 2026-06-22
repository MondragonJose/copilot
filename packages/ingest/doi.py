"""DOI metadata resolution via Crossref API.

Independent of the PDF pipeline.  Returns a ``Paper`` with metadata only
(no text, no chunks).  Errors propagate as ``IngestError`` so the caller
(``process_job`` in ``tasks.py``) can use the existing retry/backoff path.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx

from core.errors import IngestError
from core.models import Paper

CROSSREF_API = "https://api.crossref.org/works/{doi}"
HTTPX_TIMEOUT = 30.0


def _build_paper(doi: str, msg: dict[str, Any]) -> Paper:
    """Transform Crossref ``message`` dict into a ``Paper`` dataclass.

    Fields not present in the response default to ``None`` / empty.
    The full Crossref response is preserved in ``meta["crossref"]``.
    """
    title_list: list[str] = msg.get("title") or []
    title: str = title_list[0] if title_list else "Untitled"

    authors_raw: list[dict[str, Any]] = msg.get("author") or []
    authors: list[dict[str, str]] = [
        {"given": a.get("given", "") or "", "family": a.get("family", "") or ""}
        for a in authors_raw
    ]

    year: int | None = None
    for date_field in ("published-print", "published-online", "issued", "created"):
        date_parts = msg.get(date_field, {}).get("date-parts")
        if date_parts and len(date_parts) > 0 and len(date_parts[0]) > 0:
            raw = date_parts[0][0]
            if isinstance(raw, int):
                year = raw
                break

    venue_list: list[Any] = msg.get("container-title") or []
    venue: str | None = venue_list[0] if venue_list else None

    abstract: str | None = msg.get("abstract")

    return Paper(
        id=str(uuid.uuid4()),
        doi=doi,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        abstract=abstract,
        source="crossref",
        open_access=None,
        pdf_path=None,
        grobid_tei=None,
        meta={"crossref": msg},
    )


async def resolve_doi(
    doi: str,
    client: httpx.AsyncClient | None = None,
) -> Paper:
    """Resolve *doi* via the Crossref REST API and return a ``Paper``.

    Parameters
    ----------
    doi
        The DOI to resolve (e.g. ``"10.1038/nature12373"``).
    client
        Injectable ``httpx.AsyncClient`` for testing.  When ``None`` a
        short-lived client is created and closed after the request.

    Returns
    -------
    Paper
        A new ``Paper`` instance populated with Crossref metadata (no
        PDF, no chunks).

    Raises
    ------
    IngestError
        On network failure, timeout, non-200 response, or malformed data.
        All variants are *retryable* — the caller's backoff/retry loop
        determines when to give up.
    """
    close_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=HTTPX_TIMEOUT)

    url = CROSSREF_API.format(doi=doi)

    try:
        resp = await client.get(url)
        if resp.status_code != 200:
            raise IngestError(
                f"Crossref HTTP {resp.status_code} for DOI {doi}",
            )
        body = resp.json()
    except httpx.TimeoutException as exc:
        raise IngestError(f"Crossref timeout for DOI {doi}: {exc}") from exc
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(
            f"Crossref resolution failed for DOI {doi}: {exc}",
        ) from exc
    finally:
        if close_client:
            await client.aclose()

    message: dict[str, Any] | None = body.get("message") if isinstance(body, dict) else None
    if message is None:
        raise IngestError(
            f"Crossref response for DOI {doi} has no 'message' field",
        )

    return _build_paper(doi, message)
