"""Structure-aware chunking — splits parsed documents into ~512-token windows.

Never crosses section (GROBID) or page (PyMuPDF) boundaries.
Each chunk records section, page, char offsets, token count, and content_hash.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable

from core.models import Chunk, Paper

DEFAULT_TOKEN_WINDOW = 512
DEFAULT_TOKEN_OVERLAP = 64
CHARS_PER_TOKEN = 4.0


def chunk_document(
    paper: Paper,
    full_text: str,
    *,
    token_window: int = DEFAULT_TOKEN_WINDOW,
    token_overlap: int = DEFAULT_TOKEN_OVERLAP,
) -> list[Chunk]:
    """Split a parsed document into overlapping token windows.

    If *paper.meta* contains ``"sections"`` (GROBID parser output) chunks are
    constrained to never cross a section boundary.
    If it contains ``"pages"`` (PyMuPDF parser output) chunks are constrained
    to never cross a page boundary.
    """
    meta = paper.meta
    get_section: Callable[[dict[str, object]], str | None]
    get_page: Callable[[dict[str, object]], int | None]

    if "sections" in meta:
        spans: list[dict[str, object]] = list(meta["sections"])
        get_section = _get_span_section
        get_page = _get_span_page_grobid
    elif "pages" in meta:
        spans = list(meta["pages"])
        get_section = _get_span_section_pymupdf
        get_page = _get_span_page_pymupdf
    else:
        return []

    max_chars = int(token_window * CHARS_PER_TOKEN)
    overlap_chars = int(token_overlap * CHARS_PER_TOKEN)

    chunks: list[Chunk] = []
    ordinal = 0

    for span in spans:
        text = str(span.get("text", ""))
        if not text:
            continue

        section = get_section(span)
        page = get_page(span)
        raw = span.get("char_start", 0)
        span_start = int(raw) if isinstance(raw, (int, str)) else 0

        windows = _split_text(text, span_start, max_chars, overlap_chars)

        for win_text, win_start, win_end in windows:
            chunks.append(Chunk(
                id=str(uuid.uuid4()),
                paper_id=paper.id,
                ordinal=ordinal,
                section=section,
                text=win_text,
                char_start=win_start,
                char_end=win_end,
                page=page,
                token_count=_estimate_tokens(win_text),
                content_hash=hashlib.sha256(win_text.encode()).hexdigest(),
            ))
            ordinal += 1

    return chunks


# ---------------------------------------------------------------------------
# Helpers — span attribute extraction
# ---------------------------------------------------------------------------


def _get_span_section(span: dict[str, object]) -> str | None:
    val = span.get("heading")
    if isinstance(val, str) and val:
        return val
    return None


def _get_span_page_grobid(_span: dict[str, object]) -> int | None:
    return None


def _get_span_section_pymupdf(_span: dict[str, object]) -> str | None:
    return None


def _get_span_page_pymupdf(span: dict[str, object]) -> int | None:
    val = span.get("page")
    if isinstance(val, int):
        return val
    return None


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------


def _split_text(
    text: str,
    base_offset: int,
    max_chars: int,
    overlap_chars: int,
) -> list[tuple[str, int, int]]:
    """Split *text* into overlapping windows, each ≤ *max_chars* characters.

    Returns a list of ``(window_text, char_start, char_end)`` tuples where
    ``char_start`` / ``char_end`` are absolute offsets (relative to the full
    document, not the span).
    """
    if not text:
        return []

    step = max_chars - overlap_chars
    if step <= 0:
        step = max_chars  # overlap >= window → no overlap

    windows: list[tuple[str, int, int]] = []
    start = 0

    while True:
        end = min(start + max_chars, len(text))

        # Only emit a window if it has meaningful content
        if end > start:
            win_text = text[start:end]
            windows.append((win_text, base_offset + start, base_offset + end))

        if end >= len(text):
            break

        start += step

    return windows


def _estimate_tokens(text: str) -> int:
    """Rough token count via ``len(text) / chars_per_token``."""
    return max(1, round(len(text) / CHARS_PER_TOKEN))
