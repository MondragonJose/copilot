"""Goldset loader — parse, validate, and reject malformed goldset rows.

Every public symbol in this module that a caller needs:
    GoldRow, GoldsetLoader, GoldsetError
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class GoldsetError(Exception):
    """Raised when a goldset row is malformed or a gold_span cannot be verified."""


@dataclass(frozen=True)
class GoldRow:
    """A single validated goldset entry.

    Fields
    ------
    question
        Natural-language question.
    answerable
        Whether the question CAN be answered from the indexed corpus.
    gold_answer
        Expected answer (non-empty when *answerable* is ``True``, else ``None``).
    gold_span
        Exact text span from the source chunk that supports *gold_answer*.
        Empty string for unanswerable queries.
    chunk_id
        UUID of the chunk that should contain *gold_span*.
    paper_id
        UUID of the paper that the chunk belongs to.
    gold_relevant_chunk_ids
        All chunk UUIDs considered relevant for retrieval evaluation.
    """

    question: str
    answerable: bool
    gold_answer: str | None
    gold_span: str
    chunk_id: str
    paper_id: str
    gold_relevant_chunk_ids: tuple[str, ...]


_REQUIRED_FIELDS = frozenset({
    "question",
    "answerable",
    "gold_answer",
    "gold_span",
    "chunk_id",
    "paper_id",
    "gold_relevant_chunk_ids",
})


class GoldsetLoader:
    """Load and validate a JSONL goldset file.

    For each row the loader:

    1. Parses JSON, checks all required fields exist.
    2. Validates type constraints (bool, non-empty strings, etc.).
    3. Looks up the chunk text via the injected *chunk_text_resolver*.
    4. Verifies **gold_span** is a literal substring of the chunk text
       (skipped when gold_span is empty — e.g. for unanswerable queries).

    The first failure raises ``GoldsetError`` with the offending line number.
    A successful ``load()`` returns only fully validated rows.
    """

    def __init__(
        self,
        chunk_text_resolver: Callable[[str], str | None],
    ) -> None:
        """*chunk_text_resolver* maps ``chunk_id → chunk_text`` (or ``None``)."""
        self._resolver = chunk_text_resolver

    def load(self, path: str | Path) -> list[GoldRow]:
        """Load, validate, and return all goldset rows from *path*."""
        path = Path(path)
        rows: list[GoldRow] = []

        with path.open() as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = self._parse_row(line, lineno)
                self._validate_span(row, lineno)
                rows.append(row)

        return rows

    # ------------------------------------------------------------------
    # Row parsing
    # ------------------------------------------------------------------

    def _parse_row(self, line: str, lineno: int) -> GoldRow:
        try:
            data: dict = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldsetError(
                f"line {lineno}: invalid JSON — {exc}",
            ) from exc

        missing = _REQUIRED_FIELDS - data.keys()
        if missing:
            raise GoldsetError(
                f"line {lineno}: missing fields: {', '.join(sorted(missing))}",
            )

        q = data["question"]
        if not isinstance(q, str) or not q:
            raise GoldsetError(
                f"line {lineno}: question must be a non-empty string",
            )

        answerable = data["answerable"]
        if not isinstance(answerable, bool):
            raise GoldsetError(
                f"line {lineno}: answerable must be a boolean",
            )

        gold_answer = data["gold_answer"]
        if answerable:
            if not isinstance(gold_answer, str) or not gold_answer:
                raise GoldsetError(
                    f"line {lineno}: answerable rows must have a non-empty gold_answer",
                )
        else:
            if gold_answer is not None:
                raise GoldsetError(
                    f"line {lineno}: unanswerable rows must have null gold_answer",
                )

        gold_span = data["gold_span"]
        if not isinstance(gold_span, str):
            raise GoldsetError(
                f"line {lineno}: gold_span must be a string",
            )

        chunk_id = data["chunk_id"]
        if not isinstance(chunk_id, str) or not chunk_id:
            raise GoldsetError(
                f"line {lineno}: chunk_id must be a non-empty string",
            )

        paper_id = data["paper_id"]
        if not isinstance(paper_id, str) or not paper_id:
            raise GoldsetError(
                f"line {lineno}: paper_id must be a non-empty string",
            )

        ids = data["gold_relevant_chunk_ids"]
        if not isinstance(ids, list):
            raise GoldsetError(
                f"line {lineno}: gold_relevant_chunk_ids must be a list",
            )

        return GoldRow(
            question=q,
            answerable=answerable,
            gold_answer=gold_answer,
            gold_span=gold_span,
            chunk_id=chunk_id,
            paper_id=paper_id,
            gold_relevant_chunk_ids=tuple(ids),
        )

    # ------------------------------------------------------------------
    # Span validation
    # ------------------------------------------------------------------

    def _validate_span(self, row: GoldRow, lineno: int) -> None:
        chunk_text = self._resolver(row.chunk_id)
        if chunk_text is None:
            raise GoldsetError(
                f"line {lineno}: chunk_id {row.chunk_id!r} not found in resolver",
            )
        if row.gold_span and row.gold_span not in chunk_text:
            raise GoldsetError(
                f"line {lineno}: gold_span {row.gold_span!r} not found "
                f"in chunk {row.chunk_id!r}",
            )
