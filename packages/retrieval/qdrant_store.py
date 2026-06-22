"""Qdrant-backed Retriever stub — all methods raise ``NotImplementedError``.

This placeholder exists so that the type-checker can confirm the class
structurally conforms to ``core.interfaces.Retriever`` before real Qdrant
wiring is added in a follow-up issue.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.models import ScoredChunk, UpsertChunk


class QdrantStore:
    """Retriever implementation backed by Qdrant (stub)."""

    def upsert(self, items: Sequence[UpsertChunk]) -> int:
        raise NotImplementedError

    def search_dense(
        self,
        query_vector: Sequence[float],
        k: int,
        paper_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        raise NotImplementedError

    def search_lexical(
        self,
        query_text: str,
        k: int,
        paper_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        raise NotImplementedError

    def delete_by_paper(self, paper_id: str) -> int:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def health(self) -> bool:
        raise NotImplementedError
