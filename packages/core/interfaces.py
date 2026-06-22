"""Protocols for pluggable infrastructure — Retriever, Parser, Embedder, LLMClient, Verifier.

Every I/O-bound dependency lives behind one of these Protocols so that
implementations (pgvector, custom QA engine, GROBID, etc.) can be swapped without
changing callers.
"""

from collections.abc import Sequence
from typing import Protocol

from core.models import (
    Claim,
    ClaimVerdict,
    Paper,
    ScoredChunk,
    UpsertChunk,
)


class Retriever(Protocol):
    """Vector + lexical search over persisted chunks."""

    async def upsert(self, items: Sequence[UpsertChunk]) -> int:
        """Insert or update chunks and vectors idempotently (by chunk_id).
        Returns the number of affected rows.
        """
        ...

    async def search_dense(
        self,
        query_vector: Sequence[float],
        k: int,
        paper_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        """k-NN cosine similarity search. Optional filter by paper subset."""
        ...

    async def search_lexical(
        self,
        query_text: str,
        k: int,
        paper_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        """Lexical (BM25 / trigram) search. Optional filter by paper subset."""
        ...

    async def delete_by_paper(self, paper_id: str) -> int:
        """Remove all chunks and vectors for a given paper."""
        ...

    async def count(self) -> int:
        """Total number of indexed chunks."""
        ...

    async def health(self) -> bool:
        """Whether the underlying store is reachable and responsive."""
        ...


class Parser(Protocol):
    """Extract structured content from a PDF."""

    async def parse(self, pdf_path: str) -> tuple[Paper, str]:
        """Parse a PDF file and return (Paper metadata, full text)."""
        ...


class Embedder(Protocol):
    """Convert text into dense vector embeddings."""

    async def embed(self, texts: list[str]) -> list[Sequence[float]]:
        """Embed a batch of texts. Returns one vector per text."""
        ...


class LLMClient(Protocol):
    """Interface to a language model (local or remote)."""

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
    ) -> str:
        """Generate a completion for the given prompt."""
        ...


class Verifier(Protocol):
    """Check that each claim is supported by its cited span."""

    async def verify(self, claims: list[Claim]) -> list[ClaimVerdict]:
        """Verify each claim against its quoted source span.
        Returns a verdict for every input claim.
        """
        ...
