from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic_settings import BaseSettings

from core.models import ScoredChunk, UpsertChunk
from qa.engine import QAEngine
from qa.llm import LLMProvider
from retrieval.db import Pool
from retrieval.embedder import BgeM3Embedder
from retrieval.pgvector_store import PgVectorStore


class Settings(BaseSettings):
    database_url: str = "postgresql://rc:rc@localhost:5432/research_copilot"
    redis_url: str = "redis://localhost:6379/0"
    upload_dir: str = "/tmp/rc_uploads"


settings = Settings()
pool = Pool(dsn=settings.database_url)

# ---------------------------------------------------------------------------
# Lazy singletons — created on first call so that heavy imports
# (sentence-transformers) do not block application startup.
# ---------------------------------------------------------------------------

_llm: LLMProvider | None = None
_embedder: BgeM3Embedder | None = None
_pgvector: PgVectorStore | None = None


def get_llm() -> LLMProvider:
    global _llm
    if _llm is None:
        _llm = LLMProvider()
    return _llm


def get_embedder() -> BgeM3Embedder:
    global _embedder
    if _embedder is None:
        _embedder = BgeM3Embedder()
    return _embedder


def get_pgvector() -> PgVectorStore:
    global _pgvector
    if _pgvector is None:
        _pgvector = PgVectorStore(pool)
    return _pgvector


# ---------------------------------------------------------------------------
# Scoped retriever — injects paper_ids into search calls
# ---------------------------------------------------------------------------


class _PaperScopedRetriever:
    """Thin wrapper that injects *paper_ids* into every ``search_*`` call.

    Other methods (``upsert``, ``count``, etc.) pass through to the inner
    store unchanged.
    """

    def __init__(
        self,
        inner: PgVectorStore,
        paper_ids: list[str],
    ) -> None:
        self._inner = inner
        self._paper_ids = paper_ids

    async def upsert(self, items: Sequence[UpsertChunk]) -> int:
        return await self._inner.upsert(items)

    async def search_dense(
        self,
        query_vector: Sequence[float],
        k: int,
        paper_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        return await self._inner.search_dense(
            query_vector, k, paper_ids=self._paper_ids,
        )

    async def search_lexical(
        self,
        query_text: str,
        k: int,
        paper_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        return await self._inner.search_lexical(
            query_text, k, paper_ids=self._paper_ids,
        )

    async def delete_by_paper(self, paper_id: str) -> int:
        return await self._inner.delete_by_paper(paper_id)

    async def count(self) -> int:
        return await self._inner.count()

    async def health(self) -> bool:
        return await self._inner.health()


# ---------------------------------------------------------------------------
# QA engine factory
# ---------------------------------------------------------------------------


def get_qa_engine(
    paper_ids: Sequence[str] | None = None,
) -> QAEngine:
    """Return a ``QAEngine``, optionally scoped to *paper_ids*.

    When *paper_ids* is provided a ``_PaperScopedRetriever`` wraps the
    underlying ``PgVectorStore`` so that retrieval is limited to those
    papers.  Otherwise the unscoped store is used (all indexed papers).
    """
    pgvector = get_pgvector()

    if paper_ids is not None:
        paper_list = list(paper_ids) if not isinstance(paper_ids, list) else paper_ids
        retriever: Any = _PaperScopedRetriever(pgvector, paper_list)
    else:
        retriever = pgvector

    return QAEngine(
        retriever=retriever,
        llm=get_llm(),
        embedder=get_embedder(),
    )
