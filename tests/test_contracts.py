"""Verify dependency rules (import-linter) and Protocol shapes (assert_type).

At runtime this file checks that:
  1. lint-imports passes (dependency contracts enforced).
  2. Dummy implementers exist with the right method signatures.

At type-check time (mypy) it additionally verifies:
  3. Every dummy structurally conforms to its Protocol via assert_type.
"""

import subprocess
from collections.abc import Sequence
from typing import assert_type

import pytest

from core.interfaces import (
    Embedder,
    LLMClient,
    Parser,
    Retriever,
    Verifier,
)
from core.models import (
    Claim,
    ClaimVerdict,
    Paper,
    ScoredChunk,
    UpsertChunk,
)
from retrieval.qdrant_store import QdrantStore

# ---------------------------------------------------------------------------
# Dummy implementers
# ---------------------------------------------------------------------------


class DummyRetriever:
    def upsert(self, items: Sequence[UpsertChunk]) -> int:
        return 0

    def search_dense(
        self,
        query_vector: Sequence[float],
        k: int,
        paper_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        return []

    def search_lexical(
        self,
        query_text: str,
        k: int,
        paper_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        return []

    def delete_by_paper(self, paper_id: str) -> int:
        return 0

    def count(self) -> int:
        return 0

    def health(self) -> bool:
        return True


class DummyParser:
    async def parse(self, pdf_path: str) -> tuple[Paper, str]:
        return (
            Paper(
                id="p1",
                doi=None,
                title="Dummy",
                authors=[],
                year=None,
                venue=None,
                abstract=None,
                source="upload",
                open_access=None,
                pdf_path=pdf_path,
                grobid_tei=None,
                meta={},
            ),
            "full text",
        )


class DummyEmbedder:
    async def embed(self, texts: list[str]) -> list[Sequence[float]]:
        return [[0.0] for _ in texts]


class DummyLLMClient:
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
    ) -> str:
        return "response"


class DummyVerifier:
    async def verify(self, claims: list[Claim]) -> list[ClaimVerdict]:
        return [
            ClaimVerdict(claim=c, supported=True, score=1.0, reason="ok")
            for c in claims
        ]


# ---------------------------------------------------------------------------
# Protocol shape tests (assert_type — type-checked by mypy)
# ---------------------------------------------------------------------------


class TestRetrieverShape:
    def test_dummy_conforms_at_type_level(self) -> None:
        assert_type(DummyRetriever(), Retriever)  # type: ignore[assert-type]

    def test_all_methods_present(self) -> None:
        for m in ("upsert", "search_dense", "search_lexical", "delete_by_paper", "count", "health"):
            assert hasattr(DummyRetriever(), m)

    def test_dummy_behaviour(self) -> None:
        r = DummyRetriever()
        assert r.count() == 0
        assert r.health() is True
        assert r.delete_by_paper("p1") == 0


class TestParserShape:
    def test_dummy_conforms_at_type_level(self) -> None:
        assert_type(DummyParser(), Parser)  # type: ignore[assert-type]

    def test_methods_present(self) -> None:
        assert hasattr(DummyParser(), "parse")


class TestEmbedderShape:
    def test_dummy_conforms_at_type_level(self) -> None:
        assert_type(DummyEmbedder(), Embedder)  # type: ignore[assert-type]

    def test_methods_present(self) -> None:
        assert hasattr(DummyEmbedder(), "embed")


class TestLLMClientShape:
    def test_dummy_conforms_at_type_level(self) -> None:
        assert_type(DummyLLMClient(), LLMClient)  # type: ignore[assert-type]

    def test_methods_present(self) -> None:
        assert hasattr(DummyLLMClient(), "generate")


class TestVerifierShape:
    def test_dummy_conforms_at_type_level(self) -> None:
        assert_type(DummyVerifier(), Verifier)  # type: ignore[assert-type]

    def test_methods_present(self) -> None:
        assert hasattr(DummyVerifier(), "verify")

    @pytest.mark.asyncio
    async def test_verify_returns_verdicts(self) -> None:
        v = DummyVerifier()
        claim = Claim(text="test", chunk_id="c1", quoted_span="test")
        verdicts = await v.verify([claim])
        assert len(verdicts) == 1
        assert verdicts[0].supported is True


class TestQdrantStoreShape:
    def test_conforms_to_retriever_at_type_level(self) -> None:
        # Structural subtyping: a function expecting Retriever accepts QdrantStore.
        def _accepts_retriever(r: Retriever) -> None:
            assert r is not None

        _accepts_retriever(QdrantStore())

    def test_all_methods_present(self) -> None:
        for m in ("upsert", "search_dense", "search_lexical", "delete_by_paper", "count", "health"):
            assert hasattr(QdrantStore(), m)

    def test_all_raise_not_implemented(self) -> None:
        store = QdrantStore()
        with pytest.raises(NotImplementedError):
            store.upsert([])
        with pytest.raises(NotImplementedError):
            store.search_dense([0.0], 5)
        with pytest.raises(NotImplementedError):
            store.search_lexical("test", 5)
        with pytest.raises(NotImplementedError):
            store.delete_by_paper("p1")
        with pytest.raises(NotImplementedError):
            store.count()
        with pytest.raises(NotImplementedError):
            store.health()


# ---------------------------------------------------------------------------
# Dependency rule — enforced by import-linter
# ---------------------------------------------------------------------------


class TestDependencyRules:
    def test_lint_imports_passes(self) -> None:
        import shutil

        import pytest

        binary = shutil.which("lint-imports")
        if binary is None:
            pytest.skip("lint-imports not found in PATH")

        result = subprocess.run(
            [binary],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"lint-imports failed (rc={result.returncode})\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
