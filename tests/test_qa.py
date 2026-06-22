"""QAEngine verification tests — Blueprint §5 three-policy contract.

Policy 1 — answerable question → anchored citation:
    The engine returns supported=True and the quoted_span is an exact
    substring of the chunk text (literal anchor check).

Policy 2 — unanswerable → abstain:
    When no retrieval results are found, or the LLM produces no citations,
    the engine returns answerable=False / "No hay soporte suficiente".

Policy 3 — fabricated citation → dropped:
    If the LLM cites a span that does not appear in the chunk text,
    the claim is silently removed (span_not_found).

FORBIDDEN: relaxing the verification logic to make tests pass.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from core.models import ChunkRef, ScoredChunk
from qa.engine import QAEngine

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

CHUNK_ID = "11111111-1111-1111-1111-111111111111"
PAPER_ID = "22222222-2222-2222-2222-222222222222"
CHUNK_TEXT = (
    "The transformer architecture achieves state-of-the-art results "
    "on natural language processing tasks including translation and summarization."
)


def _make_scored() -> list[ScoredChunk]:
    chunk = ChunkRef(
        chunk_id=CHUNK_ID,
        paper_id=PAPER_ID,
        text=CHUNK_TEXT,
        section="Results",
        page=2,
        char_start=0,
        char_end=len(CHUNK_TEXT),
    )
    return [ScoredChunk(chunk=chunk, score=0.95, channel="dense")]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
def engine() -> tuple[QAEngine, AsyncMock, AsyncMock, AsyncMock]:
    """Return (QAEngine, embedder, retriever, llm) with fresh mocks.

    Callers configure the mocks then call ``await engine.answer(...)``.
    """
    embedder = AsyncMock()
    retriever = AsyncMock()
    llm = AsyncMock()
    qa = QAEngine(retriever=retriever, llm=llm, embedder=embedder)
    return qa, embedder, retriever, llm


# ===================================================================
# Policy 1 — Answerable → Anchored Cite
# ===================================================================


class TestAnswerableAnchoredCite:
    """An answerable question must produce at least one supported claim
    whose quoted_span is a literal substring of its chunk."""

    @pytest.mark.asyncio
    async def test_anchored_cite_survives_verification(
        self, engine: tuple[QAEngine, AsyncMock, AsyncMock, AsyncMock],
    ) -> None:
        qa, embedder, retriever, llm = engine

        embedder.embed.return_value = [[0.0] * 1024]
        retriever.search_dense.return_value = _make_scored()

        llm.generate.side_effect = [
            # Call 1: answer generation with a legitimate citation
            (
                "The transformer achieves SOTA on NLP tasks.\n\n"
                "## Citations\n"
                f"[1] chunk: {CHUNK_ID} quote: \"{CHUNK_TEXT}\""
            ),
            # Call 2: NLI entailment judge — scores it entailed
            "0.95",
        ]

        result = await qa.answer("What does the transformer achieve?")

        assert result.answerable is True
        assert result.answer is not None
        assert any(v.supported for v in result.verdicts)
        for v in result.verdicts:
            if v.supported:
                assert v.claim.quoted_span in CHUNK_TEXT

    @pytest.mark.asyncio
    async def test_partial_fabrication_keeps_real_claims(
        self, engine: tuple[QAEngine, AsyncMock, AsyncMock, AsyncMock],
    ) -> None:
        """Two citations from the LLM: one real, one fabricated.

        The real claim survives verification; the fabricated one is dropped.
        """
        qa, embedder, retriever, llm = engine

        embedder.embed.return_value = [[0.0] * 1024]
        retriever.search_dense.return_value = _make_scored()

        llm.generate.side_effect = [
            # The LLM returns one real and one fabricated citation
            (
                "The transformer achieves SOTA.\n\n"
                "## Citations\n"
                f"[1] chunk: {CHUNK_ID} quote: \"{CHUNK_TEXT}\"\n"
                f"[1] chunk: {CHUNK_ID} quote: \"This span is NOT in the chunk.\""
            ),
            # NLI for the one surviving claim (the fabricated one is dropped
            # before NLI — span_not_found kills it in layer 1)
            "0.95",
        ]

        result = await qa.answer("What does the transformer achieve?")

        assert result.answerable is True
        assert any(v.supported for v in result.verdicts)
        for v in result.verdicts:
            if v.supported:
                assert "NOT in the chunk" not in v.claim.quoted_span


# ===================================================================
# Policy 2 — Unanswerable → Abstain
# ===================================================================


class TestUnanswerableAbstains:
    """When the engine cannot produce a verifiable answer it must return
    ``answerable=False`` with an abstention message."""

    @pytest.mark.asyncio
    async def test_empty_retrieval_returns_answerable_false(
        self, engine: tuple[QAEngine, AsyncMock, AsyncMock, AsyncMock],
    ) -> None:
        qa, embedder, retriever, llm = engine

        embedder.embed.return_value = [[0.0] * 1024]
        retriever.search_dense.return_value = []

        result = await qa.answer("What is in this document?")

        assert result.answerable is False
        assert result.answer is None
        assert len(result.claims) == 0
        assert len(result.verdicts) == 0

    @pytest.mark.asyncio
    async def test_no_citations_from_llm_abstains(
        self, engine: tuple[QAEngine, AsyncMock, AsyncMock, AsyncMock],
    ) -> None:
        """Retrieval succeeds but the LLM returns an answer without a
        ``## Citations`` section → no claims are parsed → abstain."""
        qa, embedder, retriever, llm = engine

        embedder.embed.return_value = [[0.0] * 1024]
        retriever.search_dense.return_value = _make_scored()

        llm.generate.return_value = (
            "I cannot answer this question from the provided context."
        )

        result = await qa.answer("What is the meaning of life?")

        assert result.answerable is False
        assert result.answer == "No hay soporte suficiente"
        assert len(result.claims) == 0
        assert len(result.verdicts) == 0


# ===================================================================
# Policy 3 — Fabricated Citation → Dropped
# ===================================================================


class TestFabricatedCitationDropped:
    """A claim whose ``quoted_span`` does not appear in the chunk text
    must be silently removed (span_not_found → dropped in policy)."""

    @pytest.mark.asyncio
    async def test_fabricated_span_is_dropped(
        self, engine: tuple[QAEngine, AsyncMock, AsyncMock, AsyncMock],
    ) -> None:
        qa, embedder, retriever, llm = engine

        embedder.embed.return_value = [[0.0] * 1024]
        retriever.search_dense.return_value = _make_scored()

        # The LLM fabricates a span that is NOT a substring of the chunk text
        llm.generate.return_value = (
            "Some answer about AI.\n\n"
            "## Citations\n"
            f"[1] chunk: {CHUNK_ID} quote: \"This span is completely made up.\""
        )

        result = await qa.answer("What is artificial intelligence?")

        # The fabricated claim is dropped → no supported claims → abstain
        assert result.answerable is False
        assert result.answer == "No hay soporte suficiente"
        assert len(result.claims) == 0
        assert len(result.verdicts) == 0
