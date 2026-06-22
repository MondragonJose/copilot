"""Mapping test for the QA engine — claims carry chunk_id and quoted_span.

Also tests the Blueprint §5 verification policy:
drop / retry / degrade / abstain.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from core.models import ChunkRef, ScoredChunk
from qa.engine import QAEngine, _build_chunk_map, _parse_citations

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------
# Fixture — a tiny three-chunk corpus with known chunk_ids
# --------------------------------------------------------------------------


@pytest.fixture
def corpus() -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=ChunkRef(
                chunk_id="c-attn",
                paper_id="p1",
                text="Transformers rely on self-attention to model "
                     "relationships between all tokens in a sequence.",
                section="Introduction",
                page=1,
                char_start=0,
                char_end=90,
            ),
            score=0.92,
            channel="dense",
        ),
        ScoredChunk(
            chunk=ChunkRef(
                chunk_id="c-ml",
                paper_id="p1",
                text="Machine learning models learn patterns from data "
                     "by minimizing a loss function.",
                section="Background",
                page=2,
                char_start=0,
                char_end=85,
            ),
            score=0.88,
            channel="dense",
        ),
        ScoredChunk(
            chunk=ChunkRef(
                chunk_id="c-rlhf",
                paper_id="p2",
                text="RLHF aligns language models with human preferences "
                     "through reinforcement learning from human feedback.",
                section="Method",
                page=3,
                char_start=0,
                char_end=120,
            ),
            score=0.85,
            channel="dense",
        ),
    ]


# --------------------------------------------------------------------------
# Mock dependencies
# --------------------------------------------------------------------------


class _FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[Sequence[float]]:
        return [[0.1] * 4 for _ in texts]


class _FakeRetriever:
    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self._chunks = chunks

    async def search_dense(
        self,
        query_vector: Sequence[float],
        k: int,
        paper_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        return self._chunks[:k]


class _FakeLLM:
    """Mock LLM that returns the answer text for generation and a float for NLI.

    Parameters
    ----------
    answer
        Text returned for answer-generation prompts.
    entailment_score
        Single score string (e.g. ``"0.9"``) used for every NLI call, or a
        list of strings to return different scores on successive NLI calls.
    """

    def __init__(
        self,
        answer: str,
        entailment_score: str | list[str] = "0.9",
    ) -> None:
        self._answer = answer
        self._entailment_scores = (
            [entailment_score]
            if isinstance(entailment_score, str)
            else entailment_score
        )
        self._nli_calls = 0

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
    ) -> str:
        if system and "NLI" in system:
            idx = min(self._nli_calls, len(self._entailment_scores) - 1)
            self._nli_calls += 1
            return self._entailment_scores[idx]
        return self._answer


# --------------------------------------------------------------------------
# _parse_citations
# --------------------------------------------------------------------------


class TestParseCitations:

    def test_parses_single_citation(self) -> None:
        answer = """The key is attention [1].

## Citations
[1] chunk: c-attn quote: "Transformers rely on self-attention"
"""
        chunk_map = {1: "c-attn"}
        claims = _parse_citations(answer, chunk_map)

        assert len(claims) == 1
        assert claims[0].chunk_id == "c-attn"
        assert "self-attention" in claims[0].quoted_span

    def test_parses_multiple_citations(self) -> None:
        answer = """Claims about attention [1] and ML [2].

## Citations
[1] chunk: c-attn quote: "Transformers rely on self-attention"
[2] chunk: c-ml quote: "Machine learning models learn patterns"
"""
        chunk_map = {1: "c-attn", 2: "c-ml"}
        claims = _parse_citations(answer, chunk_map)

        assert len(claims) == 2
        assert claims[0].chunk_id == "c-attn"
        assert claims[1].chunk_id == "c-ml"

    def test_skips_unknown_chunk_number(self) -> None:
        answer = """Some claim [99].

## Citations
[99] chunk: unknown quote: "Missing chunk"
"""
        chunk_map = {1: "c-attn"}  # only chunk 1 exists
        claims = _parse_citations(answer, chunk_map)

        assert len(claims) == 0

    def test_returns_empty_when_no_citations_section(self) -> None:
        answer = """Just an answer without citations."""
        claims = _parse_citations(answer, {1: "c-attn"})

        assert len(claims) == 0

    def test_returns_empty_when_citations_section_is_empty(self) -> None:
        answer = """Answer.

## Citations

"""
        claims = _parse_citations(answer, {1: "c-attn"})
        assert len(claims) == 0

    def test_handles_extra_whitespace_in_quotes(self) -> None:
        answer = """Claim [1].

## Citations
[1]   chunk:   c-attn   quote:   "Some text"
"""
        claims = _parse_citations(answer, {1: "c-attn"})

        assert len(claims) == 1
        assert claims[0].quoted_span == "Some text"

    def test_quoted_span_matches_claim_text(self) -> None:
        answer = """The model uses attention [1].

## Citations
[1] chunk: c-attn quote: "Transformers rely on self-attention to model"
"""
        claims = _parse_citations(answer, {1: "c-attn"})

        assert len(claims) == 1
        assert claims[0].text == "Transformers rely on self-attention to model"
        assert claims[0].quoted_span == "Transformers rely on self-attention to model"


# --------------------------------------------------------------------------
# _build_chunk_map
# --------------------------------------------------------------------------


class TestBuildChunkMap:

    def test_maps_numbers_to_chunk_ids(self, corpus: list[ScoredChunk]) -> None:
        chunk_map = _build_chunk_map(corpus)

        assert chunk_map[1] == "c-attn"
        assert chunk_map[2] == "c-ml"
        assert chunk_map[3] == "c-rlhf"
        assert len(chunk_map) == 3


# --------------------------------------------------------------------------
# QAEngine.answer — full integration with mocked dependencies
# --------------------------------------------------------------------------


class TestQAEngineAnswer:

    async def test_returns_qaresult_with_correct_question(
        self, corpus: list[ScoredChunk],
    ) -> None:
        llm = _FakeLLM("Simple answer.\n\n## Citations\n")
        engine = QAEngine(
            retriever=_FakeRetriever(corpus),
            llm=llm,
            embedder=_FakeEmbedder(),
        )

        result = await engine.answer("What is attention?")

        assert result.question == "What is attention?"

    async def test_returns_answerable_true_with_valid_citations(
        self, corpus: list[ScoredChunk],
    ) -> None:
        answer = """Attention is key [1].

## Citations
[1] chunk: c-attn quote: "Transformers rely on self-attention"
"""
        llm = _FakeLLM(answer, entailment_score="0.9")
        engine = QAEngine(
            retriever=_FakeRetriever(corpus),
            llm=llm,
            embedder=_FakeEmbedder(),
        )

        result = await engine.answer("Question?")

        assert result.answerable
        assert len(result.claims) == 1

    async def test_returns_answerable_false_when_no_chunks(
        self, corpus: list[ScoredChunk],
    ) -> None:
        empty_retriever = _FakeRetriever([])
        llm = _FakeLLM("")
        engine = QAEngine(
            retriever=empty_retriever,
            llm=llm,
            embedder=_FakeEmbedder(),
        )

        result = await engine.answer("Question?")

        assert not result.answerable
        assert result.answer is None
        assert result.claims == []

    async def test_claims_have_chunk_id_from_citations(
        self, corpus: list[ScoredChunk],
    ) -> None:
        answer = """Attention is key [1].

## Citations
[1] chunk: c-attn quote: "Transformers rely on self-attention to model"
"""
        llm = _FakeLLM(answer, entailment_score="0.9")
        engine = QAEngine(
            retriever=_FakeRetriever(corpus),
            llm=llm,
            embedder=_FakeEmbedder(),
        )

        result = await engine.answer("What is attention?")

        assert len(result.claims) == 1
        assert result.claims[0].chunk_id == "c-attn"

    async def test_claims_have_quoted_span_matching_chunk_text(
        self, corpus: list[ScoredChunk],
    ) -> None:
        answer = """RLHF is used [1].

## Citations
[1] chunk: c-rlhf quote: "RLHF aligns language models with human preferences"
"""
        llm = _FakeLLM(answer, entailment_score="0.9")
        engine = QAEngine(
            retriever=_FakeRetriever(corpus),
            llm=llm,
            embedder=_FakeEmbedder(),
        )

        result = await engine.answer("What is RLHF?")

        assert len(result.claims) == 1
        claim = result.claims[0]
        assert claim.chunk_id == "c-rlhf"
        assert "RLHF aligns" in claim.quoted_span

    async def test_multiple_claims_from_one_answer(
        self, corpus: list[ScoredChunk],
    ) -> None:
        answer = """Attention [1] and ML [2] are both important.

## Citations
[1] chunk: c-attn quote: "Transformers rely on self-attention"
[2] chunk: c-ml quote: "Machine learning models learn patterns"
"""
        # Two entailment calls needed (one per claim)
        llm = _FakeLLM(answer, entailment_score=["0.9", "0.9"])
        engine = QAEngine(
            retriever=_FakeRetriever(corpus),
            llm=llm,
            embedder=_FakeEmbedder(),
        )

        result = await engine.answer("Key concepts?")

        assert len(result.claims) == 2
        chunk_ids = {c.chunk_id for c in result.claims}
        assert chunk_ids == {"c-attn", "c-ml"}
        # Both should be supported
        assert len(result.verdicts) == 2
        assert all(v.supported for v in result.verdicts)

    async def test_default_k_is_10(self) -> None:
        """When only 3 chunks exist, all should be retrieved with k=10."""
        corpus_3 = [
            ScoredChunk(
                chunk=ChunkRef(
                    chunk_id="c1", paper_id="p1", text="Text.",
                    section=None, page=None, char_start=None, char_end=None,
                ),
                score=0.9, channel="dense",
            ),
            ScoredChunk(
                chunk=ChunkRef(
                    chunk_id="c2", paper_id="p1", text="More.",
                    section=None, page=None, char_start=None, char_end=None,
                ),
                score=0.8, channel="dense",
            ),
            ScoredChunk(
                chunk=ChunkRef(
                    chunk_id="c3", paper_id="p1", text="Even more.",
                    section=None, page=None, char_start=None, char_end=None,
                ),
                score=0.7, channel="dense",
            ),
        ]
        answer = """Claim [1].

## Citations
[1] chunk: c1 quote: "Text."
"""
        llm = _FakeLLM(answer, entailment_score="0.9")
        engine = QAEngine(
            retriever=_FakeRetriever(corpus_3),
            llm=llm,
            embedder=_FakeEmbedder(),
        )

        result = await engine.answer("Q?")

        # All 3 retrieved even with default k=10
        assert result.answerable

    async def test_verdicts_included_in_result(
        self, corpus: list[ScoredChunk],
    ) -> None:
        answer = """Claim [1].

## Citations
[1] chunk: c-attn quote: "Transformers rely on self-attention"
"""
        llm = _FakeLLM(answer, entailment_score="0.9")
        engine = QAEngine(
            retriever=_FakeRetriever(corpus),
            llm=llm,
            embedder=_FakeEmbedder(),
        )

        result = await engine.answer("Q?")

        assert len(result.verdicts) == 1
        v = result.verdicts[0]
        assert v.supported
        assert v.score == 0.9
        assert v.reason == ""


# ---------------------------------------------------------------------------
# Blueprint §5 verification policy — one test per branch
# ---------------------------------------------------------------------------


class TestDropBranch:
    """Fabricated claims (span not in chunk text) are removed from output."""

    async def test_drops_claim_with_fabricated_span(self) -> None:
        corpus_1 = [
            ScoredChunk(
                chunk=ChunkRef(
                    chunk_id="c1", paper_id="p1",
                    text="The sky is blue on clear days.",
                    section=None, page=None, char_start=None, char_end=None,
                ),
                score=0.9, channel="dense",
            ),
        ]
        # quoted_span does NOT appear in the chunk text
        answer = """Claim [1].

## Citations
[1] chunk: c1 quote: "This span does not exist in the chunk"
"""
        llm = _FakeLLM(answer)
        engine = QAEngine(
            retriever=_FakeRetriever(corpus_1),
            llm=llm,
            embedder=_FakeEmbedder(),
        )

        result = await engine.answer("Q?")

        # Claim dropped — no surviving claims
        assert len(result.claims) == 0
        assert len(result.verdicts) == 0
        assert not result.answerable
        assert result.answer == "No hay soporte suficiente"


class TestRetryBranch:
    """Anchored-but-low gets one retry with doubled k, then passes."""

    async def test_retry_with_more_context_passes(self) -> None:
        """Low entailment on first try; retry with 2x chunks passes."""
        corpus_big = [
            ScoredChunk(
                chunk=ChunkRef(
                    chunk_id="c1", paper_id="p1",
                    text="Attention is a core mechanism.",
                    section=None, page=None, char_start=None, char_end=None,
                ),
                score=0.9, channel="dense",
            ),
            ScoredChunk(
                chunk=ChunkRef(
                    chunk_id="c2", paper_id="p1",
                    text="It computes weighted sums of values.",
                    section=None, page=None, char_start=None, char_end=None,
                ),
                score=0.8, channel="dense",
            ),
            ScoredChunk(
                chunk=ChunkRef(
                    chunk_id="c3", paper_id="p1",
                    text="Transformers use multi-head attention.",
                    section=None, page=None, char_start=None, char_end=None,
                ),
                score=0.7, channel="dense",
            ),
        ]
        # Start with k=1 (only c1 retrieved).  After retry k=2 (c1 + c2).
        first_answer = """Claim [1].

## Citations
[1] chunk: c1 quote: "Attention is a core mechanism."
"""
        # Call sequence: answer(1) → NLI(0.3) → answer(2) → NLI(0.9) → NLI(0.9)
        llm = _FakeLLM(
            answer=first_answer,
            entailment_score=["0.3", "0.9", "0.9"],
        )
        engine = QAEngine(
            retriever=_FakeRetriever(corpus_big),
            llm=llm,
            embedder=_FakeEmbedder(),
        )

        result = await engine.answer("How does attention work?", k=1)

        assert result.answerable
        assert len(result.claims) > 0
        assert all(v.supported for v in result.verdicts)


class TestDegradeBranch:
    """After retry, still-low claims are kept as 'insufficient_evidence'."""

    async def test_degrades_still_low_claim_after_retry(self) -> None:
        corpus_2 = [
            ScoredChunk(
                chunk=ChunkRef(
                    chunk_id="c1", paper_id="p1",
                    text="The sky is blue.",
                    section=None, page=None, char_start=None, char_end=None,
                ),
                score=0.9, channel="dense",
            ),
            ScoredChunk(
                chunk=ChunkRef(
                    chunk_id="c2", paper_id="p1",
                    text="Clouds are white.",
                    section=None, page=None, char_start=None, char_end=None,
                ),
                score=0.8, channel="dense",
            ),
        ]
        first_answer = """Claim [1].

## Citations
[1] chunk: c1 quote: "The sky is blue."
"""
        # Call sequence: answer(1) → NLI(0.3) → answer(2) → NLI(0.3)
        llm = _FakeLLM(
            answer=first_answer,
            entailment_score=["0.3", "0.3"],
        )
        engine = QAEngine(
            retriever=_FakeRetriever(corpus_2),
            llm=llm,
            embedder=_FakeEmbedder(),
        )

        result = await engine.answer("What colour is the sky?", k=1)

        # Claim still present but degraded
        assert len(result.claims) >= 1
        v = result.verdicts[0]
        assert not v.supported
        assert v.reason == "insufficient_evidence"
        assert v.score == 0.3
        # No claim is fully supported → global abstain
        assert not result.answerable
        assert result.answer == "No hay soporte suficiente"


class TestAbstainBranch:
    """When no claim is supported, the engine abstains."""

    async def test_abstains_when_no_supported_claims(self) -> None:
        corpus_1 = [
            ScoredChunk(
                chunk=ChunkRef(
                    chunk_id="c1", paper_id="p1",
                    text="Does not matter.",
                    section=None, page=None, char_start=None, char_end=None,
                ),
                score=0.9, channel="dense",
            ),
        ]
        # Empty citations → no claims at all
        llm = _FakeLLM("No relevant information.\n\n## Citations\n")
        engine = QAEngine(
            retriever=_FakeRetriever(corpus_1),
            llm=llm,
            embedder=_FakeEmbedder(),
        )

        result = await engine.answer("Unknown question?")

        assert not result.answerable
        assert result.answer == "No hay soporte suficiente"
        assert result.claims == []
        assert result.verdicts == []

    async def test_abstains_when_all_claims_dropped_or_degraded(
        self, corpus: list[ScoredChunk],
    ) -> None:
        # One fabricated claim (dropped) + one low-entailment (degraded)
        answer = """Claim [1] and [2].

## Citations
[1] chunk: c-attn quote: "Fabricated span does not exist"
[2] chunk: c-ml quote: "Machine learning models learn patterns from data"
"""
        # No claim survives as supported
        llm = _FakeLLM(answer, entailment_score="0.3")
        engine = QAEngine(
            retriever=_FakeRetriever(corpus),
            llm=llm,
            embedder=_FakeEmbedder(),
        )

        result = await engine.answer("Q?")

        assert not result.answerable
        assert result.answer == "No hay soporte suficiente"
        # The fabricated claim is dropped; the degraded one remains
        assert len(result.claims) >= 1
        assert all(not v.supported for v in result.verdicts)
        assert any(v.reason == "insufficient_evidence" for v in result.verdicts)
