from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.interfaces import Embedder, LLMClient, Retriever
from core.models import (
    ChunkRef,
    Claim,
    ClaimVerdict,
    ScoredChunk,
)
from qa._nli import judge_entailment
from qa.engine import (
    QAEngine,
    _build_chunk_map,
    _build_prompt,
    _format_context,
    _parse_citations,
)
from qa.llm import LLMError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_retriever() -> Retriever:
    r = MagicMock(spec=Retriever)
    r.search_dense = AsyncMock()
    return r


@pytest.fixture
def mock_llm() -> LLMClient:
    llm = MagicMock(spec=LLMClient)
    llm.generate = AsyncMock()
    return llm


@pytest.fixture
def mock_embedder() -> Embedder:
    e = MagicMock(spec=Embedder)
    e.embed = AsyncMock()
    return e


def _make_scored(chunk_id: str = "c1", text: str = "Some chunk text.",
                 paper_id: str = "p1", score: float = 0.9, section: str | None = None,
                 page: int | None = None) -> ScoredChunk:
    ref = ChunkRef(
        chunk_id=chunk_id, paper_id=paper_id, text=text,
        section=section, page=page, char_start=0, char_end=len(text),
    )
    return ScoredChunk(chunk=ref, score=score, channel="dense")


def _make_engine(
    retriever: Retriever | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    **kwargs,
) -> QAEngine:
    return QAEngine(
        retriever=retriever or MagicMock(spec=Retriever),
        llm=llm or MagicMock(spec=LLMClient),
        embedder=embedder or MagicMock(spec=Embedder),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestQAEngineConstructor:
    def test_default_k(self) -> None:
        eng = _make_engine()
        assert eng._k == 10
        assert eng._verifier_threshold == 0.5

    def test_custom_k_and_threshold(self) -> None:
        eng = _make_engine(k=5, verifier_threshold=0.8)
        assert eng._k == 5
        assert eng._verifier_threshold == 0.8

    def test_invalid_threshold_negative(self) -> None:
        with pytest.raises(ValueError, match="verifier_threshold must be in"):
            _make_engine(verifier_threshold=-0.1)

    def test_invalid_threshold_over_one(self) -> None:
        with pytest.raises(ValueError, match="verifier_threshold must be in"):
            _make_engine(verifier_threshold=1.1)

    def test_boundary_threshold_zero(self) -> None:
        eng = _make_engine(verifier_threshold=0.0)
        assert eng._verifier_threshold == 0.0

    def test_boundary_threshold_one(self) -> None:
        eng = _make_engine(verifier_threshold=1.0)
        assert eng._verifier_threshold == 1.0


# ---------------------------------------------------------------------------
# answer()
# ---------------------------------------------------------------------------

class TestAnswer:
    @pytest.mark.asyncio
    async def test_empty_question_returns_unanswerable(self) -> None:
        eng = _make_engine()
        result = await eng.answer("")
        assert result.answerable is False
        assert result.answer is None
        assert result.claims == []

    @pytest.mark.asyncio
    async def test_whitespace_question_returns_unanswerable(self) -> None:
        eng = _make_engine()
        result = await eng.answer("   \n\t  ")
        assert result.answerable is False

    @pytest.mark.asyncio
    async def test_no_chunks_returns_unanswerable(
        self, mock_retriever: Retriever, mock_embedder: Embedder,
        mock_llm: LLMClient,
    ) -> None:
        mock_embedder.embed = AsyncMock(return_value=[[0.1, 0.2]])
        mock_retriever.search_dense = AsyncMock(return_value=[])
        eng = QAEngine(retriever=mock_retriever, llm=mock_llm, embedder=mock_embedder)
        result = await eng.answer("question")
        assert result.answerable is False
        assert result.answer is None

    @pytest.mark.asyncio
    async def test_happy_path_answerable(
        self, mock_retriever: Retriever, mock_embedder: Embedder,
        mock_llm: LLMClient,
    ) -> None:
        scored = [_make_scored("c1", "The sky is blue. Very blue.")]
        mock_embedder.embed = AsyncMock(return_value=[[0.1]])
        mock_retriever.search_dense = AsyncMock(return_value=scored)

        llm_responses = [
            'Some answer.\n## Citations\n[1] chunk: c1 quote: "The sky is blue."',
            "0.95",
        ]
        mock_llm.generate = AsyncMock(side_effect=llm_responses)

        eng = QAEngine(
            retriever=mock_retriever, llm=mock_llm, embedder=mock_embedder,
            verifier_threshold=0.5,
        )
        result = await eng.answer("What color is the sky?")
        assert result.answerable is True
        assert result.answer is not None
        assert len(result.claims) >= 1
        assert any(v.supported for v in result.verdicts)

    @pytest.mark.asyncio
    async def test_all_claims_dropped_returns_unanswerable(
        self, mock_retriever: Retriever, mock_embedder: Embedder,
        mock_llm: LLMClient,
    ) -> None:
        """When every claim is fabricated (span not in chunk), answerable=False."""
        scored = [_make_scored("c1", "Some unrelated text.")]
        mock_embedder.embed = AsyncMock(return_value=[[0.1]])
        mock_retriever.search_dense = AsyncMock(return_value=scored)
        mock_llm.generate = AsyncMock(side_effect=[
            'Answer.\n## Citations\n[1] chunk: c1 quote: "Fabricated span."',
            "0.1",
        ])
        eng = QAEngine(
            retriever=mock_retriever, llm=mock_llm, embedder=mock_embedder,
        )
        result = await eng.answer("question")
        assert result.answerable is False
        assert result.answer == "No hay soporte suficiente"

    @pytest.mark.asyncio
    async def test_retry_on_low_entailment(
        self, mock_retriever: Retriever, mock_embedder: Embedder,
        mock_llm: LLMClient,
    ) -> None:
        """When first pass has entailment_below_threshold, retry with 2*k chunks."""
        scored1 = [_make_scored("c1", "The sky is blue. Very blue.",
                                 score=0.9)]
        scored2 = [
            _make_scored("c1", "The sky is blue. Very blue.", score=0.9),
            _make_scored("c2", "More context here.", score=0.8),
        ]

        mock_embedder.embed = AsyncMock(return_value=[[0.1]])
        mock_retriever.search_dense = AsyncMock(side_effect=[scored1, scored2])

        # First generation yields a claim with low entailment; retry yields a good one
        mock_llm.generate = AsyncMock(side_effect=[
            'Answer.\n## Citations\n[1] chunk: c1 quote: "blue."',
            "0.3",
            'Better answer.\n## Citations\n[1] chunk: c1 quote: "blue."',
            "0.95",
        ])

        eng = QAEngine(
            retriever=mock_retriever, llm=mock_llm, embedder=mock_embedder,
            verifier_threshold=0.5,
        )
        result = await eng.answer("What color?")
        assert result.answerable is True
        assert mock_retriever.search_dense.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_with_empty_second_retrieval(
        self, mock_retriever: Retriever, mock_embedder: Embedder,
        mock_llm: LLMClient,
    ) -> None:
        """When retry produces empty search, fall back to original verdicts."""
        scored = [_make_scored("c1", "Text.")]
        mock_embedder.embed = AsyncMock(return_value=[[0.1]])
        mock_retriever.search_dense = AsyncMock(side_effect=[scored, []])
        mock_llm.generate = AsyncMock(side_effect=[
            'Answer.\n## Citations\n[1] chunk: c1 quote: "Text."',
            "0.3",
        ])
        eng = QAEngine(
            retriever=mock_retriever, llm=mock_llm, embedder=mock_embedder,
            verifier_threshold=0.5,
        )
        result = await eng.answer("q")
        assert result.answerable is False

    @pytest.mark.asyncio
    async def test_llm_error_returns_abstention(
        self, mock_retriever: Retriever, mock_embedder: Embedder,
        mock_llm: LLMClient,
    ) -> None:
        scored = [_make_scored("c1", "The sky is blue.")]
        mock_embedder.embed = AsyncMock(return_value=[[0.1]])
        mock_retriever.search_dense = AsyncMock(return_value=scored)
        mock_llm.generate = AsyncMock(
            side_effect=LLMError("LLM unavailable"),
        )

        eng = QAEngine(
            retriever=mock_retriever, llm=mock_llm, embedder=mock_embedder,
        )
        result = await eng.answer("What color is the sky?")

        assert result.answerable is False
        assert result.answer is None
        assert result.claims == []
        assert result.verdicts == []

    @pytest.mark.asyncio
    async def test_null_k_uses_default(
        self, mock_retriever: Retriever, mock_embedder: Embedder,
        mock_llm: LLMClient,
    ) -> None:
        scored = [_make_scored("c1", "text.")]
        mock_embedder.embed = AsyncMock(return_value=[[0.1]])
        mock_retriever.search_dense = AsyncMock(return_value=scored)
        mock_llm.generate = AsyncMock(side_effect=[
            'Answer.\n## Citations\n[1] chunk: c1 quote: "text."',
            "0.9",
        ])
        eng = QAEngine(
            retriever=mock_retriever, llm=mock_llm, embedder=mock_embedder,
            k=7, verifier_threshold=0.5,
        )
        result = await eng.answer("q", k=None)
        assert result.answerable is True
        assert mock_retriever.search_dense.call_args[1]["k"] == 7

    @pytest.mark.asyncio
    async def test_explicit_k_overrides_default(
        self, mock_retriever: Retriever, mock_embedder: Embedder,
        mock_llm: LLMClient,
    ) -> None:
        scored = [_make_scored("c1", "text.")]
        mock_embedder.embed = AsyncMock(return_value=[[0.1]])
        mock_retriever.search_dense = AsyncMock(return_value=scored)
        mock_llm.generate = AsyncMock(side_effect=[
            'Answer.\n## Citations\n[1] chunk: c1 quote: "text."',
            "0.9",
        ])
        eng = QAEngine(
            retriever=mock_retriever, llm=mock_llm, embedder=mock_embedder,
            k=3, verifier_threshold=0.5,
        )
        result = await eng.answer("q", k=15)
        assert result.answerable is True
        assert mock_retriever.search_dense.call_args[1]["k"] == 15


# ---------------------------------------------------------------------------
# Entailment judge
# ---------------------------------------------------------------------------

class TestEntailmentJudge:
    @pytest.mark.asyncio
    async def test_judge_returns_clamped_score(self, mock_llm: LLMClient) -> None:
        mock_llm.generate = AsyncMock(return_value="0.85")
        score = await judge_entailment(mock_llm, "ctx", "claim")
        assert score == 0.85

    @pytest.mark.asyncio
    async def test_judge_clamps_above_one(self, mock_llm: LLMClient) -> None:
        mock_llm.generate = AsyncMock(return_value="42.0")
        score = await judge_entailment(mock_llm, "ctx", "claim")
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_judge_clamps_below_zero(self, mock_llm: LLMClient) -> None:
        mock_llm.generate = AsyncMock(return_value="-0.5")
        score = await judge_entailment(mock_llm, "ctx", "claim")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_judge_handles_non_numeric(self) -> None:
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.generate = AsyncMock(return_value="not a number")
        score = await judge_entailment(mock_llm, "ctx", "claim")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_judge_handles_empty_response(self) -> None:
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.generate = AsyncMock(return_value="")
        score = await judge_entailment(mock_llm, "ctx", "claim")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_judge_boundary_zero(self) -> None:
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.generate = AsyncMock(return_value="0.0")
        score = await judge_entailment(mock_llm, "ctx", "claim")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_judge_boundary_one(self) -> None:
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.generate = AsyncMock(return_value="1.0")
        score = await judge_entailment(mock_llm, "ctx", "claim")
        assert score == 1.0


# ---------------------------------------------------------------------------
# _verify_claims
# ---------------------------------------------------------------------------

class TestVerifyClaims:
    @pytest.mark.asyncio
    async def test_chunk_not_found(self) -> None:
        mock_llm = MagicMock(spec=LLMClient)
        eng = QAEngine(
            retriever=MagicMock(spec=Retriever),
            llm=mock_llm,
            embedder=MagicMock(spec=Embedder),
        )
        claims = [Claim(text="x", chunk_id="missing", quoted_span="x")]
        verdicts = await eng._verify_claims(claims, {"c1": "text"})
        assert len(verdicts) == 1
        assert verdicts[0].supported is False
        assert verdicts[0].reason == "chunk_not_found"

    @pytest.mark.asyncio
    async def test_span_not_found(self) -> None:
        mock_llm = MagicMock(spec=LLMClient)
        eng = QAEngine(
            retriever=MagicMock(spec=Retriever),
            llm=mock_llm,
            embedder=MagicMock(spec=Embedder),
        )
        claims = [Claim(text="x", chunk_id="c1", quoted_span="nonexistent")]
        verdicts = await eng._verify_claims(claims, {"c1": "some text here"})
        assert len(verdicts) == 1
        assert verdicts[0].supported is False
        assert verdicts[0].reason == "span_not_found"

    @pytest.mark.asyncio
    async def test_entailment_below_threshold(self) -> None:
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.generate = AsyncMock(return_value="0.3")
        eng = QAEngine(
            retriever=MagicMock(spec=Retriever),
            llm=mock_llm,
            embedder=MagicMock(spec=Embedder),
            verifier_threshold=0.5,
        )
        claims = [Claim(text="x", chunk_id="c1", quoted_span="x")]
        verdicts = await eng._verify_claims(claims, {"c1": "x is present here"})
        assert verdicts[0].supported is False
        assert verdicts[0].reason == "entailment_below_threshold"

    @pytest.mark.asyncio
    async def test_supported(self) -> None:
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.generate = AsyncMock(return_value="0.95")
        eng = QAEngine(
            retriever=MagicMock(spec=Retriever),
            llm=mock_llm,
            embedder=MagicMock(spec=Embedder),
            verifier_threshold=0.5,
        )
        claims = [Claim(text="stuff", chunk_id="c1", quoted_span="stuff")]
        verdicts = await eng._verify_claims(claims, {"c1": "stuff is here"})
        assert verdicts[0].supported is True
        assert verdicts[0].reason == ""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_format_context_no_section(self) -> None:
        scored = [_make_scored("c1", "hello")]
        result = _format_context(scored)
        assert "chunk_id: c1" in result
        assert "text: hello" in result
        assert "section:" not in result

    def test_format_context_with_section(self) -> None:
        scored = [_make_scored("c1", "hello", section="Introduction")]
        result = _format_context(scored)
        assert "section: Introduction" in result

    def test_format_context_multiple_chunks(self) -> None:
        scored = [
            _make_scored("c1", "first", section="A"),
            _make_scored("c2", "second", section="B"),
        ]
        result = _format_context(scored)
        assert "[1]" in result
        assert "[2]" in result

    def test_format_context_empty(self) -> None:
        assert _format_context([]) == ""


class TestBuildPrompt:
    def test_build_prompt(self) -> None:
        result = _build_prompt("What?", "Some context")
        assert "Question: What?" in result
        assert "Some context" in result


class TestBuildChunkMap:
    def test_build_chunk_map(self) -> None:
        scored = [
            _make_scored("c1", "a"),
            _make_scored("c2", "b"),
        ]
        m = _build_chunk_map(scored)
        assert m == {1: "c1", 2: "c2"}

    def test_empty(self) -> None:
        assert _build_chunk_map([]) == {}


class TestParseCitations:
    def test_parse_valid_citations(self) -> None:
        answer = (
            "Some answer.\n"
            "## Citations\n"
            '[1] chunk: c1 quote: "The sky is blue."\n'
            '[2] chunk: c2 quote: "Another fact."'
        )
        claims = _parse_citations(answer, {1: "c1", 2: "c2"})
        assert len(claims) == 2
        assert claims[0].chunk_id == "c1"
        assert claims[0].quoted_span == "The sky is blue."
        assert claims[1].chunk_id == "c2"

    def test_no_citations_section(self) -> None:
        claims = _parse_citations("Just an answer without citations.", {1: "c1"})
        assert claims == []

    def test_citations_reference_unknown_chunk(self) -> None:
        answer = (
            "Answer.\n"
            "## Citations\n"
            '[99] chunk: unknown quote: "text."'
        )
        claims = _parse_citations(answer, {1: "c1"})
        assert claims == []

    def test_malformed_citation_line(self) -> None:
        answer = (
            "Answer.\n"
            "## Citations\n"
            "garbage line\n"
            '[1] chunk: c1 quote: "valid."'
        )
        claims = _parse_citations(answer, {1: "c1"})
        assert len(claims) == 1

    def test_unicode_in_citations(self) -> None:
        answer = (
            "Answer.\n"
            "## Citations\n"
            '[1] chunk: c1 quote: "über cool."'
        )
        claims = _parse_citations(answer, {1: "c1"})
        assert len(claims) == 1
        assert claims[0].quoted_span == "über cool."

    def test_empty_citation_quote(self) -> None:
        answer = (
            "Answer.\n"
            "## Citations\n"
            '[1] chunk: c1 quote: ""'
        )
        claims = _parse_citations(answer, {1: "c1"})
        # empty quote passes the regex but gets stripped to ""
        assert len(claims) == 1
        assert claims[0].quoted_span == ""

    def test_no_chunk_map(self) -> None:
        claims = _parse_citations("## Citations\n[1] chunk: c1 quote: \"x\"", {})
        assert claims == []

    def test_empty_citations_section_is_empty(self) -> None:
        answer = "Answer.\n\n## Citations\n\n"
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


class TestSplitClaimVerdictPairs:
    def test_empty(self) -> None:
        c, v = QAEngine._split_claim_verdict_pairs([])
        assert c == []
        assert v == []

    def test_single(self) -> None:
        cl = Claim(text="t", chunk_id="c1", quoted_span="t")
        cv = ClaimVerdict(claim=cl, supported=True, score=1.0, reason="")
        c, v = QAEngine._split_claim_verdict_pairs([(cl, cv)])
        assert c == [cl]
        assert v == [cv]


# ---------------------------------------------------------------------------
# Integration-style fixtures for blueprint branch tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Blueprint §5 verification policy — one test per branch
# ---------------------------------------------------------------------------


class TestDropBranch:
    @pytest.mark.asyncio
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

        assert len(result.claims) == 0
        assert len(result.verdicts) == 0
        assert not result.answerable
        assert result.answer == "No hay soporte suficiente"


class TestRetryBranch:
    @pytest.mark.asyncio
    async def test_retry_with_more_context_passes(self) -> None:
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
        first_answer = """Claim [1].

## Citations
[1] chunk: c1 quote: "Attention is a core mechanism."
"""
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
    @pytest.mark.asyncio
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

        assert len(result.claims) >= 1
        v = result.verdicts[0]
        assert not v.supported
        assert v.reason == "insufficient_evidence"
        assert v.score == 0.3
        assert not result.answerable
        assert result.answer == "No hay soporte suficiente"


class TestAbstainBranch:
    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_abstains_when_all_claims_dropped_or_degraded(
        self, corpus: list[ScoredChunk],
    ) -> None:
        answer = """Claim [1] and [2].

## Citations
[1] chunk: c-attn quote: "Fabricated span does not exist"
[2] chunk: c-ml quote: "Machine learning models learn patterns from data"
"""
        llm = _FakeLLM(answer, entailment_score="0.3")
        engine = QAEngine(
            retriever=_FakeRetriever(corpus),
            llm=llm,
            embedder=_FakeEmbedder(),
        )

        result = await engine.answer("Q?")

        assert not result.answerable
        assert result.answer == "No hay soporte suficiente"
        assert len(result.claims) >= 1
        assert all(not v.supported for v in result.verdicts)
        assert any(v.reason == "insufficient_evidence" for v in result.verdicts)
