from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.errors import EmbeddingError
from core.interfaces import Embedder, LLMClient, Retriever
from core.models import (
    ChunkRef,
    Claim,
    ClaimVerdict,
    QAResult,
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
