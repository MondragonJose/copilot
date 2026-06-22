"""QA regression suite — frozen golden values, empty-guard, verifier invariants.

Guarded regressions:
  • R2-BUG-001  — Empty question crashes QA (now returns answerable=False immediately)
  • CR-BUG-003  — Citation paper_id missing (ScoredChunk + Claim carry paper_id/chunk_id)
  • R2-BUG-001  — Citation faithfulness golden thresholds
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.models import Claim, ClaimVerdict, QAResult, ScoredChunk, ChunkRef
from qa.engine import QAEngine
from qa.verifier import LiteralVerifier, TwoLayerVerifier


# ── Issue R2-BUG-001: empty / blank question ─────────────────────────────────


class TestEmptyQuestionGuard:
    """Regression: empty question must NOT crash the QA engine."""

    @pytest.mark.asyncio
    async def test_empty_string_returns_answerable_false(self) -> None:
        engine = QAEngine(
            retriever=MagicMock(),
            llm=MagicMock(),
            embedder=MagicMock(),
        )
        result = await engine.answer("")
        assert result.answerable is False
        assert result.answer is None
        assert result.claims == []
        assert result.verdicts == []

    @pytest.mark.asyncio
    async def test_whitespace_only_returns_answerable_false(self) -> None:
        engine = QAEngine(
            retriever=MagicMock(),
            llm=MagicMock(),
            embedder=MagicMock(),
        )
        result = await engine.answer("   \n  \t  ")
        assert result.answerable is False

    @pytest.mark.asyncio
    async def test_no_llm_called_on_empty_question(self) -> None:
        llm = AsyncMock()
        engine = QAEngine(
            retriever=MagicMock(),
            llm=llm,
            embedder=MagicMock(),
        )
        await engine.answer("")
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_retrieval_on_empty_question(self) -> None:
        retriever = AsyncMock()
        engine = QAEngine(
            retriever=retriever,
            llm=MagicMock(),
            embedder=MagicMock(),
        )
        await engine.answer("")
        retriever.search_dense.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_embed_on_empty_question(self) -> None:
        embedder = AsyncMock()
        engine = QAEngine(
            retriever=MagicMock(),
            llm=MagicMock(),
            embedder=embedder,
        )
        await engine.answer("")
        embedder.embed.assert_not_called()


# ── Issue R2-BUG-001: empty retrieval (no chunks returned) ──────────────────


class TestEmptyRetrievalGuard:
    """Regression: retrieval returning zero chunks must abstain, not crash."""

    @pytest.mark.asyncio
    async def test_no_chunks_returns_answerable_false(self) -> None:
        embedder = AsyncMock()
        embedder.embed.return_value = [[0.1] * 4]
        retriever = AsyncMock()
        retriever.search_dense.return_value = []

        engine = QAEngine(
            retriever=retriever,
            llm=MagicMock(),
            embedder=embedder,
        )
        result = await engine.answer("valid question")
        assert result.answerable is False
        assert result.answer is None

    @pytest.mark.asyncio
    async def test_no_chunks_no_llm_call(self) -> None:
        llm = AsyncMock()
        embedder = AsyncMock()
        embedder.embed.return_value = [[0.1] * 4]
        retriever = AsyncMock()
        retriever.search_dense.return_value = []

        engine = QAEngine(
            retriever=retriever,
            llm=llm,
            embedder=embedder,
        )
        await engine.answer("valid question")
        llm.generate.assert_not_called()


# ── Citation faithfulness golden values ──────────────────────────────────────


class TestCitationFaithfulnessGolden:
    """Frozen golden values for citation_faithfulness on fixed QAResult sets."""

    def test_all_supported_faithfulness_1(self) -> None:
        c = Claim("t", "cid", "t")
        v = ClaimVerdict(claim=c, supported=True, score=1.0, reason="")
        results = [
            QAResult("q1", "a", [c], [v], answerable=True),
            QAResult("q2", "a", [c, c], [v, v], answerable=True),
        ]
        from eval.metrics import citation_faithfulness
        assert citation_faithfulness(results) == 1.0

    def test_half_supported_faithfulness_0_5(self) -> None:
        c = Claim("t", "cid", "t")
        v_sup = ClaimVerdict(claim=c, supported=True, score=1.0, reason="")
        v_unsup = ClaimVerdict(claim=c, supported=False, score=0.3, reason="entailment_below_threshold")
        results = [
            QAResult("q", "a", [c, c], [v_sup, v_unsup], answerable=True),
        ]
        from eval.metrics import citation_faithfulness
        assert citation_faithfulness(results) == 0.5

    def test_no_verdicts_faithfulness_1(self) -> None:
        results = [
            QAResult("q", None, [], [], answerable=False),
        ]
        from eval.metrics import citation_faithfulness
        assert citation_faithfulness(results) == 1.0

    def test_mixed_results_golden(self) -> None:
        """Golden frozen value — change this only if the metric formula changes."""
        c = Claim("t", "cid", "t")
        v1 = ClaimVerdict(claim=c, supported=True, score=1.0, reason="")
        v2 = ClaimVerdict(claim=c, supported=False, score=0.2, reason="insufficient_evidence")
        results = [
            QAResult("q1", "a", [c], [v1], answerable=True),
            QAResult("q2", None, [c], [v2], answerable=False),
        ]
        from eval.metrics import citation_faithfulness
        assert citation_faithfulness(results) == 0.5


# ── Verifier regression ──────────────────────────────────────────────────────


class TestLiteralVerifierRegression:
    """LiteralVerifier behavior must remain stable."""

    def test_exact_match_supported(self) -> None:
        v = LiteralVerifier({"cid": "The quick brown fox jumps over the lazy dog."})
        claim = Claim(text="claim text", chunk_id="cid", quoted_span="quick brown fox")
        verdicts = v.verify([claim])
        assert len(verdicts) == 1
        assert verdicts[0].supported is True
        assert verdicts[0].score == 1.0
        assert verdicts[0].reason == ""

    def test_span_not_in_chunk(self) -> None:
        v = LiteralVerifier({"cid": "The quick brown fox."})
        claim = Claim(text="wrong", chunk_id="cid", quoted_span="purple elephant")
        verdicts = v.verify([claim])
        assert verdicts[0].supported is False
        assert verdicts[0].score == 0.0
        assert verdicts[0].reason == "span_not_found"

    def test_chunk_id_not_found(self) -> None:
        v = LiteralVerifier({})
        claim = Claim(text="t", chunk_id="missing", quoted_span="t")
        verdicts = v.verify([claim])
        assert verdicts[0].supported is False
        assert verdicts[0].score == 0.0
        assert verdicts[0].reason == "chunk_not_found"

    def test_multiple_claims_verify_all(self) -> None:
        texts = {"c1": "alpha", "c2": "beta"}
        v = LiteralVerifier(texts)
        claims = [
            Claim("t1", "c1", "alpha"),
            Claim("t2", "c2", "beta"),
        ]
        verdicts = v.verify(claims)
        assert all(vd.supported for vd in verdicts)


class TestTwoLayerVerifierRegression:
    """TwoLayerVerifier behavior with mocked LLM."""

    @pytest.mark.asyncio
    async def test_anchor_pass_entailment_pass(self) -> None:
        llm = AsyncMock()
        llm.generate.return_value = "0.95"
        v = TwoLayerVerifier(
            chunk_texts={"cid": "The sun rises in the east."},
            llm=llm,
            threshold=0.5,
        )
        claim = Claim(text="sun rises east", chunk_id="cid", quoted_span="sun rises in the east")
        verdicts = await v._verify_async([claim])
        assert len(verdicts) == 1
        assert verdicts[0].supported is True
        assert verdicts[0].score == 0.95

    @pytest.mark.asyncio
    async def test_anchor_pass_entailment_below_threshold(self) -> None:
        llm = AsyncMock()
        llm.generate.return_value = "0.3"
        v = TwoLayerVerifier(
            chunk_texts={"cid": "The sun rises in the east."},
            llm=llm,
            threshold=0.5,
        )
        claim = Claim(text="moon rises", chunk_id="cid", quoted_span="sun rises in the east")
        verdicts = await v._verify_async([claim])
        assert verdicts[0].supported is False
        assert verdicts[0].reason == "entailment_below_threshold"

    @pytest.mark.asyncio
    async def test_anchor_fails_no_llm_call(self) -> None:
        llm = AsyncMock()
        v = TwoLayerVerifier(
            chunk_texts={"cid": "real text"},
            llm=llm,
        )
        claim = Claim(text="t", chunk_id="cid", quoted_span="not in chunk")
        await v._verify_async([claim])
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_chunk_not_found_no_llm_call(self) -> None:
        llm = AsyncMock()
        v = TwoLayerVerifier(
            chunk_texts={},
            llm=llm,
        )
        claim = Claim(text="t", chunk_id="missing", quoted_span="t")
        await v._verify_async([claim])
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_llm_response_returns_zero(self) -> None:
        llm = AsyncMock()
        llm.generate.return_value = "not a number"
        v = TwoLayerVerifier(
            chunk_texts={"cid": "some text for context that is long enough"},
            llm=llm,
            threshold=0.5,
        )
        claim = Claim(text="t", chunk_id="cid", quoted_span="some text")
        verdicts = await v._verify_async([claim])
        assert verdicts[0].score == 0.0
        assert verdicts[0].supported is False

    @pytest.mark.asyncio
    async def test_threshold_zero_passes_any_entailment(self) -> None:
        llm = AsyncMock()
        llm.generate.return_value = "0.01"
        v = TwoLayerVerifier(
            chunk_texts={"cid": "some context text"},
            llm=llm,
            threshold=0.0,
        )
        claim = Claim(text="t", chunk_id="cid", quoted_span="some context")
        verdicts = await v._verify_async([claim])
        assert verdicts[0].supported is True

    @pytest.mark.asyncio
    async def test_threshold_one_fails_all_entailment(self) -> None:
        llm = AsyncMock()
        llm.generate.return_value = "0.99"
        v = TwoLayerVerifier(
            chunk_texts={"cid": "some text here to check against"},
            llm=llm,
            threshold=1.0,
        )
        claim = Claim(text="t", chunk_id="cid", quoted_span="some text")
        verdicts = await v._verify_async([claim])
        assert verdicts[0].supported is False
        assert verdicts[0].reason == "entailment_below_threshold"


# ── Issue CR-BUG-003: paper_id in ChunkRef (source of truth for citations) ──


class TestPaperIdInChunkRef:
    def test_chunk_ref_carries_paper_id(self) -> None:
        cr = ChunkRef("cid", "paper-uuid", "text", None, 1, 0, 10)
        assert cr.paper_id == "paper-uuid"

    def test_scored_chunk_preserves_paper_id(self) -> None:
        cr = ChunkRef("cid", "paper-uuid", "text", None, 1, 0, 10)
        sc = ScoredChunk(chunk=cr, score=0.9, channel="dense")
        assert sc.chunk.paper_id == "paper-uuid"
