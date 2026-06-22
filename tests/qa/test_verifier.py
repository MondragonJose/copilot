"""Tests for LiteralVerifier and TwoLayerVerifier — pure logic with mocked LLM."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.interfaces import LLMClient
from core.models import Claim, ClaimVerdict
from qa.verifier import LiteralVerifier, TwoLayerVerifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_claim(text: str = "claim", chunk_id: str = "c1",
                quoted_span: str = "span") -> Claim:
    return Claim(text=text, chunk_id=chunk_id, quoted_span=quoted_span)


# ---------------------------------------------------------------------------
# LiteralVerifier
# ---------------------------------------------------------------------------

class TestLiteralVerifier:
    def test_supported(self) -> None:
        verifier = LiteralVerifier({"c1": "some text with span inside"})
        claims = [_make_claim(quoted_span="span")]
        verdicts = verifier.verify(claims)
        assert len(verdicts) == 1
        assert verdicts[0].supported is True
        assert verdicts[0].score == 1.0
        assert verdicts[0].reason == ""

    def test_chunk_not_found(self) -> None:
        verifier = LiteralVerifier({})
        claims = [_make_claim(chunk_id="missing")]
        verdicts = verifier.verify(claims)
        assert verdicts[0].supported is False
        assert verdicts[0].score == 0.0
        assert verdicts[0].reason == "chunk_not_found"

    def test_span_not_found(self) -> None:
        verifier = LiteralVerifier({"c1": "some text"})
        claims = [_make_claim(quoted_span="nonexistent")]
        verdicts = verifier.verify(claims)
        assert verdicts[0].supported is False
        assert verdicts[0].reason == "span_not_found"

    def test_multiple_claims(self) -> None:
        chunk_texts = {"c1": "text with span1 here", "c2": "text with span2 here"}
        verifier = LiteralVerifier(chunk_texts)
        claims = [
            _make_claim(quoted_span="span1", chunk_id="c1"),
            _make_claim(quoted_span="span2", chunk_id="c2"),
            _make_claim(quoted_span="span3", chunk_id="c3"),
        ]
        verdicts = verifier.verify(claims)
        assert verdicts[0].supported is True
        assert verdicts[1].supported is True
        assert verdicts[2].supported is False
        assert verdicts[2].reason == "chunk_not_found"

    def test_empty_claims_list(self) -> None:
        verifier = LiteralVerifier({"c1": "text"})
        assert verifier.verify([]) == []

    def test_unicode_span(self) -> None:
        verifier = LiteralVerifier({"c1": "über cool text"})
        claims = [_make_claim(quoted_span="über")]
        verdicts = verifier.verify(claims)
        assert verdicts[0].supported is True

    def test_span_with_whitespace(self) -> None:
        verifier = LiteralVerifier({"c1": "hello   world"})
        claims = [_make_claim(quoted_span="hello   world")]
        verdicts = verifier.verify(claims)
        assert verdicts[0].supported is True

    def test_case_sensitive_match(self) -> None:
        verifier = LiteralVerifier({"c1": "Self-Attention is important."})

        verdict_lower = verifier.verify([
            _make_claim(chunk_id="c1", quoted_span="self-attention"),
        ])
        assert not verdict_lower[0].supported
        assert verdict_lower[0].reason == "span_not_found"

        verdict_exact = verifier.verify([
            _make_claim(chunk_id="c1", quoted_span="Self-Attention"),
        ])
        assert verdict_exact[0].supported


# ---------------------------------------------------------------------------
# TwoLayerVerifier
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm() -> LLMClient:
    llm = MagicMock(spec=LLMClient)
    llm.generate = AsyncMock()
    return llm


class TestTwoLayerVerifierConstructor:
    def test_default_threshold(self, mock_llm: LLMClient) -> None:
        v = TwoLayerVerifier({"c1": "text"}, mock_llm)
        assert v._threshold == 0.5

    def test_custom_threshold(self, mock_llm: LLMClient) -> None:
        v = TwoLayerVerifier({"c1": "text"}, mock_llm, threshold=0.8)
        assert v._threshold == 0.8

    def test_invalid_threshold_negative(self, mock_llm: LLMClient) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            TwoLayerVerifier({"c1": "text"}, mock_llm, threshold=-0.1)

    def test_invalid_threshold_over_one(self, mock_llm: LLMClient) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            TwoLayerVerifier({"c1": "text"}, mock_llm, threshold=1.1)

    def test_boundary_zero(self, mock_llm: LLMClient) -> None:
        v = TwoLayerVerifier({"c1": "text"}, mock_llm, threshold=0.0)
        assert v._threshold == 0.0

    def test_boundary_one(self, mock_llm: LLMClient) -> None:
        v = TwoLayerVerifier({"c1": "text"}, mock_llm, threshold=1.0)
        assert v._threshold == 1.0


@pytest.mark.asyncio
class TestTwoLayerVerifier:
    async def test_chunk_not_found(self, mock_llm: LLMClient) -> None:
        v = TwoLayerVerifier({}, mock_llm)
        claims = [_make_claim(chunk_id="missing")]
        verdicts = await v.verify(claims)
        assert verdicts[0].supported is False
        assert verdicts[0].reason == "chunk_not_found"

    async def test_span_not_found(self, mock_llm: LLMClient) -> None:
        v = TwoLayerVerifier({"c1": "text"}, mock_llm)
        claims = [_make_claim(quoted_span="nope")]
        verdicts = await v.verify(claims)
        assert verdicts[0].supported is False
        assert verdicts[0].reason == "span_not_found"

    async def test_entailment_below_threshold(self, mock_llm: LLMClient) -> None:
        mock_llm.generate = AsyncMock(return_value="0.3")
        v = TwoLayerVerifier({"c1": "text with span"}, mock_llm, threshold=0.5)
        claims = [_make_claim(quoted_span="span")]
        verdicts = await v.verify(claims)
        assert verdicts[0].supported is False
        assert verdicts[0].reason == "entailment_below_threshold"

    async def test_supported(self, mock_llm: LLMClient) -> None:
        mock_llm.generate = AsyncMock(return_value="0.9")
        v = TwoLayerVerifier({"c1": "text with span"}, mock_llm, threshold=0.5)
        claims = [_make_claim(quoted_span="span")]
        verdicts = await v.verify(claims)
        assert verdicts[0].supported is True
        assert verdicts[0].score == 0.9
        assert verdicts[0].reason == ""

    async def test_empty_claims(self, mock_llm: LLMClient) -> None:
        v = TwoLayerVerifier({"c1": "text"}, mock_llm)
        assert await v.verify([]) == []

    async def test_judge_non_numeric_returns_zero(self, mock_llm: LLMClient) -> None:
        mock_llm.generate = AsyncMock(return_value="not a number")
        v = TwoLayerVerifier({"c1": "text span"}, mock_llm, threshold=0.5)
        claims = [_make_claim(quoted_span="span")]
        verdicts = await v.verify(claims)
        assert verdicts[0].supported is False
        assert verdicts[0].score == 0.0

    async def test_judge_clamps_above_one(self, mock_llm: LLMClient) -> None:
        mock_llm.generate = AsyncMock(return_value="42.0")
        v = TwoLayerVerifier({"c1": "text span"}, mock_llm, threshold=0.5)
        claims = [_make_claim(quoted_span="span")]
        verdicts = await v.verify(claims)
        assert verdicts[0].score == 1.0

    async def test_judge_clamps_below_zero(self, mock_llm: LLMClient) -> None:
        mock_llm.generate = AsyncMock(return_value="-0.5")
        v = TwoLayerVerifier({"c1": "text span"}, mock_llm, threshold=0.5)
        claims = [_make_claim(quoted_span="span")]
        verdicts = await v.verify(claims)
        assert verdicts[0].score == 0.0

    async def test_threshold_at_zero_always_passes(self, mock_llm: LLMClient) -> None:
        mock_llm.generate = AsyncMock(return_value="0.01")
        v = TwoLayerVerifier({"c1": "Grass is green."}, mock_llm, threshold=0.0)
        claims = [_make_claim(chunk_id="c1", quoted_span="Grass is green")]
        verdicts = await v.verify(claims)
        assert verdicts[0].supported
        assert verdicts[0].score == 0.01

    async def test_threshold_at_one_requires_perfect_score(
        self, mock_llm: LLMClient,
    ) -> None:
        mock_llm.generate = AsyncMock(return_value="0.99")
        v = TwoLayerVerifier({"c1": "Grass is green."}, mock_llm, threshold=1.0)
        claims = [_make_claim(chunk_id="c1", quoted_span="Grass is green")]
        verdicts = await v.verify(claims)
        assert not verdicts[0].supported
        assert verdicts[0].reason == "entailment_below_threshold"

        mock_llm.generate = AsyncMock(return_value="1.0")
        v2 = TwoLayerVerifier({"c1": "Grass is green."}, mock_llm, threshold=1.0)
        verdicts2 = await v2.verify(claims)
        assert verdicts2[0].supported

    async def test_multiple_claims_all_pass(self, mock_llm: LLMClient) -> None:
        mock_llm.generate = AsyncMock(return_value="0.95")
        v = TwoLayerVerifier(
            {"c1": "Transformers use attention.", "c2": "RNNs process sequences."},
            mock_llm, threshold=0.5,
        )
        claims = [
            _make_claim(text="attention", chunk_id="c1", quoted_span="use attention"),
            _make_claim(text="sequences", chunk_id="c2", quoted_span="process sequences"),
        ]
        verdicts = await v.verify(claims)
        assert len(verdicts) == 2
        assert all(v.supported for v in verdicts)
        assert all(v.score == 0.95 for v in verdicts)
