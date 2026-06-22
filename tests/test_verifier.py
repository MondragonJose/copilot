"""Tests for LiteralVerifier — literal-anchor span checking."""
from __future__ import annotations

import pytest

from core.models import Claim
from qa.verifier import LiteralVerifier, TwoLayerVerifier


class _FakeLLM:
    """Mock LLMClient that returns a fixed score string."""

    def __init__(self, score: str = "0.5") -> None:
        self._score = score

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
    ) -> str:
        return self._score


class TestLiteralVerifier:
    """LiteralVerifier verifies each claim's quoted_span against chunk text."""

    def test_present_span_returns_supported(self) -> None:
        chunk_texts = {
            "c1": "Transformers rely on self-attention to model relationships.",
        }
        verifier = LiteralVerifier(chunk_texts)
        claims = [
            Claim(
                text="self-attention is key",
                chunk_id="c1",
                quoted_span="self-attention to model",
            ),
        ]

        verdicts = verifier.verify(claims)

        assert len(verdicts) == 1
        v = verdicts[0]
        assert v.supported
        assert v.score == 1.0
        assert v.reason == ""

    def test_absent_span_returns_not_supported(self) -> None:
        chunk_texts = {
            "c1": "Transformers rely on self-attention to model relationships.",
        }
        verifier = LiteralVerifier(chunk_texts)
        claims = [
            Claim(
                text="fabricated claim",
                chunk_id="c1",
                quoted_span="nothing like this exists in the chunk",
            ),
        ]

        verdicts = verifier.verify(claims)

        assert len(verdicts) == 1
        v = verdicts[0]
        assert not v.supported
        assert v.score == 0.0
        assert v.reason == "span_not_found"

    def test_unknown_chunk_id_returns_not_supported(self) -> None:
        chunk_texts: dict[str, str] = {}
        verifier = LiteralVerifier(chunk_texts)
        claims = [
            Claim(
                text="some claim",
                chunk_id="nonexistent",
                quoted_span="any text",
            ),
        ]

        verdicts = verifier.verify(claims)

        assert len(verdicts) == 1
        v = verdicts[0]
        assert not v.supported
        assert v.score == 0.0
        assert v.reason == "chunk_not_found"

    def test_mixed_claims(self) -> None:
        chunk_texts = {
            "c1": "The sky is blue.",
            "c2": "Grass is green.",
        }
        verifier = LiteralVerifier(chunk_texts)
        claims = [
            Claim(text="sky claim", chunk_id="c1", quoted_span="sky is blue"),
            Claim(text="fake claim", chunk_id="c2", quoted_span="oceans are deep"),
            Claim(text="ghost claim", chunk_id="c3", quoted_span="anything"),
        ]

        verdicts = verifier.verify(claims)

        assert len(verdicts) == 3
        assert verdicts[0].supported
        assert verdicts[0].reason == ""
        assert not verdicts[1].supported
        assert verdicts[1].reason == "span_not_found"
        assert not verdicts[2].supported
        assert verdicts[2].reason == "chunk_not_found"

    def test_empty_claims_returns_empty(self) -> None:
        verifier = LiteralVerifier({"c1": "some text"})
        verdicts = verifier.verify([])
        assert verdicts == []

    def test_case_sensitive_match(self) -> None:
        chunk_texts = {"c1": "Self-Attention is important."}
        verifier = LiteralVerifier(chunk_texts)

        verdict_lower = verifier.verify([
            Claim(text="", chunk_id="c1", quoted_span="self-attention"),
        ])
        assert not verdict_lower[0].supported
        assert verdict_lower[0].reason == "span_not_found"

        verdict_exact = verifier.verify([
            Claim(text="", chunk_id="c1", quoted_span="Self-Attention"),
        ])
        assert verdict_exact[0].supported


# ---------------------------------------------------------------------------
# TwoLayerVerifier  —  anchor + entailment judge
# ---------------------------------------------------------------------------


class TestTwoLayerVerifier:

    def test_entailment_high_score_passes_threshold(self) -> None:
        chunk_texts = {"c1": "The sky is blue on clear days."}
        llm = _FakeLLM("0.9")
        verifier = TwoLayerVerifier(chunk_texts, llm, threshold=0.5)
        claims = [
            Claim(
                text="sky is blue",
                chunk_id="c1",
                quoted_span="sky is blue",
            ),
        ]

        verdicts = verifier.verify(claims)

        assert len(verdicts) == 1
        v = verdicts[0]
        assert v.supported
        assert v.score == 0.9
        assert v.reason == ""

    def test_contradiction_low_score_below_threshold(self) -> None:
        chunk_texts = {"c1": "The sky is blue on clear days."}
        llm = _FakeLLM("0.1")
        verifier = TwoLayerVerifier(chunk_texts, llm, threshold=0.5)
        claims = [
            Claim(
                text="the sky is red",
                chunk_id="c1",
                quoted_span="sky is blue",
            ),
        ]

        verdicts = verifier.verify(claims)

        assert len(verdicts) == 1
        v = verdicts[0]
        assert not v.supported
        assert v.score == 0.1
        assert v.reason == "entailment_below_threshold"

    def test_anchor_fails_before_entailment(self) -> None:
        chunk_texts = {"c1": "The sky is blue."}
        llm = _FakeLLM("0.9")
        verifier = TwoLayerVerifier(chunk_texts, llm)
        claims = [
            Claim(
                text="fake",
                chunk_id="c1",
                quoted_span="nonexistent span",
            ),
        ]

        verdicts = verifier.verify(claims)

        assert len(verdicts) == 1
        v = verdicts[0]
        assert not v.supported
        assert v.score == 0.0
        assert v.reason == "span_not_found"

    def test_unknown_chunk_id_fails_before_entailment(self) -> None:
        chunk_texts: dict[str, str] = {}
        llm = _FakeLLM("0.9")
        verifier = TwoLayerVerifier(chunk_texts, llm)
        claims = [
            Claim(
                text="fake",
                chunk_id="missing",
                quoted_span="anything",
            ),
        ]

        verdicts = verifier.verify(claims)

        assert len(verdicts) == 1
        v = verdicts[0]
        assert not v.supported
        assert v.score == 0.0
        assert v.reason == "chunk_not_found"

    def test_threshold_at_zero_always_passes(self) -> None:
        chunk_texts = {"c1": "Grass is green."}
        llm = _FakeLLM("0.01")
        verifier = TwoLayerVerifier(chunk_texts, llm, threshold=0.0)
        claims = [
            Claim(
                text="grass is green",
                chunk_id="c1",
                quoted_span="Grass is green",
            ),
        ]

        verdicts = verifier.verify(claims)
        v = verdicts[0]
        assert v.supported
        assert v.score == 0.01

    def test_threshold_at_one_requires_perfect_score(self) -> None:
        chunk_texts = {"c1": "Grass is green."}
        llm_low = _FakeLLM("0.99")
        verifier = TwoLayerVerifier(chunk_texts, llm_low, threshold=1.0)
        claims = [
            Claim(
                text="grass is green",
                chunk_id="c1",
                quoted_span="Grass is green",
            ),
        ]

        verdicts = verifier.verify(claims)
        assert not verdicts[0].supported
        assert verdicts[0].reason == "entailment_below_threshold"

        llm_perfect = _FakeLLM("1.0")
        verifier2 = TwoLayerVerifier(chunk_texts, llm_perfect, threshold=1.0)
        verdicts2 = verifier2.verify(claims)
        assert verdicts2[0].supported

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            TwoLayerVerifier({"c1": ""}, _FakeLLM("0.5"), threshold=-0.1)
        with pytest.raises(ValueError, match="threshold must be in"):
            TwoLayerVerifier({"c1": ""}, _FakeLLM("0.5"), threshold=1.1)

    def test_llm_parse_error_defaults_to_zero(self) -> None:
        chunk_texts = {"c1": "Some text."}
        llm = _FakeLLM("not-a-number")
        verifier = TwoLayerVerifier(chunk_texts, llm, threshold=0.5)
        claims = [
            Claim(
                text="some claim",
                chunk_id="c1",
                quoted_span="Some text",
            ),
        ]

        verdicts = verifier.verify(claims)

        v = verdicts[0]
        assert not v.supported
        assert v.score == 0.0
        assert v.reason == "entailment_below_threshold"

    def test_empty_claims_returns_empty(self) -> None:
        verifier = TwoLayerVerifier({"c1": ""}, _FakeLLM("0.9"))
        verdicts = verifier.verify([])
        assert verdicts == []

    def test_multiple_claims_all_pass(self) -> None:
        chunk_texts = {
            "c1": "Transformers use attention.",
            "c2": "RNNs process sequences.",
        }
        llm = _FakeLLM("0.95")
        verifier = TwoLayerVerifier(chunk_texts, llm, threshold=0.5)
        claims = [
            Claim(text="attention", chunk_id="c1", quoted_span="use attention"),
            Claim(text="sequences", chunk_id="c2", quoted_span="process sequences"),
        ]

        verdicts = verifier.verify(claims)

        assert len(verdicts) == 2
        assert all(v.supported for v in verdicts)
        assert all(v.score == 0.95 for v in verdicts)
