"""Unit tests for every metric in ``eval.metrics`` — each with a
hand-computed expected value to catch regressions.
"""

from __future__ import annotations

import pytest

from core.models import Claim, ClaimVerdict, QAResult
from eval.metrics import (
    answer_accuracy,
    citation_faithfulness,
    correct_abstention,
    mean_reciprocal_rank,
    recall_at_k,
)


# ===================================================================
# recall_at_k
# ===================================================================


class TestRecallAtK:
    def test_all_relevant_in_top_k(self) -> None:
        """All gold items appear in top-k → 1.0."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c"}
        # top-3: [a, b, c]; hits a, c → 2/2
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_partial_recall(self) -> None:
        """Some gold items in top-k → fractional."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "d", "e"}
        # top-3: [a, b, c]; hits a → 1/3
        assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1 / 3)

    def test_no_relevant_in_top_k(self) -> None:
        """No gold items in top-k → 0.0."""
        retrieved = ["a", "b", "c"]
        relevant = {"x", "y"}
        # top-2: [a, b]; hits 0
        assert recall_at_k(retrieved, relevant, k=2) == 0.0

    def test_k_larger_than_retrieved(self) -> None:
        """k > len(retrieved) clips to available items."""
        retrieved = ["a", "b"]
        relevant = {"a", "c"}
        # top-10 clips to [a, b]; hits a → 1/2
        assert recall_at_k(retrieved, relevant, k=10) == 0.5

    def test_empty_relevant_set(self) -> None:
        """No gold items at all → 0.0."""
        assert recall_at_k(["a", "b"], set(), k=5) == 0.0


# ===================================================================
# mean_reciprocal_rank
# ===================================================================


class TestMeanReciprocalRank:
    def test_all_queries_have_relevant(self) -> None:
        """Every query has a relevant result → MRR = average RR."""
        queries_retrieved = [
            ["a", "b", "c"],  # Q0: relevant at rank 1 → RR = 1
            ["x", "y", "z"],  # Q1: relevant at rank 3 → RR = 1/3
        ]
        queries_relevant = [{"a"}, {"z"}]
        # MRR = (1 + 1/3) / 2 = 2/3
        assert mean_reciprocal_rank(queries_retrieved, queries_relevant) == pytest.approx(2 / 3)

    def test_some_no_relevant(self) -> None:
        """Queries with no relevant result contribute 0."""
        queries_retrieved = [
            ["a", "b", "c"],  # Q0: RR = 1
            ["x", "y", "z"],  # Q1: no relevant in {"m"} → RR = 0
            ["p", "q", "r"],  # Q2: relevant at rank 2 → RR = 1/2
        ]
        queries_relevant = [{"a"}, {"m"}, {"q"}]
        # MRR = (1 + 0 + 1/2) / 3 = 0.5
        assert mean_reciprocal_rank(queries_retrieved, queries_relevant) == 0.5

    def test_empty_queries(self) -> None:
        """No queries → 0.0."""
        assert mean_reciprocal_rank([], []) == 0.0

    def test_first_relevant_is_not_first(self) -> None:
        """Relevant item appears deeper in the ranking."""
        queries_retrieved = [["a", "b", "c", "d"]]
        queries_relevant = [{"d"}]
        # rank 4 → RR = 1/4
        assert mean_reciprocal_rank(queries_retrieved, queries_relevant) == 0.25


# ===================================================================
# citation_faithfulness
# ===================================================================


def _claim(text: str, chunk_id: str = "ch1", span: str | None = None) -> Claim:
    return Claim(text=text, chunk_id=chunk_id, quoted_span=span or text)


def _verdict(claim: Claim, supported: bool = True, score: float = 1.0, reason: str = "") -> ClaimVerdict:
    return ClaimVerdict(claim=claim, supported=supported, score=score, reason=reason)


class TestCitationFaithfulness:
    def test_all_supported(self) -> None:
        """Every claim passes verification → 1.0."""
        c1, c2 = _claim("A"), _claim("B")
        r = QAResult(
            question="q", answer="a",
            claims=[c1, c2],
            verdicts=[_verdict(c1, True), _verdict(c2, True)],
            answerable=True,
        )
        assert citation_faithfulness([r]) == 1.0

    def test_mixed_verdicts(self) -> None:
        """Some supported, some not → fractional."""
        c1, c2, c3 = _claim("A"), _claim("B"), _claim("C")
        r = QAResult(
            question="q", answer="a",
            claims=[c1, c2, c3],
            verdicts=[
                _verdict(c1, True),
                _verdict(c2, False),
                _verdict(c3, True),
            ],
            answerable=True,
        )
        # 2/3
        assert citation_faithfulness([r]) == pytest.approx(2 / 3)

    def test_no_claims(self) -> None:
        """No claims at all → trivially faithful → 1.0."""
        r = QAResult(question="q", answer=None, claims=[], verdicts=[], answerable=False)
        assert citation_faithfulness([r]) == 1.0

    def test_aggregates_across_results(self) -> None:
        """Multiple QAResults contribute to a single score."""
        c1, c2 = _claim("A"), _claim("B")
        r1 = QAResult(
            question="q1", answer="a1",
            claims=[c1, c2],
            verdicts=[_verdict(c1, True), _verdict(c2, False)],
            answerable=True,
        )
        c3 = _claim("C")
        r2 = QAResult(
            question="q2", answer="a2",
            claims=[c3],
            verdicts=[_verdict(c3, True)],
            answerable=True,
        )
        # r1: 1/2, r2: 1/1 → total 2/3
        assert citation_faithfulness([r1, r2]) == pytest.approx(2 / 3)

    def test_none_supported(self) -> None:
        """No claim passes verification → 0.0."""
        c1, c2 = _claim("A"), _claim("B")
        r = QAResult(
            question="q", answer="a",
            claims=[c1, c2],
            verdicts=[_verdict(c1, False), _verdict(c2, False)],
            answerable=True,
        )
        assert citation_faithfulness([r]) == 0.0


# ===================================================================
# answer_accuracy
# ===================================================================


def _qa_result(
    answerable: bool,
    answer: str | None = None,
) -> QAResult:
    return QAResult(
        question="q",
        answer=answer or ("No hay soporte suficiente" if not answerable else "ans"),
        claims=[],
        verdicts=[],
        answerable=answerable,
    )


class TestAnswerAccuracy:
    def test_all_correct(self) -> None:
        """Every gold-answerable query answered correctly → 1.0."""
        results = [_qa_result(True, "42"), _qa_result(False)]
        gold_correct = [True, False]
        gold_answerable = [True, False]
        # answerable: q0 (1); correct: q0 (1) → 1/1
        assert answer_accuracy(results, gold_correct, gold_answerable) == 1.0

    def test_mixed_correctness(self) -> None:
        """Some right, some wrong → fractional."""
        results = [
            _qa_result(True, "42"),
            _qa_result(False),
            _qa_result(True, "Paris"),
            _qa_result(True, "London"),
        ]
        gold_correct = [True, False, False, True]
        gold_answerable = [True, False, True, True]
        # answerable: q0, q2, q3 (3); correct: q0 (T), q2 (F), q3 (T) → 2
        assert answer_accuracy(results, gold_correct, gold_answerable) == pytest.approx(2 / 3)

    def test_no_answerable_queries(self) -> None:
        """No gold-answerable queries → 0.0."""
        results = [_qa_result(False)]
        assert answer_accuracy(results, [False], [False]) == 0.0

    def test_model_abstains_when_should_answer(self) -> None:
        """Model says answerable=False for a gold-answerable query → wrong."""
        results = [_qa_result(False)]
        assert answer_accuracy(results, [False], [True]) == 0.0


# ===================================================================
# correct_abstention
# ===================================================================


class TestCorrectAbstention:
    def test_all_correct(self) -> None:
        """Every gold-unanswerable query correctly abstained → 1.0."""
        results = [_qa_result(False), _qa_result(True, "42")]
        gold_answerable = [False, True]
        # unanswerable: q0 (1); abstained: q0 (1) → 1/1
        assert correct_abstention(results, gold_answerable) == 1.0

    def test_some_fail_to_abstain(self) -> None:
        """Some gold-unanswerable queries got a wrong answer → fractional."""
        results = [
            _qa_result(False),                      # correct abstention
            _qa_result(True, "42"),                 # should have abstained
            _qa_result(False),                      # correct abstention
        ]
        gold_answerable = [False, False, True]
        # unanswerable: q0, q1 (2); abstained: q0 (1) → 1/2
        assert correct_abstention(results, gold_answerable) == 0.5

    def test_no_unanswerable_queries(self) -> None:
        """No gold-unanswerable queries → 0.0."""
        results = [_qa_result(True, "a")]
        assert correct_abstention(results, [True]) == 0.0
