"""Tests for evaluation metrics — pure numeric logic."""

from core.models import Claim, ClaimVerdict, QAResult
from eval.metrics import (
    _first_relevant_rank,
    answer_accuracy,
    citation_faithfulness,
    correct_abstention,
    mean_reciprocal_rank,
    recall_at_k,
)


def _qa(verdicts: list[ClaimVerdict],
        answerable: bool = True) -> QAResult:
    return QAResult(
        question="q", answer="a",
        claims=[v.claim for v in verdicts],
        verdicts=verdicts,
        answerable=answerable,
    )


def _cv(supported: bool, score: float = 0.0,
        reason: str = "") -> ClaimVerdict:
    cl = Claim(text="t", chunk_id="c1", quoted_span="t")
    return ClaimVerdict(claim=cl, supported=supported,
                        score=score, reason=reason)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_all_relevant_retrieved(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=2) == 1.0

    def test_partial_relevant(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "d"}, k=2) == 0.5

    def test_no_relevant(self) -> None:
        assert recall_at_k(["a", "b"], {"c"}, k=2) == 0.0

    def test_empty_relevant_returns_zero(self) -> None:
        assert recall_at_k(["a", "b"], set(), k=2) == 0.0

    def test_k_zero(self) -> None:
        assert recall_at_k(["a"], {"a"}, k=0) == 0.0

    def test_k_larger_than_list(self) -> None:
        assert recall_at_k(["a", "b"], {"a", "b", "c"}, k=10) == 2 / 3

    def test_duplicates_in_retrieved(self) -> None:
        assert recall_at_k(["a", "a", "b"], {"a", "b"}, k=2) == 1.0


# ---------------------------------------------------------------------------
# mean_reciprocal_rank
# ---------------------------------------------------------------------------

class TestMeanReciprocalRank:
    def test_all_have_relevant(self) -> None:
        retrieved = [["a", "b", "c"], ["x", "y", "z"]]
        relevant = [{"b"}, {"y"}]
        mrr = mean_reciprocal_rank(retrieved, relevant)
        assert mrr == 0.5  # (1/2 + 1/2) / 2

    def test_some_no_relevant(self) -> None:
        retrieved = [["a", "b"], ["c", "d"]]
        relevant = [{"b"}, {"e"}]
        mrr = mean_reciprocal_rank(retrieved, relevant)
        assert mrr == 0.25  # (1/2 + 0) / 2

    def test_empty_retrieved(self) -> None:
        assert mean_reciprocal_rank([], []) == 0.0

    def test_first_position(self) -> None:
        retrieved = [["a", "b"], ["c", "d"]]
        relevant = [{"a"}, {"c"}]
        mrr = mean_reciprocal_rank(retrieved, relevant)
        assert mrr == 1.0

    def test_last_position(self) -> None:
        retrieved = [["a", "b", "c"], ["x", "y", "z"]]
        relevant = [{"c"}, {"z"}]
        mrr = mean_reciprocal_rank(retrieved, relevant)
        assert pytest.approx(mrr) == (1 / 3 + 1 / 3) / 2


import pytest


# ---------------------------------------------------------------------------
# citation_faithfulness
# ---------------------------------------------------------------------------

class TestCitationFaithfulness:
    def test_all_supported(self) -> None:
        qa = _qa([_cv(True), _cv(True)])
        assert citation_faithfulness([qa]) == 1.0

    def test_mixed(self) -> None:
        qa = _qa([_cv(True), _cv(False)])
        assert citation_faithfulness([qa]) == 0.5

    def test_none_supported(self) -> None:
        qa = _qa([_cv(False), _cv(False)])
        assert citation_faithfulness([qa]) == 0.0

    def test_no_verdicts_returns_one(self) -> None:
        qa = _qa([])
        assert citation_faithfulness([qa]) == 1.0

    def test_multiple_qas(self) -> None:
        qa1 = _qa([_cv(True), _cv(True)])
        qa2 = _qa([_cv(False)])
        assert citation_faithfulness([qa1, qa2]) == 2 / 3


# ---------------------------------------------------------------------------
# answer_accuracy
# ---------------------------------------------------------------------------

class TestAnswerAccuracy:
    def test_all_correct(self) -> None:
        qa = _qa([], answerable=True)
        assert answer_accuracy([qa], [True], [True]) == 1.0

    def test_some_wrong(self) -> None:
        qa_wrong = _qa([], answerable=False)
        assert answer_accuracy([qa_wrong], [False], [True]) == 0.0

    def test_skips_unanswerable_gold(self) -> None:
        qa = _qa([], answerable=True)
        assert answer_accuracy([qa], [True], [False]) == 0.0

    def test_no_answerable_gold_returns_zero(self) -> None:
        qa = _qa([], answerable=True)
        assert answer_accuracy([qa], [True], [False]) == 0.0

    def test_mixed(self) -> None:
        qa1 = _qa([], answerable=True)
        qa2 = _qa([], answerable=False)
        # qa1: gold_answerable=True, gold_correct=True → correct
        # qa2: gold_answerable=False → skipped
        assert answer_accuracy([qa1, qa2], [True, False], [True, False]) == 1.0

    def test_empty_input(self) -> None:
        assert answer_accuracy([], [], []) == 0.0

    def test_model_abstains_when_should_answer(self) -> None:
        qa = _qa([], answerable=False)
        assert answer_accuracy([qa], [False], [True]) == 0.0


# ---------------------------------------------------------------------------
# correct_abstention
# ---------------------------------------------------------------------------

class TestCorrectAbstention:
    def test_all_abstained_correctly(self) -> None:
        qa = _qa([], answerable=False)
        assert correct_abstention([qa], [False]) == 1.0

    def test_failed_to_abstain(self) -> None:
        qa = _qa([], answerable=True)
        assert correct_abstention([qa], [False]) == 0.0

    def test_skips_answerable_gold(self) -> None:
        qa = _qa([], answerable=True)
        assert correct_abstention([qa], [True]) == 0.0

    def test_no_unanswerable_gold_returns_zero(self) -> None:
        qa = _qa([], answerable=True)
        assert correct_abstention([qa], [True]) == 0.0

    def test_mixed(self) -> None:
        qa1 = _qa([], answerable=False)  # correct abstention
        qa2 = _qa([], answerable=True)   # failed to abstain
        assert correct_abstention([qa1, qa2], [False, False]) == 0.5

    def test_empty_input(self) -> None:
        assert correct_abstention([], []) == 0.0


# ---------------------------------------------------------------------------
# _first_relevant_rank
# ---------------------------------------------------------------------------

class TestFirstRelevantRank:
    def test_first_item(self) -> None:
        assert _first_relevant_rank(["a", "b", "c"], {"a"}) == 1

    def test_middle_item(self) -> None:
        assert _first_relevant_rank(["a", "b", "c"], {"b"}) == 2

    def test_last_item(self) -> None:
        assert _first_relevant_rank(["a", "b", "c"], {"c"}) == 3

    def test_not_found(self) -> None:
        assert _first_relevant_rank(["a", "b"], {"c"}) == 0

    def test_empty_retrieved(self) -> None:
        assert _first_relevant_rank([], {"a"}) == 0

    def test_empty_relevant(self) -> None:
        assert _first_relevant_rank(["a", "b"], set()) == 0

    def test_multiple_relevant_returns_first(self) -> None:
        assert _first_relevant_rank(["x", "y", "z"], {"y", "z"}) == 2
