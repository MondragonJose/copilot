"""Eval regression suite — metric thresholds, gate logic, frozen golden values.

Guarded regressions:
  • FAITHFULNESS_THRESHOLD  — citation_faithfulness must stay >= 0.95
  • ABSTENTION_THRESHOLD    — correct_abstention must stay >= 0.90
  • Gate PASS/FAIL logic    — fails loudly on drift
  • Metric formulas         — frozen golden values for known inputs
"""

from __future__ import annotations

from collections.abc import Sequence

from core.models import Claim, ClaimVerdict, QAResult
from eval.metrics import (
    answer_accuracy,
    citation_faithfulness,
    correct_abstention,
    mean_reciprocal_rank,
    recall_at_k,
)
from eval.run import EvalReport, FAITHFULNESS_THRESHOLD, ABSTENTION_THRESHOLD


# ── Threshold constants ──────────────────────────────────────────────────────


class TestThresholdConstants:
    """Thresholds must never regress below mandated minimums."""

    def test_faithfulness_threshold_meets_bar(self) -> None:
        assert FAITHFULNESS_THRESHOLD >= 0.95

    def test_abstention_threshold_meets_bar(self) -> None:
        assert ABSTENTION_THRESHOLD >= 0.90

    def test_thresholds_not_too_permissive(self) -> None:
        assert FAITHFULNESS_THRESHOLD <= 1.0
        assert ABSTENTION_THRESHOLD <= 1.0


# ── Citation faithfulness golden values ──────────────────────────────────────


class TestCitationFaithfulnessGolden:
    def test_all_supported(self) -> None:
        c = Claim("t", "cid", "t")
        v = ClaimVerdict(c, True, 1.0, "")
        results = [QAResult("q", "a", [c], [v], True)]
        assert citation_faithfulness(results) == 1.0

    def test_none_supported(self) -> None:
        c = Claim("t", "cid", "t")
        v = ClaimVerdict(c, False, 0.2, "entailment_below_threshold")
        results = [QAResult("q", None, [c], [v], False)]
        assert citation_faithfulness(results) == 0.0

    def test_no_verdicts_returns_1(self) -> None:
        results = [QAResult("q", None, [], [], False)]
        assert citation_faithfulness(results) == 1.0

    def test_mixed_golden(self) -> None:
        c = Claim("t", "cid", "t")
        v1 = ClaimVerdict(c, True, 1.0, "")
        v2 = ClaimVerdict(c, False, 0.3, "insufficient_evidence")
        results = [
            QAResult("q1", "a", [c], [v1], True),
            QAResult("q2", "a", [c, c], [v1, v2], True),
        ]
        assert citation_faithfulness(results) == 2.0 / 3.0

    def test_large_golden_set_does_not_drift(self) -> None:
        """Stability test: 100 results, 75% supported."""
        c = Claim("t", "cid", "t")
        vs = ClaimVerdict(c, True, 1.0, "")
        vu = ClaimVerdict(c, False, 0.4, "insufficient_evidence")
        results = (
            [QAResult("q", "a", [c], [vs], True) for _ in range(75)]
            + [QAResult("q", "a", [c], [vu], True) for _ in range(25)]
        )
        assert citation_faithfulness(results) == 0.75


# ── Correct abstention golden values ─────────────────────────────────────────


class TestCorrectAbstentionGolden:
    def test_all_abstained_correctly(self) -> None:
        results = [QAResult("q", None, [], [], False)]
        gold_answerable = [False]
        assert correct_abstention(results, gold_answerable) == 1.0

    def test_none_abstained(self) -> None:
        results = [QAResult("q", "a", [], [], True)]
        gold_answerable = [False]
        assert correct_abstention(results, gold_answerable) == 0.0

    def test_half_abstained_correctly(self) -> None:
        results = [
            QAResult("q1", None, [], [], False),
            QAResult("q2", "a", [], [], True),
        ]
        gold_answerable = [False, False]
        assert correct_abstention(results, gold_answerable) == 0.5

    def test_no_unanswerable_queries_returns_0(self) -> None:
        results = [QAResult("q", "a", [], [], True)]
        gold_answerable = [True]
        assert correct_abstention(results, gold_answerable) == 0.0

    def test_golden_minimum_abstention_meets_threshold(self) -> None:
        """Abstract: if exactly ABSTENTION_THRESHOLD is met, gate must PASS."""
        results = [QAResult("q", None, [], [], False) for _ in range(90)]
        results += [QAResult("q", "a", [], [], True) for _ in range(10)]
        gold_answerable = [False] * 100
        assert correct_abstention(results, gold_answerable) >= ABSTENTION_THRESHOLD

    def test_just_below_threshold_fails_gate(self) -> None:
        results = [QAResult("q", None, [], [], False) for _ in range(89)]
        results += [QAResult("q", "a", [], [], True) for _ in range(11)]
        gold_answerable = [False] * 100
        assert correct_abstention(results, gold_answerable) < ABSTENTION_THRESHOLD


# ── Answer accuracy golden values ────────────────────────────────────────────


class TestAnswerAccuracyGolden:
    def test_all_correct(self) -> None:
        c = Claim("t", "cid", "t")
        v = ClaimVerdict(c, True, 1.0, "")
        results = [QAResult("q", "gold answer", [c], [v], True)]
        gold_correct = [True]
        gold_answerable = [True]
        assert answer_accuracy(results, gold_correct, gold_answerable) == 1.0

    def test_all_wrong(self) -> None:
        results = [QAResult("q", "wrong", [], [], True)]
        gold_correct = [False]
        gold_answerable = [True]
        assert answer_accuracy(results, gold_correct, gold_answerable) == 0.0

    def test_skips_unanswerable(self) -> None:
        results = [QAResult("q", None, [], [], False)]
        gold_correct = [False]
        gold_answerable = [False]
        assert answer_accuracy(results, gold_correct, gold_answerable) == 0.0

    def test_ignores_unanswerable_queries(self) -> None:
        results = [
            QAResult("q1", "correct", [], [], True),
            QAResult("q2", None, [], [], False),
        ]
        gold_correct = [True, False]
        gold_answerable = [True, False]
        assert answer_accuracy(results, gold_correct, gold_answerable) == 1.0


# ── Recall@K golden values ───────────────────────────────────────────────────


class TestRecallAtKGolder:
    def test_all_relevant_at_k(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0

    def test_partial_relevant(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "d"}, k=3) == 0.5

    def test_none_relevant(self) -> None:
        assert recall_at_k(["a", "b"], {"c"}, k=2) == 0.0

    def test_empty_relevant_returns_0(self) -> None:
        assert recall_at_k(["a", "b"], set(), k=2) == 0.0

    def test_k_greater_than_retrieved(self) -> None:
        assert recall_at_k(["a"], {"a", "b"}, k=10) == 0.5

    def test_k_zero_returns_0(self) -> None:
        assert recall_at_k(["a", "b"], {"a"}, k=0) == 0.0

    def test_case_sensitive_match(self) -> None:
        assert recall_at_k(["A"], {"a"}, k=1) == 0.0


# ── MRR golden values ────────────────────────────────────────────────────────


class TestMeanReciprocalRankGolden:
    def test_first_rank_1(self) -> None:
        ret = [["a", "b", "c"]]
        rel = [{"a"}]
        assert mean_reciprocal_rank(ret, rel) == 1.0

    def test_first_rank_2(self) -> None:
        ret = [["x", "a", "b"]]
        rel = [{"a"}]
        assert mean_reciprocal_rank(ret, rel) == 0.5

    def test_no_relevant_returns_0(self) -> None:
        ret = [["x", "y", "z"]]
        rel = [{"a"}]
        assert mean_reciprocal_rank(ret, rel) == 0.0

    def test_average_multiple_queries(self) -> None:
        ret = [["a", "b"], ["x", "y"], ["c", "d"]]
        rel = [{"a"}, {"y"}, set()]
        assert mean_reciprocal_rank(ret, rel) == (1.0 + 0.5 + 0.0) / 3.0

    def test_empty_queries_returns_0(self) -> None:
        assert mean_reciprocal_rank([], []) == 0.0


# ── Gate logic ────────────────────────────────────────────────────────────────


class TestGateLogic:
    """Gate PASS iff faithfulness >= 0.95 AND abstention >= 0.90."""

    def test_pass_when_both_above_threshold(self) -> None:
        report = EvalReport(
            timestamp="t",
            citation_faithfulness=0.98,
            correct_abstention=0.95,
            answer_accuracy=0.5,
            recall_at_k=None,
            mrr=None,
            litqa2_accuracy=None,
            litqa2_precision_at_answered=None,
        )
        assert report.gate == "PASS"

    def test_fail_when_faithfulness_below(self) -> None:
        report = EvalReport(
            timestamp="t",
            citation_faithfulness=0.94,
            correct_abstention=0.95,
            answer_accuracy=0.5,
            recall_at_k=None,
            mrr=None,
            litqa2_accuracy=None,
            litqa2_precision_at_answered=None,
        )
        assert report.gate == "FAIL"

    def test_fail_when_abstention_below(self) -> None:
        report = EvalReport(
            timestamp="t",
            citation_faithfulness=0.98,
            correct_abstention=0.85,
            answer_accuracy=0.5,
            recall_at_k=None,
            mrr=None,
            litqa2_accuracy=None,
            litqa2_precision_at_answered=None,
        )
        assert report.gate == "FAIL"

    def test_fail_when_both_below(self) -> None:
        report = EvalReport(
            timestamp="t",
            citation_faithfulness=0.80,
            correct_abstention=0.80,
            answer_accuracy=0.5,
            recall_at_k=None,
            mrr=None,
            litqa2_accuracy=None,
            litqa2_precision_at_answered=None,
        )
        assert report.gate == "FAIL"

    def test_fail_when_faithfulness_none(self) -> None:
        report = EvalReport(
            timestamp="t",
            citation_faithfulness=None,
            correct_abstention=0.95,
            answer_accuracy=None,
            recall_at_k=None,
            mrr=None,
            litqa2_accuracy=None,
            litqa2_precision_at_answered=None,
        )
        assert report.gate == "FAIL"

    def test_fail_when_abstention_none(self) -> None:
        report = EvalReport(
            timestamp="t",
            citation_faithfulness=0.98,
            correct_abstention=None,
            answer_accuracy=None,
            recall_at_k=None,
            mrr=None,
            litqa2_accuracy=None,
            litqa2_precision_at_answered=None,
        )
        assert report.gate == "FAIL"

    def test_exact_thresholds_pass(self) -> None:
        """Boundary: exactly 0.95 faithfulness and 0.90 abstention."""
        report = EvalReport(
            timestamp="t",
            citation_faithfulness=0.95,
            correct_abstention=0.90,
            answer_accuracy=0.5,
            recall_at_k=None,
            mrr=None,
            litqa2_accuracy=None,
            litqa2_precision_at_answered=None,
        )
        assert report.gate == "PASS"

    def test_exact_faithfulness_above_abstention_below(self) -> None:
        report = EvalReport(
            timestamp="t",
            citation_faithfulness=0.95,
            correct_abstention=0.89,
            answer_accuracy=0.5,
            recall_at_k=None,
            mrr=None,
            litqa2_accuracy=None,
            litqa2_precision_at_answered=None,
        )
        assert report.gate == "FAIL"

    def test_exact_abstention_above_faithfulness_below(self) -> None:
        report = EvalReport(
            timestamp="t",
            citation_faithfulness=0.94,
            correct_abstention=0.90,
            answer_accuracy=0.5,
            recall_at_k=None,
            mrr=None,
            litqa2_accuracy=None,
            litqa2_precision_at_answered=None,
        )
        assert report.gate == "FAIL"


class TestEvalReportSummary:
    """EvalReport.summary() must contain key metrics and the gate."""

    def test_summary_contains_gate(self) -> None:
        report = EvalReport(
            timestamp="20250101_120000",
            citation_faithfulness=0.98,
            correct_abstention=0.95,
            answer_accuracy=0.8,
            recall_at_k=0.9,
            mrr=0.85,
            litqa2_accuracy=0.7,
            litqa2_precision_at_answered=0.8,
        )
        summary = report.summary()
        assert "PASS" in summary
        assert "0.980" in summary
        assert "0.950" in summary

    def test_summary_with_nones(self) -> None:
        report = EvalReport(
            timestamp="t",
            citation_faithfulness=0.98,
            correct_abstention=0.95,
            answer_accuracy=0.8,
            recall_at_k=None,
            mrr=None,
            litqa2_accuracy=None,
            litqa2_precision_at_answered=None,
        )
        summary = report.summary()
        assert "N/A" in summary


# ── Pipeline guard: check_gate ────────────────────────────────────────────────


class TestCheckGate:
    def test_check_gate_delegates_to_report_gate(self) -> None:
        from eval.run import check_gate

        report = EvalReport(
            timestamp="t",
            citation_faithfulness=0.98,
            correct_abstention=0.95,
            answer_accuracy=0.5,
            recall_at_k=None,
            mrr=None,
            litqa2_accuracy=None,
            litqa2_precision_at_answered=None,
        )
        assert check_gate(report) == "PASS"

    def test_check_gate_fail(self) -> None:
        from eval.run import check_gate

        report = EvalReport(
            timestamp="t",
            citation_faithfulness=0.50,
            correct_abstention=0.50,
            answer_accuracy=0.5,
            recall_at_k=None,
            mrr=None,
            litqa2_accuracy=None,
            litqa2_precision_at_answered=None,
        )
        assert check_gate(report) == "FAIL"
