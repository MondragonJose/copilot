"""Unit tests for eval run — gate logic with synthetic metrics."""

from __future__ import annotations

import json

import pytest

from eval.run import EvalReport, check_gate, write_report


def _report(
    faithfulness: float | None = 0.95,
    abstention: float | None = 0.90,
    accuracy: float | None = 0.70,
    recall: float | None = None,
    mrr: float | None = None,
    litqa2_acc: float | None = None,
    litqa2_prec: float | None = None,
) -> EvalReport:
    return EvalReport(
        timestamp="20250101_120000",
        citation_faithfulness=faithfulness,
        correct_abstention=abstention,
        answer_accuracy=accuracy,
        recall_at_k=recall,
        mrr=mrr,
        litqa2_accuracy=litqa2_acc,
        litqa2_precision_at_answered=litqa2_prec,
    )


# ===================================================================
# Gate logic
# ===================================================================


class TestGatePass:
    def test_both_at_threshold(self) -> None:
        """Exactly at threshold → PASS."""
        assert check_gate(_report(faithfulness=0.95, abstention=0.90)) == "PASS"

    def test_both_above_threshold(self) -> None:
        """Above threshold → PASS."""
        assert check_gate(_report(faithfulness=0.98, abstention=0.95)) == "PASS"

    def test_both_well_above(self) -> None:
        """Perfect scores → PASS."""
        assert check_gate(_report(faithfulness=1.0, abstention=1.0)) == "PASS"


class TestGateFail:
    def test_faithfulness_below(self) -> None:
        """faithfulness < 0.95, abstention OK → FAIL."""
        assert check_gate(_report(faithfulness=0.94, abstention=0.95)) == "FAIL"

    def test_abstention_below(self) -> None:
        """abstention < 0.90, faithfulness OK → FAIL."""
        assert check_gate(_report(faithfulness=0.96, abstention=0.89)) == "FAIL"

    def test_both_below(self) -> None:
        """Both below threshold → FAIL."""
        assert check_gate(_report(faithfulness=0.80, abstention=0.70)) == "FAIL"

    def test_faithfulness_none(self) -> None:
        """Missing faithfulness metric → FAIL."""
        assert check_gate(_report(faithfulness=None, abstention=0.95)) == "FAIL"

    def test_abstention_none(self) -> None:
        """Missing abstention metric → FAIL."""
        assert check_gate(_report(faithfulness=0.96, abstention=None)) == "FAIL"

    def test_both_none(self) -> None:
        """Neither gate metric available → FAIL."""
        assert check_gate(_report(faithfulness=None, abstention=None)) == "FAIL"

    def test_just_below_faithfulness(self) -> None:
        """0.949 < 0.95 → FAIL (boundary)."""
        assert check_gate(_report(faithfulness=0.949, abstention=0.90)) == "FAIL"


# ===================================================================
# Report file
# ===================================================================


class TestWriteReport:
    def test_writes_timestamped_json(self, tmp_path) -> None:
        report = _report(faithfulness=0.94, abstention=0.95)
        path = write_report(report, directory=str(tmp_path))

        assert path.exists()
        assert path.suffix == ".json"
        assert "eval_report_20250101_120000" in path.name

        data = json.loads(path.read_text())
        assert data["gate"] == "FAIL"
        assert data["metrics"]["citation_faithfulness"] == 0.94
        assert data["metrics"]["correct_abstention"] == 0.95

    def test_report_content_all_fields(self, tmp_path) -> None:
        report = _report(
            faithfulness=0.97, abstention=0.91, accuracy=0.75,
            recall=0.85, mrr=0.62,
            litqa2_acc=0.80, litqa2_prec=0.85,
        )
        path = write_report(report, directory=str(tmp_path))
        data = json.loads(path.read_text())

        assert data["gate"] == "PASS"
        assert data["thresholds"]["citation_faithfulness"] == 0.95
        assert data["thresholds"]["correct_abstention"] == 0.90
        assert data["metrics"]["recall_at_k"] == 0.85
        assert data["metrics"]["mrr"] == 0.62

    def test_report_dir_created(self, tmp_path) -> None:
        nested = tmp_path / "a" / "b"
        report = _report()
        path = write_report(report, directory=str(nested))
        assert path.exists()


# ===================================================================
# EvalReport properties
# ===================================================================


class TestEvalReport:
    def test_gate_property_matches_check(self) -> None:
        r = _report(faithfulness=0.93, abstention=0.90)
        assert r.gate == "FAIL"
        assert check_gate(r) == "FAIL"

    def test_summary_contains_metrics(self) -> None:
        r = _report(faithfulness=0.96, abstention=0.92)
        s = r.summary()
        assert "PASS" in s
        assert "0.960" in s
        assert "0.920" in s

    def test_summary_with_none(self) -> None:
        r = _report(recall=None, mrr=None)
        s = r.summary()
        assert "N/A" in s
