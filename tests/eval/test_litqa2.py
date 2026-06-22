"""Tests for LitQA2 runner — pure logic with mocked QAEngine."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from eval.litqa2 import (
    LitQA2ItemResult,
    LitQA2Question,
    LitQA2Report,
    LitQA2Runner,
    _build_report,
    _is_correct,
    _load_corpus,
)
from core.models import QAResult


class TestLitQA2Question:
    def test_fields(self) -> None:
        q = LitQA2Question(question="Q?", gold_answer="A")
        assert q.question == "Q?"
        assert q.gold_answer == "A"


class TestLitQA2ItemResult:
    def test_answered(self) -> None:
        r = LitQA2ItemResult(
            question=LitQA2Question("Q?", "A"),
            predicted_answer="A", predicted_correct=True,
        )
        assert r.predicted_correct is True

    def test_abstained(self) -> None:
        r = LitQA2ItemResult(
            question=LitQA2Question("Q?", "A"),
            predicted_answer=None, predicted_correct=None,
        )
        assert r.predicted_answer is None


class TestLitQA2Report:
    def test_summary_format(self) -> None:
        r = LitQA2Report(
            total=10, answered=5, correct=4,
            accuracy=0.4, precision_at_answered=0.8,
            _results=(),
        )
        s = r.summary()
        assert "LitQA2 Report" in s
        assert "0.400" in s
        assert "0.800" in s

    def test_zero_answered_precision(self) -> None:
        r = LitQA2Report(
            total=5, answered=0, correct=0,
            accuracy=0.0, precision_at_answered=0.0,
            _results=(),
        )
        assert r.precision_at_answered == 0.0


class TestIsCorrect:
    def test_exact_match(self) -> None:
        assert _is_correct("The answer is 42", "42") is True

    def test_case_insensitive(self) -> None:
        assert _is_correct("HELLO WORLD", "hello") is True

    def test_not_found(self) -> None:
        assert _is_correct("abc", "xyz") is False

    def test_empty_gold(self) -> None:
        assert _is_correct("abc", "") is True

    def test_empty_predicted(self) -> None:
        assert _is_correct("", "x") is False

    def test_whitespace_trim(self) -> None:
        assert _is_correct("hello world", "  HELLO  ") is True


class TestLoadCorpus:
    def test_load_simple(self, tmp_path: Path) -> None:
        p = tmp_path / "corpus.jsonl"
        p.write_text(
            '{"question": "Q1", "gold_answer": "A1"}\n'
            '{"question": "Q2", "gold_answer": "A2"}\n'
        )
        questions = _load_corpus(str(p))
        assert len(questions) == 2
        assert questions[0].question == "Q1"
        assert questions[1].gold_answer == "A2"

    def test_skip_empty_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "corpus.jsonl"
        p.write_text('{"question": "Q1", "gold_answer": "A1"}\n\n\n')
        questions = _load_corpus(str(p))
        assert len(questions) == 1

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "corpus.jsonl"
        p.write_text("")
        questions = _load_corpus(str(p))
        assert questions == []


class TestBuildReport:
    def test_all_answered_correct(self) -> None:
        results = [
            LitQA2ItemResult(
                question=LitQA2Question("Q1", "A"),
                predicted_answer="A", predicted_correct=True,
            ),
            LitQA2ItemResult(
                question=LitQA2Question("Q2", "B"),
                predicted_answer="B", predicted_correct=True,
            ),
        ]
        rep = _build_report(results)
        assert rep.total == 2
        assert rep.answered == 2
        assert rep.correct == 2
        assert rep.accuracy == 1.0
        assert rep.precision_at_answered == 1.0

    def test_some_abstained(self) -> None:
        results = [
            LitQA2ItemResult(
                question=LitQA2Question("Q1", "A"),
                predicted_answer="A", predicted_correct=True,
            ),
            LitQA2ItemResult(
                question=LitQA2Question("Q2", "B"),
                predicted_answer=None, predicted_correct=None,
            ),
        ]
        rep = _build_report(results)
        assert rep.total == 2
        assert rep.answered == 1
        assert rep.correct == 1
        assert rep.accuracy == 0.5
        assert rep.precision_at_answered == 1.0

    def test_all_abstained(self) -> None:
        results = [
            LitQA2ItemResult(
                question=LitQA2Question("Q1", "A"),
                predicted_answer=None, predicted_correct=None,
            ),
        ]
        rep = _build_report(results)
        assert rep.accuracy == 0.0
        assert rep.precision_at_answered == 0.0

    def test_empty_results(self) -> None:
        rep = _build_report([])
        assert rep.total == 0
        assert rep.accuracy == 0.0
        assert rep.precision_at_answered == 0.0

    def test_some_wrong(self) -> None:
        results = [
            LitQA2ItemResult(
                question=LitQA2Question("Q1", "A"),
                predicted_answer="B", predicted_correct=False,
            ),
            LitQA2ItemResult(
                question=LitQA2Question("Q2", "C"),
                predicted_answer="C", predicted_correct=True,
            ),
        ]
        rep = _build_report(results)
        assert rep.correct == 1
        assert rep.accuracy == 0.5


class TestLitQA2Runner:
    @pytest.mark.asyncio
    async def test_run_with_answerable(self, tmp_path: Path) -> None:
        p = tmp_path / "corpus.jsonl"
        p.write_text('{"question": "Q?", "gold_answer": "42"}')

        mock_qa = MagicMock()
        mock_qa.answer = AsyncMock(return_value=QAResult(
            question="Q?", answer="The answer is 42",
            claims=[], verdicts=[], answerable=True,
        ))
        runner = LitQA2Runner(mock_qa)
        report = await runner.run(str(p))
        assert report.total == 1
        assert report.correct == 1

    @pytest.mark.asyncio
    async def test_run_with_abstain(self, tmp_path: Path) -> None:
        p = tmp_path / "corpus.jsonl"
        p.write_text('{"question": "Q?", "gold_answer": "42"}')

        mock_qa = MagicMock()
        mock_qa.answer = AsyncMock(return_value=QAResult(
            question="Q?", answer=None,
            claims=[], verdicts=[], answerable=False,
        ))
        runner = LitQA2Runner(mock_qa)
        report = await runner.run(str(p))
        assert report.total == 1
        assert report.correct == 0
        assert report.answered == 0

    @pytest.mark.asyncio
    async def test_run_no_file(self) -> None:
        mock_qa = MagicMock()
        runner = LitQA2Runner(mock_qa)
        with pytest.raises(FileNotFoundError):
            await runner.run("/nonexistent/path.jsonl")
