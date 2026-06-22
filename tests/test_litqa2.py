"""Smoke tests for LitQA2 runner — 2–3 items, no external API calls."""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock

from core.models import QAResult
from eval.litqa2 import LitQA2Runner


@pytest_asyncio.fixture
def runner() -> tuple[LitQA2Runner, AsyncMock]:
    qa = AsyncMock()
    return LitQA2Runner(qa), qa


def _write(path, *questions: dict) -> None:
    with path.open("w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def _result(question: str, answer: str | None, answerable: bool) -> QAResult:
    return QAResult(
        question=question,
        answer=answer,
        claims=[],
        verdicts=[],
        answerable=answerable,
    )


# ===================================================================
# All answered correctly
# ===================================================================


class TestAllCorrect:
    @pytest.mark.asyncio
    async def test_two_questions_both_correct(
        self, runner: tuple[LitQA2Runner, AsyncMock], tmp_path,
    ) -> None:
        eng, qa_mock = runner
        corpus = tmp_path / "corpus.txt"
        _write(
            corpus,
            {"question": "What is attention?", "gold_answer": "self-attention"},
            {"question": "How many layers?", "gold_answer": "six"},
        )

        qa_mock.answer.side_effect = [
            _result("What is attention?", "self-attention mechanism", True),
            _result("How many layers?", "six identical layers", True),
        ]

        report = await eng.run(corpus)

        assert report.total == 2
        assert report.answered == 2
        assert report.correct == 2
        assert report.accuracy == 1.0
        assert report.precision_at_answered == 1.0


# ===================================================================
# Mixed answered / abstained
# ===================================================================


class TestMixedAnsweredAndAbstained:
    @pytest.mark.asyncio
    async def test_one_correct_one_abstained_one_wrong(
        self, runner: tuple[LitQA2Runner, AsyncMock], tmp_path,
    ) -> None:
        eng, qa_mock = runner
        corpus = tmp_path / "corpus.txt"
        _write(
            corpus,
            {"question": "Q1", "gold_answer": "A"},
            {"question": "Q2", "gold_answer": "B"},
            {"question": "Q3", "gold_answer": "C"},
        )

        qa_mock.answer.side_effect = [
            _result("Q1", "A is correct", True),       # correct
            _result("Q2", None, False),                  # abstained
            _result("Q3", "wrong guess", True),          # wrong
        ]

        report = await eng.run(corpus)

        assert report.total == 3
        assert report.answered == 2
        assert report.correct == 1
        assert report.accuracy == pytest.approx(1 / 3)
        assert report.precision_at_answered == 0.5  # 1/2


# ===================================================================
# All abstained
# ===================================================================


class TestAllAbstained:
    @pytest.mark.asyncio
    async def test_no_answers_given(
        self, runner: tuple[LitQA2Runner, AsyncMock], tmp_path,
    ) -> None:
        eng, qa_mock = runner
        corpus = tmp_path / "corpus.txt"
        _write(corpus, {"question": "Q1", "gold_answer": "A"})

        qa_mock.answer.return_value = _result("Q1", None, False)

        report = await eng.run(corpus)

        assert report.total == 1
        assert report.answered == 0
        assert report.correct == 0
        assert report.accuracy == 0.0
        assert report.precision_at_answered == 0.0


# ===================================================================
# Answer matching (no external API)
# ===================================================================


class TestAnswerMatching:
    @pytest.mark.asyncio
    async def test_case_insensitive_match(
        self, runner: tuple[LitQA2Runner, AsyncMock], tmp_path,
    ) -> None:
        eng, qa_mock = runner
        corpus = tmp_path / "corpus.txt"
        _write(corpus, {"question": "Q", "gold_answer": "Self-Attention"})

        # Predicted has different casing
        qa_mock.answer.return_value = _result("Q", "Self-attention is key", True)

        report = await eng.run(corpus)
        assert report.correct == 1

    @pytest.mark.asyncio
    async def test_gold_not_in_predicted(
        self, runner: tuple[LitQA2Runner, AsyncMock], tmp_path,
    ) -> None:
        eng, qa_mock = runner
        corpus = tmp_path / "corpus.txt"
        _write(corpus, {"question": "Q", "gold_answer": "six"})

        qa_mock.answer.return_value = _result("Q", "seven layers", True)

        report = await eng.run(corpus)
        assert report.correct == 0
