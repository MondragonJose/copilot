"""LitQA2 evaluation runner — local subset through our QA pipeline.

Reports ``accuracy`` (correct / total) and ``precision@answered``
(correct / answered) as specified in Blueprint §6.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from qa.engine import QAEngine


@dataclass(frozen=True)
class LitQA2Question:
    """A single LitQA2 question with its gold answer."""

    question: str
    gold_answer: str


@dataclass(frozen=True)
class LitQA2ItemResult:
    """Outcome for one question after running through the pipeline."""

    question: LitQA2Question
    predicted_answer: str | None  # ``None`` when the model abstained
    predicted_correct: bool | None  # ``None`` when abstained, else True/False


@dataclass(frozen=True)
class LitQA2Report:
    """Aggregate metrics over the full question set."""

    total: int
    answered: int
    correct: int
    accuracy: float  # correct / total
    precision_at_answered: float  # correct / answered (0 if none answered)
    _results: tuple[LitQA2ItemResult, ...] = field(repr=False)

    def summary(self) -> str:
        return (
            f"LitQA2 Report\n"
            f"  Total questions:     {self.total}\n"
            f"  Answered:            {self.answered}\n"
            f"  Correct:             {self.correct}\n"
            f"  Accuracy:            {self.accuracy:.3f}\n"
            f"  Precision@answered:  {self.precision_at_answered:.3f}\n"
        )


class LitQA2Runner:
    """Run a LitQA2 question set through a ``QAEngine`` instance."""

    def __init__(self, qa_engine: QAEngine) -> None:
        self._qa = qa_engine

    async def run(self, corpus_path: str | Path) -> LitQA2Report:
        """Load *corpus_path*, run every question, and return a report."""
        questions = _load_corpus(corpus_path)
        results: list[LitQA2ItemResult] = []

        for q in questions:
            qa_result = await self._qa.answer(q.question)

            if qa_result.answerable and qa_result.answer:
                correct = _is_correct(qa_result.answer, q.gold_answer)
                results.append(LitQA2ItemResult(
                    question=q,
                    predicted_answer=qa_result.answer,
                    predicted_correct=correct,
                ))
            else:
                results.append(LitQA2ItemResult(
                    question=q,
                    predicted_answer=None,
                    predicted_correct=None,
                ))

        return _build_report(results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_correct(predicted: str, gold: str) -> bool:
    """Does *gold* appear in *predicted* (case-insensitive)?"""
    return gold.lower().strip() in predicted.lower().strip()


def _load_corpus(path: str | Path) -> list[LitQA2Question]:
    path = Path(path)
    questions: list[LitQA2Question] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            questions.append(LitQA2Question(
                question=data["question"],
                gold_answer=data["gold_answer"],
            ))
    return questions


def _build_report(results: Sequence[LitQA2ItemResult]) -> LitQA2Report:
    total = len(results)
    answered = sum(1 for r in results if r.predicted_answer is not None)
    correct = sum(1 for r in results if r.predicted_correct is True)
    accuracy = correct / total if total > 0 else 0.0
    precision = correct / answered if answered > 0 else 0.0
    return LitQA2Report(
        total=total,
        answered=answered,
        correct=correct,
        accuracy=accuracy,
        precision_at_answered=precision,
        _results=tuple(results),
    )
