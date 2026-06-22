"""Evaluation metrics for Research Copilot — Blueprint §6.

Every public function in this module is a metric.  Each has a docstring
with its formula and the meaning of its return value.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from core.models import QAResult

T = TypeVar("T")


def recall_at_k(
    retrieved: Sequence[T],
    relevant: set[T],
    k: int,
) -> float:
    """Recall@K = |relevant ∩ top-k| / |relevant|  (0 if |relevant| = 0).

    Fraction of gold-relevant items that appear in the first *k* positions of
    the ranked *retrieved* list.
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(
    queries_retrieved: Sequence[Sequence[T]],
    queries_relevant: Sequence[set[T]],
) -> float:
    """MRR = (1 / |Q|) * Σ 1 / rank_i

    For each query the reciprocal rank is 1/(rank of first relevant item),
    or 0 when no relevant item is found.  MRR is the average across queries.
    """
    if not queries_retrieved:
        return 0.0

    total = 0.0
    for retrieved, relevant in zip(queries_retrieved, queries_relevant):
        rank = _first_relevant_rank(retrieved, relevant)
        total += 1.0 / rank if rank > 0 else 0.0
    return total / len(queries_retrieved)


def citation_faithfulness(results: Sequence[QAResult]) -> float:
    """Citation faithfulness = supported_claims / total_claims  (1.0 if none).

    A claim is "supported" when its ``ClaimVerdict.supported`` is True.
    Only claims that survived the policy drop are counted (fabricated claims
    do not appear in ``QAResult.verdicts``).

    Formula::

        faithfulness = Σ v.supported / |verdicts|
    """
    total = 0
    supported = 0
    for r in results:
        for v in r.verdicts:
            total += 1
            if v.supported:
                supported += 1
    if total == 0:
        return 1.0
    return supported / total


def answer_accuracy(
    results: Sequence[QAResult],
    gold_correct: Sequence[bool],
    gold_answerable: Sequence[bool],
) -> float:
    """Answer accuracy = correct_count / answerable_count  (0 if none).

    Only queries where ``gold_answerable[i]`` is True are considered.
    A query counts as correct when the model returns ``answerable=True``
    AND ``gold_correct[i]`` is True.

    Formula::

        accuracy = Σ(gold_answerable[i] and gold_correct[i]) / Σ gold_answerable[i]
    """
    correct = 0
    total = 0
    for r, gc, ga in zip(results, gold_correct, gold_answerable):
        if not ga:
            continue
        total += 1
        if r.answerable and gc:
            correct += 1
    if total == 0:
        return 0.0
    return correct / total


def correct_abstention(
    results: Sequence[QAResult],
    gold_answerable: Sequence[bool],
) -> float:
    """Correct abstention = abstained_count / unanswerable_count  (0 if none).

    Only queries where ``gold_answerable[i]`` is False are considered.
    A query counts as correctly abstained when the model returns
    ``answerable=False``.

    Formula::

        abstention = Σ(not gold_answerable[i] and not r.answerable) / Σ(not gold_answerable[i])
    """
    abstained = 0
    total = 0
    for r, ga in zip(results, gold_answerable):
        if ga:
            continue
        total += 1
        if not r.answerable:
            abstained += 1
    if total == 0:
        return 0.0
    return abstained / total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_relevant_rank(retrieved: Sequence[T], relevant: set[T]) -> int:
    """1-based rank of first relevant item, or 0 if none found."""
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return i
    return 0
