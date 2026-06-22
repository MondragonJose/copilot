"""Eval orchestrator — run all metrics, emit a timestamped report, gate PASS/FAIL.

Usage::

    python -m eval.run

Gate (Blueprint §6):
  PASS iff citation_faithfulness >= 0.95 AND correct_abstention >= 0.90.
  Recall@10 and MRR are reported as objectives (≥0.85 / ≥0.6) but do NOT
  block the gate per the spec.
Exits with code 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from core.models import QAResult
from eval.goldset import GoldRow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAITHFULNESS_THRESHOLD = 0.95
ABSTENTION_THRESHOLD = 0.90
DEFAULT_REPORT_DIR = "eval_data/reports"

# Minimum goldset size for statistically meaningful evaluation (§6)
MIN_GOLDSET_SIZE = 50

# Retrieval evaluation
RECALL_K = 10

GateResult = Literal["PASS", "FAIL"]

# ---------------------------------------------------------------------------
# Report data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalReport:
    """Aggregate evaluation results for one run.

    Every field that was actually computed carries a float value.  Metrics
    that could not be computed (e.g. missing infrastructure) are ``None``.
    """

    timestamp: str
    citation_faithfulness: float | None
    correct_abstention: float | None
    answer_accuracy: float | None
    recall_at_k: float | None
    mrr: float | None
    litqa2_accuracy: float | None
    litqa2_precision_at_answered: float | None
    goldset_size: int = 0
    goldset_note: str = ""
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def gate(self) -> GateResult:
        """PASS iff both threshold constraints are met."""
        if self.citation_faithfulness is None or self.correct_abstention is None:
            return "FAIL"
        if (
            self.citation_faithfulness >= FAITHFULNESS_THRESHOLD
            and self.correct_abstention >= ABSTENTION_THRESHOLD
        ):
            return "PASS"
        return "FAIL"

    def summary(self) -> str:
        lines = [
            "=" * 56,
            "Research Copilot — Evaluation Report",
            f"  Timestamp:              {self.timestamp}",
            f"  Goldset size:           {self.goldset_size}",
        ]
        if self.goldset_note:
            lines.append(f"  Goldset note:           {self.goldset_note}")
        lines += [
            f"  Gate:                   {self.gate}",
            "=" * 56,
            f"  Citation faithfulness   "
            f"{self.citation_faithfulness or 0.0:.3f}    (threshold {FAITHFULNESS_THRESHOLD:.2f})",
            f"  Correct abstention      "
            f"{self.correct_abstention or 0.0:.3f}    (threshold {ABSTENTION_THRESHOLD:.2f})",
            f"  Answer accuracy         {self.answer_accuracy or 0.0:.3f}",
            "  Recall@{:<2d}              {}    (target >= 0.85)".format(
                RECALL_K,
                f"{self.recall_at_k:.3f}" if self.recall_at_k is not None else "N/A",
            ),
            "  MRR                     {}    (target >= 0.60)".format(
                f"{self.mrr:.3f}" if self.mrr is not None else "N/A",
            ),
            "  LitQA2 accuracy         {}".format(
                f"{self.litqa2_accuracy:.3f}" if self.litqa2_accuracy is not None else "N/A",
            ),
            "  LitQA2 precision@ans    {}".format(
                f"{self.litqa2_precision_at_answered:.3f}"
                if self.litqa2_precision_at_answered is not None else "N/A",
            ),
            "-" * 56,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gate check
# ---------------------------------------------------------------------------


def check_gate(report: EvalReport) -> GateResult:
    """Evaluate the go/no-go gate."""
    return report.gate


# ---------------------------------------------------------------------------
# Report persistence
# ---------------------------------------------------------------------------


def write_report(report: EvalReport, directory: str = DEFAULT_REPORT_DIR) -> Path:
    """Write a timestamped JSON report file.  Returns the file path."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    report_path = path / f"eval_report_{report.timestamp}.json"

    payload = {
        "timestamp": report.timestamp,
        "gate": report.gate,
        "goldset_size": report.goldset_size,
        "goldset_note": report.goldset_note,
        "thresholds": {
            "citation_faithfulness": FAITHFULNESS_THRESHOLD,
            "correct_abstention": ABSTENTION_THRESHOLD,
        },
        "objectives": {
            "recall_at_10": 0.85,
            "mrr": 0.6,
        },
        "metrics": {
            "citation_faithfulness": report.citation_faithfulness,
            "correct_abstention": report.correct_abstention,
            "answer_accuracy": report.answer_accuracy,
            "recall_at_k": report.recall_at_k,
            "mrr": report.mrr,
            "litqa2_accuracy": report.litqa2_accuracy,
            "litqa2_precision_at_answered": report.litqa2_precision_at_answered,
        },
    }

    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return report_path


# ---------------------------------------------------------------------------
# Retrieval metrics helper
# ---------------------------------------------------------------------------


async def compute_retrieval_metrics(
    gold_rows: Sequence[GoldRow],
    retriever: Any,
    embedder: Any,
    k: int = RECALL_K,
) -> tuple[float | None, float | None]:
    """Compute recall@k and MRR across all answerable goldset rows.

    For each answerable row (``gold_relevant_chunk_ids`` non-empty), run
    dense search, collect ranked chunk IDs, and compare against the gold
    relevant set.

    Returns
    -------
    ``(recall_at_k, mrr)`` where both are ``None`` if no answerable rows
    have relevant chunks defined.
    """
    from eval.metrics import mean_reciprocal_rank, recall_at_k

    answerable_rows = [r for r in gold_rows if r.gold_relevant_chunk_ids]
    if not answerable_rows:
        return None, None

    retrieved_per_q: list[list[str]] = []
    relevant_per_q: list[set[str]] = []

    for row in answerable_rows:
        vectors = await embedder.embed([row.question])
        scored = await retriever.search_dense(
            vectors[0], k=k,
        )
        retrieved_ids = [sc.chunk.chunk_id for sc in scored]
        retrieved_per_q.append(retrieved_ids)
        relevant_per_q.append(set(row.gold_relevant_chunk_ids))

    recall_scores = [
        recall_at_k(retrieved, relevant, k=k)
        for retrieved, relevant in zip(retrieved_per_q, relevant_per_q)
    ]
    recall_val = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    mrr_val = mean_reciprocal_rank(retrieved_per_q, relevant_per_q)
    return recall_val, mrr_val


# ---------------------------------------------------------------------------
# Goldset note helper
# ---------------------------------------------------------------------------


def goldset_note(goldset_size: int, min_size: int = MIN_GOLDSET_SIZE) -> str:
    """Return a note string for the report, or empty string if goldset is adequate."""
    if goldset_size >= min_size:
        return ""
    return (
        f"INSUFFICIENT EVIDENCE — goldset has {goldset_size} rows; "
        f"Blueprint §6 requires ≥{min_size} for statistically "
        f"meaningful evaluation. All metrics computed on available data."
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


async def main() -> None:
    """Orchestrate the full eval pipeline: load data, run QA, compute metrics,
    gate, write report, print summary, exit with code."""
    logging.basicConfig(level=logging.INFO)

    # ── Infrastructure ──────────────────────────────────────────────────
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://rc:rc@localhost:5432/research_copilot",
    )

    from qa.engine import QAEngine
    from qa.llm import LLMProvider
    from retrieval.db import Pool
    from retrieval.embedder import BgeM3Embedder
    from retrieval.pgvector_store import PgVectorStore

    pool = Pool(dsn=database_url)
    await pool.open()
    logger.info("Connected to database")

    try:
        embedder = BgeM3Embedder()
        llm = LLMProvider()
        pgvector = PgVectorStore(pool)
        qa = QAEngine(retriever=pgvector, llm=llm, embedder=embedder)
        logger.info("Pipeline components initialised")

        # ── Goldset ────────────────────────────────────────────────────
        from eval.goldset import GoldsetLoader
        from eval.metrics import (
            answer_accuracy,
            citation_faithfulness,
            correct_abstention,
        )

        goldset_path = "eval_data/goldset_v2.jsonl"

        # First pass: collect chunk_ids referenced in the goldset
        import json as _json

        chunk_ids: set[str] = set()
        with open(goldset_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = _json.loads(line)
                    chunk_ids.add(data["chunk_id"])
                    for cid in data.get("gold_relevant_chunk_ids", []):
                        chunk_ids.add(cid)

        # Fetch chunk texts from the database
        chunk_texts: dict[str, str] = {}
        for cid in chunk_ids:
            row = await pool.fetchrow(
                "SELECT text FROM chunks WHERE id = $1::uuid", cid,
            )
            if row:
                chunk_texts[cid] = row["text"]

        loader = GoldsetLoader(chunk_texts.get)
        gold_rows = loader.load(goldset_path)
        logger.info("Loaded %d goldset rows", len(gold_rows))

        # Run QA on each goldset question
        gold_results: list[QAResult] = []
        gold_answerable: list[bool] = []
        gold_correct: list[bool] = []

        for row in gold_rows:
            qa_result = await qa.answer(row.question)
            gold_results.append(qa_result)
            gold_answerable.append(row.answerable)
            if row.answerable:
                matched = (row.gold_answer or "").lower().strip() in (
                    qa_result.answer or ""
                ).lower().strip()
                gold_correct.append(matched)
            else:
                gold_correct.append(False)

        logger.info("Goldset QA complete (%d questions)", len(gold_rows))

        # ── Goldset metrics ─────────────────────────────────────────────
        faithfulness = citation_faithfulness(gold_results)
        accuracy = answer_accuracy(gold_results, gold_correct, gold_answerable)
        abstention = correct_abstention(gold_results, gold_answerable)

        # ── Retrieval metrics (recall@K, MRR) ───────────────────────────
        recall_val, mrr_val = await compute_retrieval_metrics(
            gold_rows, pgvector, embedder, k=RECALL_K,
        )
        if recall_val is not None:
            logger.info(
                "Retrieval metrics: recall@%d=%.3f, MRR=%.3f",
                RECALL_K, recall_val, mrr_val,
            )
        else:
            logger.warning(
                "No answerable questions have relevant chunks — "
                "cannot compute recall/MRR",
            )

        # ── Goldset note (INSUFFICIENT EVIDENCE if below target) ─────────
        gs_note = goldset_note(len(gold_rows))

        # ── LitQA2 ──────────────────────────────────────────────────────
        from eval.litqa2 import LitQA2Runner

        litqa2_runner = LitQA2Runner(qa)
        litqa2_report = await litqa2_runner.run("eval_data/corpus.txt")
        logger.info("LitQA2 complete (%d questions)", litqa2_report.total)

        # ── Assemble report ─────────────────────────────────────────────
        report = EvalReport(
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            citation_faithfulness=faithfulness,
            correct_abstention=abstention,
            answer_accuracy=accuracy,
            recall_at_k=recall_val,
            mrr=mrr_val,
            litqa2_accuracy=litqa2_report.accuracy,
            litqa2_precision_at_answered=litqa2_report.precision_at_answered,
            goldset_size=len(gold_rows),
            goldset_note=gs_note,
        )

        gate = check_gate(report)
        report_path = write_report(report)

        # ── Output ─────────────────────────────────────────────────────
        print()
        print(report.summary())
        print(f"  Gate:                   {gate}")
        print(f"  Report file:            {report_path}")
        print("=" * 56)
        print()

        if gate == "FAIL":
            sys.exit(1)

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
