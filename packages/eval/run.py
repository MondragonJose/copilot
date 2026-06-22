"""Eval orchestrator — run all metrics, emit a timestamped report, gate PASS/FAIL.

Usage::

    python -m eval.run

Gate: PASS iff citation_faithfulness >= 0.95 AND correct_abstention >= 0.90.
Exits with code 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAITHFULNESS_THRESHOLD = 0.95
ABSTENTION_THRESHOLD = 0.90
DEFAULT_REPORT_DIR = "eval_data/reports"

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
    _raw: dict = field(default_factory=dict, repr=False)

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
            f"  Gate:                   {self.gate}",
            "=" * 56,
            "  Citation faithfulness   {:.3f}    (threshold {:.2f})".format(
                self.citation_faithfulness or 0.0, FAITHFULNESS_THRESHOLD,
            ),
            "  Correct abstention      {:.3f}    (threshold {:.2f})".format(
                self.correct_abstention or 0.0, ABSTENTION_THRESHOLD,
            ),
            "  Answer accuracy         {:.3f}".format(
                self.answer_accuracy or 0.0,
            ),
            "  Recall@K                {}".format(
                f"{self.recall_at_k:.3f}" if self.recall_at_k is not None else "N/A",
            ),
            "  MRR                     {}".format(
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
        "thresholds": {
            "citation_faithfulness": FAITHFULNESS_THRESHOLD,
            "correct_abstention": ABSTENTION_THRESHOLD,
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

    from retrieval.db import Pool
    from retrieval.embedder import BgeM3Embedder
    from retrieval.pgvector_store import PgVectorStore
    from qa.engine import QAEngine
    from qa.llm import LLMProvider

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

        goldset_path = "eval_data/goldset_v1.jsonl"

        # First pass: collect chunk_ids referenced in the goldset
        import json as _json

        chunk_ids: set[str] = set()
        with open(goldset_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    chunk_ids.add(_json.loads(line)["chunk_id"])

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
        gold_results = []
        gold_answerable: list[bool] = []
        gold_correct: list[bool] = []

        for row in gold_rows:
            qa_result = await qa.answer(row.question)
            gold_results.append(qa_result)
            gold_answerable.append(row.answerable)
            if row.answerable:
                matched = row.gold_answer.lower().strip() in (
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
            recall_at_k=None,
            mrr=None,
            litqa2_accuracy=litqa2_report.accuracy,
            litqa2_precision_at_answered=litqa2_report.precision_at_answered,
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
