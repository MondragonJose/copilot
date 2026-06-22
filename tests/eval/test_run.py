"""Tests for eval.run — EvalReport, check_gate, write_report, helpers."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.models import QAResult
from eval.run import (
    EvalReport,
    FAITHFULNESS_THRESHOLD,
    ABSTENTION_THRESHOLD,
    RECALL_K,
    check_gate,
    compute_retrieval_metrics,
    goldset_note,
    write_report,
)


class TestEvalReport:
    def test_minimal_report(self) -> None:
        r = EvalReport(
            timestamp="20250101_120000",
            citation_faithfulness=0.96,
            correct_abstention=0.92,
            answer_accuracy=0.80,
            recall_at_k=0.75,
            mrr=0.60,
            litqa2_accuracy=0.50,
            litqa2_precision_at_answered=0.70,
        )
        assert r.timestamp == "20250101_120000"
        assert r.gate == "PASS"

    def test_fail_low_faithfulness(self) -> None:
        r = EvalReport(
            timestamp="t", citation_faithfulness=0.80,
            correct_abstention=0.92, answer_accuracy=0.5,
            recall_at_k=None, mrr=None,
            litqa2_accuracy=None, litqa2_precision_at_answered=None,
        )
        assert r.gate == "FAIL"

    def test_fail_low_abstention(self) -> None:
        r = EvalReport(
            timestamp="t", citation_faithfulness=0.96,
            correct_abstention=0.50, answer_accuracy=0.5,
            recall_at_k=None, mrr=None,
            litqa2_accuracy=None, litqa2_precision_at_answered=None,
        )
        assert r.gate == "FAIL"

    def test_fail_when_none(self) -> None:
        r = EvalReport(
            timestamp="t", citation_faithfulness=None,
            correct_abstention=None, answer_accuracy=None,
            recall_at_k=None, mrr=None,
            litqa2_accuracy=None, litqa2_precision_at_answered=None,
        )
        assert r.gate == "FAIL"

    def test_boundary_pass(self) -> None:
        r = EvalReport(
            timestamp="t",
            citation_faithfulness=FAITHFULNESS_THRESHOLD,
            correct_abstention=ABSTENTION_THRESHOLD,
            answer_accuracy=0.0, recall_at_k=None, mrr=None,
            litqa2_accuracy=None, litqa2_precision_at_answered=None,
        )
        assert r.gate == "PASS"

    def test_summary_includes_gate(self) -> None:
        r = EvalReport(
            timestamp="t", citation_faithfulness=0.96,
            correct_abstention=0.92, answer_accuracy=0.5,
            recall_at_k=0.3, mrr=0.2,
            litqa2_accuracy=0.5, litqa2_precision_at_answered=0.7,
        )
        s = r.summary()
        assert "PASS" in s
        assert "0.960" in s
        assert "0.920" in s

    def test_summary_none_metrics(self) -> None:
        r = EvalReport(
            timestamp="t", citation_faithfulness=0.96,
            correct_abstention=0.92, answer_accuracy=0.5,
            recall_at_k=None, mrr=None,
            litqa2_accuracy=None, litqa2_precision_at_answered=None,
        )
        s = r.summary()
        assert "N/A" in s


class TestCheckGate:
    def test_pass(self) -> None:
        r = EvalReport(
            timestamp="t", citation_faithfulness=0.95,
            correct_abstention=0.90, answer_accuracy=0.5,
            recall_at_k=None, mrr=None,
            litqa2_accuracy=None, litqa2_precision_at_answered=None,
        )
        assert check_gate(r) == "PASS"

    def test_fail(self) -> None:
        r = EvalReport(
            timestamp="t", citation_faithfulness=0.5,
            correct_abstention=0.5, answer_accuracy=0.5,
            recall_at_k=None, mrr=None,
            litqa2_accuracy=None, litqa2_precision_at_answered=None,
        )
        assert check_gate(r) == "FAIL"

    def test_just_below_faithfulness(self) -> None:
        r = EvalReport(
            timestamp="t", citation_faithfulness=0.949,
            correct_abstention=0.90, answer_accuracy=0.5,
            recall_at_k=None, mrr=None,
            litqa2_accuracy=None, litqa2_precision_at_answered=None,
        )
        assert check_gate(r) == "FAIL"


class TestWriteReport:
    def test_writes_json_file(self) -> None:
        r = EvalReport(
            timestamp="20250101_120000",
            citation_faithfulness=0.96,
            correct_abstention=0.92,
            answer_accuracy=0.80,
            recall_at_k=None,
            mrr=None,
            litqa2_accuracy=None,
            litqa2_precision_at_answered=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_report(r, directory=tmp)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["gate"] == "PASS"
            assert data["metrics"]["citation_faithfulness"] == 0.96
            assert data["thresholds"]["citation_faithfulness"] == FAITHFULNESS_THRESHOLD

    def test_default_directory(self) -> None:
        r = EvalReport(
            timestamp="t", citation_faithfulness=0.5,
            correct_abstention=0.5, answer_accuracy=0.5,
            recall_at_k=None, mrr=None,
            litqa2_accuracy=None, litqa2_precision_at_answered=None,
        )
        with patch("eval.run.Path.mkdir") as mkdir:
            with patch("eval.run.Path.write_text"):
                path = write_report(r, directory="/tmp/eval_test_reports")
                assert "tmp" in str(path)
                mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestMain:
    @pytest.mark.asyncio
    async def test_main_passes_on_good_metrics(self) -> None:
        """Fully mocked main() — exercises the CLI pipeline code path."""
        import eval.run as evrun
        with (
            patch("retrieval.db.Pool") as mpool,
            patch("retrieval.embedder.BgeM3Embedder"),
            patch("qa.llm.LLMProvider"),
            patch("retrieval.pgvector_store.PgVectorStore"),
            patch("qa.engine.QAEngine") as mqa_cls,
            patch("eval.goldset.GoldsetLoader") as mloader_cls,
            patch("eval.litqa2.LitQA2Runner") as mrunner_cls,
            patch("builtins.open") as mopen,
            patch.object(evrun, "write_report") as mwrite,
            patch.object(evrun.logger, "info"),
        ):
            pool = mpool.return_value
            pool.open = AsyncMock()
            pool.close = AsyncMock()
            pool.fetchrow = AsyncMock(
                return_value=MagicMock(**{"__getitem__": lambda s, k: "text"}),
            )

            qa = mqa_cls.return_value
            qa.answer = AsyncMock(side_effect=[
                QAResult(
                    question="Q1?", answer="42",
                    claims=[], verdicts=[], answerable=True,
                ),
                QAResult(
                    question="Q2?", answer=None,
                    claims=[], verdicts=[], answerable=False,
                ),
            ])

            loader = mloader_cls.return_value
            loader.load.return_value = [
                MagicMock(
                    question="Q1?", answerable=True, gold_answer="42",
                    gold_span="42", chunk_id="c1", paper_id="p1",
                    gold_relevant_chunk_ids=(),
                ),
                MagicMock(
                    question="Q2?", answerable=False, gold_answer=None,
                    gold_span="", chunk_id="c2", paper_id="p2",
                    gold_relevant_chunk_ids=(),
                ),
            ]

            runner = mrunner_cls.return_value
            runner.run = AsyncMock(
                return_value=MagicMock(
                    total=1, answered=1, correct=1,
                    accuracy=1.0, precision_at_answered=1.0,
                ),
            )

            mopen.return_value.__enter__.return_value.__iter__.return_value = [
                '{"question": "Q1?", "answerable": true, "gold_answer": "42", '
                '"gold_span": "42", "chunk_id": "c1", "paper_id": "p1", '
                '"gold_relevant_chunk_ids": ["c1"]}\n',
                '{"question": "Q2?", "answerable": false, "gold_answer": null, '
                '"gold_span": "", "chunk_id": "c2", "paper_id": "p2", '
                '"gold_relevant_chunk_ids": []}\n',
            ]

            await evrun.main()
            mwrite.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_exits_on_bad_metrics(self) -> None:
        """Fully mocked main() with FAIL gate — exercises sys.exit(1)."""
        import eval.run as evrun
        with (
            patch("retrieval.db.Pool") as mpool,
            patch("retrieval.embedder.BgeM3Embedder"),
            patch("qa.llm.LLMProvider"),
            patch("retrieval.pgvector_store.PgVectorStore"),
            patch("qa.engine.QAEngine") as mqa_cls,
            patch("eval.goldset.GoldsetLoader") as mloader_cls,
            patch("eval.litqa2.LitQA2Runner") as mrunner_cls,
            patch("builtins.open") as mopen,
            patch.object(evrun, "write_report"),
            patch.object(evrun.logger, "info"),
            patch.object(evrun, "sys") as msys,
        ):
            pool = mpool.return_value
            pool.open = AsyncMock()
            pool.close = AsyncMock()
            pool.fetchrow = AsyncMock(
                return_value=MagicMock(**{"__getitem__": lambda s, k: "text"}),
            )

            qa = mqa_cls.return_value
            qa.answer = AsyncMock(side_effect=[
                QAResult(
                    question="Q1?", answer="wrong",
                    claims=[], verdicts=[], answerable=True,
                ),
                QAResult(
                    question="Q2?", answer="not empty",
                    claims=[], verdicts=[], answerable=True,
                ),
            ])

            loader = mloader_cls.return_value
            loader.load.return_value = [
                MagicMock(
                    question="Q1?", answerable=True, gold_answer="42",
                    gold_span="42", chunk_id="c1", paper_id="p1",
                    gold_relevant_chunk_ids=(),
                ),
                MagicMock(
                    question="Q2?", answerable=False, gold_answer=None,
                    gold_span="", chunk_id="c2", paper_id="p2",
                    gold_relevant_chunk_ids=(),
                ),
            ]

            runner = mrunner_cls.return_value
            runner.run = AsyncMock(
                return_value=MagicMock(
                    total=1, answered=1, correct=1,
                    accuracy=1.0, precision_at_answered=1.0,
                ),
            )

            mopen.return_value.__enter__.return_value.__iter__.return_value = [
                '{"question": "Q1?", "answerable": true, "gold_answer": "42", '
                '"gold_span": "42", "chunk_id": "c1", "paper_id": "p1", '
                '"gold_relevant_chunk_ids": ["c1"]}\n',
                '{"question": "Q2?", "answerable": false, "gold_answer": null, '
                '"gold_span": "", "chunk_id": "c2", "paper_id": "p2", '
                '"gold_relevant_chunk_ids": []}\n',
            ]

            await evrun.main()
            msys.exit.assert_called_once_with(1)


class TestComputeRetrievalMetrics:
    """compute_retrieval_metrics — recall@k and MRR from synthetic rankings."""

    @staticmethod
    def _scored(chunk_id: str) -> MagicMock:
        return MagicMock(chunk=MagicMock(chunk_id=chunk_id))

    @pytest.mark.asyncio
    async def test_all_relevant_retrieved(self) -> None:
        """Perfect retrieval: every gold chunk in top-k."""
        from eval.goldset import GoldRow
        rows = [
            GoldRow(
                question="Q1?", answerable=True, gold_answer="A1",
                gold_span="span1", chunk_id="c1", paper_id="p1",
                gold_relevant_chunk_ids=("c1", "c2"),
            ),
        ]
        retriever = MagicMock()
        retriever.search_dense = AsyncMock(return_value=[
            self._scored("c1"),
            self._scored("c2"),
        ])
        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[[0.1, 0.2]])

        recall, mrr = await compute_retrieval_metrics(rows, retriever, embedder, k=10)
        assert recall == 1.0
        assert mrr == 1.0

    @pytest.mark.asyncio
    async def test_partial_relevant_retrieved(self) -> None:
        """Only half of gold chunks in top-k."""
        from eval.goldset import GoldRow
        rows = [
            GoldRow(
                question="Q1?", answerable=True, gold_answer="A1",
                gold_span="span1", chunk_id="c1", paper_id="p1",
                gold_relevant_chunk_ids=("c1", "c2", "c3"),
            ),
        ]
        retriever = MagicMock()
        retriever.search_dense = AsyncMock(return_value=[
            self._scored("c1"),
            self._scored("c4"),
        ])
        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[[0.1, 0.2]])

        recall, mrr = await compute_retrieval_metrics(rows, retriever, embedder, k=10)
        assert recall == 1.0 / 3.0
        assert mrr == 1.0  # first result is relevant

    @pytest.mark.asyncio
    async def test_no_relevant_retrieved(self) -> None:
        """No gold chunks retrieved."""
        from eval.goldset import GoldRow
        rows = [
            GoldRow(
                question="Q1?", answerable=True, gold_answer="A1",
                gold_span="span1", chunk_id="c1", paper_id="p1",
                gold_relevant_chunk_ids=("c2",),
            ),
        ]
        retriever = MagicMock()
        retriever.search_dense = AsyncMock(return_value=[
            self._scored("c1"),
            self._scored("c3"),
        ])
        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[[0.1, 0.2]])

        recall, mrr = await compute_retrieval_metrics(rows, retriever, embedder, k=10)
        assert recall == 0.0
        assert mrr == 0.0

    @pytest.mark.asyncio
    async def test_multiple_questions_mrr(self) -> None:
        """MRR averaged over multiple queries."""
        from eval.goldset import GoldRow
        rows = [
            GoldRow(
                question="Q1?", answerable=True, gold_answer="A1",
                gold_span="span1", chunk_id="c1", paper_id="p1",
                gold_relevant_chunk_ids=("a",),
            ),
            GoldRow(
                question="Q2?", answerable=True, gold_answer="A2",
                gold_span="span2", chunk_id="c2", paper_id="p1",
                gold_relevant_chunk_ids=("z",),
            ),
        ]
        retriever = MagicMock()
        retriever.search_dense = AsyncMock(side_effect=[
            [self._scored("x"), self._scored("a"), self._scored("y")],
            [self._scored("z")],
        ])
        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[[0.1, 0.2]])

        recall, mrr = await compute_retrieval_metrics(rows, retriever, embedder, k=10)
        # Q1: 1 relevant (a), retrieved [x,a,y] → recall=1/1=1.0, RR=1/2
        # Q2: 1 relevant (z), retrieved [z] → recall=1/1=1.0, RR=1/1
        # Avg recall = 1.0, MRR = (0.5 + 1.0)/2 = 0.75
        assert recall == 1.0
        assert mrr == 0.75

    @pytest.mark.asyncio
    async def test_no_answerable_rows_returns_none(self) -> None:
        """When no rows have gold_relevant_chunk_ids, returns None, None."""
        from eval.goldset import GoldRow
        rows = [
            GoldRow(
                question="Q?", answerable=False, gold_answer=None,
                gold_span="", chunk_id="c1", paper_id="p1",
                gold_relevant_chunk_ids=(),
            ),
        ]
        recall, mrr = await compute_retrieval_metrics(
            rows, MagicMock(), MagicMock(), k=10,
        )
        assert recall is None
        assert mrr is None


class TestGoldsetNote:
    """goldset_note — insufficiency detection."""

    def test_adequate_size_returns_empty(self) -> None:
        assert goldset_note(50) == ""

    def test_above_minimum_returns_empty(self) -> None:
        assert goldset_note(100) == ""

    def test_below_minimum_returns_note(self) -> None:
        note = goldset_note(3)
        assert "INSUFFICIENT EVIDENCE" in note
        assert "3" in note
        assert "50" in note

    def test_at_boundary_49_returns_note(self) -> None:
        note = goldset_note(49)
        assert "INSUFFICIENT EVIDENCE" in note


class TestEvalReportGoldsetFields:
    """EvalReport goldset_size / goldset_note defaults."""

    def test_defaults(self) -> None:
        r = EvalReport(
            timestamp="t", citation_faithfulness=0.95,
            correct_abstention=0.90, answer_accuracy=0.5,
            recall_at_k=None, mrr=None,
            litqa2_accuracy=None, litqa2_precision_at_answered=None,
        )
        assert r.goldset_size == 0
        assert r.goldset_note == ""

    def test_explicit_values(self) -> None:
        r = EvalReport(
            timestamp="t", citation_faithfulness=0.95,
            correct_abstention=0.90, answer_accuracy=0.5,
            recall_at_k=0.9, mrr=0.8,
            litqa2_accuracy=None, litqa2_precision_at_answered=None,
            goldset_size=50, goldset_note="INSUFFICIENT EVIDENCE",
        )
        assert r.goldset_size == 50
        assert "INSUFFICIENT EVIDENCE" in r.goldset_note

    def test_summary_includes_size(self) -> None:
        r = EvalReport(
            timestamp="t", citation_faithfulness=0.95,
            correct_abstention=0.90, answer_accuracy=0.5,
            recall_at_k=None, mrr=None,
            litqa2_accuracy=None, litqa2_precision_at_answered=None,
            goldset_size=50,
        )
        s = r.summary()
        assert "50" in s

    def test_summary_includes_note_when_present(self) -> None:
        r = EvalReport(
            timestamp="t", citation_faithfulness=0.95,
            correct_abstention=0.90, answer_accuracy=0.5,
            recall_at_k=None, mrr=None,
            litqa2_accuracy=None, litqa2_precision_at_answered=None,
            goldset_size=3, goldset_note="INSUFFICIENT EVIDENCE — goldset has 3 rows",
        )
        s = r.summary()
        assert "INSUFFICIENT EVIDENCE" in s
