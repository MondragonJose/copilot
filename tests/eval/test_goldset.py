"""Tests for GoldsetLoader — pure logic with a simple dict resolver."""

import json
import tempfile
from pathlib import Path

import pytest

from eval.goldset import GoldRow, GoldsetError, GoldsetLoader


def _resolver(chunk_texts: dict[str, str]):
    return lambda cid: chunk_texts.get(cid)


class TestGoldRow:
    def test_frozen(self) -> None:
        import dataclasses
        r = GoldRow(
            question="Q?", answerable=True, gold_answer="A",
            gold_span="span", chunk_id="c1", paper_id="p1",
            gold_relevant_chunk_ids=("c1",),
        )
        with pytest.raises(Exception):
            r.question = "changed"

    def test_all_fields(self) -> None:
        r = GoldRow(
            question="Q?", answerable=False, gold_answer=None,
            gold_span="", chunk_id="c1", paper_id="p1",
            gold_relevant_chunk_ids=(),
        )
        assert r.answerable is False
        assert r.gold_answer is None
        assert r.gold_span == ""


class TestGoldsetLoader:
    def _write_jsonl(self, rows: list[dict]) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            return f.name

    def test_load_valid_answerable(self) -> None:
        rows = [{
            "question": "What is X?",
            "answerable": True,
            "gold_answer": "X is Y",
            "gold_span": "X is Y",
            "chunk_id": "c1",
            "paper_id": "p1",
            "gold_relevant_chunk_ids": ["c1", "c2"],
        }]
        path = self._write_jsonl(rows)
        loader = GoldsetLoader(_resolver({"c1": "X is Y indeed"}))
        gold = loader.load(path)
        assert len(gold) == 1
        assert gold[0].question == "What is X?"
        assert gold[0].gold_answer == "X is Y"
        assert gold[0].answerable is True
        Path(path).unlink()

    def test_load_valid_unanswerable(self) -> None:
        rows = [{
            "question": "What?",
            "answerable": False,
            "gold_answer": None,
            "gold_span": "",
            "chunk_id": "c1",
            "paper_id": "p1",
            "gold_relevant_chunk_ids": [],
        }]
        path = self._write_jsonl(rows)
        loader = GoldsetLoader(_resolver({"c1": "some text"}))
        gold = loader.load(path)
        assert len(gold) == 1
        assert gold[0].answerable is False
        assert gold[0].gold_answer is None
        Path(path).unlink()

    def test_skip_empty_lines(self) -> None:
        path = self._write_jsonl([])
        loader = GoldsetLoader(_resolver({}))
        gold = loader.load(path)
        assert gold == []
        Path(path).unlink()

    def test_blank_lines_between_rows(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            f.write("\n")
            f.write(json.dumps({
                "question": "Q1", "answerable": True, "gold_answer": "A1",
                "gold_span": "span1", "chunk_id": "c1", "paper_id": "p1",
                "gold_relevant_chunk_ids": [],
            }) + "\n")
            f.write("\n")
            f.write("\n")
            f.write(json.dumps({
                "question": "Q2", "answerable": False, "gold_answer": None,
                "gold_span": "", "chunk_id": "c2", "paper_id": "p2",
                "gold_relevant_chunk_ids": [],
            }) + "\n")
            p = f.name
        loader = GoldsetLoader(_resolver({"c1": "span1 here", "c2": "x"}))
        gold = loader.load(p)
        assert len(gold) == 2
        Path(p).unlink()

    def test_invalid_json(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            f.write("{invalid json\n")
            p = f.name
        loader = GoldsetLoader(_resolver({}))
        with pytest.raises(GoldsetError, match="invalid JSON"):
            loader.load(p)
        Path(p).unlink()

    def test_missing_fields(self) -> None:
        rows = [{"question": "Q?"}]  # missing most fields
        path = self._write_jsonl(rows)
        loader = GoldsetLoader(_resolver({}))
        with pytest.raises(GoldsetError, match="missing fields"):
            loader.load(path)
        Path(path).unlink()

    def test_empty_question(self) -> None:
        rows = [{
            "question": "", "answerable": True, "gold_answer": "A",
            "gold_span": "s", "chunk_id": "c1", "paper_id": "p1",
            "gold_relevant_chunk_ids": [],
        }]
        path = self._write_jsonl(rows)
        loader = GoldsetLoader(_resolver({"c1": "s"}))
        with pytest.raises(GoldsetError, match="non-empty string"):
            loader.load(path)
        Path(path).unlink()

    def test_answerable_not_bool(self) -> None:
        rows = [{
            "question": "Q?", "answerable": "yes", "gold_answer": "A",
            "gold_span": "s", "chunk_id": "c1", "paper_id": "p1",
            "gold_relevant_chunk_ids": [],
        }]
        path = self._write_jsonl(rows)
        loader = GoldsetLoader(_resolver({"c1": "s"}))
        with pytest.raises(GoldsetError, match="boolean"):
            loader.load(path)
        Path(path).unlink()

    def test_answerable_but_empty_gold_answer(self) -> None:
        rows = [{
            "question": "Q?", "answerable": True, "gold_answer": "",
            "gold_span": "s", "chunk_id": "c1", "paper_id": "p1",
            "gold_relevant_chunk_ids": [],
        }]
        path = self._write_jsonl(rows)
        loader = GoldsetLoader(_resolver({"c1": "s"}))
        with pytest.raises(GoldsetError, match="non-empty gold_answer"):
            loader.load(path)
        Path(path).unlink()

    def test_unanswerable_but_gold_answer_not_null(self) -> None:
        rows = [{
            "question": "Q?", "answerable": False, "gold_answer": "A",
            "gold_span": "", "chunk_id": "c1", "paper_id": "p1",
            "gold_relevant_chunk_ids": [],
        }]
        path = self._write_jsonl(rows)
        loader = GoldsetLoader(_resolver({"c1": "s"}))
        with pytest.raises(GoldsetError, match="unanswerable rows must have null"):
            loader.load(path)
        Path(path).unlink()

    def test_gold_span_not_string(self) -> None:
        rows = [{
            "question": "Q?", "answerable": False, "gold_answer": None,
            "gold_span": 123, "chunk_id": "c1", "paper_id": "p1",
            "gold_relevant_chunk_ids": [],
        }]
        path = self._write_jsonl(rows)
        loader = GoldsetLoader(_resolver({"c1": "s"}))
        with pytest.raises(GoldsetError, match="gold_span must be a string"):
            loader.load(path)
        Path(path).unlink()

    def test_chunk_id_empty(self) -> None:
        rows = [{
            "question": "Q?", "answerable": False, "gold_answer": None,
            "gold_span": "", "chunk_id": "", "paper_id": "p1",
            "gold_relevant_chunk_ids": [],
        }]
        path = self._write_jsonl(rows)
        loader = GoldsetLoader(_resolver({"c1": "s"}))
        with pytest.raises(GoldsetError, match="non-empty string"):
            loader.load(path)
        Path(path).unlink()

    def test_paper_id_empty(self) -> None:
        rows = [{
            "question": "Q?", "answerable": False, "gold_answer": None,
            "gold_span": "", "chunk_id": "c1", "paper_id": "",
            "gold_relevant_chunk_ids": [],
        }]
        path = self._write_jsonl(rows)
        loader = GoldsetLoader(_resolver({"c1": "s"}))
        with pytest.raises(GoldsetError, match="non-empty string"):
            loader.load(path)
        Path(path).unlink()

    def test_relevant_ids_not_list(self) -> None:
        rows = [{
            "question": "Q?", "answerable": False, "gold_answer": None,
            "gold_span": "", "chunk_id": "c1", "paper_id": "p1",
            "gold_relevant_chunk_ids": "not_a_list",
        }]
        path = self._write_jsonl(rows)
        loader = GoldsetLoader(_resolver({"c1": "s"}))
        with pytest.raises(GoldsetError, match="must be a list"):
            loader.load(path)
        Path(path).unlink()

    def test_chunk_id_not_found_in_resolver(self) -> None:
        rows = [{
            "question": "Q?", "answerable": False, "gold_answer": None,
            "gold_span": "", "chunk_id": "missing", "paper_id": "p1",
            "gold_relevant_chunk_ids": [],
        }]
        path = self._write_jsonl(rows)
        loader = GoldsetLoader(_resolver({}))
        with pytest.raises(GoldsetError, match="not found in resolver"):
            loader.load(path)
        Path(path).unlink()

    def test_gold_span_not_in_chunk(self) -> None:
        rows = [{
            "question": "Q?", "answerable": True, "gold_answer": "A",
            "gold_span": "nonexistent", "chunk_id": "c1", "paper_id": "p1",
            "gold_relevant_chunk_ids": [],
        }]
        path = self._write_jsonl(rows)
        loader = GoldsetLoader(_resolver({"c1": "actual text"}))
        with pytest.raises(GoldsetError, match="not found in chunk"):
            loader.load(path)
        Path(path).unlink()

    def test_unicode_content(self) -> None:
        rows = [{
            "question": "¿Qué?",
            "answerable": True,
            "gold_answer": "über cool",
            "gold_span": "über",
            "chunk_id": "c1",
            "paper_id": "p1",
            "gold_relevant_chunk_ids": ["c1"],
        }]
        path = self._write_jsonl(rows)
        loader = GoldsetLoader(_resolver({"c1": "über cool stuff"}))
        gold = loader.load(path)
        assert gold[0].question == "¿Qué?"
        assert gold[0].gold_answer == "über cool"
        Path(path).unlink()

    def test_multiple_rows(self) -> None:
        rows = [
            {
                "question": "Q1", "answerable": True, "gold_answer": "A1",
                "gold_span": "span1", "chunk_id": "c1", "paper_id": "p1",
                "gold_relevant_chunk_ids": [],
            },
            {
                "question": "Q2", "answerable": False, "gold_answer": None,
                "gold_span": "", "chunk_id": "c2", "paper_id": "p2",
                "gold_relevant_chunk_ids": [],
            },
        ]
        path = self._write_jsonl(rows)
        loader = GoldsetLoader(_resolver({"c1": "span1 here", "c2": "x"}))
        gold = loader.load(path)
        assert len(gold) == 2
        assert gold[0].question == "Q1"
        assert gold[1].answerable is False
        Path(path).unlink()
