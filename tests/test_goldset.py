"""Unit tests for the goldset loader — valid rows accepted, invalid rejected."""

from __future__ import annotations

import json

import pytest

from eval.goldset import GoldRow, GoldsetError, GoldsetLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHUNK_TEXTS: dict[str, str] = {
    "chunk-a": (
        "self-attention mechanism weighs the importance of each token "
        "in the input sequence"
    ),
    "chunk-b": "The encoder is composed of a stack of six identical layers",
}


def _resolver(chunk_id: str) -> str | None:
    return CHUNK_TEXTS.get(chunk_id)


def _row(
    *,
    question: str = "What is attention?",
    answerable: bool = True,
    gold_answer: str | None = "Attention",
    gold_span: str = "self-attention mechanism weighs the importance",
    chunk_id: str = "chunk-a",
    paper_id: str = "paper-1",
    relevant_ids: list[str] | None = None,
) -> dict:
    return {
        "question": question,
        "answerable": answerable,
        "gold_answer": gold_answer,
        "gold_span": gold_span,
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "gold_relevant_chunk_ids": relevant_ids or [chunk_id],
    }


def _write_jsonl(path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ===================================================================
# Valid rows
# ===================================================================


class TestValidRowsAccepted:
    def test_answerable_row(self, tmp_path) -> None:
        path = tmp_path / "goldset.jsonl"
        _write_jsonl(path, [_row()])

        loader = GoldsetLoader(_resolver)
        rows = loader.load(path)

        assert len(rows) == 1
        r = rows[0]
        assert r.question == "What is attention?"
        assert r.answerable is True
        assert r.gold_answer == "Attention"
        assert r.gold_span == "self-attention mechanism weighs the importance"
        assert r.chunk_id == "chunk-a"
        assert r.paper_id == "paper-1"
        assert r.gold_relevant_chunk_ids == ("chunk-a",)

    def test_unanswerable_row(self, tmp_path) -> None:
        path = tmp_path / "goldset.jsonl"
        _write_jsonl(path, [
            _row(
                answerable=False,
                gold_answer=None,
                gold_span="",
                relevant_ids=[],
            ),
        ])

        loader = GoldsetLoader(_resolver)
        rows = loader.load(path)

        assert len(rows) == 1
        r = rows[0]
        assert r.answerable is False
        assert r.gold_answer is None
        assert r.gold_span == ""

    def test_multiple_rows(self, tmp_path) -> None:
        """Multiple valid rows are all accepted."""
        path = tmp_path / "goldset.jsonl"
        _write_jsonl(path, [
            _row(question="Q1"),
            _row(
                question="Q2",
                answerable=False,
                gold_answer=None,
                gold_span="",
                chunk_id="chunk-b",
                relevant_ids=[],
            ),
            _row(
                question="Q3",
                gold_span="six identical layers",
                chunk_id="chunk-b",
            ),
        ])

        loader = GoldsetLoader(_resolver)
        rows = loader.load(path)

        assert len(rows) == 3
        assert [r.question for r in rows] == ["Q1", "Q2", "Q3"]

    def test_blank_lines_skipped(self, tmp_path) -> None:
        path = tmp_path / "goldset.jsonl"
        with path.open("w") as f:
            f.write("\n")
            f.write(json.dumps(_row()) + "\n")
            f.write("\n")
            f.write("\n")
            f.write(json.dumps(_row(question="Q2", gold_span="six identical layers", chunk_id="chunk-b")) + "\n")

        loader = GoldsetLoader(_resolver)
        rows = loader.load(path)

        assert len(rows) == 2


# ===================================================================
# Invalid — span absent
# ===================================================================


class TestSpanAbsentRejected:
    def test_gold_span_not_in_chunk(self, tmp_path) -> None:
        """gold_span is absent from the referenced chunk text → rejected."""
        path = tmp_path / "goldset.jsonl"
        _write_jsonl(path, [
            _row(gold_span="This span does NOT exist in the chunk text."),
        ])

        loader = GoldsetLoader(_resolver)
        with pytest.raises(GoldsetError, match="gold_span.*not found"):
            loader.load(path)

    def test_chunk_id_not_found(self, tmp_path) -> None:
        """chunk_id not in resolver → rejected."""
        path = tmp_path / "goldset.jsonl"
        _write_jsonl(path, [
            _row(chunk_id="nonexistent-chunk"),
        ])

        loader = GoldsetLoader(_resolver)
        with pytest.raises(GoldsetError, match="not found in resolver"):
            loader.load(path)


# ===================================================================
# Invalid — malformed rows
# ===================================================================


class TestMalformedRowsRejected:
    def test_missing_fields(self, tmp_path) -> None:
        path = tmp_path / "goldset.jsonl"
        with path.open("w") as f:
            f.write('{"question": "Is this valid?"}\n')

        loader = GoldsetLoader(_resolver)
        with pytest.raises(GoldsetError, match="missing fields"):
            loader.load(path)

    def test_invalid_json(self, tmp_path) -> None:
        path = tmp_path / "goldset.jsonl"
        with path.open("w") as f:
            f.write("{invalid json}\n")

        loader = GoldsetLoader(_resolver)
        with pytest.raises(GoldsetError, match="invalid JSON"):
            loader.load(path)

    def test_answerable_with_null_gold_answer(self, tmp_path) -> None:
        path = tmp_path / "goldset.jsonl"
        _write_jsonl(path, [
            _row(gold_answer=None),
        ])

        loader = GoldsetLoader(_resolver)
        with pytest.raises(GoldsetError, match="answerable rows must have"):
            loader.load(path)

    def test_unanswerable_with_non_null_gold_answer(self, tmp_path) -> None:
        path = tmp_path / "goldset.jsonl"
        _write_jsonl(path, [
            _row(answerable=False, gold_answer="should be null"),
        ])

        loader = GoldsetLoader(_resolver)
        with pytest.raises(GoldsetError, match="unanswerable rows must have null"):
            loader.load(path)

    def test_non_bool_answerable(self, tmp_path) -> None:
        path = tmp_path / "goldset.jsonl"
        _write_jsonl(path, [_row(answerable="yes")])  # type: ignore[arg-type]

        loader = GoldsetLoader(_resolver)
        with pytest.raises(GoldsetError, match="answerable must be a boolean"):
            loader.load(path)

    def test_empty_question(self, tmp_path) -> None:
        path = tmp_path / "goldset.jsonl"
        _write_jsonl(path, [_row(question="")])

        loader = GoldsetLoader(_resolver)
        with pytest.raises(GoldsetError, match="question must be a non-empty"):
            loader.load(path)

    def test_non_list_relevant_ids(self, tmp_path) -> None:
        path = tmp_path / "goldset.jsonl"
        _write_jsonl(path, [_row(relevant_ids="not-a-list")])  # type: ignore[arg-type]

        loader = GoldsetLoader(_resolver)
        with pytest.raises(GoldsetError, match="must be a list"):
            loader.load(path)
