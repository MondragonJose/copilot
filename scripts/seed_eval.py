#!/usr/bin/env python3
"""Seed the evaluation database and generate the expanded goldset (goldset_v2.jsonl).

Usage::

    # Seed DB and write goldset
    python scripts/seed_eval.py

    # Only seed DB (goldset already exists)
    python scripts/seed_eval.py --db-only

Requires:
    - DATABASE_URL (default: postgresql://rc:rc@localhost:5432/research_copilot)
    - ``pip install ".[embedding]" `` for BgeM3Embedder (fallback zero-vector if missing)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import asyncpg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("seed_eval")

# ---------------------------------------------------------------------------
# Known identifiers — must match goldset_v2.jsonl
# ---------------------------------------------------------------------------
CHUNK_1_ID = "00000000-0000-0000-0000-000000000001"
CHUNK_2_ID = "00000000-0000-0000-0000-000000000002"
CHUNK_3_ID = "00000000-0000-0000-0000-000000000003"
CHUNK_4_ID = "00000000-0000-0000-0000-000000000004"
CHUNK_5_ID = "00000000-0000-0000-0000-000000000005"
CHUNK_6_ID = "00000000-0000-0000-0000-000000000006"
PAPER_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

# Default fields for every goldset row
_DEFAULT_FIELDS = {"paper_id": PAPER_ID}
CHUNK_IDS = [CHUNK_1_ID, CHUNK_2_ID, CHUNK_3_ID, CHUNK_4_ID, CHUNK_5_ID, CHUNK_6_ID]

# ---------------------------------------------------------------------------
# Full chunk texts — source of truth for gold_span validation
# ---------------------------------------------------------------------------
CHUNK_TEXTS: dict[str, str] = {
    CHUNK_1_ID: (
        "The self-attention mechanism weighs the importance of each token in the "
        "input sequence relative to every other token. This allows the model to "
        "capture long-range dependencies without the sequential processing constraints "
        "of RNNs. The attention function maps a query and a set of key-value pairs "
        "to an output, where the output is computed as a weighted sum of the values."
    ),
    CHUNK_2_ID: (
        "The encoder is composed of a stack of six identical layers. Each layer has "
        "two sub-layers: a multi-head self-attention mechanism and a position-wise "
        "feed-forward network. A residual connection surrounds each of the two "
        "sub-layers, followed by layer normalization."
    ),
    CHUNK_3_ID: (
        "The decoder is also composed of a stack of six identical layers. In addition "
        "to the two sub-layers in each encoder layer, the decoder inserts a third "
        "sub-layer, which performs multi-head attention over the output of the encoder "
        "stack. The self-attention sub-layer in the decoder stack is modified to "
        "prevent positions from attending to subsequent positions."
    ),
    CHUNK_4_ID: (
        "We use the Adam optimizer with a learning rate that varies according to a "
        "specific schedule over the course of training. We increase the learning rate "
        "linearly for the first 4000 training steps and decrease it thereafter "
        "proportionally to the inverse square root of the step number. We used a "
        "batch size of 64 for training."
    ),
    CHUNK_5_ID: (
        "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation "
        "task. On the WMT 2014 English-to-French translation task our model achieves "
        "41.0 BLEU. These results outperform previously reported state-of-the-art "
        "results including ensembles, while requiring significantly less time to train."
    ),
    CHUNK_6_ID: (
        "Positional encodings are added to the input embeddings at the bottoms of the "
        "encoder and decoder stacks. We use sine and cosine functions of different "
        "frequencies. The positional encoding has the same dimension as the embeddings, "
        "so that they can be summed."
    ),
}

# ---------------------------------------------------------------------------
# Goldset questions (50 pairs: 35 answerable + 15 unanswerable)
# ---------------------------------------------------------------------------
QUESTIONS: list[dict[str, Any]] = []


def _q(
    question: str, answerable: bool, gold_answer: str | None,
    gold_span: str, chunk_id: str,
    gold_relevant_chunk_ids: list[str],
) -> dict[str, Any]:
    """Build a question dict with required default fields."""
    return {
        "question": question,
        "answerable": answerable,
        "gold_answer": gold_answer,
        "gold_span": gold_span,
        "chunk_id": chunk_id,
        "paper_id": PAPER_ID,
        "gold_relevant_chunk_ids": gold_relevant_chunk_ids,
    }


QUESTIONS = [
    # ---- Chunk 1: self-attention (5 questions) ----
    _q(
        "What mechanism allows transformers to weigh token importance?",
        True, "Self-attention",
        "self-attention mechanism weighs the importance of each token",
        CHUNK_1_ID, [CHUNK_1_ID],
    ),
    _q(
        "What does the attention function use to compute its output?",
        True, "A query and a set of key-value pairs",
        "attention function maps a query and a set of key-value pairs",
        CHUNK_1_ID, [CHUNK_1_ID],
    ),
    _q(
        "How is the output of the attention function computed?",
        True, "As a weighted sum of the values",
        "output is computed as a weighted sum of the values",
        CHUNK_1_ID, [CHUNK_1_ID],
    ),
    _q(
        "What constraint does self-attention avoid compared to RNNs?",
        True, "Sequential processing constraints",
        "without the sequential processing constraints of RNNs",
        CHUNK_1_ID, [CHUNK_1_ID],
    ),
    _q(
        "What type of dependencies can self-attention capture?",
        True, "Long-range dependencies",
        "capture long-range dependencies",
        CHUNK_1_ID, [CHUNK_1_ID],
    ),
    # ---- Chunk 2: encoder layers (5 questions) ----
    _q(
        "How many layers does the transformer encoder contain?",
        True, "Six",
        "encoder is composed of a stack of six identical layers",
        CHUNK_2_ID, [CHUNK_2_ID],
    ),
    _q(
        "What two sub-layers does each encoder layer have?",
        True, "Multi-head self-attention and position-wise feed-forward network",
        "multi-head self-attention mechanism and a position-wise feed-forward network",
        CHUNK_2_ID, [CHUNK_2_ID],
    ),
    _q(
        "What surrounds each sub-layer in the encoder?",
        True, "A residual connection",
        "A residual connection surrounds each of the two sub-layers",
        CHUNK_2_ID, [CHUNK_2_ID],
    ),
    _q(
        "What follows the residual connection in each encoder sub-layer?",
        True, "Layer normalization",
        "followed by layer normalization",
        CHUNK_2_ID, [CHUNK_2_ID],
    ),
    _q(
        "How many sub-layers are in each encoder layer?",
        True, "Two",
        "two sub-layers",
        CHUNK_2_ID, [CHUNK_2_ID],
    ),
    # ---- Chunk 3: decoder layers (5 questions) ----
    _q(
        "How many layers does the transformer decoder contain?",
        True, "Six",
        "decoder is also composed of a stack of six identical layers",
        CHUNK_3_ID, [CHUNK_3_ID],
    ),
    _q(
        "What additional sub-layer does the decoder have compared to the encoder?",
        True, "Multi-head attention over the encoder output",
        "inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack",
        CHUNK_3_ID, [CHUNK_3_ID],
    ),
    _q(
        "How is self-attention modified in the decoder?",
        True, "It is prevented from attending to subsequent positions",
        "prevent positions from attending to subsequent positions",
        CHUNK_3_ID, [CHUNK_3_ID],
    ),
    _q(
        "What does the decoder's cross-attention layer attend to?",
        True, "The output of the encoder stack",
        "multi-head attention over the output of the encoder stack",
        CHUNK_3_ID, [CHUNK_3_ID, CHUNK_2_ID],
    ),
    _q(
        "How many sub-layers are in each decoder layer?",
        True, "Three",
        "inserts a third sub-layer",
        CHUNK_3_ID, [CHUNK_3_ID],
    ),
    # ---- Chunk 4: optimizer (5 questions) ----
    _q(
        "What optimizer does the transformer use?",
        True, "Adam",
        "We use the Adam optimizer",
        CHUNK_4_ID, [CHUNK_4_ID],
    ),
    _q(
        "How does the learning rate vary during training?",
        True, "Increases linearly then decreases proportionally to inverse square root",
        "increase the learning rate linearly for the first 4000 training steps and decrease it thereafter proportionally to the inverse square root",
        CHUNK_4_ID, [CHUNK_4_ID],
    ),
    _q(
        "For how many steps is the learning rate increased?",
        True, "4000",
        "first 4000 training steps",
        CHUNK_4_ID, [CHUNK_4_ID],
    ),
    _q(
        "What batch size is used for transformer training?",
        True, "64",
        "We used a batch size of 64 for training",
        CHUNK_4_ID, [CHUNK_4_ID],
    ),
    _q(
        "What learning rate schedule does the transformer use?",
        True, "Linear warmup with inverse square root decay",
        "increase the learning rate linearly for the first 4000 training steps and decrease it thereafter proportionally to the inverse square root",
        CHUNK_4_ID, [CHUNK_4_ID],
    ),
    # ---- Chunk 5: results/BLEU (6 questions) ----
    _q(
        "What BLEU score does the transformer achieve on English-to-German?",
        True, "28.4",
        "28.4 BLEU on the WMT 2014 English-to-German translation",
        CHUNK_5_ID, [CHUNK_5_ID],
    ),
    _q(
        "What BLEU score does the transformer achieve on English-to-French?",
        True, "41.0",
        "WMT 2014 English-to-French translation task our model achieves 41.0 BLEU",
        CHUNK_5_ID, [CHUNK_5_ID],
    ),
    _q(
        "How does the transformer's training time compare to prior methods?",
        True, "Requires significantly less time",
        "requiring significantly less time to train",
        CHUNK_5_ID, [CHUNK_5_ID],
    ),
    _q(
        "What metric is used to evaluate translation performance?",
        True, "BLEU",
        "28.4 BLEU on the WMT 2014 English-to-German translation",
        CHUNK_5_ID, [CHUNK_5_ID],
    ),
    _q(
        "What translation tasks were used for model evaluation?",
        True, "WMT 2014 English-to-German and English-to-French",
        "WMT 2014 English-to-German translation task",
        CHUNK_5_ID, [CHUNK_5_ID],
    ),
    _q(
        "Do transformer results outperform previous state-of-the-art?",
        True, "Yes",
        "outperform previously reported state-of-the-art results",
        CHUNK_5_ID, [CHUNK_5_ID],
    ),
    # ---- Chunk 6: positional encodings (5 questions) ----
    _q(
        "What is added to the input embeddings in the transformer?",
        True, "Positional encodings",
        "Positional encodings are added to the input embeddings",
        CHUNK_6_ID, [CHUNK_6_ID],
    ),
    _q(
        "What functions are used for positional encodings?",
        True, "Sine and cosine functions of different frequencies",
        "We use sine and cosine functions of different frequencies",
        CHUNK_6_ID, [CHUNK_6_ID],
    ),
    _q(
        "Why do positional encodings have the same dimension as the embeddings?",
        True, "So that they can be summed",
        "positional encoding has the same dimension as the embeddings, so that they can be summed",
        CHUNK_6_ID, [CHUNK_6_ID],
    ),
    _q(
        "Where are positional encodings added in the transformer?",
        True, "At the bottoms of the encoder and decoder stacks",
        "added to the input embeddings at the bottoms of the encoder and decoder stacks",
        CHUNK_6_ID, [CHUNK_6_ID],
    ),
    _q(
        "What is the purpose of positional encodings?",
        True, "To encode token position information",
        "Positional encodings are added to the input embeddings",
        CHUNK_6_ID, [CHUNK_6_ID],
    ),
    # ---- Cross-chunk questions (4 questions) ----
    _q(
        "How many total encoder+decoder layers are in the transformer?",
        True, "Twelve",
        "encoder is composed of a stack of six identical layers",
        CHUNK_2_ID, [CHUNK_2_ID, CHUNK_3_ID],
    ),
    _q(
        "What type of normalization does the transformer use?",
        True, "Layer normalization",
        "followed by layer normalization",
        CHUNK_2_ID, [CHUNK_2_ID],
    ),
    _q(
        "What WMT year was used for translation evaluation?",
        True, "2014",
        "WMT 2014 English-to-German translation",
        CHUNK_5_ID, [CHUNK_5_ID],
    ),
    _q(
        "What two language pairs were used for evaluation?",
        True, "English-to-German and English-to-French",
        "English-to-German translation task",
        CHUNK_5_ID, [CHUNK_5_ID, CHUNK_5_ID],
    ),
    # ---- Unanswerable questions (15 questions) ----
    _q(
        "What dataset was used for the reinforcement learning experiments?",
        False, None, "", CHUNK_1_ID, [],
    ),
    _q(
        "How does the model handle multimodal inputs?",
        False, None, "", CHUNK_1_ID, [],
    ),
    _q(
        "What was the total training cost in dollars?",
        False, None, "", CHUNK_1_ID, [],
    ),
    _q(
        "What hyperparameter tuning method was used?",
        False, None, "", CHUNK_1_ID, [],
    ),
    _q(
        "How does the model compare to BERT on GLUE?",
        False, None, "", CHUNK_1_ID, [],
    ),
    _q(
        "What GPU specifications were used for training?",
        False, None, "", CHUNK_1_ID, [],
    ),
    _q(
        "What data augmentation techniques were applied?",
        False, None, "", CHUNK_1_ID, [],
    ),
    _q(
        "What was the dropout rate used in the model?",
        False, None, "", CHUNK_1_ID, [],
    ),
    _q(
        "What ethical considerations are discussed in the paper?",
        False, None, "", CHUNK_1_ID, [],
    ),
    _q(
        "What was the carbon footprint of training?",
        False, None, "", CHUNK_1_ID, [],
    ),
    _q(
        "How does the model handle out-of-vocabulary words?",
        False, None, "", CHUNK_1_ID, [],
    ),
    _q(
        "What ablation studies were performed?",
        False, None, "", CHUNK_1_ID, [],
    ),
    _q(
        "What was the inference latency of the model?",
        False, None, "", CHUNK_1_ID, [],
    ),
    _q(
        "What software framework was used for implementation?",
        False, None, "", CHUNK_1_ID, [],
    ),
    _q(
        "How does the architecture scale to larger models?",
        False, None, "", CHUNK_1_ID, [],
    ),
]

GOLDSET_PATH = Path("eval_data/goldset_v2.jsonl")

# ---------------------------------------------------------------------------
# DB helpers — fully async
# ---------------------------------------------------------------------------


async def _paper_row(conn: Any, paper_id: str) -> None:
    """Upsert a minimal paper row."""
    await conn.execute(
        "INSERT INTO papers (id, doi, title, authors, source, year, venue, abstract, open_access, pdf_path, grobid_tei) "
        "VALUES ($1::uuid, $2, $3, $4::jsonb, $5, $6, $7, $8, $9, $10, $11) "
        "ON CONFLICT (id) DO NOTHING",
        paper_id,
        "10.1234/transformer",
        "Attention Is All You Need",
        json.dumps([{"name": "Vaswani et al."}]),
        "crossref",
        2017,
        "NeurIPS",
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        False,
        None,
        None,
    )


async def _chunk_row(conn: Any, chunk_id: str, paper_id: str, text: str, ordinal: int) -> None:
    """Upsert a chunk row."""
    await conn.execute(
        "INSERT INTO chunks (id, paper_id, ordinal, section, text, char_start, char_end, page, token_count, content_hash) "
        "VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10) "
        "ON CONFLICT (id) DO NOTHING",
        chunk_id,
        paper_id,
        ordinal,
        "Methods",
        text,
        0,
        len(text),
        1,
        len(text.split()),
        str(hash(text)),
    )


async def _embedding_row(conn: Any, chunk_id: str, model: str, vector: Sequence[float]) -> None:
    """Upsert an embedding row."""
    await conn.execute(
        "INSERT INTO embeddings (chunk_id, model, dim, vector) "
        "VALUES ($1::uuid, $2, $3, $4::vector) "
        "ON CONFLICT (chunk_id) DO NOTHING",
        chunk_id,
        model,
        len(vector),
        "[" + ",".join(str(v) for v in vector) + "]" if vector else "[]",
    )


async def seed_db(conn: Any) -> None:
    """Insert papers and chunks into the database."""
    logger.info("Seeding paper (id=%s)", PAPER_ID)
    await _paper_row(conn, PAPER_ID)

    for i, cid in enumerate(CHUNK_IDS, start=1):
        text = CHUNK_TEXTS[cid]
        logger.info("Seeding chunk %s (ordinal=%d)", cid, i)
        await _chunk_row(conn, cid, PAPER_ID, text, ordinal=i)

    # Generate embeddings via BgeM3Embedder (or zero-vector fallback)
    try:
        from retrieval.embedder import BgeM3Embedder
        embedder = BgeM3Embedder()
        texts = [CHUNK_TEXTS[cid] for cid in CHUNK_IDS]
        logger.info("Generating embeddings for %d chunks ...", len(texts))
        vectors: Sequence[Sequence[float]] = await embedder.embed(texts)
    except Exception as exc:
        logger.warning(
            "BgeM3Embedder unavailable (%s); using zero-vector fallback. "
            "Dense search will not work — recall@10/MRR will be 0.",
            exc,
        )
        dim = 1024
        vectors = [[0.0] * dim for _ in CHUNK_IDS]

    for cid, vec in zip(CHUNK_IDS, vectors):
        await _embedding_row(conn, cid, "bge-m3", vec)

    logger.info("DB seeding complete (%d chunks)", len(CHUNK_IDS))


def write_goldset() -> Path:
    """Write the expanded goldset to disk and return the path."""
    GOLDSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLDSET_PATH.open("w") as f:
        for q in QUESTIONS:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    logger.info("Wrote %s (%d rows)", GOLDSET_PATH, len(QUESTIONS))
    return GOLDSET_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed eval DB and generate goldset",
    )
    parser.add_argument(
        "--db-only", action="store_true",
        help="Only seed the database (skip writing goldset file)",
    )
    args = parser.parse_args()

    if not args.db_only:
        write_goldset()

    # Seed DB
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://rc:rc@localhost:5432/research_copilot",
    )

    async def _run() -> None:
        try:
            conn = await asyncpg.connect(database_url, timeout=10)
        except Exception as exc:
            logger.error(
                "Cannot connect to database at %s — %s. "
                "Skipping DB seed. Run `docker compose up -d` first.",
                database_url, exc,
            )
            return
        try:
            await seed_db(conn)
        finally:
            await conn.close()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
