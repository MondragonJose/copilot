"""Integration tests: verify full schema against a live pgvector database."""

from typing import Any

import pytest

pytestmark = pytest.mark.asyncio

# --- Extensions ---

EXPECTED_EXTENSIONS: set[str] = {"vector", "pg_trgm", "uuid-ossp"}


async def test_extensions_exist(db: Any) -> None:
    rows = await db.fetch("SELECT extname FROM pg_extension")
    installed: set[str] = {r["extname"] for r in rows}
    for ext in EXPECTED_EXTENSIONS:
        assert ext in installed, f"Extension '{ext}' is not installed"


# --- Tables ---

EXPECTED_TABLES: set[str] = {"papers", "chunks", "embeddings", "annotations", "jobs"}


async def test_tables_exist(db: Any) -> None:
    rows = await db.fetch(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'",
    )
    found: set[str] = {r["table_name"] for r in rows}
    for table in EXPECTED_TABLES:
        assert table in found, f"Table '{table}' not found"


# --- Columns ---

COLUMNS: dict[str, set[str]] = {
    "papers": {
        "id", "doi", "title", "authors", "year", "venue",
        "abstract", "source", "open_access", "pdf_path",
        "grobid_tei", "meta", "created_at", "updated_at",
    },
    "chunks": {
        "id", "paper_id", "ordinal", "section", "text",
        "char_start", "char_end", "page", "token_count",
        "content_hash", "created_at",
    },
    "embeddings": {"chunk_id", "model", "dim", "vector", "created_at"},
    "annotations": {
        "id", "paper_id", "chunk_id", "page", "char_start",
        "char_end", "quote", "note", "kind", "created_at",
    },
    "jobs": {
        "id", "kind", "status", "paper_id", "payload",
        "attempts", "max_attempts", "error", "stage",
        "created_at", "updated_at",
    },
}


async def assert_columns(db: Any, table: str, expected: set[str]) -> None:
    rows = await db.fetch(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = $1",
        table,
    )
    actual: set[str] = {r["column_name"] for r in rows}
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"Missing columns in '{table}': {missing}"
    assert not extra, f"Unexpected columns in '{table}': {extra}"


async def test_papers_columns(db: Any) -> None:
    await assert_columns(db, "papers", COLUMNS["papers"])


async def test_chunks_columns(db: Any) -> None:
    await assert_columns(db, "chunks", COLUMNS["chunks"])


async def test_embeddings_columns(db: Any) -> None:
    await assert_columns(db, "embeddings", COLUMNS["embeddings"])


async def test_annotations_columns(db: Any) -> None:
    await assert_columns(db, "annotations", COLUMNS["annotations"])


async def test_jobs_columns(db: Any) -> None:
    await assert_columns(db, "jobs", COLUMNS["jobs"])


# --- Constraints ---

UNIQUE_CONSTRAINTS: dict[str, set[str]] = {
    "chunks": {"chunks_paper_id_ordinal_key"},
}


async def test_unique_constraints(db: Any) -> None:
    rows = await db.fetch(
        "SELECT tablename, indexname FROM pg_indexes "
        "WHERE tablename = ANY($1) AND indexdef ILIKE '%UNIQUE%'",
        list(UNIQUE_CONSTRAINTS.keys()),
    )
    actual: set[str] = {r["indexname"] for r in rows}
    for table, expected_indexes in UNIQUE_CONSTRAINTS.items():
        for idx in expected_indexes:
            assert idx in actual, f"Unique index '{idx}' on '{table}' not found"


# --- Indexes ---

EXPECTED_INDEXES: dict[str, set[str]] = {
    "papers": {"idx_papers_doi", "idx_papers_title_trgm"},
    "chunks": {"idx_chunks_paper", "idx_chunks_text_trgm"},
    "embeddings": {"idx_embeddings_hnsw"},
    "annotations": {"idx_annotations_paper"},
    "jobs": {"idx_jobs_status"},
}


async def assert_indexes(db: Any, table: str, expected: set[str]) -> None:
    rows = await db.fetch(
        "SELECT indexname FROM pg_indexes WHERE tablename = $1",
        table,
    )
    actual: set[str] = {r["indexname"] for r in rows}
    for idx in expected:
        assert idx in actual, f"Index '{idx}' on '{table}' not found"


async def test_papers_indexes(db: Any) -> None:
    await assert_indexes(db, "papers", EXPECTED_INDEXES["papers"])


async def test_chunks_indexes(db: Any) -> None:
    await assert_indexes(db, "chunks", EXPECTED_INDEXES["chunks"])


async def test_embeddings_indexes(db: Any) -> None:
    await assert_indexes(db, "embeddings", EXPECTED_INDEXES["embeddings"])


async def test_annotations_indexes(db: Any) -> None:
    await assert_indexes(db, "annotations", EXPECTED_INDEXES["annotations"])


async def test_jobs_indexes(db: Any) -> None:
    await assert_indexes(db, "jobs", EXPECTED_INDEXES["jobs"])


# --- FK constraints (spot-check) ---

async def test_foreign_keys(db: Any) -> None:
    expected_fks: dict[str, set[str]] = {
        "chunks": {"paper_id"},
        "embeddings": {"chunk_id"},
        "annotations": {"paper_id", "chunk_id"},
        "jobs": {"paper_id"},
    }
    rows = await db.fetch(
        "SELECT tc.table_name, kcu.column_name "
        "FROM information_schema.table_constraints tc "
        "JOIN information_schema.key_column_usage kcu "
        "  ON tc.constraint_name = kcu.constraint_name "
        "WHERE tc.constraint_type = 'FOREIGN KEY'",
    )
    actual: dict[str, set[str]] = {}
    for r in rows:
        actual.setdefault(r["table_name"], set()).add(r["column_name"])

    for table, columns in expected_fks.items():
        present = actual.get(table, set())
        missing = columns - present
        assert not missing, (
            f"Missing FK columns in '{table}': {missing}. "
            f"Actual FKs: {present}"
        )


# --- Defaults & not-null (spot-check key columns) ---

NOT_NULL_COLUMNS: dict[str, set[str]] = {
    "papers": {"id", "title", "authors", "source", "meta", "created_at", "updated_at"},
    "chunks": {"id", "paper_id", "ordinal", "text", "content_hash", "created_at"},
    "embeddings": {"chunk_id", "model", "dim", "vector", "created_at"},
    "annotations": {"id", "paper_id", "quote", "kind", "created_at"},
    "jobs": {
        "id", "kind", "status", "payload", "attempts",
        "max_attempts", "error", "stage",
        "created_at", "updated_at",
    },
}


async def test_not_null_constraints(db: Any) -> None:
    for table, expected_not_null in NOT_NULL_COLUMNS.items():
        rows = await db.fetch(
            "SELECT column_name, is_nullable FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = $1",
            table,
        )
        for row in rows:
            col: str = row["column_name"]
            if col in expected_not_null:
                assert row["is_nullable"] == "NO", (
                    f"Column '{table}.{col}' should be NOT NULL but is nullable"
                )
