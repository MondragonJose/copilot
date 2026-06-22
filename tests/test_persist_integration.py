"""Integration tests for persist_document against a real Postgres.

Tests the chunks table schema, column persistence, UNIQUE constraint, and
idempotency via ON CONFLICT directly against a real PostgreSQL instance.

Required external services (see ``docker-compose.test.yml``):
  - PostgreSQL 16 + pgvector + pg_trgm + uuid-ossp

Usage:
  docker compose -f docker-compose.test.yml up -d --wait
  DATABASE_URL=postgresql://rc:rc@localhost:5432/research_copilot \\
    pytest tests/test_persist_integration.py -v
"""

from __future__ import annotations

import os
import pathlib

import asyncpg
import pytest
import pytest_asyncio

from retrieval.db import Pool

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://rc:rc@localhost:5432/research_copilot",
)
MIGRATIONS_DIR = pathlib.Path(__file__).resolve().parent.parent / "migrations"

PAPER_ID = "30000000-0000-0000-0000-000000000001"
CHUNK_IDS = [
    "30000000-0000-0000-0000-000000000010",
    "30000000-0000-0000-0000-000000000011",
]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module")
async def apply_migrations() -> None:
    try:
        conn = await asyncpg.connect(DSN, timeout=5)
    except Exception:
        pytest.skip(
            f"Database not reachable at {DSN}. "
            "Start services: docker compose -f docker-compose.test.yml up -d --wait",
        )
        return
    try:
        for f in sorted(MIGRATIONS_DIR.glob("*.sql")):
            sql = f.read_text()
            if sql.strip():
                await conn.execute(sql)
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def pool(apply_migrations: None) -> Pool:
    p = Pool(DSN)
    try:
        await p.open()
    except Exception:
        pytest.skip("Database not available after migration")
    return p


async def _cleanup(pool: Pool) -> None:
    for cid in CHUNK_IDS:
        await pool.execute(
            "DELETE FROM chunks WHERE id = $1::uuid", cid,
        )
    await pool.execute(
        "DELETE FROM chunks WHERE paper_id = $1::uuid", PAPER_ID,
    )
    await pool.execute(
        "DELETE FROM papers WHERE id = $1::uuid", PAPER_ID,
    )


async def _insert_paper(pool: Pool) -> None:
    await pool.execute(
        "INSERT INTO papers (id, title, source, meta, authors) "
        "VALUES ($1::uuid, 'Test Paper', 'test', '{}'::jsonb, '[]'::jsonb) "
        "ON CONFLICT (id) DO NOTHING",
        PAPER_ID,
    )


# ============================================================================
# (a) Column persistence
# ============================================================================


class TestColumnPersistence:
    """Verify every column of the chunks table is persisted correctly."""

    async def test_all_columns_persisted_correctly(
        self, pool: Pool,
    ) -> None:
        await _cleanup(pool)
        await _insert_paper(pool)

        await pool.execute(
            "INSERT INTO chunks "
            "(id, paper_id, ordinal, section, text, "
            " char_start, char_end, page, token_count, content_hash) "
            "VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10)",
            CHUNK_IDS[0], PAPER_ID, 0, "Introduction",
            "Attention mechanisms allow the model to focus.",
            10, 58, 1, 8, "hash_a",
        )

        row = await pool.fetchrow(
            "SELECT id, paper_id, ordinal, section, text, "
            "char_start, char_end, page, token_count, content_hash "
            "FROM chunks WHERE id = $1::uuid",
            CHUNK_IDS[0],
        )
        assert row is not None
        assert str(row["id"]) == CHUNK_IDS[0]
        assert str(row["paper_id"]) == PAPER_ID
        assert row["ordinal"] == 0
        assert row["section"] == "Introduction"
        assert row["text"] == "Attention mechanisms allow the model to focus."
        assert row["char_start"] == 10
        assert row["char_end"] == 58
        assert row["page"] == 1
        assert row["token_count"] == 8
        assert row["content_hash"] == "hash_a"

        await _cleanup(pool)

    async def test_nullable_columns_default_to_null(
        self, pool: Pool,
    ) -> None:
        await _insert_paper(pool)

        await pool.execute(
            "INSERT INTO chunks (id, paper_id, ordinal, text, content_hash) "
            "VALUES ($1::uuid, $2::uuid, 99, 'minimal row', 'hash_minimal')",
            "30000000-0000-0000-0000-000000000099", PAPER_ID,
        )

        row = await pool.fetchrow(
            "SELECT ordinal, section, char_start, char_end, page, token_count "
            "FROM chunks WHERE id = $1::uuid",
            "30000000-0000-0000-0000-000000000099",
        )
        assert row is not None
        assert row["ordinal"] == 99
        assert row["section"] is None
        assert row["char_start"] is None
        assert row["char_end"] is None
        assert row["page"] is None
        assert row["token_count"] is None

        await pool.execute(
            "DELETE FROM chunks WHERE id = $1::uuid",
            "30000000-0000-0000-0000-000000000099",
        )
        await _cleanup(pool)


# ============================================================================
# (b) UNIQUE(paper_id, ordinal) constraint
# ============================================================================


class TestUniqueOrdinalConstraint:
    """UNIQUE(paper_id, ordinal) must reject duplicate ordinals."""

    async def test_duplicate_ordinal_raises_integrity_error(
        self, pool: Pool,
    ) -> None:
        await _insert_paper(pool)

        await pool.execute(
            "INSERT INTO chunks (id, paper_id, ordinal, text, content_hash) "
            "VALUES ($1::uuid, $2::uuid, 0, 'first chunk', 'hash1')",
            CHUNK_IDS[0], PAPER_ID,
        )

        with pytest.raises(Exception) as exc_info:
            await pool.execute(
                "INSERT INTO chunks (id, paper_id, ordinal, text, content_hash) "
                "VALUES ($1::uuid, $2::uuid, 0, "
                "'second chunk same ordinal', 'hash2')",
                CHUNK_IDS[1], PAPER_ID,
            )

        err = str(exc_info.value)
        assert "unique" in err.lower() or "integrity" in err.lower(), (
            f"Expected unique-constraint error, got: {err}"
        )

        await _cleanup(pool)

    async def test_different_paper_same_ordinal_allowed(
        self, pool: Pool,
    ) -> None:
        paper2_id = "30000000-0000-0000-0000-000000000002"
        await _insert_paper(pool)
        await pool.execute(
            "INSERT INTO papers (id, title, source, meta, authors) "
            "VALUES ($1::uuid, 'Paper 2', 'test', '{}'::jsonb, '[]'::jsonb) "
            "ON CONFLICT (id) DO NOTHING",
            paper2_id,
        )

        await pool.execute(
            "INSERT INTO chunks (id, paper_id, ordinal, text, content_hash) "
            "VALUES ($1::uuid, $2::uuid, 0, 'paper1 ordinal 0', 'h1')",
            CHUNK_IDS[0], PAPER_ID,
        )
        await pool.execute(
            "INSERT INTO chunks (id, paper_id, ordinal, text, content_hash) "
            "VALUES ($1::uuid, $2::uuid, 0, 'paper2 ordinal 0', 'h2')",
            CHUNK_IDS[1], paper2_id,
        )

        rows = await pool.fetch(
            "SELECT paper_id, ordinal FROM chunks "
            "WHERE (paper_id = $1::uuid AND ordinal = 0) "
            "   OR (paper_id = $2::uuid AND ordinal = 0) "
            "ORDER BY paper_id",
            PAPER_ID, paper2_id,
        )
        assert len(rows) == 2

        await _cleanup(pool)
        await pool.execute(
            "DELETE FROM chunks WHERE paper_id = $1::uuid", paper2_id,
        )
        await pool.execute(
            "DELETE FROM papers WHERE id = $1::uuid", paper2_id,
        )


# ============================================================================
# (c) Idempotency via ON CONFLICT (id)
# ============================================================================


class TestOnConflictIdempotent:
    """INSERT … ON CONFLICT (id) DO UPDATE must be idempotent."""

    async def test_same_id_does_not_create_duplicate(
        self, pool: Pool,
    ) -> None:
        await _insert_paper(pool)

        insert_sql = (
            "INSERT INTO chunks "
            "(id, paper_id, ordinal, text, content_hash) "
            "VALUES ($1::uuid, $2::uuid, 0, 'original text', 'hash_orig') "
            "ON CONFLICT (id) DO UPDATE "
            "SET text = EXCLUDED.text, content_hash = EXCLUDED.content_hash"
        )

        await pool.execute(insert_sql, CHUNK_IDS[0], PAPER_ID)

        await pool.execute(
            insert_sql, CHUNK_IDS[0], PAPER_ID,
        )

        row = await pool.fetchrow(
            "SELECT COUNT(*) AS cnt FROM chunks WHERE id = $1::uuid",
            CHUNK_IDS[0],
        )
        assert row["cnt"] == 1, "Same id inserted twice — duplicate row"

        await _cleanup(pool)

    async def test_content_hash_updated_on_re_upsert(
        self, pool: Pool,
    ) -> None:
        await _insert_paper(pool)

        await pool.execute(
            "INSERT INTO chunks (id, paper_id, ordinal, text, content_hash) "
            "VALUES ($1::uuid, $2::uuid, 0, 'v1 text', 'hash_v1')",
            CHUNK_IDS[0], PAPER_ID,
        )

        await pool.execute(
            "INSERT INTO chunks (id, paper_id, ordinal, text, content_hash) "
            "VALUES ($1::uuid, $2::uuid, 0, 'v2 text', 'hash_v2') "
            "ON CONFLICT (id) DO UPDATE "
            "SET text = EXCLUDED.text, content_hash = EXCLUDED.content_hash",
            CHUNK_IDS[0], PAPER_ID,
        )

        row = await pool.fetchrow(
            "SELECT text, content_hash FROM chunks WHERE id = $1::uuid",
            CHUNK_IDS[0],
        )
        assert row["text"] == "v2 text"
        assert row["content_hash"] == "hash_v2"

        await _cleanup(pool)

    async def test_unique_violation_on_different_id_same_ordinal(
        self, pool: Pool,
    ) -> None:
        """Document behavior: ON CONFLICT (id) does NOT protect against
        UNIQUE(paper_id, ordinal) violations when different chunk ids
        have the same (paper_id, ordinal)."""
        await _insert_paper(pool)

        await pool.execute(
            "INSERT INTO chunks (id, paper_id, ordinal, text, content_hash) "
            "VALUES ($1::uuid, $2::uuid, 0, 'first', 'hash1')",
            CHUNK_IDS[0], PAPER_ID,
        )

        with pytest.raises(Exception) as exc_info:
            await pool.execute(
                "INSERT INTO chunks (id, paper_id, ordinal, text, content_hash) "
                "VALUES ($1::uuid, $2::uuid, 0, 'second same ordinal', 'hash2') "
                "ON CONFLICT (id) DO UPDATE "
                "SET text = EXCLUDED.text",
                CHUNK_IDS[1], PAPER_ID,
            )

        err = str(exc_info.value)
        assert "unique" in err.lower() or "integrity" in err.lower(), (
            f"Expected unique-constraint error despite ON CONFLICT (id), "
            f"got: {err}"
        )

        await _cleanup(pool)
