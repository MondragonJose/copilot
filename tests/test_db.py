"""Integration tests: async Pool open/close, query helpers, parametrised queries."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from core.errors import RetrievalError
from retrieval.db import Pool

pytestmark = pytest.mark.asyncio

DEFAULT_DSN = "postgresql://rc:rc@localhost:5432/research_copilot"


async def _open_or_skip(p: Pool) -> None:
    try:
        await p.open()
    except RetrievalError:
        pytest.skip("Database not available")


@pytest_asyncio.fixture
async def pool() -> AsyncGenerator[Pool, None]:
    dsn = os.environ.get("DATABASE_URL", DEFAULT_DSN)
    p = Pool(dsn)
    try:
        await p.open()
    except Exception:
        pytest.skip("Database not available")
        return
    yield p
    await p.close()


class TestPoolLifecycle:
    async def test_open_close(self) -> None:
        dsn = os.environ.get("DATABASE_URL", DEFAULT_DSN)
        p = Pool(dsn)
        assert p.size is None
        await _open_or_skip(p)
        assert p.size is not None and p.size >= 0
        await p.close()
        assert p.size is None

    async def test_close_idempotent(self) -> None:
        dsn = os.environ.get("DATABASE_URL", DEFAULT_DSN)
        p = Pool(dsn)
        await _open_or_skip(p)
        await p.close()
        await p.close()
        assert p.size is None

    async def test_context_manager(self) -> None:
        dsn = os.environ.get("DATABASE_URL", DEFAULT_DSN)
        try:
            async with Pool(dsn) as p:
                assert p.size is not None and p.size >= 0
        except RetrievalError:
            pytest.skip("Database not available")
            return
        assert p.size is None


class TestQueryHelpers:
    async def test_select_1(self, pool: Pool) -> None:
        rows = await pool.fetch("SELECT 1 AS v")
        assert len(rows) == 1
        assert rows[0]["v"] == 1

    async def test_fetchrow(self, pool: Pool) -> None:
        row = await pool.fetchrow("SELECT 1 AS v")
        assert row is not None
        assert row["v"] == 1

    async def test_fetchrow_none(self, pool: Pool) -> None:
        row = await pool.fetchrow(
            "SELECT 1 WHERE FALSE",
        )
        assert row is None

    async def test_parametrized_query(self, pool: Pool) -> None:
        rows = await pool.fetch(
            "SELECT $1::int + $2::int AS s",
            1,
            2,
        )
        assert rows[0]["s"] == 3

    async def test_execute_returns_tag(self, pool: Pool) -> None:
        tag = await pool.execute("SELECT 1")
        assert isinstance(tag, str)

    async def test_no_sql_injection(self, pool: Pool) -> None:
        malicious = "'; DROP TABLE papers; --"
        row = await pool.fetchrow(
            "SELECT $1::text AS v",
            malicious,
        )
        assert row is not None
        assert row["v"] == malicious


class TestErrors:
    async def test_not_open_raises(self) -> None:
        p = Pool(DEFAULT_DSN)
        with pytest.raises(RetrievalError, match="not open"):
            await p.fetch("SELECT 1")

    async def test_bad_query_raises(self, pool: Pool) -> None:
        with pytest.raises(RetrievalError):
            await pool.fetch("SELECT BAD")
