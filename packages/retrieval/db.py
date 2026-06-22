"""Async connection pool and query helpers for the retrieval store.

Usage:
    pool = Pool()
    await pool.open()
    rows = await pool.fetch("SELECT $1::int + $2::int", 1, 2)
    await pool.close()

Or as an async context manager:
    async with Pool() as pool:
        row = await pool.fetchrow("SELECT 1 AS v")
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

import asyncpg

from core.errors import RetrievalError


class Pool:
    """Asyncpg connection pool configured from DATABASE_URL."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or os.environ["DATABASE_URL"]
        self._pool: asyncpg.Pool | None = None

    async def open(self) -> None:
        """Create the underlying asyncpg pool."""
        try:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=2,
                max_size=10,
            )
        except Exception as exc:
            raise RetrievalError(
                f"Failed to create database pool: {exc}"
            ) from exc

    async def close(self) -> None:
        """Close the underlying asyncpg pool and release all connections."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def fetch(self, query: str, *params: Any) -> list[asyncpg.Record]:
        """Execute a parametrized query and return all result rows.

        Raises RetrievalError on database failures.
        """
        result = await self._run(lambda c: c.fetch(query, *params))
        return cast("list[asyncpg.Record]", result)

    async def fetchrow(self, query: str, *params: Any) -> asyncpg.Record | None:
        """Execute a parametrized query and return the first row (or None).

        Raises RetrievalError on database failures.
        """
        result = await self._run(lambda c: c.fetchrow(query, *params))
        return cast("asyncpg.Record | None", result)

    async def execute(self, query: str, *params: Any) -> str:
        """Execute a parametrized query and return the command status tag.

        Raises RetrievalError on database failures.
        """
        result = await self._run(lambda c: c.execute(query, *params))
        return cast("str", result)

    async def _run(self, op: Any) -> Any:
        """Acquire a connection from the pool and run *op* on it."""
        pool = self._pool
        if pool is None:
            raise RetrievalError("Pool is not open; call open() first")
        try:
            async with pool.acquire() as conn:
                return await op(conn)
        except asyncpg.PostgresError as exc:
            raise RetrievalError(str(exc)) from exc

    @property
    def size(self) -> int | None:
        """Current number of idle + active connections, or None if closed."""
        return self._pool.size if self._pool is not None else None

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Acquire a raw connection for transactional operations.

        Yields an ``asyncpg.Connection`` from the pool.  The connection is
        returned to the pool when the context exits.
        """
        if self._pool is None:
            raise RetrievalError("Pool is not open")
        async with self._pool.acquire() as conn:
            yield conn

    async def __aenter__(self) -> Pool:
        await self.open()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __aiter__(self) -> AsyncGenerator[asyncpg.Record, None]:
        raise TypeError("Pool is not directly iterable; use .fetch()")
