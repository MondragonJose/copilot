"""Tests for db.Pool — minimal mocks for the asyncpg wrapper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.errors import RetrievalError
from retrieval.db import Pool


class TestPool:
    def test_init_default_dsn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
        p = Pool()
        assert p._dsn == "postgresql://u:p@localhost/db"

    def test_init_custom_dsn(self) -> None:
        p = Pool(dsn="postgresql://custom/db")
        assert p._dsn == "postgresql://custom/db"

    def test_size_when_closed(self) -> None:
        p = Pool(dsn="postgresql://x/x")
        assert p.size is None

    @pytest.mark.asyncio
    async def test_open_failure_raises_retrieval_error(self) -> None:
        with patch("retrieval.db.asyncpg.create_pool",
                   side_effect=Exception("connection refused")):
            p = Pool(dsn="postgresql://x/x")
            with pytest.raises(RetrievalError, match="Failed to create"):
                await p.open()

    @pytest.mark.asyncio
    async def test_execute_before_open_raises(self) -> None:
        p = Pool(dsn="postgresql://x/x")
        with pytest.raises(RetrievalError, match="Pool is not open"):
            await p.execute("SELECT 1")

    @pytest.mark.asyncio
    async def test_connection_before_open_raises(self) -> None:
        p = Pool(dsn="postgresql://x/x")
        with pytest.raises(RetrievalError, match="Pool is not open"):
            async with p.connection():
                pass

    @pytest.mark.asyncio
    async def test_open_close(self) -> None:
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock()
        mock_pool.size = 5

        async def fake_create_pool(*a, **kw):
            return mock_pool

        with patch("retrieval.db.asyncpg.create_pool",
                   new=fake_create_pool):
            p = Pool(dsn="postgresql://x/x")
            await p.open()
            assert p.size == 5
            await p.close()
            assert p.size is None

    @pytest.mark.asyncio
    async def test_aenter_aexit(self) -> None:
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock()

        async def fake_create_pool(*a, **kw):
            return mock_pool

        with patch("retrieval.db.asyncpg.create_pool",
                   new=fake_create_pool):
            async with Pool(dsn="postgresql://x/x") as p:
                assert p._pool is mock_pool
            mock_pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_aiter_raises(self) -> None:
        p = Pool(dsn="postgresql://x/x")
        with pytest.raises(TypeError, match="not directly iterable"):
            async for _ in p:
                pass
