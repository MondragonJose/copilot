"""Shared fixtures for Research Copilot tests.

Optional dependencies (fitz, asyncpg) are imported lazily so unit
tests that mock all I/O do not require them.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio


@pytest_asyncio.fixture(scope="module")
async def db() -> AsyncGenerator:  # type: ignore[misc]
    import asyncpg

    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://rc:rc@localhost:5432/research_copilot",
    )
    try:
        conn = await asyncpg.connect(url, timeout=5)
    except Exception:
        pytest.skip(f"Database not available (DATABASE_URL={url})")
        return
    yield conn
    await conn.close()


def _make_native_pdf(path: pathlib.Path, text_pages: list[str] | None = None) -> None:
    import fitz

    if text_pages is None:
        text_pages = ["Page one content.", "Page two content."]
    doc = fitz.open()
    for text in text_pages:
        page = doc.new_page()
        page.insert_text((50, 50), text, fontname="helv", fontsize=12)
    doc.save(str(path))
    doc.close()


def _make_scanned_pdf(path: pathlib.Path) -> None:
    import fitz

    doc = fitz.open()
    doc.new_page()
    doc.save(str(path))
    doc.close()


@pytest.fixture
def native_pdf(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "test_article.pdf"
    _make_native_pdf(path)
    return path


@pytest.fixture
def scanned_pdf(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "scanned.pdf"
    _make_scanned_pdf(path)
    return path
