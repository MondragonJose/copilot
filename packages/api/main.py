from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.deps import pool
from api.routes.health import router as health_router
from api.routes.ingest import router as ingest_router
from api.routes.jobs import router as jobs_router
from api.routes.qa import router as qa_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await pool.open()
    yield
    await pool.close()


app = FastAPI(title="Research Copilot", version="0.1.0", lifespan=lifespan)
app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(jobs_router)
app.include_router(qa_router)
