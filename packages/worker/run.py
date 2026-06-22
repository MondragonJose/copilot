"""Minimal ingest worker — polls the database for pending jobs."""

from __future__ import annotations

import asyncio
import logging
import os

from ingest.router import ParserRouter
from ingest.tasks import process_job
from retrieval.db import Pool
from retrieval.embedder import BgeM3Embedder
from retrieval.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)

POLL_INTERVAL = 5.0


async def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")

    pool = Pool(dsn=db_url)
    await pool.open()

    embedder = BgeM3Embedder()
    retriever = PgVectorStore(pool)
    router = ParserRouter()

    logger.info("Worker started, polling every %.1fs", POLL_INTERVAL)

    while True:
        try:
            row = await pool.fetchrow(
                "UPDATE jobs SET status = 'running', updated_at = now() "
                "WHERE id = ("
                "  SELECT id FROM jobs "
                "  WHERE status = 'queued' "
                "     OR (status = 'running' AND updated_at < now() - interval '5 minutes')"
                "     OR (status = 'failed' AND attempts < max_attempts "
                "         AND (next_attempt_at IS NULL OR next_attempt_at <= now()))"
                "  LIMIT 1 FOR UPDATE SKIP LOCKED"
                ") RETURNING id::text, attempts, max_attempts",
            )
            if row is not None:
                job_id = row["id"]
                logger.info("Processing job %s", job_id)
                await process_job(pool, retriever, embedder, router, job_id)
        except Exception:
            logger.exception("Worker loop error")

        await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(main())
