# Changelog

## [Unreleased]

### Highlights
- **PDF Ingest Pipeline** — Upload PDFs (file or DOI), GROBID/pyMuPDF parsing, structure-aware chunking, pgvector embeddings
- **QA Engine** — Retrieve-then-generate with literal + two-layer citation verification (LLM-as-judge)
- **Eval Framework** — 6 metrics (recall@k, MRR, citation faithfulness, answer accuracy, correct abstention, LitQA2 accuracy/precision@answered), goldset-driven gating
- **React Frontend** — Corpus import screen, PDF reader with text selection → annotation flow, Ask screen with grounded answer + citation deep-links
- **One-command self-host** — `docker compose up` starts db, redis, GROBID, API, frontend, and worker

### Features
- `packages/core` — Frozen dataclasses for Paper, Chunk, ChunkRef, ScoredChunk, UpsertChunk, Claim, ClaimVerdict, QAResult; Protocols for Retriever, Parser, Embedder, LLMClient, Verifier
- `packages/retrieval` — PgVectorStore with raw SQL (hybrid + dense search, paper_id filter), BGE-M3 embedder (lazy-loaded), Qdrant stub
- `packages/ingest` — GROBID → PyMuPDF fallback router, section-aware + page-aware chunking, Redis-backed task queue
- `packages/qa` — QAEngine (retrieve-then-generate) with two-layer inline citation verification, LLMProvider (OpenAI / Ollama)
- `packages/eval` — GoldsetLoader, 6 metric functions, LitQA2 runner, EvalReport with PASS/FAIL gate (faithfulness ≥ 0.95 AND abstention ≥ 0.90)
- `packages/api` — FastAPI app with health, ingest, jobs, and QA routes
- `frontend/` — Vite + React 18 + TanStack Query 5; Corpus import (file upload + DOI), PDF viewer (react-pdf → text selection → char offset mapping), Reader with annotation sidebar, Ask screen with citation deep-links to exact spans; 53 unit tests
- `docker/` — Multi-stage Dockerfile.frontend (dev hot-reload + nginx production)
- `docker-compose.yml` — Full-stack orchestration (db, redis, grobid, api, frontend, worker)
- `Makefile` — `make release` gated on `make eval` (PASS required before tagging)

### Infrastructure
- PostgreSQL 16 + pgvector, Redis 7, GROBID 0.8.1
- SQL migrations (idempotent via `IF NOT EXISTS`, loaded by docker-entrypoint-initdb.d)
- CI: ruff check → mypy packages → lint-imports → pytest + coverage (`--cov-fail-under=91`)
- `make eval` — gate must PASS before `make release` creates the release tag

### Fixed
- P0.2.1 — QA engine now parallelizes claim verification (asyncio.gather) instead of sequential LLM calls; reduces answer latency proportionally to claim count
- P0.2.2 — `POST /ingest` file uploads no longer crash with HTTP 500; `python-multipart` is now declared as a runtime dependency
- P0.2.3 — Application no longer crashes on startup when test dependencies are not installed; `httpx` moved from `[test]` to `[dependencies]` (used by GROBID parser and LLM provider at runtime)
- P0.2.4 — Health endpoint reuses the shared connection pool instead of opening a new `asyncpg` connection per health check; prevents connection leak under frequent polling

### Changed
- P0.2.5 — Production Docker images (`Dockerfile.api`, `Dockerfile.worker`) no longer install dev/testing tooling (pytest, mypy, ruff, pre-commit); images are ~100 MB smaller
- P0.2.6 — Redis dependency pinned to `>=5,<9` (was `>=5`) to prevent accidental breakage on major-version upgrades

### Deprecated
- P0.2.7 — MinIO service and `S3_ENDPOINT`/`S3_ACCESS_KEY`/`S3_SECRET_KEY` env vars removed from docker-compose; S3-backed PDF storage deferred to v0.2 per blueprint S8
