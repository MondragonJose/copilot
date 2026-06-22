# Research Copilot — Agent Playbook

Architecture is **APPROVED and FROZEN**. Do NOT redesign, add features, or expand scope.
Blueprint: `RESEARCH_COPILOT_PHASE_0.1.md`.

---

## Dependency Rule (NON-NEGOTIABLE)

All dependencies point toward `core`. `core` has zero I/O and imports nothing internal.

```
core ← retrieval, ingest, qa, eval ← api
```

- `retrieval` — imports `core` only
- `ingest` — imports `core`, `retrieval` only
- `qa` — imports `core`, `retrieval` only
- `eval` — imports `core`, `qa`, `retrieval` only
- `api` — imports everything

Enforced by `lint-imports` (configured in `pyproject.toml`).

---

## Packages (actual state)

| Package | Path | What's implemented |
|---------|------|--------------------|
| `core` | `packages/core/` | `models.py`, `interfaces.py` (Protocols), `errors.py`. Frozen dataclasses for all value objects. |
| `retrieval` | `packages/retrieval/` | `pgvector_store.py` (raw SQL, no ORM), `hybrid.py`, `embedder.py` (BGE-M3, lazy-loaded), `qdrant_store.py` (stub). |
| `ingest` | `packages/ingest/` | `router.py` (GROBID → PyMuPDF fallback), `parsers/grobid.py`, `parsers/pymupdf.py`, `chunking.py`, `tasks.py`, `persist.py`. |
| `qa` | `packages/qa/` | `engine.py` (retrieve-then-generate), `llm.py` (OpenAI/Ollama), `verifier.py` (LiteralVerifier + TwoLayerVerifier). |
| `eval` | `packages/eval/` | **Empty scaffold** (only `__init__.py`). |
| `api` | `packages/api/` | FastAPI app, routes: `health`, `ingest`, `jobs`, `qa`. DI in `deps.py`. |

---

## Commands

```bash
pip install -e ".[dev,test]"          # install all packages + dev + test deps
make install-dev                       # same but `.[dev]` only
make lint                              # ruff check . → mypy packages/core → lint-imports
make test                              # pytest (unit tests only, no infra needed)
make test-integration                  # integration tests (needs compose stack up)
make precommit-run                     # pre-commit run --all-files
ruff check .                           # line-length=100, double quotes
ruff format .                          # auto-format
mypy packages/core                     # strict mode only enforced on core in CI
lint-imports                           # verify inter-package dependency contracts

# Integration test setup
docker compose -f docker-compose.test.yml up -d --wait
DATABASE_URL=postgresql://rc:rc@localhost:5432/research_copilot \
  pytest tests/test_integration.py -v --timeout=120
```

---

## CI (`.github/workflows/ci.yml`)

- **lint job**: `ruff check .` → `mypy packages/core` → `lint-imports`
- **test job**: `pytest -v` with postgres (pgvector/pg16) + redis (7-alpine) service containers
- mypy overrides for `asyncpg`, `redis`, `pytest_asyncio` (`ignore_missing_imports`)

---

## Testing quirks

- **DB-dependent tests skip gracefully** if no database is reachable (`conftest.py` calls `pytest.skip()`)
- Integration tests need a running pgvector + Redis (provided by CI service containers or `docker compose -f docker-compose.test.yml up -d --wait`)
- PDF fixtures are ephemeral documents created in-memory via PyMuPDF (`fitz`) in `conftest.py`
- Use `TestClient` from FastAPI for API route tests (mock async deps with `AsyncMock`)

---

## Key conventions

- **Protocol mismatch**: `core/interfaces.py` uses sync method signatures but all implementations (`PgVectorStore`, `LLMProvider`, etc.) use `async def`. Works at runtime; mypy on non-core packages may flag it.
- **Pydantic only at API boundaries** — request/response validation. Domain objects are frozen dataclasses.
- **Config from env vars** via `pydantic-settings.BaseSettings` / `os.environ`. LLM provider: `LLM_PROVIDER` (openai|ollama), `LLM_API_KEY`, etc.
- **Migrations**: sequential SQL files in `migrations/` mounted into `docker-entrypoint-initdb.d`. Idempotent (`IF NOT EXISTS`). No Alembic yet.
- **No `print()`** — use `logging.getLogger()` (standard library, no loguru/structlog dependency added yet).

---

## Key files

| File | Purpose |
|------|---------|
| `packages/core/interfaces.py` | Protocols: Retriever, Parser, Embedder, LLMClient, Verifier |
| `packages/core/models.py` | Paper, Chunk, ChunkRef, ScoredChunk, UpsertChunk, Claim, ClaimVerdict, QAResult |
| `packages/core/errors.py` | RCError, IngestError, ParseError, RetrievalError, EmbeddingError, VerificationError |
| `pyproject.toml` | ruff config, mypy config, import-linter contracts, pytest config |
| `docker-compose.yml` | Local stack: pgvector, Redis, GROBID, MinIO, rc-api, rc-worker |
| `docker-compose.test.yml` | Test infra: pgvector, Redis, GROBID, MinIO (no app containers) |
| `tests/test_integration.py` | 5-scenario integration tests: GROBID fallback, partial embedding, forced dead, re-ingest idempotency, unanswerable abstain |
| `RESEARCH_COPILOT_PHASE_0.1.md` | Full blueprint (DDL, contracts, pipeline, eval targets) |

---

## Session Summary — Three-Task Sprint + Two Bug-Fix Rounds

### High-level

Three feature tasks + two adversarial sanity-check rounds. Architecture stayed **frozen** per playbook.

### Task 1 — PDF Reader with Annotations

| What | Files |
|------|-------|
| `PdfViewer` with `react-pdf`, selection-to-offset mapping | `frontend/src/components/PdfViewer.tsx` |
| `SelectionPopover` — tooltip on text selection | `frontend/src/components/SelectionPopover.tsx` |
| `Reader` screen — dual-panel (viewer + sidebar) | `frontend/src/screens/Reader.tsx` |
| `AnnotationList` sidebar | `frontend/src/screens/AnnotationList.tsx` |
| `useAnnotations` hook (CRUD via API) | `frontend/src/hooks/useAnnotations.ts` |
| PDF URL resolution (api.ts serves file/signed URL) | `frontend/src/api.ts` |
| Page-to-offset resolution | `frontend/src/pdf-utils.ts` |

### Task 2 — Ask Screen with Grounded Answers

| What | Files |
|------|-------|
| `Ask.tsx` — question input + answer display + citations + abstain | `frontend/src/screens/Ask.tsx` |
| `QuestionForm` — text area + scope selector | `frontend/src/components/QuestionForm.tsx` |
| `AnswerCard` — verdict display | `frontend/src/components/AnswerCard.tsx` |
| `CitationLink` — chunk-level cite with page | `frontend/src/components/CitationLink.tsx` |
| `AbstainNotice` — shown when `answerable=false` | `frontend/src/components/AbstainNotice.tsx` |
| Frontend types (QARequest, QAResponse, ClaimOut, etc.) | `frontend/src/types.ts` |

### Task 3 — Self-Host + Release

| What | Files |
|------|-------|
| Multi-stage Dockerfile (Python 3.12 + Node → slim) | `Dockerfile` |
| docker-compose with rc-api + rc-worker services | `docker-compose.yml` |
| Makefile targets: `docker-up`, `docker-down`, `docker-smoke`, `release` | `Makefile` |
| CHANGELOG | `CHANGELOG.md` |
| README with self-hosting instructions | `README.md` |
| E2E smoke test (health → ingest → poll → ask) | `tests/smoke_docker.sh` |
| Release gates (`release` target runs `eval` first) | `Makefile:52-55` |

### Bug-Fix Round 1 — Sanity-Check

1. **SelectionPopover memory leak** — `AbortController` cleanup in `useEffect` return
2. **Worker missing from docker-compose** — added `rc-worker` service
3. **Citation `paper_id` missing** — `_fetch_chunk_meta` now returns `paper_id`; `ClaimOut.paper_id` exposed
4. **E2E smoke test wrong port** — `smoke_docker.sh` uses correct container mapping
5. **`any` types in frontend** — replaced with proper types in PdfViewer, SelectionPopover, api.ts, types.ts
6. **`BASE_URL` stale under HMR** — examined, no fix needed (Vite env pattern is correct for SPA)
7. **Error parsing in `ingestFile`** — `api.ts:62` uses `.json()` fallback to `statusText`

### Architecture Enforcement Fixes

- **Worker import-linter contract** — `worker/run.py` imports only `ingest.*` + `retrieval.*` (no `api`, `qa`, `eval`)
- **`ParserRouter` → `Parser` Protocol** — `router.py` implements `Parser` Protocol; route handler calls `.parse()` instead of `.route()`

### Bug-Fix Round 2 — Adversarial Sanity-Check

1. **`paper_id` stale on DB conflict** — `persist.py` uses `ON CONFLICT (doi) DO UPDATE SET updated_at = NOW() RETURNING id`
2. **Worker race on chunk upsert** — `tasks.py` batches `existing = set()` queries into groups of 100
3. **Stuck jobs on partial failure** — `IngestTaskManager.fail()` is idempotent (checks status before UPDATE)
4. **Partial PDF visibility** — `pymupdf.py:82` excludes space-before-float glyphs via `char.startswith(" ")` + bbox width < font size
5. **Empty question crashes QA** — `QAEngine.answer()` returns `QAResult(answerable=False, ...)` immediately when input is blank
6. **`k <= 0` crashes retrieval** — `PgVectorStore._search_dense/_search_lexical` clamps `k = max(k, 1)`
7. **Stale `stage` on re-run** — `router.py:90` resets `stage="parsing"` before re-processing
