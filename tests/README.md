# Test Strategy

## Structure

```
tests/
├── README.md               # This file
├── conftest.py              # Shared fixtures (db, native_pdf, scanned_pdf)
├── __init__.py
│
├── test_*.py                # Flat root: integration, schema, pipeline, contract
│
├── core/                    # Unit tests for packages/core/
├── eval/                    # Unit tests for packages/eval/
├── ingest/                  # Unit tests for packages/ingest/
├── qa/                      # Unit tests for packages/qa/
├── retrieval/               # Unit tests for packages/retrieval/
│
└── regression/              # Frozen golden-value guards (bug-specific)
```

## File Classification

### Flat root (`tests/test_*.py`)

| File | Nature | Requires Infra |
|------|--------|---------------|
| `test_ingest.py` | Pipeline unit tests — exercises `process_job()` end-to-end via mocks | No |
| `test_retrieval.py` | **Integration** — real asyncpg Pool + pgvector | Postgres |
| `test_integration.py` | **Integration** — real Postgres + BGE-M3 embedder + GROBID | Postgres, Redis, GROBID |
| `test_schema.py` | **Schema** — queries `information_schema` | Postgres |
| `test_db.py` | **Integration** — real asyncpg Pool lifecycle | Postgres |
| `test_health.py` | Unit — mocked `/health` endpoint | No |
| `test_imports.py` | Unit — package importability | No |
| `test_contracts.py` | Unit — Protocol shapes + `lint-imports` | No (lint-imports binary) |

### Nested directories (`tests/<package>/`)

Every file in these directories is a **unit test** that mocks all I/O.

| Directory | Tests for |
|-----------|-----------|
| `core/` | `core/errors.py`, `core/interfaces.py`, `core/models.py` |
| `ingest/` | `ingest/chunking.py`, `ingest/doi.py`, `ingest/parsers/grobid.py`, `ingest/persist.py`, `ingest/parsers/pymupdf.py`, `ingest/router.py`, `ingest/tasks.py` |
| `qa/` | `qa/engine.py`, `qa/llm.py`, `qa/verifier.py` |
| `retrieval/` | `retrieval/db.py`, `retrieval/embedder.py`, `retrieval/hybrid.py`, `retrieval/pgvector_store.py` |
| `eval/` | `eval/goldset.py`, `eval/litqa2.py`, `eval/metrics.py`, `eval/run.py` |

### Regression (`tests/regression/`)

Frozen golden-value tests that guard against specific bugs. Each file duplicates a
handful of unit tests from the corresponding package directory, annotated with
bug IDs (e.g., `R2-BUG-001`, `CR-BUG-003`). These exist to make regressions
visible even if the original unit test is refactored.

## Flat-vs-Nested Rationale

Flat root and nested directories are **complementary, not redundant**:

- **Flat `test_retrieval.py`** = integration tests with a real Postgres connection.
  **Nested `tests/retrieval/`** = unit tests with mocked `asyncpg`. Both test the
  same `PgVectorStore` class at different levels of the test pyramid.

- **Flat `test_ingest.py`** = pipeline-style unit tests that wire multiple
  components (parser, chunker, persister) together through `process_job()`.
  **Nested `tests/ingest/`** = focused unit tests of each component in isolation.

- **Flat `test_db.py`** = integration tests for the real `asyncpg` Pool wrapper.
  **Nested `tests/retrieval/test_db.py`** = unit tests that mock `asyncpg.create_pool`.

- **Unique flat files** (`test_schema.py`, `test_health.py`, `test_imports.py`,
  `test_contracts.py`) have no nested equivalent because they test
  cross-cutting concerns (DB schema, endpoint health, importability, contracts).

## Running Tests

```bash
# All tests (unit + integration via service containers)
pytest -v --cov=packages --cov-fail-under=91

# Unit tests only (no infra required)
pytest tests/core/ tests/eval/ tests/ingest/ tests/qa/ tests/retrieval/ \
       tests/regression/ tests/test_ingest.py tests/test_health.py \
       tests/test_imports.py tests/test_contracts.py -v

# Integration tests only (needs Postgres)
pytest tests/test_retrieval.py tests/test_db.py tests/test_schema.py \
       tests/test_integration.py -v
```

Integration tests skip gracefully when the database is unreachable
(see `tests/conftest.py` `db` fixture).
