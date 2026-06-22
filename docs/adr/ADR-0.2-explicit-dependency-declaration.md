# ADR-0.2: Explicit Dependency Declaration for Runtime Packages

**Status:** Accepted

---

## Context

The monorepo uses `pyproject.toml` as the single source of truth for Python
dependencies.  A dependency audit revealed several packages that were imported
at runtime but either undeclared or misclassified:

| Package | Declared as | Actually used by |
|---------|-------------|------------------|
| `httpx` | `[test]` only | `ingest/parsers/grobid.py`, `qa/llm.py` |
| `PyMuPDF` | not declared | `ingest/parsers/pymupdf.py` |
| `python-multipart` | not declared | `api/routes/ingest.py` (FastAPI `request.form()`) |
| `sentence-transformers` | not declared | `retrieval/embedder.py` (lazy) |

These gaps mean a production install using `pip install -e .` (without `[test]`)
would crash on startup or on first file upload.  The `redis` pin was
unbounded (`>=5`), risking breakage across major versions (5→8).

A secondary finding: Dockerfiles installed `".[dev]"` in production images,
pulling pytest/mypy/ruff into deployed containers.

---

## Decision

1. **Declare every runtime import** in `pyproject.toml` `[dependencies]`:
   - `httpx>=0.27`
   - `PyMuPDF>=1.27`
   - `python-multipart>=0.0.18`

2. **Pin `redis>=5,<9`** to prevent accidental major-version upgrades.

3. **Classify heavy/lazy deps as optional extras**:
   - `sentence-transformers>=3` → `[embedding]`

4. **Production Dockerfiles install `.[dev]` → `"."`** to exclude dev tooling.

5. **Remove stale infra** that has no consuming code:
   - `minio` service, `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` from
     `docker-compose.yml` and `docker-compose.test.yml`.

---

## Consequences

- `pip install -e .` now pulls all packages needed for runtime.
- `pip install -e ".[embedding]"` adds the ~800 MB PyTorch dependency only
  when embeddings are needed (lazy-loaded, so non-blocking at startup).
- Production images are ~100 MB smaller (no pytest/mypy/ruff).
- Redis upgrades to v8 are safe; v9 would require explicit pin update.
- MinIO can be re-added when S3 storage is implemented (deferred per S8).

---

## Reversibility

**Not easily reversible** once downstream CI/packaging depends on the new
classifications.  Moving a package back to implicit or `[test]` would break
production deployments silently.

---

## Alternatives Considered

- **Keep httpx in `[test]` and add `httpx` to API runtime deps only** —
  rejected because `grobid.py` and `llm.py` are runtime modules, not test
  fixtures.
- **Wrap httpx/PyMuPDF imports in lazy try/except guards** — rejected as
  fragile; a missing dep should fail fast at import time, not silently at
  runtime.
- **Keep MinIO as dead config** — rejected; the blueprint §8 defers S3 to
  v0.2, and dead config adds complexity (port conflict, resource usage,
  misleading maintainer expectations).

---

## Blueprint References

- **S2** (pgvector) — `asyncpg` was already correctly declared.
- **S3** (Redis) — pin tightened from `>=5` to `>=5,<9`.
- **S5** (LLM) — `httpx` now in `[dependencies]` for OpenAI/Ollama HTTP calls.
- **S6** (BGE-M3 embeddings) — `sentence-transformers` as optional `[embedding]`.
- **S7** (PDF parsing) — `PyMuPDF` and `httpx` (GROBID) explicitly declared.
- **S8** (S3 deferred) — MinIO removed until v0.2.
