# Research Copilot

MVP v0.1 — Academic research assistant with grounded citation verification.

Extracts claims from PDF literature, retrieves evidence, verifies every citation against its source span, and surfaces grounded answers with deep-linked citations.

---

## Quick Start

### Prerequisites

- Docker & Docker Compose v2
- Git
- Make (optional, for `make` targets)

### One-command self-host

```bash
git clone <repo-url> research-copilot
cd research-copilot
cp .env.example .env          # edit LLM_API_KEY if you have one
docker compose up --build -d
```

Wait for all services to report healthy (≈90 s):

```bash
make docker-smoke
# or manually:
./tests/smoke_docker.sh
```

Open **http://localhost:5173** in your browser.

Services:

| Service  | URL                          |
|----------|------------------------------|
| Frontend | http://localhost:5173        |
| API      | http://localhost:8000        |
| GROBID   | http://localhost:8070        |

### Shutdown

```bash
make docker-down
# or: docker compose down
```

---

## Usage Walkthrough

### 1. Import a paper

**Upload a PDF** — go to the **Corpus** tab, select a `.pdf` file, click Upload, and wait for the job to complete.

**Import by DOI** — paste a DOI (e.g. `10.48550/arxiv.1706.03762`) and click Import.

### 2. Read & annotate

When a job completes, click **View** to open the PDF reader. Select text to create annotations and ask the AI to explain passages.

### 3. Ask questions

Switch to the **Ask** tab, optionally scope your question to specific papers, type your question, and click Ask.

- **Answerable questions** show the generated answer with numbered citation cards. Each citation links to the exact page + span in the Reader.
- **Unanswerable questions** display an explicit abstain notice — never hidden.

### 4. Evaluation & Release Gate

Run the evaluation suite (requires a running database with goldset data):

```bash
make eval
```

The gate PASSes only when `citation_faithfulness >= 0.95` AND `correct_abstention >= 0.90`. On PASS, create the release tag:

```bash
make release
```

This runs `make eval` first. If the gate FAILs, the tag is **not** created.

---

## Architecture

```
core ← ingest, retrieval, qa, eval ← api
```

- `core` — Frozen dataclasses, Protocols, errors. Zero I/O, imports nothing internal.
- `ingest` — PDF parsing (GROBID → PyMuPDF fallback), structure-aware chunking, Redis task queue.
- `retrieval` — pgvector hybrid/dense search, BGE-M3 embeddings.
- `qa` — Retrieve-then-generate with literal + two-layer LLM verification.
- `eval` — Goldset loader, 6 metrics, LitQA2 runner, PASS/FAIL gate.
- `api` — FastAPI with health, ingest, jobs, QA routes.
- `frontend` — Vite + React 18 + TanStack Query 5. Corpus import, PDF reader, Ask screen.

See `RESEARCH_COPILOT_PHASE_0.1.md` for the full blueprint.

---

## Development

### Without Docker

```bash
# Install Python packages
pip install -e ".[dev,test]"

# Install frontend
cd frontend && npm install

# Run tests
make test          # backend
cd frontend && npm test   # frontend

# Lint
make lint
```

### With Docker (hot-reload)

```bash
# Start stack
docker compose up --build -d

# Watch logs
docker compose logs -f

# Rebuild a single service (after dependency changes)
docker compose build api
docker compose up -d api
```

### Frontend-only

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
npm test           # 53 unit tests
npm run build      # production bundle
```

### Environment variables

Copy `.env.example` to `.env` and edit:

| Variable         | Required | Default                     | Description                      |
|------------------|----------|-----------------------------|----------------------------------|
| `LLM_API_KEY`    | No*      | —                           | OpenAI / Ollama API key          |
| `DATABASE_URL`   | No       | `postgresql://rc:rc@...`    | PostgreSQL connection string     |
| `REDIS_URL`      | No       | `redis://redis:6379/0`      | Redis connection string          |
| `GROBID_URL`     | No       | `http://grobid:8070`        | GROBID service URL               |

*Required only for the QA loop — local dev works without it (no LLM = abstention).

### End-to-end smoke test

The smoke test verifies all Docker services are healthy:

```bash
make docker-smoke
```

#### Full E2E (manual)

```bash
# 1. Start the stack
docker compose up --build -d

# 2. Wait for all services
./tests/smoke_docker.sh

# 3. Verify API health
curl http://localhost:8000/health

# 4. Import a test PDF
curl -X POST http://localhost:8000/ingest \
  -F "file=@test.pdf"

# 5. Check job status (replace <job-id>)
curl http://localhost:8000/jobs/<job-id>

# 6. Ask a question
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What mechanism does the transformer use?"}'

# 7. Open the frontend at http://localhost:5173
```

---

## Release process

1. Ensure all tests pass: `make test && cd frontend && npm test`
2. Ensure lint is clean: `make lint`
3. Run the eval gate: `make eval`
4. If the gate reports **PASS**, create and push the release tag: `make release && git push origin --tags`

> Tagging is **forbidden** when `make eval` reports FAIL.
