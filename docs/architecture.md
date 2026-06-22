# Architecture

## 2026-06-22 — Phase 0.2: Dependency Audit, Drift Remediation & Maintainability

### Component Diagram — Changed Components

```mermaid
graph TB
    subgraph internal["Internal Packages"]
        core["core"]
        retrieval["retrieval"]
        ingest["ingest"]
        qa["qa"]
        api["api"]
        worker["worker"]
    end

    subgraph external["External Dependencies"]
        asyncpg["asyncpg"]
        redis["redis"]
        httpx["httpx"]
        pymupdf["PyMuPDF"]
        sentence_t["sentence-transformers"]
        python_mp["python-multipart"]
        fastapi["FastAPI / uvicorn"]
        pydantic_s["pydantic-settings"]
    end

    core --> retrieval
    core --> ingest
    core --> qa
    retrieval --> ingest
    core --> api
    retrieval --> api
    qa --> api
    ingest --> worker
    retrieval --> worker

    retrieval --> asyncpg
    retrieval -.-> sentence_t
    ingest --> asyncpg
    ingest --> httpx
    ingest --> pymupdf
    qa --> httpx
    api --> asyncpg
    api --> redis
    api --> fastapi
    api --> pydantic_s
    api --> python_mp

    style sentence_t stroke-dasharray: 5 5
    style pymupdf fill:#d4edda,stroke:#28a745
    style httpx fill:#d4edda,stroke:#28a745
    style python_mp fill:#d4edda,stroke:#28a745
    style redis fill:#fff3cd,stroke:#ffc107

    linkStyle 0,1,2,3,4,5,6,7,8 stroke:#6c757d,stroke-width:1px
    linkStyle 9,10,11,12,13,14,15,16 stroke:#6c757d,stroke-width:1px,stroke-dasharray: 3 3
```

**Legend:**
- 🟢 `PyMuPDF`, `httpx`, `python-multipart` — added to explicit `[dependencies]` in this phase
- 🟡 `redis` — pin tightened from `>=5` to `>=5,<9`
- ⚪ `sentence-transformers` — moved to optional `[embedding]` extra
- Removed from infra: MinIO service & `S3_ENDPOINT` env vars (deferred to v0.2)

### Sequence Diagram — QA Verification Loop (Parallelized)

```mermaid
sequenceDiagram
    participant Client
    participant Route as "POST /qa route"
    participant Engine as "QAEngine.answer()"
    participant Embedder as "Embedder"
    participant Retriever as "Retriever"
    participant LLM as "LLM (generate)"
    participant DB as "Postgres"

    Client->>Route: POST /qa {question, paper_ids}
    Route->>Engine: answer(question)
    Engine->>Embedder: embed([question])
    Embedder-->>Engine: [vector]
    Engine->>Retriever: search_dense(vector, k)
    Retriever-->>Engine: [ScoredChunk]
    Engine->>LLM: generate(prompt, system)
    LLM-->>Engine: answer_text + ##Citations section
    Engine->>Engine: _parse_citations() → [Claim]

    par Parallel Verification (asyncio.gather)
        Engine->>Engine: _verify_one(claim[0])
        Engine->>Engine: _verify_one(claim[1])
        Engine->>Engine: _verify_one(claim[N])
    end

    Note over Engine: Layer 1 — literal anchor<br/>(quoted_span in chunk_text?)
    Note over Engine: Layer 2 — entailment judge<br/>(LLM NLI score >= threshold)

    Engine->>Engine: _apply_policy() → drop/retry/degrade/abstain
    Engine-->>Route: QAResult

    opt enrich with chunk meta
        Route->>DB: fetch chunk_id → page, offsets
        DB-->>Route: {chunk_id → meta}
    end

    Route-->>Client: 200 QAResponse
```

**What changed in this phase:** The verification loop (highlighted `par` block) was parallelized from sequential `for claim in claims: await verify(claim)` to `asyncio.gather(...)`. This reduces answer latency proportionally to claim count — the dominant cost (LLM NLI calls) now runs concurrently instead of serially.
