# Research Copilot — Blueprint Técnico v0.1 (MVP)

> **Propósito**: Documento de contexto para la implementación del MVP v0.1 de Research Copilot.
> No incluye código de aplicación; solo artefactos de blueprint: DDL, interfaces, contratos, pipeline, docker-compose e issues.
> La fuente de verdad es la *Especificación Consolidada v2*.

---

## 0. Supuestos y Decisiones por Defecto

| # | Decisión | Default | Reversibilidad |
|---|----------|---------|----------------|
| S1 | Lenguaje backend | Python 3.11 + FastAPI | ❌ No reversible |
| S2 | Vector store | Postgres 16 + pgvector 0.7+ | ✅ Reversible vía interfaz |
| S3 | Cola async | Redis + Arq | ✅ Reversible |
| S4 | Motor QA | Custom retrieve-then-generate (Retriever + LLMClient) | ✅ Reversible (tras interfaz) |
| S5 | LLM dev | API opt-in (Claude/GPT); Ollama local opcional | ✅ Reversible |
| S6 | Embeddings | BGE-M3 (sentence-transformers, dim 1024) | ❌ No reversible sin reindexar |
| S7 | Parsing PDF | GROBID + PyMuPDF fallback | ✅ Reversible (router) |
| S8 | Grafo conocimiento | Fuera de v0.1 → v0.2 | ✅ Reversible |
| S9 | Auth | Single-user, API key en .env | ✅ Reversible |
| S10 | Frontend | React + Vite + TS mínimo (3 pantallas) | ✅ Reversible |

> **Corpus v0.1**: PDFs importados por el usuario (carpeta local/subida) + metadatos vía Crossref/OpenAlex por DOI. Sin búsqueda en vivo PubMed/Scopus.

---

## 1. Estructura del Monorepo

```
research-copilot/
├── pyproject.toml              # workspace raíz (uv/pip), ruff, pytest, mypy
├── docker-compose.yml          # stack local
├── .env.example
├── packages/
│   ├── core/                   # rc_core: dominio puro, SIN side-effects
│   │   ├── models.py           #   Paper, Chunk, Citation, QAResult
│   │   ├── interfaces.py       #   Protocols: Retriever, Parser, Embedder, LLMClient, Verifier
│   │   └── errors.py
│   ├── ingest/                 # rc_ingest: pipeline de ingesta
│   │   ├── router.py           #   selección de parser por tipo/calidad
│   │   ├── parsers/            #   grobid.py, pymupdf.py
│   │   ├── chunking.py         #   chunking estructura-aware
│   │   └── tasks.py            #   jobs Arq
│   ├── retrieval/              # rc_retrieval: implementaciones de Retriever
│   │   ├── pgvector_store.py
│   │   ├── qdrant_store.py     #   stub para swap futuro
│   │   └── hybrid.py           #   fusión BM25 + denso (off por flag)
│   ├── qa/                     # rc_qa: loop de QA + verificación
│   │   ├── engine.py           #   retrieve-then-generate con verificación
│   │   ├── verifier.py         #   verificación de citas
│   │   └── prompts/
│   ├── eval/                   # rc_eval: arnés de evaluación
│   │   ├── litqa2.py
│   │   ├── goldset.py
│   │   └── metrics.py          #   recall@k, MRR, citation faithfulness
│   └── api/                    # rc_api: FastAPI, único punto con I/O HTTP
│       ├── main.py
│       ├── routes/             #   ingest, search, qa, jobs, health
│       └── deps.py             #   DI/wiring
├── migrations/                 # SQL versionado
├── frontend/                   # React + Vite + TS
├── eval_data/                  # Gold set propio (JSONL) + fixtures LitQA2
├── docker/                     # Dockerfile.api, Dockerfile.worker, grobid config
└── tests/                      # unit + integración (pytest)
```

### Regla de dependencias

```
core ← ingest, retrieval, qa, eval ← api
```

**core** no importa nada interno. **api** importa todo. Las dependencias apuntan siempre hacia core.

---

## 2. Esquema PostgreSQL + pgvector (DDL)

### Extensiones
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

### `papers`
```sql
CREATE TABLE papers (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doi           TEXT UNIQUE,
    title         TEXT NOT NULL,
    authors       JSONB NOT NULL DEFAULT '[]',
    year          INT,
    venue         TEXT,
    abstract      TEXT,
    source        TEXT NOT NULL,                -- 'upload' | 'openalex' | 'crossref'
    open_access   BOOLEAN,
    pdf_path      TEXT,
    grobid_tei    TEXT,
    meta          JSONB NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_papers_doi ON papers (doi);
CREATE INDEX idx_papers_title_trgm ON papers USING gin (title gin_trgm_ops);
```

### `chunks`
```sql
CREATE TABLE chunks (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id      UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    ordinal       INT NOT NULL,
    section       TEXT,
    text          TEXT NOT NULL,
    char_start    INT,
    char_end      INT,
    page          INT,
    token_count   INT,
    content_hash  TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (paper_id, ordinal)
);
CREATE INDEX idx_chunks_paper ON chunks (paper_id);
CREATE INDEX idx_chunks_text_trgm ON chunks USING gin (text gin_trgm_ops);
```

### `embeddings`
```sql
CREATE TABLE embeddings (
    chunk_id      UUID PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model         TEXT NOT NULL,
    dim           INT NOT NULL,
    vector        vector(1024) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_embeddings_hnsw ON embeddings
    USING hnsw (vector vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```

### `annotations`
```sql
CREATE TABLE annotations (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id      UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    chunk_id      UUID REFERENCES chunks(id) ON DELETE SET NULL,
    page          INT,
    char_start    INT,
    char_end      INT,
    quote         TEXT NOT NULL,
    note          TEXT,
    kind          TEXT NOT NULL DEFAULT 'user',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_annotations_paper ON annotations (paper_id);
```

### `jobs`
```sql
CREATE TABLE jobs (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kind          TEXT NOT NULL,                 -- 'ingest_pdf' | 'reindex' | 'enrich_meta'
    status        TEXT NOT NULL DEFAULT 'queued',
    paper_id      UUID REFERENCES papers(id) ON DELETE CASCADE,
    payload       JSONB NOT NULL DEFAULT '{}',
    attempts      INT NOT NULL DEFAULT 0,
    max_attempts  INT NOT NULL DEFAULT 3,
    error         TEXT,
    stage         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_jobs_status ON jobs (status, kind);
```

> **Nota**: `char_start/char_end/page` son el sustrato de la cita a nivel de pasaje. Sin offsets exactos la verificación no puede anclar la afirmación al span de origen. Se diseña desde el día 1.

---

## 3. Contrato de Interfaz de Retrieval (Protocol)

```python
# packages/core/interfaces.py — firmas sin implementación

@dataclass(frozen=True)
class ChunkRef:
    chunk_id: str
    paper_id: str
    text: str
    section: Optional[str]
    page: Optional[int]
    char_start: Optional[int]
    char_end: Optional[int]

@dataclass(frozen=True)
class ScoredChunk:
    chunk: ChunkRef
    score: float           # normalizado 0..1
    channel: str           # 'dense' | 'lexical' | 'fused'

@dataclass(frozen=True)
class UpsertChunk:
    chunk_id: str
    paper_id: str
    text: str
    vector: Sequence[float]
    metadata: dict

class Retriever(Protocol):
    def upsert(self, items: Sequence[UpsertChunk]) -> int: ...
    def search_dense(self, query_vector: Sequence[float], k: int,
                     paper_ids: Optional[Sequence[str]] = None) -> list[ScoredChunk]: ...
    def search_lexical(self, query_text: str, k: int,
                       paper_ids: Optional[Sequence[str]] = None) -> list[ScoredChunk]: ...
    def delete_by_paper(self, paper_id: str) -> int: ...
    def count(self) -> int: ...
    def health(self) -> bool: ...
```

### Invariantes
- `upsert` es idempotente por `chunk_id`
- `score` se devuelve normalizado 0..1, comparable entre dense y lexical
- El filtro `paper_ids` es obligatorio (scoping a colección del usuario)
- La fusión dense+lexical NO vive en el Retriever; vive en `retrieval/hybrid.py`

---

## 4. Pipeline de Ingesta Async

```
POST /ingest (PDF o DOI)
  → crea row en jobs (queued)
  → Arq worker toma el job
    → Router de parsers:
        ¿PDF con capa de texto + GROBID responde?
          → Sí: GROBID (secciones + refs + meta)
          → No: Fallback PyMuPDF (texto plano)
          → PDF escaneado: needs_ocr → job dead (diferido a v0.2)
    → Chunking estructura-aware (512 tokens, solape 64)
    → Embedder BGE-M3 (batch)
    → Retriever.upsert + persist papers/chunks
    → job done
```

### Chunking
- **Con TEI (GROBID)**: cortar por sección → ventanas de ~512 tokens, solape ~64. Nunca cruzar fronteras de sección.
- **Solo texto (PyMuPDF)**: ventanas de ~512 tokens con solape, registrando `page` y offsets `char_start/char_end`.
- Cada chunk persiste `content_hash = sha256(text)` para dedupe e idempotencia.

### Manejo de fallos
- Reintentos con backoff exponencial hasta `max_attempts=3`
- Luego `status='dead'` con `error` y `stage` poblados
- Fallo parcial de embeddings: el job no se marca `done` hasta que todos los chunks tengan embedding

---

## 5. Loop de QA con Verificación de Citas

```
Pregunta + scope (paper_ids)
  → Retrieval (dense + lexical, fusión)
  → Rerank top-k (cross-encoder, opcional v0.1)
  → QA engine (Retriever + LLMClient): genera respuesta con citas {claim → chunk_id}
  → Verifier: cada claim vs su span citado
    → ¿faithfulness >= umbral?
      → Sí: respuesta entregada con citas ancladas a span+página
      → No: política de fallo
```

### Contrato del Verifier

```python
@dataclass(frozen=True)
class Claim:
    text: str
    chunk_id: str
    quoted_span: str

@dataclass(frozen=True)
class ClaimVerdict:
    claim: Claim
    supported: bool
    score: float          # 0..1
    reason: str

class Verifier(Protocol):
    def verify(self, claims: list[Claim]) -> list[ClaimVerdict]: ...
```

### Verificación en dos capas
1. **Anclaje literal** (barato, determinista): `quoted_span` debe aparecer textualmente en `chunks.text`. Si no → cita inventada → `supported=False`.
2. **Entailment** (semántico): LLM juez o cross-encoder NLI evalúa si el span implica el claim.

### Política ante fallo

| Situación | Acción |
|-----------|--------|
| quoted_span no existe en el chunk | Descartar claim. Log como cita fabricada. |
| Anclaje ok pero entailment < umbral | 1 reintento con más contexto |
| Reintento sigue bajo umbral | Marcar "evidencia insuficiente" |
| Ningún claim supera umbral | Respuesta: "no hay soporte suficiente" |

> **Umbral objetivo**: citation faithfulness ≥ 0.95 sobre el gold set. Si no se alcanza ≥0.90, no se publican features de síntesis.

---

## 6. Arnés de Evaluación

### LitQA2 (`eval/litqa2.py`)
- Benchmark externo reproducible de FutureHouse/PaperQA
- Métricas: accuracy y abstención (precision@answered)
- Subconjunto local con PDFs ingeridos por nuestro pipeline

### Gold Set Propio (50–100 pares)
- 15–25 papers open-access del dominio real
- 3–5 preguntas factuales por paper
- 70% factuales de un paper, 20%跨 2 papers, 10% sin respuesta (abstención)
- Formato JSONL con `paper_doi`, `char_start/char_end`, `gold_span`
- Versionado en git (`goldset_v1.jsonl`)

### Métricas

| Métrica | Objetivo v0.1 |
|---------|--------------|
| Recall@10 | ≥ 0.85 |
| MRR | ≥ 0.6 |
| Citation faithfulness | ≥ 0.95 (go/no-go) |
| Answer accuracy (answerable) | ≥ 0.70 |
| Abstención correcta (unanswerable) | ≥ 0.90 |

> **Go/No-Go**: se libera si y solo si `faithfulness ≥ 0.95` y `abstención ≥ 0.90`. Un fallo de faithfulness bloquea el release.

---

## 7. docker-compose.yml

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: rc
      POSTGRES_PASSWORD: rc
      POSTGRES_DB: research_copilot
    ports: ["5432:5432"]
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./migrations:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rc -d research_copilot"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]

  grobid:
    image: lfoppiano/grobid:0.8.1
    ports: ["8070:8070"]
    deploy:
      resources:
        limits:
          memory: 4g

  api:
    build: { context: ., dockerfile: docker/Dockerfile.api }
    env_file: [.env]
    environment:
      DATABASE_URL: postgresql://rc:rc@db:5432/research_copilot
      REDIS_URL: redis://redis:6379/0
      GROBID_URL: http://grobid:8070
    ports: ["8000:8000"]
    depends_on:
      db: { condition: service_healthy }
      redis: { condition: service_healthy }

  worker:
    build: { context: ., dockerfile: docker/Dockerfile.worker }
    env_file: [.env]
    environment:
      DATABASE_URL: postgresql://rc:rc@db:5432/research_copilot
      REDIS_URL: redis://redis:6379/0
      GROBID_URL: http://grobid:8070
    depends_on:
      db: { condition: service_healthy }
      redis: { condition: service_healthy }
      grobid: { condition: service_started }

  frontend:
    build: { context: ./frontend }
    ports: ["5173:5173"]
    depends_on: [api]

volumes:
  pgdata:
```

> Ollama no se incluye por defecto (S5: LLM por API opt-in). Se añade vía `.env` si se desea ruta 100% local.

---

## 8. Issues (Orden de Dependencia)

| # | Issue | Depende de | Criterio de Aceptación |
|---|-------|-----------|----------------------|
| 1 | Scaffold monorepo + docker-compose | — | `docker compose up` deja todos los servicios healthy; `GET /health` → 200 |
| 2 | Migraciones + extensiones | 1 | 5 tablas creadas; extensiones vector, pg_trgm, uuid-ossp presentes |
| 3 | core: modelos + Protocols | — | mypy pasa; Protocols importan sin dependencias externas |
| 4 | PgVectorStore implementa Retriever | 2, 3 | upsert idempotente; search_dense devuelve top-1 esperado |
| 5 | Embedder BGE-M3 + batching | 3 | Vectores dim=1024 normalizados; throughput documentado |
| 6 | Parser GROBID + PyMuPDF + router | 1, 3 | 3 PDFs fixture: GROBID produce secciones; timeout cae a PyMuPDF |
| 7 | Chunking + persistencia | 4, 6 | Chunks con section, offsets, page; sin cruzar secciones |
| 8 | Pipeline ingesta async (Arq) | 5, 7 | POST /ingest → done; fallo forzado → dead con stage/error |
| 9 | Loop QA + Verifier | 4, 8 | Pregunta answerable → cita con quoted_span literal en chunk |
| 10 | Arnés eval + goldset_v1 + reporte | 9 | `make eval` emite métricas; go/no-go evaluado automáticamente |

---

## 9. Resumen de Decisiones

### Tomadas (ya decididas)
- Python/FastAPI, Postgres+pgvector, Arq, custom retrieve-then-generate QA, BGE-M3, GROBID+PyMuPDF
- Single-user sin auth, verificación en dos capas con anclaje literal obligatorio

### Diferidas con default (reversibles)
- BM25 real → pg_trgm
- Qdrant/Neo4j → tras interfaz
- OCR → needs_ocr/dead
- Grafo conocimiento → v0.2
- Ollama local → opt-in por .env

### No reversibles (decidir bien ahora)
- Lenguaje backend (S1)
- Modelo de embedding y su dimensión (S6)
- Principio: faithfulness bloquea release (§5/§6)
