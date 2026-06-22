-- 005_jobs.sql
-- Research Copilot — Jobs table (MVP v0.1)
-- Pipeline async, §4

CREATE TABLE jobs (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kind          TEXT NOT NULL,                 -- 'ingest_pdf' | 'reindex' | 'enrich_meta'
    status        TEXT NOT NULL DEFAULT 'queued',-- queued|running|done|failed|dead
    paper_id      UUID REFERENCES papers(id) ON DELETE CASCADE,
    payload       JSONB NOT NULL DEFAULT '{}',
    attempts      INT NOT NULL DEFAULT 0,
    max_attempts  INT NOT NULL DEFAULT 3,
    error         TEXT,                          -- último error (para inspección)
    stage         TEXT,                          -- etapa donde quedó (parse|chunk|embed|persist)
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_jobs_status ON jobs (status, kind);
