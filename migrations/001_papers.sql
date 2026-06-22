-- 001_papers.sql
-- Research Copilot — Papers table (MVP v0.1)

CREATE TABLE IF NOT EXISTS papers (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doi           TEXT UNIQUE,                  -- puede ser NULL (preprint sin DOI)
    title         TEXT NOT NULL,
    authors       JSONB NOT NULL DEFAULT '[]',  -- [{name, orcid?}]  (preserva orden)
    year          INT,
    venue         TEXT,
    abstract      TEXT,
    source        TEXT NOT NULL,                -- 'upload' | 'openalex' | 'crossref'
    open_access   BOOLEAN,
    pdf_path      TEXT,                          -- ruta en object storage (MinIO)
    grobid_tei    TEXT,                          -- TEI/XML crudo (auditoría)
    meta          JSONB NOT NULL DEFAULT '{}',  -- campos extra sin esquema fijo
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers (doi);
CREATE INDEX IF NOT EXISTS idx_papers_title_trgm ON papers USING gin (title gin_trgm_ops);
