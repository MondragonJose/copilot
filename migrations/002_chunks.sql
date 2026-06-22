-- 002_chunks.sql
-- Research Copilot — Chunks table (MVP v0.1)
-- Un chunk = unidad recuperable con procedencia EXACTA (clave para citas a nivel de pasaje).

CREATE TABLE IF NOT EXISTS chunks (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id      UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    ordinal       INT NOT NULL,                 -- orden dentro del paper
    section       TEXT,                          -- 'abstract'|'methods'|... (estructura-aware)
    text          TEXT NOT NULL,
    char_start    INT,                           -- offset en el texto plano del paper
    char_end      INT,
    page          INT,                           -- página PDF de origen (para el lector)
    token_count   INT,
    content_hash  TEXT NOT NULL,                 -- dedupe + idempotencia de ingesta
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (paper_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks (paper_id);
CREATE INDEX IF NOT EXISTS idx_chunks_text_trgm ON chunks USING gin (text gin_trgm_ops);
