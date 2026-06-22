-- 004_annotations.sql
-- Research Copilot — Annotations table (MVP v0.1)
-- highlight-to-explain del lector PDF

CREATE TABLE IF NOT EXISTS annotations (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id      UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    chunk_id      UUID REFERENCES chunks(id) ON DELETE SET NULL,
    page          INT,
    char_start    INT,
    char_end      INT,
    quote         TEXT NOT NULL,                 -- texto exacto resaltado
    note          TEXT,                          -- nota del usuario o explicación IA
    kind          TEXT NOT NULL DEFAULT 'user',  -- 'user' | 'ai_explain'
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_annotations_paper ON annotations (paper_id);
