-- 003_embeddings.sql
-- Research Copilot — Embeddings table (MVP v0.1)
-- Separada de chunks: permite reindexar/migrar modelo sin reescribir texto ni offsets.

CREATE TABLE embeddings (
    chunk_id      UUID PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model         TEXT NOT NULL,                 -- p.ej. 'bge-m3' (audita qué modelo generó el vector)
    dim           INT NOT NULL,
    vector        vector(1024) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- HNSW: buen recall con escritura incremental. Coseno por defecto (BGE normaliza).
CREATE INDEX idx_embeddings_hnsw ON embeddings
    USING hnsw (vector vector_cosine_ops) WITH (m = 16, ef_construction = 64);
