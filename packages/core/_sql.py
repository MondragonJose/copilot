"""SQL snippet constants for idempotent upserts.

Every string in this module is a pure PostgreSQL template parameterised
for ``asyncpg`` (``$N`` placeholders).  Shared between ``ingest/persist.py``
and ``retrieval/pgvector_store.py`` so that the two codebases cannot
drift apart.
"""

UPSERT_CHUNK_SQL = (
    "INSERT INTO chunks (id, paper_id, ordinal, section, text, "
    "char_start, char_end, page, token_count, content_hash) "
    "VALUES ($1::uuid, $2::uuid, 0, NULL, $3, "
    "NULL, NULL, NULL, NULL, '') "
    "ON CONFLICT (id) DO UPDATE "
    "SET text = EXCLUDED.text, "
    "    section = EXCLUDED.section, "
    "    paper_id = EXCLUDED.paper_id"
)

UPSERT_EMBEDDING_SQL = (
    "INSERT INTO embeddings (chunk_id, model, dim, vector) "
    "VALUES ($1::uuid, $2, $3, $4::vector) "
    "ON CONFLICT (chunk_id) DO UPDATE "
    "SET vector = EXCLUDED.vector, "
    "    model = EXCLUDED.model"
)

UPSERT_PAPER_SQL = (
    "INSERT INTO papers (id, doi, title, authors, year, venue, abstract, "
    "source, open_access, pdf_path, grobid_tei, meta) "
    "VALUES ($1::uuid, $2, $3, $4::jsonb, $5, $6, $7, $8, $9, $10, $11, "
    "$12::jsonb) "
    "ON CONFLICT (id) DO UPDATE "
    "SET title = EXCLUDED.title, "
    "    authors = EXCLUDED.authors, "
    "    source = EXCLUDED.source"
)

PAPER_COLUMNS = (
    "id, doi, title, authors, year, venue, abstract, "
    "source, open_access, pdf_path, grobid_tei, meta"
)
