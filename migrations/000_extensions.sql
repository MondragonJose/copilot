-- 000_extensions.sql
-- Research Copilot — Bootstrap PostgreSQL extensions (MVP v0.1)
-- Loaded by docker-entrypoint-initdb.d on first container start.
-- See migrations/README.md for the migration strategy.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
