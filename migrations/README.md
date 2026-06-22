# Migrations

Database schema management for Research Copilot.

## Current approach (v0.1)

SQL files in this directory are mounted into the PostgreSQL container via
`docker-compose.yml`:

```yaml
volumes:
  - ./migrations:/docker-entrypoint-initdb.d
```

PostgreSQL runs files in alphabetical order on **first database initialization**
(when the `pgdata` volume is empty). This is sufficient for local dev and MVP.

### Conventions

- **Prefix**: `NNN_description.sql` (e.g. `000_extensions.sql`, `001_papers.sql`)
- **Idempotent**: all statements use `IF NOT EXISTS` / `IF EXISTS` so re-runs are safe
- **No destructive changes**: never `DROP` in a numbered migration — use a new file

### Applying manually

```bash
# Connect and run a single file
docker compose exec -T db psql -U rc -d research_copilot < migrations/000_extensions.sql

# Or apply all pending files in order
for f in $(ls migrations/*.sql | sort); do
  docker compose exec -T db psql -U rc -d research_copilot < "$f"
done
```

## Future: Alembic upgrade path

For multi-environment deploys (staging, prod) and collaborative schema evolution,
the project will adopt [Alembic](https://alembic.sqlalchemy.org/).

```
migrations/
├── 000_extensions.sql          # (this file) bootstrap extensions for docker-entrypoint
├── README.md                   # this file
└── versions/                   # alembic revision files (future)
    ├── 001_papers.py
    ├── 002_chunks.py
    └── ...
```

When Alembic is introduced, the initial `000_extensions.sql` will become Alembic's
first revision so that both paths converge on the same schema.
