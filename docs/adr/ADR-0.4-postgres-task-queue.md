# ADR-0.4: Postgres-Based Task Queue Instead of Arq

**Status:** Accepted

---

## Context

The MVP blueprint (RESEARCH_COPILOT_PHASE_0.1.md) specifies an async task queue
backed by "Redis + Arq" (S3). The implementation in `packages/worker/run.py`
uses a polling loop over a `jobs` table in Postgres with `FOR UPDATE SKIP LOCKED`
for safe concurrent job dispatch, without any Redis-based queue or the `arq`
library.

Arq (`arq` / `python-arq`) is a popular Redis-backed job queue for Python
asyncio. It provides retries, scheduling, and concurrency control out of the
box. The project already has Redis as a dependency (for caching and future use).

Key factors that led to a Postgres-native approach:

- **Single dependency, lower operational complexity** — Postgres is already
  required for data storage (papers, chunks, embeddings). Adding Arq means
  deploying and monitoring an additional queue abstraction on top of Redis.
- **Job state visibility** — with a `jobs` table, every job's status, attempts,
  stage, error, and `next_attempt_at` is immediately queryable via SQL without
  an extra admin UI. This is critical for debug workflows.
- **Exponential backoff with `next_attempt_at`** — the worker selects jobs
  whose `next_attempt_at <= now()`, which is a natural SQL expression. No
  scheduler needed.
- **Dead-letter semantics** — jobs whose `attempts >= max_attempts` are marked
  `dead` with the last error preserved. This maps cleanly to a column value
  rather than a separate dead-letter queue.
- **Transactional integrity** — paper insert and job status update can share
  the same Postgres transaction, eliminating the risk of a job completing
  but its results being lost between Redis and Postgres.
- **No additional Python dependency** — `arq` would add ~5 transitive
  dependencies and requires `redis` to be configured and available at worker
  startup.

---

## Decision

1. **Do NOT use `arq`** — the worker polls Postgres directly.

2. **Implement the task queue as a `jobs` table** (`migrations/005_jobs.sql`)
   with columns: `id`, `status`, `kind`, `payload`, `attempts`,
   `max_attempts`, `stage`, `error`, `next_attempt_at`, `created_at`,
   `updated_at`.

3. **Use `FOR UPDATE SKIP LOCKED`** for safe concurrent dispatch — multiple
   worker replicas can poll without stealing each other's jobs.

4. **Implement exponential backoff** inline in the polling SQL via
   `next_attempt_at` (see `migrations/006_jobs_next_attempt_at.sql`).

---

## Consequences

- No dependency on `arq` or its transitive dependencies. The `pyproject.toml`
  runtime dependencies remain unchanged.
- Job lifecycle is fully visible in Postgres — any SQL client can inspect
  pending, running, failed, and dead jobs.
- The polling loop introduces latency equal to `POLL_INTERVAL` (currently
  5 seconds) between job enqueue and processing. Arq would have lower
  sub-second latency via Redis pub/sub.
- High-frequency job dispatch could create contention on the `jobs` table.
  Mitigated by the fact that ingest jobs are human-triggered (file upload
  or DOI import), not high-throughput.
- Worker crash during processing loses the in-flight job until the
  stale-running timeout fires (5 minutes). Arq would re-enqueue immediately
  on worker disconnect.

---

## Reversibility

**Easily reversible** — the `jobs` table abstraction is simple enough that
migrating to Arq would require:
1. Replacing the polling loop in `packages/worker/run.py` with an `arq`
   worker decorator.
2. Adding `arq` to `pyproject.toml` dependencies.
3. Replacing `jobs` status writes with Arq enqueue calls.
The `process_job` function (`packages/ingest/tasks.py`) would remain
unchanged — it receives a `job_id` and operates the same way.

---

## Alternatives Considered

- **Arq** — rejected as unnecessary operational complexity for the
  human-scale ingest workload.
- **Celery** — rejected; too heavy for a single-purpose ingest pipeline.
- **Redis Streams (custom consumer)** — would provide lower latency than
  polling but adds Redis Stream protocol knowledge and consumer group
  management. Deferred to v0.2 if throughput requirements grow.

---

## Blueprint References

- **S3** (Cola async) — implemented as Postgres polling, not Redis + Arq.
- **§4** (Ingest pipeline) — `packages/worker/run.py` lines 34–54 implement
  the polling loop. `packages/ingest/tasks.py` contains `process_job` and
  `_handle_failure` with exponential backoff.

## Supersedes

- RESEARCH_COPILOT_PHASE_0.1.md S3 — "Redis + Arq" amended by this ADR.
