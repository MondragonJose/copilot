-- 006_jobs_next_attempt_at.sql
-- Research Copilot — Add next_attempt_at for retry backoff support

ALTER TABLE jobs
  ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ;
