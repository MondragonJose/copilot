// ---------------------------------------------------------------------------
// Request / response shapes matching the Research Copilot API (FastAPI routes)
// ---------------------------------------------------------------------------

export interface HealthResponse {
  db: boolean;
  redis: boolean;
}

export interface QARequest {
  question: string;
  paper_ids?: string[];
}

export interface ClaimOut {
  text: string;
  chunk_id: string;
  quoted_span: string;
  supported: boolean;
  score: number;
  reason: string;
  page: number | null;
  char_start: number | null;
  char_end: number | null;
  paper_id: string | null;
}

export interface QAResponse {
  question: string;
  answer: string | null;
  answerable: boolean;
  claims: ClaimOut[];
}

export interface IngestResponse {
  job_id: string;
}

export interface JobStatusResponse {
  id: string;
  kind: string;
  status: string;
  stage: string | null;
  error: string | null;
  attempts: number;
  max_attempts: number;
  created_at: string | null;
  updated_at: string | null;
  result?: string | null;
}

export interface Annotation {
  id: string;
  paper_id: string;
  chunk_id: string | null;
  page: number;
  char_start: number;
  char_end: number;
  quote: string;
  note: string | null;
  kind: "user" | "ai_explain";
  created_at: string;
}

export interface CreateAnnotationRequest {
  page: number;
  char_start: number;
  char_end: number;
  quote: string;
  note?: string;
}

export interface ErrorResponse {
  error: string;
}
