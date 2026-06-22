// ---------------------------------------------------------------------------
// Typed HTTP client for the Research Copilot API
// ---------------------------------------------------------------------------

import type {
  Annotation,
  CreateAnnotationRequest,
  HealthResponse,
  IngestResponse,
  JobStatusResponse,
  QARequest,
  QAResponse,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export const BASE_URL =
  (typeof import.meta !== "undefined" &&
    (import.meta as { env?: { VITE_API_URL?: string } }).env
      ?.VITE_API_URL) ||
  "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(
      res.status,
      (body as { error?: string }).error ?? res.statusText,
    );
  }
  return res.json() as Promise<T>;
}

export const api = {
  /** GET /health */
  health(): Promise<HealthResponse> {
    return request<HealthResponse>("/health");
  },

  /** POST /ingest — file upload (multipart) */
  async ingestFile(file: File): Promise<IngestResponse> {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${BASE_URL}/ingest`, { method: "POST", body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new ApiError(
        res.status,
        (body as { error?: string }).error ?? res.statusText,
      );
    }
    return res.json() as Promise<IngestResponse>;
  },

  /** POST /ingest — DOI (JSON) */
  ingestDoi(doi: string): Promise<IngestResponse> {
    return request<IngestResponse>("/ingest", {
      method: "POST",
      body: JSON.stringify({ doi }),
    });
  },

  /** GET /jobs/{jobId} */
  getJob(jobId: string): Promise<JobStatusResponse> {
    return request<JobStatusResponse>(`/jobs/${jobId}`);
  },

  /** POST /qa */
  ask(body: QARequest): Promise<QAResponse> {
    return request<QAResponse>("/qa", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  /** GET /papers/{paperId}/annotations */
  getAnnotations(paperId: string): Promise<Annotation[]> {
    return request<Annotation[]>(`/papers/${paperId}/annotations`);
  },

  /** POST /papers/{paperId}/annotations */
  createAnnotation(
    paperId: string,
    body: CreateAnnotationRequest,
  ): Promise<Annotation> {
    return request<Annotation>(`/papers/${paperId}/annotations`, {
      method: "POST",
      body: JSON.stringify(body),
    });
  },
};
