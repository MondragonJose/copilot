import { describe, it, expect, vi, beforeEach } from "vitest";
import { api, ApiError } from "../api";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

function ok(body: unknown) {
  return Promise.resolve({
    ok: true,
    json: () => Promise.resolve(body),
    status: 200,
    statusText: "OK",
  });
}

function notOk(status: number, error: string) {
  return Promise.resolve({
    ok: false,
    status,
    statusText: error,
    json: () => Promise.resolve({ error }),
  });
}

beforeEach(() => {
  mockFetch.mockReset();
});

// ===================================================================
// GET /health
// ===================================================================

describe("api.health", () => {
  it("returns health status on 200", async () => {
    mockFetch.mockResolvedValue(ok({ db: true, redis: true }));
    const res = await api.health();
    expect(res).toEqual({ db: true, redis: true });
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/health"),
      expect.objectContaining({
        headers: { "Content-Type": "application/json" },
      }),
    );
  });

  it("throws ApiError on failure", async () => {
    mockFetch.mockResolvedValue(notOk(503, "Service Unavailable"));
    await expect(api.health()).rejects.toThrow(ApiError);
    await expect(api.health()).rejects.toThrow("Service Unavailable");
  });
});

// ===================================================================
// POST /qa
// ===================================================================

describe("api.ask", () => {
  it("sends JSON body and returns response", async () => {
    mockFetch.mockResolvedValue(
      ok({
        question: "test",
        answer: "42",
        answerable: true,
        claims: [],
      }),
    );

    const res = await api.ask({ question: "test" });
    expect(res.answerable).toBe(true);
    expect(res.answer).toBe("42");

    const call = mockFetch.mock.calls[0];
    expect(call[0]).toContain("/qa");
    expect(call[1].method).toBe("POST");
    expect(JSON.parse(call[1].body)).toEqual({ question: "test" });
  });
});

// ===================================================================
// GET /jobs/{id}
// ===================================================================

describe("api.getJob", () => {
  it("returns job status on 200", async () => {
    mockFetch.mockResolvedValue(
      ok({
        id: "abc-123",
        kind: "ingest_pdf",
        status: "done",
        stage: null,
        error: null,
        attempts: 0,
        max_attempts: 3,
        created_at: "2025-01-01T00:00:00",
        updated_at: "2025-01-01T01:00:00",
      }),
    );

    const res = await api.getJob("abc-123");
    expect(res.id).toBe("abc-123");
    expect(res.status).toBe("done");
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("/jobs/abc-123"),
      expect.any(Object),
    );
  });
});

// ===================================================================
// ApiError
// ===================================================================

describe("ApiError", () => {
  it("carries status and message", () => {
    const err = new ApiError(404, "Not Found");
    expect(err.status).toBe(404);
    expect(err.message).toBe("Not Found");
    expect(err.name).toBe("ApiError");
  });
});

// ===================================================================
// POST /ingest (DOI)
// ===================================================================

describe("api.ingestDoi", () => {
  it("sends DOI and returns job_id", async () => {
    mockFetch.mockResolvedValue(ok({ job_id: "job-42" }));
    const res = await api.ingestDoi("10.1234/test");
    expect(res.job_id).toBe("job-42");
    const call = mockFetch.mock.calls[0];
    expect(JSON.parse(call[1].body)).toEqual({ doi: "10.1234/test" });
  });
});
