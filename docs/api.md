# Research Copilot — API Reference

Base URL: `http://localhost:8000`

---

## `GET /health`

Check whether the API, database, and Redis are reachable.

### Response `200`

```json
{"db": true, "redis": true}
```

### Response `503`

At least one dependency is unreachable.

```json
{"db": false, "redis": true}
```

### curl

```bash
curl http://localhost:8000/health
```

---

## `POST /ingest`

Queue a PDF or DOI for ingestion.  Two content-type variants.

### Multipart upload (PDF file)

```
Content-Type: multipart/form-data
```

| Field | Type   | Required | Description         |
|-------|--------|----------|---------------------|
| file  | File   | yes      | PDF document to ingest |

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@paper.pdf"
```

### JSON (DOI)

```
Content-Type: application/json
```

```json
{"doi": "10.1234/example"}
```

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"doi": "10.1234/example"}'
```

### Response `202`

```json
{"job_id": "a1b2c3d4-...-uuid"}
```

### Errors

| Code | Condition                      | Body                               |
|------|--------------------------------|------------------------------------|
| 415  | Unsupported Content-Type       | `{"error": "Send multipart/form-data (file) or application/json (doi)"}` |
| 422  | Missing file field             | `{"error": "Missing file field"}`  |
| 422  | Missing or invalid DOI         | `{"error": "Missing or invalid doi field"}` |

---

## `GET /jobs/{job_id}`

Poll the status of an ingest job.

### Path parameters

| Parameter | Type   | Description            |
|-----------|--------|------------------------|
| job_id    | string | UUID of the ingest job |

### Response `200`

```json
{
  "id": "a1b2c3d4-...",
  "kind": "ingest_pdf",
  "status": "done",
  "stage": null,
  "error": null,
  "attempts": 1,
  "max_attempts": 3,
  "created_at": "2026-06-22T10:00:00+00:00",
  "updated_at": "2026-06-22T10:00:15+00:00"
}
```

Possible `status` values: `queued`, `running`, `failed`, `done`, `dead`.
`stage` tracks the pipeline phase: `parse`, `chunk`, `persist`.

### Response `404`

```json
{"error": "Job not found"}
```

### curl

```bash
curl http://localhost:8000/jobs/a1b2c3d4-...-uuid
```

---

## `POST /qa`

Answer a research question against the ingested corpus.

### Request

```json
{
  "question": "What is RLHF?",
  "paper_ids": null
}
```

| Field      | Type            | Required | Description                                      |
|------------|-----------------|----------|--------------------------------------------------|
| question   | string          | yes      | Natural-language research question               |
| paper_ids  | list[string]    | no       | Scope to specific papers (null = all ingested)   |

### Response `200`

```json
{
  "question": "What is RLHF?",
  "answer": "Reinforcement Learning from Human Feedback (RLHF) ...",
  "answerable": true,
  "claims": [
    {
      "text": "RLHF uses human preferences to fine-tune language models.",
      "chunk_id": "c1c2c3d4-...",
      "quoted_span": "RLHF uses human preferences to fine-tune language models.",
      "supported": true,
      "score": 0.92,
      "reason": "",
      "page": 3,
      "char_start": 1200,
      "char_end": 1280,
      "paper_id": "p1p2p3d4-..."
    }
  ]
}
```

When no answer can be supported `answer` is `"No hay soporte suficiente"` and
`answerable` is `false`.

### curl

```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RLHF?"}'
```

### Error codes

| Code | Condition        | Body                                                     |
|------|------------------|----------------------------------------------------------|
| 422  | Missing question | `{"detail": [{"loc": ["body","question"], "msg": "field required", ...}]}` |
| 422  | Invalid UUID     | `{"detail": [{"loc": ["path","job_id"], ...}]}`          |
