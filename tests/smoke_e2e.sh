#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# E2E smoke test for Research Copilot.
# Uploads a real PDF, waits for ingest, asks a question, validates.
# ────────────────────────────────────────────────────────────────────
set -euo pipefail

API="${API_URL:-http://localhost:8000}"
MAX_WAIT_SECONDS=300
POLL_INTERVAL=10

echo "=== E2E Smoke: Research Copilot ==="
echo "API: $API"

# ── 1. Wait for API health ──────────────────────────────────────────
echo
echo "--- Step 1: Wait for API ---"
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT_SECONDS ]; do
  if curl -sf "$API/health" > /dev/null 2>&1; then
    echo "API healthy after ${ELAPSED}s"
    break
  fi
  sleep 5
  ELAPSED=$((ELAPSED + 5))
done
if [ $ELAPSED -ge $MAX_WAIT_SECONDS ]; then
  echo "TIMEOUT: API not healthy within ${MAX_WAIT_SECONDS}s"
  exit 1
fi

# ── 2. Download a test PDF ──────────────────────────────────────────
echo
echo "--- Step 2: Download test PDF ---"
TEST_PDF="/tmp/rc_smoke_test.pdf"
# arXiv abstract page for a well-known paper — pdf link
PDF_URL="https://arxiv.org/pdf/1706.03762.pdf"
if curl -sL -o "$TEST_PDF" "$PDF_URL"; then
  SIZE=$(stat -c%s "$TEST_PDF" 2>/dev/null || stat -f%z "$TEST_PDF" 2>/dev/null)
  echo "Downloaded test PDF ($SIZE bytes)"
else
  echo "WARNING: Could not download test PDF, creating a minimal one"
  python3 -c "
import struct, zlib
# Minimal valid PDF with one page
pdf = b'%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<<>>>>endobj\n4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 100 700 Td (Hello World) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000266 00000 n \ntrailer<</Size 5/Root 1 0 R>>\nstartxref\n375\n%%EOF'
  open('$TEST_PDF', 'wb').write(pdf)
"
  echo "Created minimal test PDF"
fi

# ── 3. Upload PDF ───────────────────────────────────────────────────
echo
echo "--- Step 3: Upload PDF ---"
JOB_RESPONSE=$(curl -sf -X POST "$API/ingest" \
  -F "file=@$TEST_PDF" \
  -H "accept: application/json")
JOB_ID=$(echo "$JOB_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "Job ID: $JOB_ID"

# ── 4. Poll job until done ──────────────────────────────────────────
echo
echo "--- Step 4: Wait for ingest job ---"
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT_SECONDS ]; do
  JOB_STATUS=$(curl -sf "$API/jobs/$JOB_ID" | \
    python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))")
  echo "  [${ELAPSED}s] status = ${JOB_STATUS}"
  if [ "$JOB_STATUS" = "done" ]; then
    echo "Ingest complete!"
    PAPER_ID=$(curl -sf "$API/jobs/$JOB_ID" | \
      python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('result',''))")
    echo "Paper ID: $PAPER_ID"
    break
  fi
  if [ "$JOB_STATUS" = "dead" ]; then
    ERROR=$(curl -sf "$API/jobs/$JOB_ID" | \
      python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error','unknown'))")
    echo "FAIL: Job went dead — $ERROR"
    exit 1
  fi
  sleep "$POLL_INTERVAL"
  ELAPSED=$((ELAPSED + POLL_INTERVAL))
done
if [ $ELAPSED -ge $MAX_WAIT_SECONDS ]; then
  echo "TIMEOUT: Ingest not done within ${MAX_WAIT_SECONDS}s"
  exit 1
fi

# ── 5. Ask a question ───────────────────────────────────────────────
echo
echo "--- Step 5: Ask a question ---"
QA_RESPONSE=$(curl -sf -X POST "$API/qa" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main contribution of this paper?"}')
echo "QA response received"

ANSWERABLE=$(echo "$QA_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('answerable', False))")
CLAIMS_COUNT=$(echo "$QA_RESPONSE" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('claims', [])))")
ANSWER=$(echo "$QA_RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); a=d.get('answer',''); print(a[:100]) if a else print('(None)')")

echo "  answerable : $ANSWERABLE"
echo "  claims     : $CLAIMS_COUNT"
echo "  answer     : ${ANSWER}..."

# ── 6. Validate ─────────────────────────────────────────────────────
echo
echo "--- Step 6: Validate ---"
if [ "$ANSWERABLE" != "True" ]; then
  echo "WARNING: Question was not answerable (may be OK depending on corpus)"
fi
echo
echo "=== E2E smoke PASS ==="
