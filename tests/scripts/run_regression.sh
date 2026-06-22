#!/usr/bin/env bash
# ── Research Copilot — Regression Suite ─────────────────────────────────────
# Fails loudly on any regression (set -e, pipefail).
# Usage:  bash tests/scripts/run_regression.sh
#
# Exits 0 on PASS, 1 on FAIL.

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../..")"

PACKAGES="core qa eval retrieval ingest"
COV_FLAGS=""
for pkg in $PACKAGES; do
    COV_FLAGS="$COV_FLAGS --cov=packages/$pkg"
done

echo "========================================"
echo " Research Copilot — Regression Suite"
echo "========================================"
echo ""

# ── 1. Regression unit tests ────────────────────────────────────────────────
echo "--- 1a. Regression tests (unit, no infra) ---"
python3 -m pytest tests/regression/ -v --tb=short -W ignore::DeprecationWarning "$@"
echo ""

# ── 2. Existing unit tests ──────────────────────────────────────────────────
echo "--- 1b. Full unit test suite ---"
python3 -m pytest tests/core/ tests/qa/ tests/eval/ tests/retrieval/ tests/ingest/ \
    -v --tb=short -x -W ignore::DeprecationWarning "$@"
echo ""

# ── 3. Coverage (regression + existing) ─────────────────────────────────────
echo "--- 2. Coverage report ---"
python3 -m pytest tests/regression/ tests/core/ tests/qa/ tests/eval/ tests/retrieval/ tests/ingest/ \
    $COV_FLAGS --cov-report=term-missing -W ignore::DeprecationWarning 2>&1 \
    | tail -30
echo ""

# ── 4. Golden metric assertions ─────────────────────────────────────────────
echo "--- 3. Golden thresholds ---"
python3 -c "
from eval.run import FAITHFULNESS_THRESHOLD, ABSTENTION_THRESHOLD
assert FAITHFULNESS_THRESHOLD >= 0.95, 'Faithfulness threshold regressed!'
assert ABSTENTION_THRESHOLD >= 0.90, 'Abstention threshold regressed!'
print(f'  FAITHFULNESS_THRESHOLD = {FAITHFULNESS_THRESHOLD}  ✓')
print(f'  ABSTENTION_THRESHOLD   = {ABSTENTION_THRESHOLD}  ✓')
print('  All golden threshold assertions PASS')
"
echo ""

# ── Summary ─────────────────────────────────────────────────────────────────
echo "========================================"
echo " Regression Suite — ALL PASS"
echo "========================================"
