#!/usr/bin/env bash
# Resume / run the SROIE harness pilot once the free-tier quota is available.
# Safe to re-run: successful API responses are cached under research/outputs/cache,
# and cached docs are skipped (no re-spend). Stops cleanly when quota is hit.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "Missing .env with gemini_api_key=..." >&2; exit 1
fi
set -a; . ./.env; set +a

MAX_DOCS="${MAX_DOCS:-24}"
N_SAMPLES="${N_SAMPLES:-2}"
WORKERS="${WORKERS:-3}"

echo "=== SROIE harness pilot (docs=$MAX_DOCS, samples=$N_SAMPLES, workers=$WORKERS) ==="
venv/bin/python3 research/run_experiment.py
echo
echo "=== Summarizing ==="
venv/bin/python3 research/summarize.py
echo
echo "Done. See research/outputs/FINDINGS.md"
