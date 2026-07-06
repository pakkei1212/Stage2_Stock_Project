#!/bin/sh
# ── run_notebook.sh ──────────────────────────────────────────────────────────
# Executed by the `runner` service on each scheduled run.
# Converts + executes the notebook, saves the output (with cell outputs filled
# in) to /app/data/runs/ with a timestamp in the filename.
# ─────────────────────────────────────────────────────────────────────────────

set -e

NOTEBOOK="${NOTEBOOK_NAME:-nasdaq_stage2_screener.ipynb}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
INPUT="/app/notebooks/${NOTEBOOK}"
OUTPUT="/app/data/runs/${TIMESTAMP}_${NOTEBOOK}"

echo "──────────────────────────────────────────"
echo "Stage 2 Screener — scheduled run"
echo "Timestamp : ${TIMESTAMP}"
echo "Notebook  : ${INPUT}"
echo "Output    : ${OUTPUT}"
echo "──────────────────────────────────────────"

# nbconvert executes all cells in order and writes a new .ipynb with outputs.
# --ExecutePreprocessor.timeout controls max seconds per cell (default: 3600 = 1hr).
jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output "${OUTPUT}" \
    "${INPUT}"

echo "✅ Run complete → ${OUTPUT}"
