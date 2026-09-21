#!/bin/sh
# ── run_weekly.sh ────────────────────────────────────────────────────────────
# Manual one-shot helper (the `pipeline` service now runs pipeline.scheduler).
# Runs the native-Python screen + rank + chart + VCP-vision pipeline
# (pipeline/run_pipeline.py) — no notebook execution involved.
# Outputs: data/reports/watchlist_<date>.csv, data/reports/vcp_analysis_<date>.csv,
# data/charts/<symbol>.png
# ─────────────────────────────────────────────────────────────────────────────

set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "──────────────────────────────────────────"
echo "Stage 2 Screener — weekly pipeline run"
echo "Timestamp : ${TIMESTAMP}"
echo "──────────────────────────────────────────"

cd /app
python -m pipeline.run_pipeline

echo "Run complete."
