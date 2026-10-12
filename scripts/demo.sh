#!/usr/bin/env bash
# One-command demo on the committed 200-trace sample.
#   scripts/demo.sh                              uses Ollama (nomic-embed-text and qwen3:8b)
#   CONFIG=configs/sample-offline.json scripts/demo.sh   needs no model server at all
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-demo}"
CONFIG="${CONFIG:-configs/sample.json}"

uv run goldenset run --config "$CONFIG" --run-id "$RUN_ID" --force
uv run goldenset eval --run "runs/$RUN_ID" --truth data/sample/ground_truth.jsonl --what scrub,dedup,coverage,enrich
uv run goldenset report --run "runs/$RUN_ID"

echo
echo "Report:   runs/$RUN_ID/report.md"
echo "Review:   uv run goldenset review --run runs/$RUN_ID"
echo "Finalise: uv run goldenset finalize --run runs/$RUN_ID"
