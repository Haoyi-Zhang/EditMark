#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "PostEditBench watermark artifact bootstrap"
echo "Project root: $ROOT"
echo "Prerequisites:"
echo "  - Python 3.10+"
echo "  - A POSIX shell for the release packaging helper"
echo "  - Optional: jq for inspecting JSON outputs"
echo

mkdir -p "$ROOT/data/release/sources" "$ROOT/results/runs" "$ROOT/model_cache"

if command -v python >/dev/null 2>&1; then
  python "$ROOT/scripts/validate_setup.py" --check-redaction --config "$ROOT/configs/debug.yaml"
else
  echo "python is not available on PATH"
fi

echo "Local directories prepared. No external model downloads are required for the deterministic sample pipeline."
