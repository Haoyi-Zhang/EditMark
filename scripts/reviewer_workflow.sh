#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI_ARGS=("$@")
CLI_PYTHON=""
for ((i=0; i<${#CLI_ARGS[@]}; i++)); do
  if [[ "${CLI_ARGS[$i]}" == "--python" && $((i + 1)) -lt ${#CLI_ARGS[@]} ]]; then
    CLI_PYTHON="${CLI_ARGS[$((i + 1))]}"
    break
  fi
done

if [[ -n "$CLI_PYTHON" ]]; then
  RESOLVED_PYTHON="$CLI_PYTHON"
  PYTHON_SOURCE="--python"
elif [[ -n "${PYTHON_BIN:-}" ]]; then
  RESOLVED_PYTHON="$PYTHON_BIN"
  PYTHON_SOURCE="PYTHON_BIN"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  RESOLVED_PYTHON="$ROOT/.venv/bin/python"
  PYTHON_SOURCE="repo-local .venv"
else
  ACTIVE_PYTHON="$(command -v python3 || command -v python || true)"
  if [[ -n "${VIRTUAL_ENV:-}" && -n "$ACTIVE_PYTHON" ]]; then
    CANDIDATE="$("$ACTIVE_PYTHON" - <<'PY'
import os
import sys
from pathlib import Path

current = Path(sys.executable).resolve()
active = str(os.environ.get("VIRTUAL_ENV", "")).strip()
if active:
    try:
        current.relative_to(Path(active).resolve())
    except Exception:
        pass
    else:
        print(current)
PY
)"
    if [[ -n "$CANDIDATE" ]]; then
      RESOLVED_PYTHON="$CANDIDATE"
      PYTHON_SOURCE="active virtualenv"
    fi
  fi
  if [[ -z "${RESOLVED_PYTHON:-}" && -n "$ACTIVE_PYTHON" ]]; then
    CANDIDATE="$("$ACTIVE_PYTHON" - <<'PY'
import sys
from pathlib import Path

current = Path(sys.executable).resolve()
if any(token in current.parts for token in (".venv", "posteditbench_release", "posteditbench_release_clean")):
    print(current)
PY
)"
    if [[ -n "$CANDIDATE" ]]; then
      RESOLVED_PYTHON="$CANDIDATE"
      PYTHON_SOURCE="current interpreter"
    fi
  fi
  if [[ -z "${RESOLVED_PYTHON:-}" ]]; then
    echo "[reviewer_workflow.sh] Missing Python interpreter. Set PYTHON_BIN, activate a dedicated virtualenv, or create $ROOT/.venv/bin/python." >&2
    exit 1
  fi
fi

echo "[reviewer_workflow.sh] Using $PYTHON_SOURCE: $RESOLVED_PYTHON" >&2

exec "$RESOLVED_PYTHON" "$ROOT/scripts/reviewer_workflow.py" "$@"
