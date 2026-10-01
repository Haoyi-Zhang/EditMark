#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
METHOD="${1:-all}"

checkout_dirty() {
  local target="$1"
  local status line entry
  status="$(git -C "$target" status --porcelain 2>/dev/null || true)"
  if [ -z "$status" ]; then
    return 1
  fi
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    entry="${line:3}"
    if printf '%s\n' "$entry" | grep -Eq '(^|/)(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache)(/|$)|(\.pyc|\.pyo)$'; then
      continue
    fi
    return 0
  done <<<"$status"
  return 1
}

safe_external_checkout_root() {
  local candidate="$1"
  local managed_root="$2"
  local resolved_root resolved_candidate resolved_workspace
  resolved_workspace="$(cd "$ROOT" 2>/dev/null && pwd -P)"
  resolved_root="$(cd "$managed_root" 2>/dev/null && pwd -P)"
  resolved_candidate="$(cd "$candidate" 2>/dev/null && pwd -P)"
  if [ -z "$resolved_workspace" ] || [ -z "$resolved_root" ] || [ -z "$resolved_candidate" ]; then
    return 1
  fi
  case "$resolved_root/" in
    "$resolved_workspace"/*) ;;
    *) return 1 ;;
  esac
  case "$resolved_candidate/" in
    "$resolved_root"/*) return 0 ;;
  esac
  return 1
}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  exit 1
fi

FETCH_ENTRIES_RAW="$("$PYTHON_BIN" - "$ROOT" "$METHOD" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1]).resolve()
method = str(sys.argv[2]).strip().lower()


def _sanitize_external_root(value: str) -> str | None:
    stripped = str(value).strip().replace("\\", "/")
    if not stripped:
        return None
    candidate = PurePosixPath(stripped)
    if candidate.is_absolute():
        return None
    if any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    if any(part.startswith(".") for part in candidate.parts):
        return None
    normalized = candidate.as_posix().strip()
    if normalized != "external_checkout" and not normalized.startswith("external_checkout/"):
        return None
    return normalized

entries = {
    "stone_runtime": root / "third_party" / "STONE-watermarking.UPSTREAM.json",
    "sweet_runtime": root / "third_party" / "SWEET-watermark.UPSTREAM.json",
    "ewd_runtime": root / "third_party" / "EWD.UPSTREAM.json",
    "kgw_runtime": root / "third_party" / "KGW-lm-watermarking.UPSTREAM.json",
}
default_external_roots = {
    "stone_runtime": "external_checkout/STONE-watermarking",
    "sweet_runtime": "external_checkout/sweet-watermark",
    "ewd_runtime": "external_checkout/EWD",
    "kgw_runtime": "external_checkout/lm-watermarking",
}
selected = entries if method in {"", "all"} else {method: entries[method]}
for runtime_name, manifest_path in selected.items():
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested_root = str(
        payload.get("local_external_root")
        or payload.get("external_root")
        or default_external_roots[runtime_name]
    ).strip()
    local_external_root = _sanitize_external_root(requested_root)
    if local_external_root is None:
        raise SystemExit(
            f"{manifest_path}: external checkout root must stay under external_checkout/ without hidden or parent traversal components: {requested_root!r}"
        )
    print("\t".join(
            [
                runtime_name,
                str(manifest_path),
                str(payload["repo_url"]),
                str(payload["pinned_commit"]),
                str(root / local_external_root),
                str(root / local_external_root),
            ]
        ))
PY
)" || exit $?
mapfile -t FETCH_ENTRIES <<<"$FETCH_ENTRIES_RAW"

for entry in "${FETCH_ENTRIES[@]}"; do
  IFS=$'\t' read -r RUNTIME_NAME MANIFEST_PATH REPO_URL PINNED_COMMIT TARGET MANAGED_ROOT <<<"$entry"
  echo "[fetch_runtime_upstreams] $RUNTIME_NAME -> $TARGET"

  if [ -e "$TARGET" ] && [ ! -d "$TARGET/.git" ] && [ ! -f "$TARGET/.git" ]; then
    echo "Target exists but is not a git checkout: $TARGET" >&2
    exit 1
  fi

  FRESH_CLONE=0
  if [ ! -d "$TARGET/.git" ] && [ ! -f "$TARGET/.git" ]; then
    mkdir -p "$(dirname "$TARGET")"
    git -c http.version=HTTP/1.1 clone --filter=blob:none --no-checkout "$REPO_URL" "$TARGET"
    FRESH_CLONE=1
  fi

  REMOTE_URL="$(git -C "$TARGET" remote get-url origin 2>/dev/null || true)"
  if [ -z "$REMOTE_URL" ]; then
    git -C "$TARGET" remote add origin "$REPO_URL"
    REMOTE_URL="$REPO_URL"
  fi

  if [ "$REMOTE_URL" != "$REPO_URL" ]; then
    echo "Remote mismatch for $RUNTIME_NAME" >&2
    echo "expected: $REPO_URL" >&2
    echo "actual:   $REMOTE_URL" >&2
    exit 1
  fi

  if [ "$FRESH_CLONE" != "1" ] && ! checkout_dirty "$TARGET"; then
    HEAD_COMMIT="$(git -C "$TARGET" rev-parse HEAD 2>/dev/null || true)"
    if [ "$HEAD_COMMIT" = "$PINNED_COMMIT" ]; then
      echo "[fetch_runtime_upstreams] ready: $RUNTIME_NAME @ $HEAD_COMMIT (existing checkout)"
      continue
    fi
  fi

  if [ "$FRESH_CLONE" != "1" ] && checkout_dirty "$TARGET"; then
    if ! safe_external_checkout_root "$TARGET" "$MANAGED_ROOT"; then
      echo "Checkout is dirty before refresh outside the managed external root: $TARGET" >&2
      exit 1
    fi
    echo "Checkout is dirty before refresh; recreating managed external checkout: $TARGET" >&2
    rm -rf "$TARGET"
    mkdir -p "$(dirname "$TARGET")"
    git -c http.version=HTTP/1.1 clone --filter=blob:none --no-checkout "$REPO_URL" "$TARGET"
  fi

  git -C "$TARGET" -c http.version=HTTP/1.1 fetch --depth 1 origin "$PINNED_COMMIT"
  git -C "$TARGET" checkout --force --detach FETCH_HEAD

  HEAD_COMMIT="$(git -C "$TARGET" rev-parse HEAD)"
  if [ "$HEAD_COMMIT" != "$PINNED_COMMIT" ]; then
    echo "Pinned commit mismatch for $RUNTIME_NAME" >&2
    echo "expected: $PINNED_COMMIT" >&2
    echo "actual:   $HEAD_COMMIT" >&2
    exit 1
  fi

  if checkout_dirty "$TARGET"; then
    echo "Checkout is dirty after refresh: $TARGET" >&2
    exit 1
  fi

  echo "[fetch_runtime_upstreams] ready: $RUNTIME_NAME @ $HEAD_COMMIT"
done
