from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

EXCLUDED_PREFIXES = (
    ".git/",
    ".coordination/",
    ".venv/",
    ".pytest_cache/",
    "external_checkout/",
    "model_cache/",
    "tmp/",
    "_remote_preview_figures/",
    "_review_outputs/",
    "results/archive/",
    "results/release_bundle/",
    "results/test_release_bundle/",
    "results/fetched_suite/",
    "results/tmp/",
    "results/article_preflight/",
    "results/matrix/",
    "results/matrix_shards/",
    "results/audits/",
    "results/certifications/",
    "results/environment/",
    "results/figures/",
    "results/launchers/",
    "results/tables/",
)
EXCLUDED_NAMES = {".DS_Store"}
EXCLUDED_DIR_NAMES = {"__pycache__", ".git"}


def _is_excluded(rel: str, path: Path) -> bool:
    if path.name in EXCLUDED_NAMES:
        return True
    if any(part.startswith(".tmp_pytest_") for part in path.parts):
        return True
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return True
    return any(rel == prefix[:-1] or rel.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def repo_snapshot_entries(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if _is_excluded(rel, path):
            continue
        if path.is_symlink():
            entries.append({"path": rel, "symlink": os.readlink(path)})
            continue
        if not path.is_file():
            continue
        entries.append(
            {
                "path": rel,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
        )
    return entries


def repo_snapshot_sha256(root: Path) -> str:
    payload = json.dumps(repo_snapshot_entries(root), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
