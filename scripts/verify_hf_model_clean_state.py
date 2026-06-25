from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _hf_readiness import cache_entry_paths, resolve_cache_roots

_ARTIFACT_SUFFIXES = (
    "",
    ".py",
    ".log",
    ".status",
    ".pid",
    ".json",
    ".jsonl",
    ".state",
    ".state.json",
    ".tmp",
    ".tar",
    ".tar.tmp",
    ".tar.part",
    ".part",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail closed unless a Hugging Face model cache entry is fully absent, along with "
            "matching lock files, incomplete downloads, and optional process/path residuals."
        )
    )
    parser.add_argument("--model", required=True, help="Model ID, for example Qwen/Qwen2.5-Coder-7B-Instruct.")
    parser.add_argument("--cache-dir", required=True, help="Hugging Face cache root.")
    parser.add_argument(
        "--process-pattern",
        action="append",
        default=[],
        help="Optional process substring that must not appear in `ps -ef`.",
    )
    parser.add_argument(
        "--extra-path",
        action="append",
        default=[],
        help="Optional extra file or directory path that must not exist.",
    )
    parser.add_argument(
        "--artifact-prefix",
        action="append",
        default=[],
        help="Optional artifact prefix; common relay/smoke residue suffixes under this prefix must not exist.",
    )
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path.")
    return parser.parse_args()


def _matching_processes(pattern: str, *, exclude_pid: int | None = None) -> list[str]:
    if not str(pattern).strip():
        return []
    completed = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="],
        check=False,
        capture_output=True,
        text=True,
    )
    matches: list[str] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=2)
        if len(parts) < 3:
            continue
        pid, _ppid, args = parts
        if pattern not in args:
            continue
        if exclude_pid is not None and pid == str(exclude_pid):
            continue
        matches.append(stripped)
    return matches


def _artifact_family(prefix: Path) -> list[Path]:
    entries: list[Path] = []
    seen: set[str] = set()
    for suffix in _ARTIFACT_SUFFIXES:
        candidate = Path(str(prefix) + suffix)
        normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        entries.append(candidate)
    return entries


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def inspect_clean_state(
    *,
    model: str,
    cache_dir: str | Path,
    process_patterns: Sequence[str] = (),
    extra_paths: Sequence[str | Path] = (),
    artifact_prefixes: Sequence[str | Path] = (),
) -> dict[str, Any]:
    cache_root = Path(cache_dir)
    root_cache, hub_cache = resolve_cache_roots(str(cache_root))
    root_entry, hub_entry = cache_entry_paths(model, str(cache_root))
    model_token = model.replace("/", "--")
    lock_matches_set: set[str] = set()
    for lock_dir in {root_cache / ".locks", hub_cache / ".locks"}:
        if not lock_dir.exists():
            continue
        for path in lock_dir.rglob(f"*{model_token}*"):
            lock_matches_set.add(str(path))
    lock_matches = sorted(lock_matches_set)
    incomplete_matches = sorted(
        str(path)
        for path in cache_root.rglob("*.incomplete")
        if model_token in str(path)
    )
    process_matches = {
        str(pattern): _matching_processes(str(pattern), exclude_pid=os.getpid())
        for pattern in process_patterns
        if str(pattern).strip()
    }
    extra_path_entries = [Path(str(path)) for path in extra_paths if str(path).strip()]
    existing_extra_paths = [str(path) for path in extra_path_entries if _path_present(path)]
    artifact_prefix_entries = [Path(str(path)) for path in artifact_prefixes if str(path).strip()]
    existing_artifact_family_paths = sorted(
        {
            str(path)
            for prefix in artifact_prefix_entries
            for path in _artifact_family(prefix)
            if _path_present(path)
        }
    )

    issues: list[str] = []
    if _path_present(root_entry):
        issues.append(f"root cache entry still exists: {root_entry}")
    if _path_present(hub_entry):
        issues.append(f"hub cache entry still exists: {hub_entry}")
    if lock_matches:
        issues.append(f"lock files still exist: {lock_matches}")
    if incomplete_matches:
        issues.append(f"incomplete downloads still exist: {incomplete_matches}")
    for pattern, matches in process_matches.items():
        if matches:
            issues.append(f"matching process still running for {pattern!r}: {matches}")
    if existing_extra_paths:
        issues.append(f"extra paths still exist: {existing_extra_paths}")
    if existing_artifact_family_paths:
        issues.append(f"artifact-prefix residuals still exist: {existing_artifact_family_paths}")

    return {
        "model": model,
        "cache_dir": str(cache_root),
        "root_entry": str(root_entry),
        "hub_entry": str(hub_entry),
        "root_entry_exists": _path_present(root_entry),
        "hub_entry_exists": _path_present(hub_entry),
        "lock_matches": lock_matches,
        "incomplete_matches": incomplete_matches,
        "process_matches": process_matches,
        "existing_extra_paths": existing_extra_paths,
        "artifact_prefixes": [str(path) for path in artifact_prefix_entries],
        "existing_artifact_family_paths": existing_artifact_family_paths,
        "status": "ok" if not issues else "failed",
        "issues": issues,
    }


def main() -> int:
    args = parse_args()
    payload = inspect_clean_state(
        model=args.model,
        cache_dir=args.cache_dir,
        process_patterns=args.process_pattern,
        extra_paths=args.extra_path,
        artifact_prefixes=args.artifact_prefix,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
