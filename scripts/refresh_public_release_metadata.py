from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RELEASE_SOURCES_ROOT = ROOT / "data" / "release" / "sources"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh release-facing benchmark rows/manifests with scrubbed path labels."
    )
    parser.add_argument(
        "--root",
        dest="roots",
        action="append",
        type=Path,
        help="Benchmark root containing */normalized.jsonl files. Defaults to the public and canonical release roots.",
    )
    return parser.parse_args()


def _scrub_path(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    lowered = text.lower()
    if "data/interim/public_snapshots/_cache/" in lowered:
        return Path(text).name
    if ".coordination/" in lowered:
        return Path(text).name
    return text


def _normalize_repo_relative_path(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    lowered = text.lower()
    for marker in ("data/release/", "results/", "docs/", "scripts/", "third_party/"):
        index = lowered.find(marker)
        if index >= 0:
            return text[index:]
    return text


def _refresh_row(row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    if "source_path" in updated:
        updated["source_path"] = _scrub_path(updated.get("source_path"))
    if "official_problem_file" in updated:
        updated["official_problem_file"] = _scrub_path(updated.get("official_problem_file"))
    return updated


def _refresh_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    updated = dict(payload)
    if "source_path" in updated:
        updated["source_path"] = _scrub_path(updated.get("source_path"))
    if "normalized_path" in updated:
        updated["normalized_path"] = _normalize_repo_relative_path(updated.get("normalized_path"))
    if "sample_ids_path" in updated:
        updated["sample_ids_path"] = _normalize_repo_relative_path(updated.get("sample_ids_path"))
    source_manifests = []
    for item in list(updated.get("source_manifests", []) or []):
        if isinstance(item, dict):
            refreshed = dict(item)
            if "source_path" in refreshed:
                refreshed["source_path"] = _scrub_path(refreshed.get("source_path"))
            source_manifests.append(refreshed)
        else:
            source_manifests.append(item)
    if source_manifests:
        updated["source_manifests"] = source_manifests
    return updated


def main() -> int:
    args = parse_args()
    roots = [path.resolve() for path in (args.roots or [RELEASE_SOURCES_ROOT])]
    normalized_files_set: set[Path] = set()
    for root in roots:
        normalized_files_set.update(root.glob("*/normalized.jsonl"))
        normalized_files_set.update(root.glob("*.normalized.jsonl"))
    normalized_files = sorted(normalized_files_set)
    updated_rows_files = 0
    updated_manifest_files = 0
    for normalized in normalized_files:
        rows = [json.loads(line) for line in normalized.read_text(encoding="utf-8").splitlines() if line.strip()]
        refreshed_rows = [_refresh_row(row) for row in rows]
        if refreshed_rows != rows:
            normalized.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in refreshed_rows) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            updated_rows_files += 1
        manifest_path = normalized.with_suffix(".manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            refreshed_manifest = _refresh_manifest(manifest)
            if refreshed_manifest != manifest:
                manifest_path.write_text(
                    json.dumps(refreshed_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                updated_manifest_files += 1
    print(
        json.dumps(
            {
                "roots": [str(root) for root in roots],
                "normalized_files": len(normalized_files),
                "updated_rows_files": updated_rows_files,
                "updated_manifest_files": updated_manifest_files,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
