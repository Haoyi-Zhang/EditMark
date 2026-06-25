from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from posteditbench.language_support import source_relative_to


ROOT = Path(__file__).resolve().parents[1]
RELEASE_SOURCES_ROOT = ROOT / "data" / "release" / "sources"


def _sanitize_path(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    candidate = source_relative_to(ROOT, Path(text))
    if ".coordination" in candidate.replace("/", "\\").lower():
        return Path(candidate).name
    return candidate


def _sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    for key in ("source_path", "official_problem_file"):
        if key in item:
            item[key] = _sanitize_path(item[key])
    return item


def _count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "")).strip().lower()
        if value:
            counts[value] = counts.get(value, 0) + 1
    return counts


def _reference_kind_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = _count(rows, "reference_kind")
    if "canonical" not in counts:
        counts["canonical"] = 0
    return dict(sorted(counts.items()))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            handle.write("\n")


def _sanitize_manifest(manifest: dict[str, Any], rows: list[dict[str, Any]], source_path: Path) -> dict[str, Any]:
    data = dict(manifest)
    data["source_path"] = _sanitize_path(data.get("source_path") or source_path)
    if data.get("normalized_path"):
        data["normalized_path"] = _sanitize_path(data["normalized_path"])
    if data.get("source_manifests") and isinstance(data["source_manifests"], list):
        sanitized_source_manifests = []
        for entry in data["source_manifests"]:
            if not isinstance(entry, dict):
                continue
            item = dict(entry)
            if item.get("source_path"):
                item["source_path"] = _sanitize_path(item["source_path"])
            sanitized_source_manifests.append(item)
        data["source_manifests"] = sanitized_source_manifests
    data["reference_kind_counts"] = _reference_kind_counts(rows)
    data["reference_kind_total"] = len(rows)
    data["canonical_reference_count"] = data["reference_kind_counts"].get("canonical", 0)
    data["smoke_overlay_reference_count"] = data["reference_kind_counts"].get("smoke_overlay", 0)
    return data


def _build_manifest(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    languages = sorted({str(row.get("language", "")).lower() for row in rows if str(row.get("language", "")).strip()})
    datasets = sorted({str(row.get("dataset", "")).strip() for row in rows if str(row.get("dataset", "")).strip()})
    validation_supported_languages = sorted({str(row.get("language", "")).lower() for row in rows if bool(row.get("validation_supported"))})
    source_group = str(rows[0].get("source_group", path.parent.name)) if rows else path.parent.name
    manifest = {
        "schema_version": 2,
        "benchmark": str(rows[0].get("public_source", path.parent.name)) if rows else path.parent.name,
        "dataset_label": datasets[0] if datasets else path.parent.name,
        "source_group": source_group,
        "source_url": str(rows[0].get("source_url", "")) if rows else "",
        "source_revision": str(rows[0].get("source_revision", "")) if rows else "",
        "source_archive_sha256": str(rows[0].get("source_sha256", "")) if rows else "",
        "source_path": str(path),
        "normalized_path": str(path),
        "split": str(rows[0].get("split", "test")) if rows else "test",
        "license_note": str(rows[0].get("license_note", "")) if rows else "",
        "adapter_name": str(rows[0].get("adapter_name", "")) if rows else "",
        "validation_scope": str(rows[0].get("validation_scope", "python_first")) if rows else "python_first",
        "task_count": len(rows),
        "expected_task_count": len(rows),
        "observed_languages": languages,
        "datasets": datasets,
        "validation_supported_languages": validation_supported_languages,
        "source_group_counts": {source_group: len(rows)} if source_group else {},
        "origin_type_counts": _count(rows, "origin_type"),
        "difficulty_counts": _count(rows, "difficulty"),
        "source_manifests": [],
        "notes": "",
    }
    if rows:
        manifest["source_manifests"] = [
            {
                "benchmark": manifest["benchmark"],
                "dataset_label": manifest["dataset_label"],
                "language": language,
                "license_note": manifest["license_note"],
                "source_group": source_group,
                "source_path": _sanitize_path(rows[0].get("source_path", "")),
                "source_revision": manifest["source_revision"],
                "source_url": manifest["source_url"],
                "split": manifest["split"],
                "task_count": sum(1 for row in rows if str(row.get("language", "")).lower() == language),
            }
            for language in languages
        ]
    return _sanitize_manifest(manifest, rows, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh release-source benchmark snapshots in place.")
    parser.add_argument(
        "--root",
        dest="roots",
        action="append",
        type=Path,
        help="Optional root containing *.normalized.jsonl release sources. Defaults to data/release/sources.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without rewriting files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = [path.resolve() for path in (args.roots or [RELEASE_SOURCES_ROOT])]
    targets = sorted({path for root in roots for path in root.glob("*.normalized.jsonl")})

    for target in targets:
        rows = _load_jsonl(target)
        if not rows:
            continue
        sanitized_rows = [_sanitize_row(row) for row in rows]
        manifest = {}
        manifest_path = target.with_suffix(".manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = _build_manifest(target, sanitized_rows)
        manifest = _sanitize_manifest(manifest, sanitized_rows, target)
        if args.dry_run:
            print(f"Would refresh {target} ({len(sanitized_rows)} rows)")
            continue
        _write_jsonl(target, sanitized_rows)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"Refreshed {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
