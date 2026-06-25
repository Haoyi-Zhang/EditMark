from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from posteditbench.models import BenchmarkRow
from posteditbench.report import resolve_benchmark_source_metadata


_BENCHMARK_ROW_FIELDS = {field.name for field in fields(BenchmarkRow)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh report summary metadata without recomputing experiment results.")
    parser.add_argument("--matrix-index", type=Path, required=True, help="Matrix index JSON for the finished full run.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print changes without writing files.")
    return parser.parse_args()


def _repo_path(path: str | Path, *, base_dir: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (base_dir / candidate)


def _row_from_payload(payload: Mapping[str, Any]) -> BenchmarkRow:
    data = {key: value for key, value in payload.items() if key in _BENCHMARK_ROW_FIELDS}
    data.setdefault("metadata", {})
    return BenchmarkRow(**data)


def _row_source_groups(rows: list[BenchmarkRow]) -> list[str]:
    values = {str(row.source_group).strip() or str(row.dataset).strip() for row in rows}
    return sorted(value for value in values if value)


def _manifest_source_groups(manifest: Mapping[str, Any]) -> list[str]:
    source_group_counts = manifest.get("source_group_counts", {})
    if isinstance(source_group_counts, Mapping) and source_group_counts:
        return sorted(str(source_group).strip() for source_group in source_group_counts.keys() if str(source_group).strip())
    source_groups = manifest.get("source_groups", [])
    if isinstance(source_groups, list):
        return sorted(str(source_group).strip() for source_group in source_groups if str(source_group).strip())
    return []


def _configured_source(report: Mapping[str, Any]) -> str:
    config = dict(report.get("config", {}))
    corpus_parameters = dict(config.get("corpus_parameters", {}))
    return str(corpus_parameters.get("public_source") or corpus_parameters.get("source") or "").strip()


def _refresh_report(report_path: Path) -> tuple[bool, str]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows_payload = payload.get("rows", [])
    if not isinstance(rows_payload, list):
        raise ValueError(f"{report_path}: rows payload is not a list")
    rows = [_row_from_payload(row) for row in rows_payload if isinstance(row, Mapping)]
    summary = dict(payload.get("summary", {}))
    manifest = dict(summary.get("benchmark_manifest", {}))
    manifest_sources = _manifest_source_groups(manifest)
    row_sources = _row_source_groups(rows)
    if manifest_sources and row_sources and manifest_sources != row_sources:
        raise ValueError(
            f"{report_path}: manifest/row source mismatch "
            f"(manifest={manifest_sources}, rows={row_sources})"
        )
    metadata = resolve_benchmark_source_metadata(
        rows,
        benchmark_manifest=manifest,
        configured_source=_configured_source(payload),
    )
    changed = False
    for key, value in metadata.items():
        if summary.get(key) != value:
            summary[key] = value
            changed = True
    if changed:
        payload["summary"] = summary
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return changed, metadata["benchmark_source"]


def main() -> int:
    args = _parse_args()
    matrix_index_path = args.matrix_index.resolve()
    base_dir = matrix_index_path.parents[3]
    matrix_index = json.loads(matrix_index_path.read_text(encoding="utf-8"))
    runs = matrix_index.get("runs", [])
    changed = 0
    checked = 0
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        report_path_value = run.get("report_path")
        if not report_path_value:
            continue
        report_path = _repo_path(report_path_value, base_dir=base_dir)
        checked += 1
        if args.dry_run:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            summary = dict(payload.get("summary", {}))
            rows = [_row_from_payload(row) for row in payload.get("rows", []) if isinstance(row, Mapping)]
            manifest = dict(summary.get("benchmark_manifest", {}))
            manifest_sources = _manifest_source_groups(manifest)
            row_sources = _row_source_groups(rows)
            if manifest_sources and row_sources and manifest_sources != row_sources:
                raise ValueError(
                    f"{report_path}: manifest/row source mismatch "
                    f"(manifest={manifest_sources}, rows={row_sources})"
                )
            metadata = resolve_benchmark_source_metadata(
                rows,
                benchmark_manifest=manifest,
                configured_source=_configured_source(payload),
            )
            print(f"[dry-run] {report_path}: {summary.get('benchmark_source')} -> {metadata['benchmark_source']}")
            continue
        report_changed, benchmark_source = _refresh_report(report_path)
        changed += 1 if report_changed else 0
        print(f"{report_path}: benchmark_source={benchmark_source} changed={report_changed}")
    print(f"checked={checked} changed={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

