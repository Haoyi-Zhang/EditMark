from __future__ import annotations

import argparse
import json
from pathlib import Path

import export_full_run_tables as eft

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE_DIR = ROOT / "results" / "tables" / "suite_all_models_methods"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize repository-tracked materialized summary tables in place.")
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table_dir = args.table_dir.resolve()
    updated = 0
    scanned = 0
    for json_path in sorted(table_dir.glob("*.json")):
        scanned += 1
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            continue
        normalized_rows = [eft._presentation_row(dict(row)) for row in payload]
        if normalized_rows == payload:
            continue
        eft._write_rows_json(json_path, normalized_rows)
        csv_path = json_path.with_suffix(".csv")
        if csv_path.exists():
            eft._write_rows_csv(csv_path, normalized_rows)
        updated += 1
    print(
        json.dumps(
            {
                "table_dir": str(table_dir),
                "scanned_json_files": scanned,
                "updated_json_files": updated,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
