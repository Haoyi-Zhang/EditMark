from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from posteditbench.public_benchmarks import available_public_sources, prepare_public_benchmark, resolve_public_source_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and normalize public benchmark snapshots.")
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        default=None,
        help="Public benchmark source to fetch (repeatable). Defaults to all registered sources.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/interim/public_snapshots"),
        help="Root directory for fetched normalized snapshots before release slicing.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/interim/public_snapshots/_cache"),
        help="Directory used to store downloaded archives.",
    )
    parser.add_argument("--fetch", action="store_true", help="Download the upstream archive when the cached copy is missing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = args.sources or list(available_public_sources())
    for name in sources:
        resolved = resolve_public_source_name(name)
        output = args.output_root / resolved / "normalized.jsonl"
        manifest_path = output.with_suffix(".manifest.json")
        if output.exists() and manifest_path.exists() and not args.fetch:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            print(f"{resolved}: {manifest['task_count']} tasks -> {output}")
            continue
        manifest = prepare_public_benchmark(resolved, output_path=output, fetch=args.fetch, cache_dir=args.cache_root / resolved)
        print(f"{resolved}: {manifest['task_count']} tasks -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
