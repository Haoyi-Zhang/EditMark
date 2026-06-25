from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from posteditbench.baselines import runtime_family_names
from posteditbench.baselines.stone_family import runtime_watermark_names as stone_runtime_watermark_names
from posteditbench.config import build_experiment_config, load_config
from posteditbench.pipeline import run_experiment
from posteditbench.utils import scrub_paths


FAMILY_TO_WATERMARKS = {
    "runtime_official": stone_runtime_watermark_names(),
    "stone_family": stone_runtime_watermark_names(),
}

DEFAULT_CONFIG_BY_FAMILY = {
    "runtime_official": Path("configs/public_humaneval_plus_stone_runtime.yaml"),
    "stone_family": Path("configs/public_humaneval_plus_stone_runtime.yaml"),
}

DEFAULT_CONFIG_BY_WATERMARK = {
    "stone_runtime": Path("configs/public_humaneval_plus_stone_runtime.yaml"),
    "sweet_runtime": Path("configs/public_humaneval_plus_sweet_runtime.yaml"),
    "ewd_runtime": Path("configs/public_humaneval_plus_ewd_runtime.yaml"),
    "kgw_runtime": Path("configs/public_humaneval_plus_kgw_runtime.yaml"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline families from a single config template.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Base runtime config. The script overrides only the watermark scheme and output path.",
    )
    parser.add_argument(
        "--watermark",
        action="append",
        dest="watermarks",
        choices=runtime_family_names(include_extensions=False),
        default=None,
        help="Runtime watermark to include. Repeat to restrict the family.",
    )
    parser.add_argument(
        "--family",
        choices=tuple(FAMILY_TO_WATERMARKS),
        default="runtime_official",
        help="Runtime family to expand when --watermark is omitted. 'stone_family' remains a backward-compatible alias.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/runs/runtime_family"),
        help="Directory that will receive one subdirectory per runtime baseline.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the experiment seed.")
    parser.add_argument("--count", type=int, default=None, help="Override the benchmark example limit.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned matrix and stop.")
    return parser.parse_args()


def _selected_watermarks(values: list[str] | None, family: str) -> tuple[str, ...]:
    if not values:
        return FAMILY_TO_WATERMARKS[str(family)]
    seen: dict[str, None] = {}
    for value in values:
        seen[str(value).lower()] = None
    return tuple(seen)


def _default_config_path(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return args.config
    return DEFAULT_CONFIG_BY_FAMILY[str(args.family)]


def _default_config_path_for_watermark(args: argparse.Namespace, watermark_name: str) -> Path:
    if args.config is not None:
        return args.config
    return DEFAULT_CONFIG_BY_WATERMARK.get(str(watermark_name), _default_config_path(args))


def _build_runtime_raw(base_raw: dict[str, object], *, watermark_name: str, seed: int | None, count: int | None) -> dict[str, object]:
    raw = deepcopy(base_raw)
    project = dict(raw.get("project", {})) if isinstance(raw.get("project"), dict) else {}
    benchmark = dict(raw.get("benchmark", {})) if isinstance(raw.get("benchmark"), dict) else {}
    watermark = dict(raw.get("watermark", {})) if isinstance(raw.get("watermark"), dict) else {}

    base_name = str(project.get("name", "posteditbench-runtime-family")).strip() or "posteditbench-runtime-family"
    if not base_name.endswith(f"-{watermark_name}"):
        project["name"] = f"{base_name}-{watermark_name}"
    watermark["scheme"] = watermark_name
    if seed is not None:
        project["seed"] = seed
    if count is not None:
        benchmark["limit"] = count

    raw["project"] = project
    raw["benchmark"] = benchmark
    raw["watermark"] = watermark
    return raw


def _output_token(raw: dict[str, object]) -> str:
    benchmark = raw.get("benchmark", {})
    if isinstance(benchmark, dict):
        limit = benchmark.get("limit")
        if limit is None:
            return "full"
        return str(int(limit))
    return "full"


def main() -> int:
    args = parse_args()
    payload = []
    for watermark_name in _selected_watermarks(args.watermarks, args.family):
        config_path = _default_config_path_for_watermark(args, watermark_name)
        source = load_config(config_path)
        if not source.raw:
            raise ValueError("runtime family runner requires a non-empty base config")
        raw = _build_runtime_raw(source.raw, watermark_name=watermark_name, seed=args.seed, count=args.count)
        output_path = args.output_root / watermark_name / "report.json"
        config = build_experiment_config(raw, output_path=str(output_path))

        if args.dry_run:
            payload.append(
                {
                    "config": str(config_path),
                    "watermark": watermark_name,
                    "family": args.family,
                    "output_path": scrub_paths(str(output_path)),
                    "seed": config.seed,
                    "corpus_size": config.corpus_size,
                    "benchmark_limit_mode": _output_token(raw),
                    "benchmark_path": scrub_paths(str(config.corpus_parameters.get("prepared_benchmark", ""))),
                }
            )
            continue

        result = run_experiment(config)
        payload.append(
            {
                "watermark": watermark_name,
                "family": args.family,
                "output_path": scrub_paths(str(result.report.output_path or output_path)),
                "example_count": len(result.examples),
                "row_count": len(result.report.rows),
                "semantic_validation_rate": result.report.summary.get("semantic_validation_rate", 0.0),
            }
        )

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
