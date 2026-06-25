from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from posteditbench.config import build_experiment_config, load_config, validate_experiment_config
from posteditbench.pipeline import run_experiment
from posteditbench.utils import ensure_parent, scrub_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PostEditBench benchmark pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON-compatible YAML config file. Use an explicit canonical runtime config such as configs/public_humaneval_plus_stone_runtime.yaml.",
    )
    parser.add_argument("--benchmark", type=Path, default=None, help="Override the canonical normalized benchmark fixture.")
    parser.add_argument("--output", type=Path, default=None, help="Report JSON file or output directory.")
    parser.add_argument("--seed", type=int, default=None, help="Override the benchmark seed.")
    parser.add_argument("--count", type=int, default=None, help="Override the benchmark corpus size.")
    parser.add_argument("--attack", action="append", dest="attacks", default=None, help="Restrict to one or more registry attacks.")
    parser.add_argument("--watermark", type=str, default=None, help="Override the watermark registry name.")
    parser.add_argument("--provider-mode", type=str, default=None, help="Completion provider mode (offline_mock, local_hf, local_command).")
    parser.add_argument("--provider-model", type=str, default=None, help="Model name for provider-backed runs.")
    parser.add_argument("--provider-command", type=str, default=None, help="Shell command for local-command providers.")
    parser.add_argument("--provider-timeout", type=float, default=None, help="Provider timeout in seconds.")
    parser.add_argument("--provider-temperature", type=float, default=None, help="Provider sampling temperature.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned run and stop before execution.")
    return parser.parse_args()


def _default_output_path(config_path: Path, raw_config: dict[str, Any]) -> Path:
    project = raw_config.get("project", {})
    benchmark = raw_config.get("benchmark", {})
    name = str(project.get("name", config_path.stem)).replace(" ", "-")
    seed = int(project.get("seed", raw_config.get("seed", 2026)))
    limit = benchmark.get("limit")
    corpus = "full" if limit is None else str(int(limit))
    return Path("results") / "runs" / f"{name}-{seed}-{corpus}" / "report.json"


def _benchmark_limit_mode(raw_config: dict[str, Any]) -> str:
    benchmark = raw_config.get("benchmark", {})
    if isinstance(benchmark, dict) and benchmark.get("limit") is None:
        return "full"
    return "sample"


def _resolve_output_path(config_path: Path, raw_config: dict[str, Any], output: Path | None) -> Path:
    if output is None:
        return _default_output_path(config_path, raw_config)
    if output.suffix.lower() == ".json":
        return output
    return output / "report.json"


def main() -> int:
    args = parse_args()
    source = load_config(args.config)
    overrides: dict[str, Any] = {}
    benchmark_overrides: dict[str, Any] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.count is not None:
        benchmark_overrides["limit"] = args.count
    if args.attacks is not None:
        overrides["attacks"] = tuple(args.attacks)
    if args.watermark is not None:
        overrides["watermark_name"] = args.watermark
    provider_overrides: dict[str, Any] = {}
    if args.provider_mode is not None:
        provider_overrides["mode"] = args.provider_mode
    provider_parameters: dict[str, Any] = {}
    if args.provider_model is not None:
        provider_parameters["model"] = args.provider_model
    if args.provider_command is not None:
        provider_parameters["command"] = args.provider_command
    if args.provider_timeout is not None:
        provider_parameters["timeout"] = args.provider_timeout
    if args.provider_temperature is not None:
        provider_parameters["temperature"] = args.provider_temperature
    if provider_parameters:
        provider_overrides["parameters"] = provider_parameters
    if provider_overrides:
        overrides["provider"] = provider_overrides
    if benchmark_overrides:
        existing_benchmark = dict(source.raw.get("benchmark", {}))
        existing_benchmark.update(benchmark_overrides)
        overrides["benchmark"] = existing_benchmark
    if args.benchmark is not None:
        benchmark_override = dict(overrides.get("benchmark", source.raw.get("benchmark", {})))
        benchmark_override.update({"prepared_output": str(args.benchmark), "source": str(args.benchmark)})
        overrides["benchmark"] = benchmark_override

    merged_raw_config = dict(source.raw)
    if "benchmark" in overrides:
        merged_raw_config["benchmark"] = dict(overrides["benchmark"])
    merged_raw_config.update({key: value for key, value in overrides.items() if key != "benchmark"})
    report_path = _resolve_output_path(args.config, merged_raw_config, args.output)
    config = build_experiment_config(source, output_path=str(report_path), **overrides)

    issues = validate_experiment_config(config)
    if issues:
        raise SystemExit("; ".join(issues))

    if args.dry_run:
        planned = {
            "config": str(args.config),
            "output_path": str(report_path),
            "seed": config.seed,
            "corpus_size": config.corpus_size,
            "benchmark_limit_mode": _benchmark_limit_mode(source.raw if hasattr(source, "raw") else {}),
            "watermark_name": config.watermark_name,
            "attacks": list(config.attacks),
            "provider_mode": config.provider_mode,
            "provider_summary": dict(config.metadata.get("provider_summary", {})) if isinstance(config.metadata, dict) else {},
        }
        print(json.dumps(planned, indent=2, sort_keys=True))
        return 0

    result = run_experiment(config)
    ensure_parent(result.report.output_path or report_path)
    print(
        json.dumps(
            {
                "output_path": scrub_paths(str(result.report.output_path or report_path)),
                "example_count": len(result.examples),
                "row_count": len(result.report.rows),
                "semantic_validation_rate": result.report.summary.get("semantic_validation_rate", 0.0),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
