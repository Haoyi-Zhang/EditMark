from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _shared import dump_json, markdown_table, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PostEditBench benchmark reports.")
    parser.add_argument("--input", type=Path, required=True, help="Report JSON file or run directory containing report.json.")
    parser.add_argument("--output", type=Path, default=None, help="Destination JSON summary file.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of worst cases to include.")
    return parser.parse_args()


def resolve_report_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "report.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"{path} does not contain report.json")
    return path


def load_report(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def analyze_report(report: dict[str, Any], top_k: int) -> dict[str, Any]:
    rows = list(report.get("rows", []))
    summary = dict(report.get("summary", {}))
    ordered = sorted(rows, key=lambda row: (row.get("robustness_score", row.get("attacked_score", 0.0)), row.get("quality_score", 0.0)))
    semantic_rows = [row for row in rows if row.get("semantic_validation_available") and row.get("semantic_preserving")]
    budget_curves = summary.get("budget_curves", {})
    benchmark_manifest = summary.get("benchmark_manifest", {})
    return {
        "record_count": len(rows),
        "summary": summary,
        "top_failures": ordered[:top_k],
        "semantic_subset_count": len(semantic_rows),
        "budget_curves": budget_curves,
        "attack_breakdown": summary.get("by_attack") or summary.get("attack_breakdown", {}),
        "by_language": summary.get("by_language", {}),
        "semantic_attack_robustness": summary.get("semantic_attack_robustness", {}),
        "benchmark_manifest": benchmark_manifest,
        "coverage_gaps": summary.get("coverage_gaps", {}),
        "by_reference_kind": summary.get("by_reference_kind", {}),
        "by_baseline_family": summary.get("by_baseline_family", {}),
        "watermarked_functional_metrics": summary.get("watermarked_functional_metrics", {}),
    }


def main() -> int:
    args = parse_args()
    report_path = resolve_report_path(args.input)
    report = load_report(report_path)
    analysis = analyze_report(report, args.top_k)

    print(f"Analyzing {report_path} ({analysis['record_count']} rows)")
    if analysis["attack_breakdown"]:
        rows = [
            [
                attack,
                stats["count"],
                round(float(stats.get("mean_quality_score", stats.get("avg_quality", 0.0))), 4),
                round(float(stats.get("mean_robustness_score", 0.0)), 4),
            ]
            for attack, stats in sorted(analysis["attack_breakdown"].items())
        ]
        print("\nAttack overview")
        print(markdown_table(["attack", "count", "mean_quality", "mean_robustness"], rows))

    if analysis["by_language"]:
        rows = [
            [
                language,
                stats["count"],
                round(float(stats.get("mean_quality_score", stats.get("avg_quality", 0.0))), 4),
                round(float(stats.get("semantic_validation_rate", 0.0)), 4),
                round(float(stats.get("semantic_preservation_rate", 0.0)), 4),
            ]
            for language, stats in sorted(analysis["by_language"].items())
        ]
        print("\nLanguage overview")
        print(markdown_table(["language", "count", "mean_quality", "validation_rate", "semantic_preserve"], rows))

    declared_validation = summary.get("declared_semantic_validation_by_language", {})
    if declared_validation:
        rows = [
            [
                language,
                stats["count"],
                round(float(stats.get("declared_semantic_validation_rate", 0.0)), 4),
            ]
            for language, stats in sorted(declared_validation.items())
        ]
        print("\nDeclared validation overview")
        print(markdown_table(["language", "count", "declared_validation_rate"], rows))

    runtime_validation = summary.get("runtime_semantic_validation_by_language", {})
    if runtime_validation:
        rows = [
            [
                language,
                stats["count"],
                round(float(stats.get("semantic_validation_rate", 0.0)), 4),
                round(float(stats.get("semantic_preservation_rate", 0.0)), 4),
            ]
            for language, stats in sorted(runtime_validation.items())
        ]
        print("\nRuntime validation overview")
        print(markdown_table(["language", "count", "runtime_validation_rate", "semantic_preserve"], rows))

    if analysis.get("by_reference_kind"):
        rows = [
            [kind, stats["count"], round(float(stats.get("semantic_validation_rate", 0.0)), 4)]
            for kind, stats in sorted(analysis["by_reference_kind"].items())
        ]
        print("\nReference-kind overview")
        print(markdown_table(["reference_kind", "count", "validation_rate"], rows))

    if analysis.get("by_baseline_family"):
        rows = [
            [family, stats["count"], round(float(stats.get("semantic_validation_rate", 0.0)), 4)]
            for family, stats in sorted(analysis["by_baseline_family"].items())
        ]
        print("\nBaseline-family overview")
        print(markdown_table(["baseline_family", "count", "validation_rate"], rows))

    manifest = analysis["benchmark_manifest"]
    if manifest:
        summary = analysis["summary"]
        coverage = manifest.get("coverage", {})
        rows = [
            ["observed_languages", ", ".join(manifest.get("observed_languages", [])) or "-"],
            ["claimed_languages", ", ".join(manifest.get("claimed_languages", [])) or "-"],
            ["observed_coverage_rate", coverage.get("observed_coverage_rate", 0.0)],
            [
                "manifest_declared_semantic_validation_rate",
                coverage.get("declared_semantic_validation_rate", coverage.get("semantic_validation_rate", 0.0)),
            ],
            [
                "manifest_declared_semantic_validation_language_rate",
                coverage.get(
                    "declared_semantic_validation_language_rate",
                    coverage.get("semantic_validation_language_rate", 0.0),
                ),
            ],
            ["summary_declared_semantic_validation_rate", summary_metric(summary, "declared_semantic_validation_rate")],
            [
                "summary_declared_semantic_validation_language_rate",
                summary_metric(summary, "declared_semantic_validation_language_rate"),
            ],
            ["summary_runtime_semantic_validation_rate", summary_metric(summary, "runtime_semantic_validation_rate")],
            [
                "summary_runtime_semantic_validation_language_rate",
                summary_metric(summary, "runtime_semantic_validation_language_rate"),
            ],
        ]
        print("\nCoverage summary")
        print(markdown_table(["metric", "value"], rows))

    if analysis["semantic_attack_robustness"]:
        rows = [[attack, score] for attack, score in sorted(analysis["semantic_attack_robustness"].items())]
        print("\nSemantic-preserving robustness")
        print(markdown_table(["attack", "robustness"], rows))

    if analysis["budget_curves"]:
        rows = []
        for attack, curve in sorted(analysis["budget_curves"].items()):
            for point in curve:
                rows.append(
                    [
                        attack,
                        point["budget"],
                        point["count"],
                        point["mean_detector_score"],
                        point["mean_quality_score"],
                        point["semantic_preserving_rate"],
                    ]
                )
        print("\nBudget curves")
        print(markdown_table(["attack", "budget", "count", "detector", "quality", "semantic_rate"], rows))

    if analysis["top_failures"]:
        worst_rows = [
            [row["example_id"], row["attack_name"], row["attacked_score"], row["quality_score"], row["semantic_preserving"]]
            for row in analysis["top_failures"]
        ]
        print("\nWorst cases")
        print(markdown_table(["example", "attack", "attacked_score", "quality", "semantic_preserving"], worst_rows))

    if analysis["watermarked_functional_metrics"]:
        rows = [[key, value] for key, value in analysis["watermarked_functional_metrics"].items()]
        print("\nWatermarked functional metrics")
        print(markdown_table(["metric", "value"], rows))

    if args.output is not None:
        dump_json(args.output, analysis)
        print(f"Wrote analysis summary to {args.output}")
    return 0


def summary_metric(summary: dict[str, Any], key: str) -> Any:
    return summary.get(key, summary.get(key.replace("declared_", "").replace("runtime_", ""), 0.0))


if __name__ == "__main__":
    raise SystemExit(main())
