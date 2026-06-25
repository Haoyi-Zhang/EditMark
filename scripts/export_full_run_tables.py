from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from posteditbench.models import BenchmarkRow
from posteditbench.attacks.registry import attack_tier
from posteditbench.leaderboards import (
    GENERATION_TIME_TRACK,
    build_suite_method_master_leaderboard,
    build_suite_method_model_leaderboard,
)
from posteditbench.report import (
    _attacked_functional_metrics,
    _clean_functional_metrics,
    _stage_timing_metrics,
    _watermarked_functional_metrics,
)
from posteditbench.scorecard import HEADLINE_SCORE_FIELD, scorecard_for_rows
from posteditbench.suite import (
    OFFICIAL_RUNTIME_BASELINES,
    SUITE_AGGREGATE_SOURCES,
    SUITE_AGGREGATE_SOURCE_GROUPS,
    CANONICAL_SUITE_MODELS,
    SUITE_MODEL_ROSTER,
    normalize_source_group,
)

try:
    import _repo_snapshot
except Exception:  # pragma: no cover - keep export usable in minimal script copies.
    _repo_snapshot = None  # type: ignore[assignment]


_BENCHMARK_ROW_FIELDS = {field.name for field in fields(BenchmarkRow)}
# Public-facing headline metric name. The exported exact-value tables still use
# HEADLINE_SCORE_FIELD as the schema key, but the canonical paper/reviewer label
# for that field is PostEditScore.
PUBLIC_HEADLINE_METRIC = "PostEditScore"
_GENERALIZATION_STATUS_ORDER = {
    "unsupported": 0,
    "supported_zero": 1,
    "supported_nonzero": 2,
}
_ROLLUP_SCORECARD_SEMANTICS = "scorecard_recomputed_from_grouped_benchmark_rows"
_MODEL_SUMMARY_SCORE_SEMANTICS = "descriptive_mean_of_model_method_scorecard_rollups"
_PROJECTED_MODEL_METHOD_SCORE_SEMANTICS = "projected_from_model_method_scorecard_rollup"
_TIMING_ONLY_SCORE_SEMANTICS = "generation_stage_timing_descriptive_only"
_RUN_INVENTORY_SCORE_SEMANTICS = "per_run_scorecard_from_report_rows"
_LEADERBOARD_SCORE_SEMANTICS = "scorecard_recomputed_from_grouped_benchmark_rows"

_PRESENTATION_DATASET_LABELS = {
    "CraftedOriginal": "Crafted Original",
    "CraftedTranslation": "Crafted Translation",
    "CraftedStress": "Crafted Stress",
    "HumanEval-X": "HumanEval-X (5-language balanced slice)",
    "MBXP 5-language subset": "MBXP-5lang (5-language balanced slice)",
    "MBXP-5lang": "MBXP-5lang (5-language balanced slice)",
}
_PRESENTATION_METHOD_LABELS = {
    "stone_runtime": "STONE",
    "sweet_runtime": "SWEET",
    "ewd_runtime": "EWD",
    "kgw_runtime": "KGW",
}
_MODEL_ORDER = {model: index for index, model in enumerate(SUITE_MODEL_ROSTER)}
_METHOD_ORDER = {
    **{method: index for index, method in enumerate(OFFICIAL_RUNTIME_BASELINES)},
}
_CANONICAL_SUITE_MANIFEST = "configs/matrices/suite_all_models_methods.json"
_CANONICAL_SUITE_PROFILE = "suite_all_models_methods"
_CANONICAL_SUITE_MANIFEST_DIGEST = hashlib.sha256((ROOT / _CANONICAL_SUITE_MANIFEST).read_bytes()).hexdigest()
_FORMAL_CANONICAL_EXECUTION_MODE = "single_host_canonical"
_EXPECTED_SHARDED_EXECUTION_MODE = "sharded_identical_execution_class"
_KNOWN_EXECUTION_MODES = {_FORMAL_CANONICAL_EXECUTION_MODE, _EXPECTED_SHARDED_EXECUTION_MODE}
_SUMMARY_EXPORT_IDENTITY_STEM = f"{_CANONICAL_SUITE_PROFILE}_export_identity"
_SUMMARY_EXPORT_IDENTITY_TABLES = (
    "method_summary.json",
    "suite_all_models_methods_method_master_leaderboard.json",
    "suite_all_models_methods_method_model_leaderboard.json",
    "suite_all_models_methods_model_method_functional_quality.json",
    "per_attack_robustness_breakdown.csv",
    "per_attack_robustness_breakdown.json",
    "core_vs_stress_robustness_summary.csv",
    "core_vs_stress_robustness_summary.json",
    "robustness_factor_decomposition.csv",
    "robustness_factor_decomposition.json",
    "utility_factor_decomposition.csv",
    "utility_factor_decomposition.json",
    "generalization_axis_breakdown.csv",
    "generalization_axis_breakdown.json",
    "gate_decomposition.csv",
    "gate_decomposition.json",
)
_PUBLICATION_FIGURE_STEMS = (
    "suite_all_models_methods_score_decomposition",
    "suite_all_models_methods_detection_vs_utility",
)
_RELEASE_ENVIRONMENT_CAPTURE = Path("results/environment/runtime_environment.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export full-run aggregate tables from existing report.json files.")
    parser.add_argument("--matrix-index", type=Path, required=True, help="Finished matrix_index.json path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV/JSON summary tables.")
    return parser.parse_args()


def _repo_path(path: str | Path, *, base_dir: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (base_dir / candidate)


def _exportable_runs(matrix_index: Mapping[str, Any], *, base_dir: Path) -> list[Mapping[str, Any]]:
    exportable: list[Mapping[str, Any]] = []
    blocked: list[str] = []
    raw_runs = [run for run in matrix_index.get("runs", []) if isinstance(run, Mapping)]
    expected_count = int(matrix_index.get("run_count", len(raw_runs)) or len(raw_runs))
    for run in raw_runs:
        run_id = str(run.get("run_id", "")).strip() or "<unknown>"
        status = str(run.get("status", "")).strip()
        reason = str(run.get("reason", "")).strip()
        report_path = str(run.get("report_path", "")).strip()
        if status == "success":
            if reason == "resume_existing_report":
                blocked.append(f"{run_id}:resume_existing_report")
                continue
            if report_path and _repo_path(report_path, base_dir=base_dir).exists():
                exportable.append(run)
                continue
            blocked.append(f"{run_id}:success_without_report")
            continue
        blocked.append(f"{run_id}:{status or 'missing_status'}")
    if blocked or len(exportable) != expected_count:
        preview = ", ".join(blocked[:8])
        if len(blocked) > 8:
            preview += ", ..."
        raise SystemExit(
            "export_full_run_tables requires a complete result-backed matrix with only success runs. "
            f"exportable={len(exportable)} expected={expected_count} blocked=[{preview}]"
        )
    if not exportable:
        raise SystemExit("export_full_run_tables found no exportable runs in the matrix index.")
    return exportable


def _row_from_payload(payload: Mapping[str, Any]) -> BenchmarkRow:
    data = {key: value for key, value in payload.items() if key in _BENCHMARK_ROW_FIELDS}
    data.setdefault("metadata", {})
    return BenchmarkRow(**data)


def _round(value: float) -> float:
    return round(float(value), 4)


def _nullable_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_rows_json(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(rows, indent=2, ensure_ascii=False))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_environment_fingerprint(matrix_index: Mapping[str, Any]) -> str:
    fallback = str(matrix_index.get("execution_environment_fingerprint", "")).strip()
    environment_path = ROOT / _RELEASE_ENVIRONMENT_CAPTURE
    if not environment_path.exists():
        return fallback
    try:
        payload = json.loads(environment_path.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    execution = payload.get("execution", {}) if isinstance(payload, Mapping) else {}
    if isinstance(execution, Mapping):
        release_fingerprint = str(execution.get("execution_environment_fingerprint", "")).strip()
        if release_fingerprint:
            return release_fingerprint
    return fallback


def _presentation_text(value: str) -> str:
    normalized = str(value)
    for raw, pretty in _PRESENTATION_DATASET_LABELS.items():
        if raw == pretty:
            continue
        if pretty in normalized:
            continue
        normalized = normalized.replace(raw, pretty)
    return normalized


def _presentation_dataset_label(value: Any) -> Any:
    raw = str(value).strip()
    if not raw:
        return value
    return _PRESENTATION_DATASET_LABELS.get(raw, _presentation_text(raw))


def _presentation_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if isinstance(normalized.get("datasets"), list):
        normalized["datasets"] = [_presentation_dataset_label(value) for value in normalized["datasets"]]
    elif isinstance(normalized.get("datasets"), str):
        normalized["datasets"] = _presentation_text(normalized["datasets"])
    if isinstance(normalized.get("dataset"), str):
        normalized["dataset"] = _presentation_dataset_label(normalized["dataset"])
    if isinstance(normalized.get("method"), str):
        raw_method = normalized["method"].strip()
        normalized["method"] = _PRESENTATION_METHOD_LABELS.get(raw_method, raw_method)
    if isinstance(normalized.get("methods"), list):
        normalized["methods"] = [
            _PRESENTATION_METHOD_LABELS.get(str(value).strip(), str(value).strip())
            for value in normalized["methods"]
        ]
    return normalized


def _write_table(output_dir: Path, stem: str, rows: list[dict[str, Any]]) -> None:
    normalized_rows = [_presentation_row(row) for row in rows]
    _write_rows_json(output_dir / f"{stem}.json", normalized_rows)
    _write_rows_csv(output_dir / f"{stem}.csv", normalized_rows)


def _write_summary_export_identity(
    output_dir: Path,
    *,
    matrix_index_path: Path,
    matrix_index: Mapping[str, Any],
    score_version: str,
    observed_models: list[str],
) -> None:
    table_hashes: dict[str, str] = {}
    for filename in _SUMMARY_EXPORT_IDENTITY_TABLES:
        path = output_dir / filename
        if not path.exists():
            raise SystemExit(
                f"summary export identity requires {path} to exist before the identity sidecar is written."
            )
        table_hashes[filename] = _sha256_file(path)
    try:
        matrix_index_relpath = matrix_index_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        matrix_index_relpath = str(matrix_index_path.resolve())
    normalized_models = [str(model).strip() for model in observed_models if str(model).strip()]
    canonical_model_set = set(SUITE_MODEL_ROSTER)
    observed_model_set = set(normalized_models)
    if observed_model_set != canonical_model_set:
        raise SystemExit(
            "summary export identity requires the complete canonical five-model release roster; "
            f"observed={sorted(observed_model_set)} expected={sorted(canonical_model_set)}."
        )
    model_roster = [
        {
            "model": spec.name,
            "model_revision": spec.revision,
            "model_slug": spec.slug,
            "model_family": spec.family,
        }
        for spec in CANONICAL_SUITE_MODELS
    ]
    matrix_code_snapshot_digest = str(matrix_index.get("code_snapshot_digest", "")).strip()
    release_code_snapshot_digest = matrix_code_snapshot_digest
    if _repo_snapshot is not None:
        try:
            release_code_snapshot_digest = str(_repo_snapshot.repo_snapshot_sha256(ROOT)).strip()
        except Exception:
            release_code_snapshot_digest = matrix_code_snapshot_digest
    identity_payload = {
        "artifact_role": "suite_all_models_methods_release_summary_export_identity",
        "schema_version": 1,
        "matrix_index": matrix_index_relpath,
        "matrix_index_sha256": _sha256_file(matrix_index_path),
        "manifest": str(matrix_index.get("manifest", "")).strip(),
        "profile": str(matrix_index.get("profile", "")).strip(),
        "canonical_manifest_digest": str(matrix_index.get("canonical_manifest_digest", "")).strip(),
        "execution_mode": str(matrix_index.get("execution_mode", "")).strip(),
        "run_count": int(matrix_index.get("run_count", 0) or 0),
        "success_count": int(matrix_index.get("success_count", 0) or 0),
        "failed_count": int(matrix_index.get("failed_count", 0) or 0),
        "code_snapshot_digest": release_code_snapshot_digest,
        "matrix_code_snapshot_digest": matrix_code_snapshot_digest,
        "execution_environment_fingerprint": _release_environment_fingerprint(matrix_index),
        "assembly_source_execution_modes": [
            str(mode).strip()
            for mode in matrix_index.get("assembly_source_execution_modes", [])
            if str(mode).strip()
        ],
        "assembly_source_index_count": len(
            [item for item in matrix_index.get("assembly_source_indexes", []) if isinstance(item, Mapping)]
        ),
        "score_version": str(score_version).strip(),
        "model_roster": model_roster,
        "required_table_hashes": table_hashes,
        "required_figure_hashes": {},
        "figure_stems": list(_PUBLICATION_FIGURE_STEMS),
    }
    identity_path = output_dir / f"{_SUMMARY_EXPORT_IDENTITY_STEM}.json"
    with identity_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(identity_payload, indent=2, ensure_ascii=False) + "\n")


def _annotate_release_rows(rows: list[dict[str, Any]], **fields: Any) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        merged = dict(row)
        for key, value in fields.items():
            merged[key] = value
        annotated.append(merged)
    return annotated


def _model_rank(value: Any) -> int:
    return _MODEL_ORDER.get(str(value).strip(), len(_MODEL_ORDER) + 1)


def _method_rank(value: Any) -> int:
    return _METHOD_ORDER.get(str(value).strip(), len(_METHOD_ORDER) + 1)


def _paper_functional_quality_rows(model_method_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in model_method_summary:
        rows.append(
            {
                "table_role": "paper_functional_quality",
                "aggregation_view": "descriptive_model_method_rollup",
                "score_semantics": _PROJECTED_MODEL_METHOD_SCORE_SEMANTICS,
                "model": row["model"],
                "method": row["method"],
                "report_count": row["report_count"],
                "row_count": row["row_count"],
                "clean_compile_success_rate": row["clean_compile_success_rate"],
                "clean_test_pass_rate": row["clean_test_pass_rate"],
                "clean_pass@1": row["clean_pass@1"],
                "watermarked_test_pass_rate": row["watermarked_test_pass_rate"],
                "watermarked_pass@1": row["watermarked_pass@1"],
                "watermarked_pass_preservation": row["watermarked_pass_preservation"],
                "attacked_test_pass_rate": row["attacked_test_pass_rate"],
                "attacked_pass_preservation": row["attacked_pass_preservation"],
                HEADLINE_SCORE_FIELD: row[HEADLINE_SCORE_FIELD],
                "negative_control_fpr": row["negative_control_fpr"],
            }
        )
    rows.sort(key=lambda item: (_model_rank(item["model"]), _method_rank(item["method"]), -float(item[HEADLINE_SCORE_FIELD])))
    return rows


def _timing_hours(value: Any) -> float:
    return _round(float(value or 0.0) / 3600.0)


def _paper_timing_rows(model_method_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in model_method_summary:
        rows.append(
            {
                "table_role": "paper_timing",
                "aggregation_view": "descriptive_model_method_rollup",
                "score_semantics": _TIMING_ONLY_SCORE_SEMANTICS,
                "model": row["model"],
                "method": row["method"],
                "task_count": row["task_count"],
                "attack_row_count": row["attack_row_count"],
                "clean_generation_hours_total": _timing_hours(row["clean_generation_seconds_total"]),
                "watermarked_generation_hours_total": _timing_hours(row["watermarked_generation_seconds_total"]),
                "attack_hours_total": _timing_hours(row["attack_seconds_total"]),
                "validation_hours_total": _timing_hours(row["validation_seconds_total"]),
                "detection_hours_total": _timing_hours(row["detection_seconds_total"]),
                "total_example_hours_total": _timing_hours(row["total_example_seconds_total"]),
                "total_example_seconds_mean_per_task": row["total_example_seconds_mean_per_task"],
                "clean_generation_seconds_per_1k_token": row["clean_generation_seconds_per_1k_token"],
                "watermarked_generation_seconds_per_1k_token": row["watermarked_generation_seconds_per_1k_token"],
            }
        )
    rows.sort(key=lambda item: (_model_rank(item["model"]), _method_rank(item["method"])))
    return rows


def _paper_utility_robustness_rows(leaderboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in leaderboard:
        utility = _nullable_float(row.get("utility"))
        robustness = _nullable_float(row.get("robustness"))
        rows.append(
            {
                "table_role": "paper_utility_robustness_summary",
                "aggregation_view": "suite_method_utility_robustness_summary",
                "score_semantics": _LEADERBOARD_SCORE_SEMANTICS,
                "method": row["method"],
                "origin": row["origin"],
                "utility": utility,
                "robustness": robustness,
                "utility_robustness_drop": (
                    round(max(0.0, float(utility) - float(robustness)), 4)
                    if utility is not None and robustness is not None
                    else None
                ),
                "detection_separability": float(row.get("detection_separability", 0.0)),
                "gate": float(row.get("gate", 0.0)),
                "headline_core_score": float(row.get("headline_core_score", 0.0)),
                "generalization_status": str(row.get("generalization_status", "unsupported")),
                HEADLINE_SCORE_FIELD: float(row.get(HEADLINE_SCORE_FIELD, 0.0)),
            }
        )
    rows.sort(
        key=lambda item: (
            -float(item["robustness"] or -1.0),
            -float(item["utility"] or -1.0),
            str(item["method"]),
        )
    )
    return rows


def _score_coverage_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = row.get("score_coverage")
    return payload if isinstance(payload, Mapping) else {}


def _per_attack_robustness_breakdown(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in method_rows:
        method = row["method"]
        coverage = _score_coverage_payload(row)
        attack_breakdown = dict(coverage.get("attack_breakdown", {}))
        for attack_name, payload in attack_breakdown.items():
            rows.append(
                {
                    "table_role": "repo_attack_robustness_breakdown",
                    "aggregation_view": "suite_method_master_attack_breakdown_descriptive",
                    "score_semantics": "descriptive_unbalanced_attack_breakdown_from_master_score_coverage",
                    "method": method,
                    "attack": attack_name,
                    "attack_tier": str(payload.get("attack_tier", attack_tier(attack_name))),
                    "source_balanced_headline_robustness": row.get("robustness"),
                    "source_balanced_headline_raw_robustness_strict": row.get("raw_robustness_strict"),
                    "source_balanced_headline_support_rate": row.get("robustness_support_rate"),
                    "reconstructs_source_balanced_headline": False,
                    "attack_support_rate": float(payload.get("attack_support_rate", 0.0) or 0.0),
                    "attack_robustness": _nullable_float(payload.get("attack_robustness")),
                    "raw_attack_robustness_strict": _nullable_float(payload.get("raw_attack_robustness_strict")),
                    "attack_retention": _nullable_float(payload.get("attack_retention")),
                    "attack_attacked_detected_semantic_rate": _nullable_float(
                        payload.get("attack_attacked_detected_semantic_rate")
                    ),
                    "attack_attacked_pass_preservation": _nullable_float(
                        payload.get("attack_attacked_pass_preservation")
                    ),
                    "attack_status": str(payload.get("attack_status", "unsupported")),
                }
            )
    rows.sort(
        key=lambda item: (
            str(item["attack_tier"]),
            str(item["attack"]),
            -float(item.get("attack_robustness") or 0.0),
            str(item["method"]),
        )
    )
    return rows


def _core_vs_stress_robustness_summary_rows(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "table_role": "repo_core_vs_stress_robustness_summary",
            "aggregation_view": row.get("aggregation_view", "suite_method_master_leaderboard"),
            "score_semantics": row.get("score_semantics", _LEADERBOARD_SCORE_SEMANTICS),
            "method": row["method"],
            "row_count": row.get("row_count"),
            "attack_support_rate": row.get("attack_support_rate"),
            "robustness": row.get("robustness"),
            "raw_robustness_strict": row.get("raw_robustness_strict"),
            "robustness_status": row.get("robustness_status"),
            "robustness_support_rate": row.get("robustness_support_rate"),
            "stress_robustness": row.get("stress_robustness"),
            HEADLINE_SCORE_FIELD: row.get(HEADLINE_SCORE_FIELD),
        }
        for row in method_rows
    ]
    rows.sort(key=lambda item: (-float(item.get("robustness") or 0.0), str(item["method"])))
    return rows


def _robustness_factor_decomposition_rows(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "table_role": "repo_robustness_factor_decomposition",
            "aggregation_view": row.get("aggregation_view", "suite_method_master_leaderboard"),
            "score_semantics": row.get("score_semantics", _LEADERBOARD_SCORE_SEMANTICS),
            "method": row["method"],
            "row_count": row.get("row_count"),
            "attack_support_rate": row.get("attack_support_rate"),
            "robustness": row.get("robustness"),
            "raw_robustness_strict": row.get("raw_robustness_strict"),
            "robustness_status": row.get("robustness_status"),
            "robustness_support_rate": row.get("robustness_support_rate"),
            "watermark_retention_mean": row.get("watermark_retention_mean"),
            "attacked_detected_semantic_rate": row.get("attacked_detected_semantic_rate"),
            "attacked_pass_preservation": row.get("attacked_pass_preservation"),
        }
        for row in method_rows
    ]
    rows.sort(key=lambda item: (-float(item.get("robustness") or 0.0), str(item["method"])))
    return rows


def _utility_factor_decomposition_rows(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "table_role": "repo_utility_factor_decomposition",
            "aggregation_view": row.get("aggregation_view", "suite_method_master_leaderboard"),
            "score_semantics": row.get("score_semantics", _LEADERBOARD_SCORE_SEMANTICS),
            "method": row["method"],
            "row_count": row.get("row_count"),
            "utility": row.get("utility"),
            "raw_utility_strict": row.get("raw_utility_strict"),
            "utility_status": row.get("utility_status"),
            "utility_support_rate": row.get("utility_support_rate"),
            "quality_score_mean": row.get("quality_score_mean"),
            "semantic_preservation_rate": row.get("semantic_preservation_rate"),
            "semantic_validation_rate": row.get("semantic_validation_rate"),
            "declared_semantic_validation_rate": row.get("declared_semantic_validation_rate"),
        }
        for row in method_rows
    ]
    rows.sort(key=lambda item: (-float(item.get("utility") or 0.0), str(item["method"])))
    return rows


def _generalization_axis_breakdown_rows(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "table_role": "repo_generalization_axis_breakdown",
            "aggregation_view": row.get("aggregation_view", "suite_method_master_leaderboard"),
            "score_semantics": row.get("score_semantics", _LEADERBOARD_SCORE_SEMANTICS),
            "method": row["method"],
            "row_count": row.get("row_count"),
            "generalization": row.get("generalization"),
            "raw_generalization_strict": row.get("raw_generalization_strict"),
            "headline_generalization": row.get("headline_generalization"),
            "generalization_status": row.get("generalization_status"),
            "generalization_available_axes": row.get("generalization_available_axes", []),
            "source_stability": row.get("source_stability"),
            "task_stability": row.get("task_stability"),
            "language_stability": row.get("language_stability"),
            "cross_family_transfer": row.get("cross_family_transfer"),
        }
        for row in method_rows
    ]
    rows.sort(key=lambda item: (-float(item.get("headline_generalization") or 0.0), str(item["method"])))
    return rows


def _gate_decomposition_rows(
    method_rows: list[dict[str, Any]],
    *,
    descriptive_method_rows: Mapping[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    descriptive_method_rows = descriptive_method_rows or {}
    rows = [
        {
            "table_role": "repo_gate_decomposition",
            "aggregation_view": row.get("aggregation_view", "suite_method_master_leaderboard"),
            "score_semantics": row.get("score_semantics", _LEADERBOARD_SCORE_SEMANTICS),
            "method": row["method"],
            "row_count": row.get("row_count"),
            "gate": row.get("gate"),
            "watermarked_pass_preservation": row.get("watermarked_pass_preservation"),
            "negative_control_fpr": row.get("negative_control_fpr"),
            "descriptive_clean_test_pass_rate": descriptive_method_rows.get(str(row["method"]), {}).get(
                "clean_test_pass_rate"
            ),
            "descriptive_watermarked_test_pass_rate": descriptive_method_rows.get(str(row["method"]), {}).get(
                "watermarked_test_pass_rate"
            ),
            "negative_control_support_rate": row.get("negative_control_support_rate"),
        }
        for row in method_rows
    ]
    rows.sort(key=lambda item: (-float(item.get("gate") or 0.0), str(item["method"])))
    return rows


def _run_duration_seconds(run: Mapping[str, Any]) -> float:
    return float(run.get("duration_seconds", 0.0) or 0.0)


def _group_rows(
    grouped_rows: Mapping[Any, list[BenchmarkRow]],
    grouped_runs: Mapping[Any, list[Mapping[str, Any]]],
    *,
    extra_fields: callable | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, benchmark_rows in grouped_rows.items():
        run_records = list(grouped_runs.get(key, []))
        scorecard = scorecard_for_rows(benchmark_rows, balance_by_source_group=True)
        clean = _clean_functional_metrics(benchmark_rows)
        watermarked = _watermarked_functional_metrics(benchmark_rows)
        attacked = _attacked_functional_metrics(benchmark_rows)
        stage_timing = _stage_timing_metrics(benchmark_rows)
        durations = [_run_duration_seconds(run) for run in run_records]
        payload = {
            "aggregation_view": "descriptive_rollup",
            "score_semantics": _ROLLUP_SCORECARD_SEMANTICS,
            "row_count": len(benchmark_rows),
            "report_count": len(run_records),
            "duration_seconds_total": _round(sum(durations)),
            "duration_hours_total": _round(sum(durations) / 3600.0),
            "duration_seconds_mean": _round(mean(durations)) if durations else 0.0,
            "duration_hours_mean": _round(mean(durations) / 3600.0) if durations else 0.0,
            HEADLINE_SCORE_FIELD: float(scorecard.get(HEADLINE_SCORE_FIELD, 0.0)),
            "detection_separability": float(scorecard.get("detection_separability", 0.0)),
            "robustness": _nullable_float(scorecard.get("robustness")),
            "raw_robustness_strict": _nullable_float(scorecard.get("raw_robustness_strict")),
            "robustness_status": str(scorecard.get("robustness_status", "unsupported")),
            "robustness_support_rate": float(scorecard.get("robustness_support_rate", 0.0) or 0.0),
            "stress_robustness": _nullable_float(scorecard.get("stress_robustness")),
            "utility": _nullable_float(scorecard.get("utility")),
            "raw_utility_strict": _nullable_float(scorecard.get("raw_utility_strict")),
            "utility_status": str(scorecard.get("utility_status", "unsupported")),
            "utility_support_rate": float(scorecard.get("utility_support_rate", 0.0) or 0.0),
            "stealth": float(scorecard.get("stealth", 0.0)),
            "efficiency": float(scorecard.get("efficiency", 0.0)),
            "stealth_conditioned": float(scorecard.get("stealth_conditioned", 0.0)),
            "efficiency_conditioned": float(scorecard.get("efficiency_conditioned", 0.0)),
            "core_score": float(scorecard.get("core_score", 0.0)),
            "raw_core_score_strict": float(scorecard.get("raw_core_score_strict", 0.0)),
            "headline_core_score": float(scorecard.get("headline_core_score", 0.0)),
            "generalization": _nullable_float(scorecard.get("generalization")),
            "raw_generalization_strict": _nullable_float(scorecard.get("raw_generalization_strict")),
            "headline_generalization": float(scorecard.get("headline_generalization", 1.0)),
            "generalization_supported": bool(scorecard.get("generalization_supported", False)),
            "generalization_status": str(scorecard.get("generalization_status", "unsupported")),
            "generalization_available_axes": list(scorecard.get("generalization_available_axes", [])),
            "source_stability": scorecard.get("source_stability"),
            "task_stability": scorecard.get("task_stability"),
            "language_stability": scorecard.get("language_stability"),
            "cross_family_transfer": scorecard.get("cross_family_transfer"),
            "scale_consistency": scorecard.get("scale_consistency"),
            "scale_supported_families": list(scorecard.get("scale_supported_families", [])),
            "scale_supported_family_count": int(scorecard.get("scale_supported_family_count", 0)),
            "negative_control_fpr": float(scorecard.get("negative_control_fpr", 0.0)),
            "negative_control_support_rate": float(scorecard.get("negative_control_support_rate", 0.0)),
            "clean_compile_success_rate": float(clean.get("compile_success_rate", 0.0)),
            "clean_test_pass_rate": float(clean.get("test_pass_rate", 0.0)),
            "clean_pass@1": float(clean.get("pass@1", 0.0)),
            "watermarked_test_pass_rate": float(watermarked.get("test_pass_rate", 0.0)),
            "watermarked_pass@1": float(watermarked.get("pass@1", 0.0)),
            "attacked_test_pass_rate": float(attacked.get("test_pass_rate", 0.0)),
            "watermarked_pass_preservation": float(scorecard.get("watermarked_pass_preservation", 0.0)),
            "attacked_pass_preservation": float(scorecard.get("attacked_pass_preservation", 0.0)),
            "gate": float(scorecard.get("gate", 0.0)),
            "raw_composite_strict": float(scorecard.get("raw_composite_strict", 0.0)),
            "score_version": str(scorecard.get("score_version", "")),
        }
        payload.update(stage_timing)
        if extra_fields is not None:
            payload.update(extra_fields(key, benchmark_rows, run_records))
        rows.append(payload)
    return rows


def _mean_of(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row.get(field, 0.0) or 0.0) for row in rows]
    return _round(mean(values)) if values else 0.0


def _mean_optional_of(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row.get(field)) for row in rows if row.get(field) is not None]
    if not values:
        return None
    return _round(mean(values))


def _union_preserving_order(rows: list[dict[str, Any]], field: str) -> list[Any]:
    seen: set[Any] = set()
    merged: list[Any] = []
    for row in rows:
        values = row.get(field, []) or []
        if not isinstance(values, list):
            continue
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            merged.append(value)
    return merged


def _merge_status(rows: list[dict[str, Any]], field: str, *, default: str = "unsupported") -> str:
    seen: list[str] = []
    seen_set: set[str] = set()
    for row in rows:
        status = str(row.get(field, "")).strip() or default
        if status in seen_set:
            continue
        seen_set.add(status)
        seen.append(status)
    if not seen:
        return default
    if len(seen) == 1:
        return seen[0]
    return "descriptive_mixed"


def _merge_generalization_status(rows: list[dict[str, Any]]) -> str:
    return _merge_status(rows, "generalization_status")


def _aggregate_model_summary(model_method_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in model_method_summary:
        by_model[str(row.get("model", "")).strip() or "unspecified"].append(row)
    aggregated: list[dict[str, Any]] = []
    for model, method_rows in by_model.items():
        report_count = sum(int(row.get("report_count", 0) or 0) for row in method_rows)
        duration_seconds_total = _round(sum(float(row.get("duration_seconds_total", 0.0) or 0.0) for row in method_rows))
        payload = {
            "model": model,
            "methods": [str(row.get("method", "")).strip() for row in method_rows if str(row.get("method", "")).strip()],
            "method_count": len(method_rows),
            "table_role": "repo_descriptive_model_rollup",
            "aggregation_view": "descriptive_model_rollup",
            "score_semantics": _MODEL_SUMMARY_SCORE_SEMANTICS,
            "row_count": sum(int(row.get("row_count", 0) or 0) for row in method_rows),
            "report_count": report_count,
            "duration_seconds_total": duration_seconds_total,
            "duration_hours_total": _round(duration_seconds_total / 3600.0),
            "duration_seconds_mean": _round(duration_seconds_total / max(report_count, 1)),
            "duration_hours_mean": _round((duration_seconds_total / max(report_count, 1)) / 3600.0),
            "task_count": sum(int(row.get("task_count", 0) or 0) for row in method_rows),
            "attack_row_count": sum(int(row.get("attack_row_count", 0) or 0) for row in method_rows),
            "clean_generation_seconds_total": _round(sum(float(row.get("clean_generation_seconds_total", 0.0) or 0.0) for row in method_rows)),
            "watermarked_generation_seconds_total": _round(sum(float(row.get("watermarked_generation_seconds_total", 0.0) or 0.0) for row in method_rows)),
            "attack_seconds_total": _round(sum(float(row.get("attack_seconds_total", 0.0) or 0.0) for row in method_rows)),
            "validation_seconds_total": _round(sum(float(row.get("validation_seconds_total", 0.0) or 0.0) for row in method_rows)),
            "detection_seconds_total": _round(sum(float(row.get("detection_seconds_total", 0.0) or 0.0) for row in method_rows)),
            "total_example_seconds_total": _round(sum(float(row.get("total_example_seconds_total", 0.0) or 0.0) for row in method_rows)),
            "clean_generation_seconds_per_1k_token": _mean_of(method_rows, "clean_generation_seconds_per_1k_token"),
            "watermarked_generation_seconds_per_1k_token": _mean_of(method_rows, "watermarked_generation_seconds_per_1k_token"),
            HEADLINE_SCORE_FIELD: _mean_of(method_rows, HEADLINE_SCORE_FIELD),
            "detection_separability": _mean_of(method_rows, "detection_separability"),
            "robustness": _mean_optional_of(method_rows, "robustness"),
            "raw_robustness_strict": _mean_optional_of(method_rows, "raw_robustness_strict"),
            "robustness_status": _merge_status(method_rows, "robustness_status"),
            "robustness_support_rate": _mean_of(method_rows, "robustness_support_rate"),
            "stress_robustness": _mean_optional_of(method_rows, "stress_robustness"),
            "utility": _mean_optional_of(method_rows, "utility"),
            "raw_utility_strict": _mean_optional_of(method_rows, "raw_utility_strict"),
            "utility_status": _merge_status(method_rows, "utility_status"),
            "utility_support_rate": _mean_of(method_rows, "utility_support_rate"),
            "stealth": _mean_of(method_rows, "stealth"),
            "efficiency": _mean_of(method_rows, "efficiency"),
            "stealth_conditioned": _mean_of(method_rows, "stealth_conditioned"),
            "efficiency_conditioned": _mean_of(method_rows, "efficiency_conditioned"),
            "core_score": _mean_of(method_rows, "core_score"),
            "raw_core_score_strict": _mean_of(method_rows, "raw_core_score_strict"),
            "headline_core_score": _mean_of(method_rows, "headline_core_score"),
            "generalization": _mean_optional_of(method_rows, "generalization"),
            "raw_generalization_strict": _mean_optional_of(method_rows, "raw_generalization_strict"),
            "headline_generalization": _mean_of(method_rows, "headline_generalization"),
            "generalization_supported": any(bool(row.get("generalization_supported", False)) for row in method_rows),
            "generalization_status": _merge_generalization_status(method_rows),
            "generalization_available_axes": _union_preserving_order(method_rows, "generalization_available_axes"),
            "source_stability": _mean_optional_of(method_rows, "source_stability"),
            "task_stability": _mean_optional_of(method_rows, "task_stability"),
            "language_stability": _mean_optional_of(method_rows, "language_stability"),
            "cross_family_transfer": _mean_optional_of(method_rows, "cross_family_transfer"),
            "scale_consistency": _mean_optional_of(method_rows, "scale_consistency"),
            "scale_supported_families": _union_preserving_order(method_rows, "scale_supported_families"),
            "scale_supported_family_count": max(int(row.get("scale_supported_family_count", 0) or 0) for row in method_rows),
            "negative_control_fpr": _mean_of(method_rows, "negative_control_fpr"),
            "negative_control_support_rate": _mean_of(method_rows, "negative_control_support_rate"),
            "clean_compile_success_rate": _mean_of(method_rows, "clean_compile_success_rate"),
            "clean_test_pass_rate": _mean_of(method_rows, "clean_test_pass_rate"),
            "clean_pass@1": _mean_of(method_rows, "clean_pass@1"),
            "watermarked_test_pass_rate": _mean_of(method_rows, "watermarked_test_pass_rate"),
            "watermarked_pass@1": _mean_of(method_rows, "watermarked_pass@1"),
            "attacked_test_pass_rate": _mean_of(method_rows, "attacked_test_pass_rate"),
            "watermarked_pass_preservation": _mean_of(method_rows, "watermarked_pass_preservation"),
            "attacked_pass_preservation": _mean_of(method_rows, "attacked_pass_preservation"),
            "gate": _mean_of(method_rows, "gate"),
            "raw_composite_strict": _mean_of(method_rows, "raw_composite_strict"),
            "score_version": str(method_rows[0].get("score_version", "")) if method_rows else "",
        }
        aggregated.append(payload)
    return aggregated


def _require_complete_suite_atomic_roster(entries: list[dict[str, Any]], *, label: str) -> None:
    if not entries:
        raise SystemExit(f"{label} requires a complete result-backed suite matrix and cannot be exported from partial coverage.")
    incomplete = [
        f"{str(entry.get('method', '')).strip() or 'unspecified'}"
        + (
            f"/{str(entry.get('model', '')).strip()}"
            if str(entry.get("model", "")).strip()
            else ""
        )
        + f":{','.join(entry.get('missing_source_groups', []))}"
        for entry in entries
        if not bool(entry.get("suite_atomic_source_complete", False))
    ]
    if incomplete:
        preview = ", ".join(incomplete[:8])
        if len(incomplete) > 8:
            preview += ", ..."
        raise SystemExit(f"{label} requires complete seven-source suite coverage; incomplete entries: {preview}")


def _require_canonical_suite_identity(
    matrix_index: Mapping[str, Any],
    *,
    label: str,
) -> None:
    manifest = str(matrix_index.get("manifest", "")).strip()
    profile = str(matrix_index.get("profile", "")).strip()
    run_count = int(matrix_index.get("run_count", 0) or 0)
    if manifest != _CANONICAL_SUITE_MANIFEST or profile != _CANONICAL_SUITE_PROFILE or run_count != 140:
        raise SystemExit(
            f"{label} requires the canonical suite identity "
            f"(manifest={_CANONICAL_SUITE_MANIFEST}, profile={_CANONICAL_SUITE_PROFILE}, run_count=140). "
            f"Observed manifest={manifest or '<missing>'}, profile={profile or '<missing>'}, run_count={run_count}."
        )
    canonical_manifest_digest = str(matrix_index.get("canonical_manifest_digest", "")).strip()
    execution_mode = str(matrix_index.get("execution_mode", "")).strip()
    if not canonical_manifest_digest:
        raise SystemExit(f"{label} is missing canonical_manifest_digest.")
    if canonical_manifest_digest != _CANONICAL_SUITE_MANIFEST_DIGEST:
        raise SystemExit(
            f"{label} canonical_manifest_digest mismatch: expected {_CANONICAL_SUITE_MANIFEST_DIGEST}, "
            f"observed {canonical_manifest_digest}."
        )
    if not execution_mode:
        raise SystemExit(f"{label} is missing execution_mode.")
    if execution_mode not in _KNOWN_EXECUTION_MODES:
        raise SystemExit(
            f"{label} execution_mode must be one of {sorted(_KNOWN_EXECUTION_MODES)}; observed {execution_mode}."
        )
    shard_profiles = matrix_index.get("shard_profiles", [])
    if execution_mode != _EXPECTED_SHARDED_EXECUTION_MODE and shard_profiles:
        raise SystemExit(
            f"{label} shard_profiles require execution_mode={_EXPECTED_SHARDED_EXECUTION_MODE}; observed {execution_mode}."
        )
    if execution_mode == _EXPECTED_SHARDED_EXECUTION_MODE:
        code_snapshot_digest = str(matrix_index.get("code_snapshot_digest", "")).strip()
        if not code_snapshot_digest:
            raise SystemExit(f"{label} is missing code_snapshot_digest.")
        execution_environment_fingerprint = str(matrix_index.get("execution_environment_fingerprint", "")).strip()
        if not execution_environment_fingerprint:
            raise SystemExit(f"{label} is missing execution_environment_fingerprint.")


def _require_formal_single_host_execution_mode(matrix_index: Mapping[str, Any], *, label: str) -> None:
    execution_mode = str(matrix_index.get("execution_mode", "")).strip()
    if execution_mode != _FORMAL_CANONICAL_EXECUTION_MODE:
        raise SystemExit(
            f"{label} requires execution_mode={_FORMAL_CANONICAL_EXECUTION_MODE}; observed {execution_mode or '<missing>'}."
        )
    code_snapshot_digest = str(matrix_index.get("code_snapshot_digest", "")).strip()
    if not code_snapshot_digest:
        raise SystemExit(f"{label} is missing code_snapshot_digest.")
    execution_environment_fingerprint = str(matrix_index.get("execution_environment_fingerprint", "")).strip()
    if not execution_environment_fingerprint:
        raise SystemExit(f"{label} is missing execution_environment_fingerprint.")
    source_modes = matrix_index.get("assembly_source_execution_modes", [])
    source_indexes = matrix_index.get("assembly_source_indexes", [])
    if bool(source_modes) != bool(source_indexes):
        raise SystemExit(
            f"{label} must provide both assembly_source_execution_modes and assembly_source_indexes when publication "
            "provenance is assembled from multiple source indexes."
        )
    if not source_modes:
        raise SystemExit(
            f"{label} must retain non-empty assembly_source_execution_modes and assembly_source_indexes for the "
            "publication-facing canonical single-host matrix."
        )
    if source_modes:
        normalized_source_modes = sorted(
            {
                str(mode).strip()
                for mode in source_modes
                if str(mode).strip()
            }
        )
        if not normalized_source_modes:
            raise SystemExit(f"{label} is missing non-empty assembly_source_execution_modes.")
        invalid_modes = [mode for mode in normalized_source_modes if mode not in _KNOWN_EXECUTION_MODES]
        if invalid_modes:
            raise SystemExit(
                f"{label} assembly_source_execution_modes must be drawn from {sorted(_KNOWN_EXECUTION_MODES)}; "
                f"observed invalid modes {invalid_modes}."
            )
        if not isinstance(source_indexes, list) or not source_indexes:
            raise SystemExit(f"{label} is missing assembly_source_indexes.")
        index_modes: list[str] = []
        for index, source_index in enumerate(source_indexes, start=1):
            if not isinstance(source_index, Mapping):
                raise SystemExit(f"{label} assembly_source_indexes[{index}] is not an object.")
            source_mode = str(source_index.get("execution_mode", "")).strip()
            if not source_mode:
                raise SystemExit(f"{label} assembly_source_indexes[{index}] is missing execution_mode.")
            if source_mode not in _KNOWN_EXECUTION_MODES:
                raise SystemExit(
                    f"{label} assembly_source_indexes[{index}] execution_mode must be one of "
                    f"{sorted(_KNOWN_EXECUTION_MODES)}; observed {source_mode}."
                )
            index_modes.append(source_mode)
        if sorted(set(index_modes)) != normalized_source_modes:
            raise SystemExit(
                f"{label} assembly provenance is inconsistent: assembly_source_execution_modes={normalized_source_modes} "
                f"but assembly_source_indexes encode {sorted(set(index_modes))}."
            )
        if len(source_indexes) != 1:
            raise SystemExit(
                f"{label} must be backed by exactly one source index for the formal one-shot release path; "
                f"observed {len(source_indexes)}."
            )
        if normalized_source_modes != [_FORMAL_CANONICAL_EXECUTION_MODE]:
            raise SystemExit(
                f"{label} must be backed only by {_FORMAL_CANONICAL_EXECUTION_MODE} source indexes; "
                f"observed {normalized_source_modes}."
            )


def _report_track_set(payload: Mapping[str, Any]) -> list[str]:
    summary = dict(payload.get("summary", {}))
    tracks = summary.get("evaluation_tracks", [])
    if isinstance(tracks, list):
        normalized = sorted({str(track).strip() for track in tracks if str(track).strip()})
        if normalized:
            return normalized
    track = str(summary.get("paper_primary_track", "")).strip()
    return [track] if track else []


def _report_row_track_violations(payload: Mapping[str, Any]) -> list[str]:
    raw_rows = payload.get("rows", [])
    if not isinstance(raw_rows, list) or not raw_rows:
        return ["rows=<missing_or_empty>"]
    violations: list[str] = []
    for index, row in enumerate(raw_rows, start=1):
        if not isinstance(row, Mapping):
            violations.append(f"row[{index}]=<non-mapping>")
            continue
        track = str(row.get("evaluation_track", "")).strip()
        if track != GENERATION_TIME_TRACK:
            violations.append(f"row[{index}].evaluation_track={track or '<missing>'}")
    return violations


def _require_generation_time_release_contract(report_payloads: list[Mapping[str, Any]], *, label: str) -> None:
    if not report_payloads:
        raise SystemExit(f"{label} found no report payloads to validate.")
    violations: list[str] = []
    for payload in report_payloads:
        config = dict(payload.get("config", {}))
        watermark_scheme = str(config.get("watermark_scheme", "")).strip() or "<unknown>"
        summary = dict(payload.get("summary", {}))
        track_set = _report_track_set(payload)
        paper_track_ready = bool(summary.get("paper_track_ready"))
        row_track_violations = _report_row_track_violations(payload)
        if track_set != [GENERATION_TIME_TRACK] or not paper_track_ready or row_track_violations:
            row_preview = ", ".join(row_track_violations[:3]) if row_track_violations else "<none>"
            if len(row_track_violations) > 3:
                row_preview += ", ..."
            violations.append(
                f"{watermark_scheme}:tracks={track_set or ['<missing>']}:paper_track_ready={paper_track_ready}:"
                f"row_contract={row_preview}"
            )
    if violations:
        preview = ", ".join(violations[:8])
        if len(violations) > 8:
            preview += ", ..."
        raise SystemExit(
            f"{label} requires generation-time-only report payloads with paper_track_ready=true and per-row "
            "evaluation_track=generation_time; "
            f"violations=[{preview}]"
        )


def main() -> int:
    args = _parse_args()
    matrix_index_path = args.matrix_index.resolve()
    base_dir = matrix_index_path.parents[3]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_index = json.loads(matrix_index_path.read_text(encoding="utf-8"))
    runs = _exportable_runs(matrix_index, base_dir=base_dir)

    reports: list[tuple[Mapping[str, Any], list[BenchmarkRow], Mapping[str, Any]]] = []
    report_payloads: list[Mapping[str, Any]] = []
    for run in runs:
        report_path = _repo_path(run["report_path"], base_dir=base_dir)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        rows = [_row_from_payload(row) for row in payload.get("rows", []) if isinstance(row, Mapping)]
        reports.append((run, rows, payload))
        report_payloads.append(payload)

    _require_canonical_suite_identity(matrix_index, label="suite_all_models_methods exact-value exports")
    _require_formal_single_host_execution_mode(
        matrix_index,
        label="suite_all_models_methods exact-value exports",
    )
    _require_generation_time_release_contract(
        report_payloads,
        label="suite_all_models_methods exact-value exports",
    )

    by_method_rows: dict[str, list[BenchmarkRow]] = defaultdict(list)
    by_model_rows: dict[str, list[BenchmarkRow]] = defaultdict(list)
    by_model_method_rows: dict[tuple[str, str], list[BenchmarkRow]] = defaultdict(list)
    by_method_source_rows: dict[tuple[str, str], list[BenchmarkRow]] = defaultdict(list)
    by_method_language_rows: dict[tuple[str, str], list[BenchmarkRow]] = defaultdict(list)
    by_method_attack_rows: dict[tuple[str, str], list[BenchmarkRow]] = defaultdict(list)
    by_method_runs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_model_runs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_model_method_runs: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_method_source_runs: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_method_language_runs: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_method_attack_runs: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)

    run_inventory: list[dict[str, Any]] = []
    multilingual_source_groups = {
        normalize_source_group(source.source_group)
        for source in SUITE_AGGREGATE_SOURCES
        if len(tuple(source.languages)) > 1
    }

    for run, rows, _payload in reports:
        if not rows:
            continue
        exemplar = rows[0]
        method = str(exemplar.watermark_scheme).strip() or "unspecified"
        model = str(exemplar.model_label).strip() or "unspecified"
        source = normalize_source_group(str(exemplar.source_group).strip() or str(exemplar.dataset).strip()) or "unspecified"

        by_method_rows[method].extend(rows)
        by_model_rows[model].extend(rows)
        by_model_method_rows[(model, method)].extend(rows)
        by_method_source_rows[(method, source)].extend(rows)
        by_method_runs[method].append(run)
        by_model_runs[model].append(run)
        by_model_method_runs[(model, method)].append(run)
        by_method_source_runs[(method, source)].append(run)

        multilingual_language_groups: dict[str, list[BenchmarkRow]] = defaultdict(list)
        for row in rows:
            row_source = normalize_source_group(str(row.source_group).strip() or str(row.dataset).strip())
            if row_source not in multilingual_source_groups:
                continue
            language = str(row.language).strip().lower()
            if not language:
                continue
            multilingual_language_groups[language].append(row)
        for language, language_rows in multilingual_language_groups.items():
            by_method_language_rows[(method, language)].extend(language_rows)
            by_method_language_runs[(method, language)].append(run)

        run_inventory.append(
            {
                "run_id": str(run.get("run_id", "")),
                "status": str(run.get("status", "")),
                "duration_hours": _round(_run_duration_seconds(run) / 3600.0),
                "method": method,
                "model": model,
                "source_group": source,
                "row_count": len(rows),
                "score_semantics": _RUN_INVENTORY_SCORE_SEMANTICS,
                HEADLINE_SCORE_FIELD: float(
                    scorecard_for_rows(rows, balance_by_source_group=True).get(HEADLINE_SCORE_FIELD, 0.0)
                ),
            }
        )

        attack_groups: dict[str, list[BenchmarkRow]] = defaultdict(list)
        for row in rows:
            attack = str(row.attack_name).strip() or "unspecified"
            attack_groups[attack].append(row)
        for attack, attack_rows in attack_groups.items():
            by_method_attack_rows[(method, attack)].extend(attack_rows)
            by_method_attack_runs[(method, attack)].append(run)

    method_summary = _group_rows(
        by_method_rows,
        by_method_runs,
        extra_fields=lambda key, _rows, _runs: {
            "method": key,
            "table_role": "repo_descriptive_method_rollup",
            "aggregation_view": "descriptive_method_rollup",
        },
    )
    method_summary.sort(key=lambda row: (-float(row[HEADLINE_SCORE_FIELD]), str(row["method"])))

    model_method_summary = _group_rows(
        by_model_method_rows,
        by_model_method_runs,
        extra_fields=lambda key, _rows, _runs: {
            "model": key[0],
            "method": key[1],
            "table_role": "repo_descriptive_model_method_rollup",
            "aggregation_view": "descriptive_model_method_rollup",
        },
    )
    model_method_summary.sort(key=lambda row: (_model_rank(row["model"]), _method_rank(row["method"]), -float(row[HEADLINE_SCORE_FIELD])))

    model_summary = _aggregate_model_summary(model_method_summary)
    model_summary.sort(key=lambda row: (_model_rank(row["model"]), str(row["model"])))

    method_source_summary = _group_rows(
        by_method_source_rows,
        by_method_source_runs,
        extra_fields=lambda key, _rows, _runs: {
            "method": key[0],
            "source_group": key[1],
            "table_role": "repo_descriptive_method_source_rollup",
            "aggregation_view": "descriptive_method_source_rollup",
        },
    )
    method_source_summary.sort(key=lambda row: (str(row["source_group"]), -float(row[HEADLINE_SCORE_FIELD]), str(row["method"])))

    method_language_summary = _group_rows(
        by_method_language_rows,
        by_method_language_runs,
        extra_fields=lambda key, _rows, _runs: {
            "method": key[0],
            "language": key[1],
            "table_role": "repo_descriptive_method_language_rollup",
            "aggregation_view": "descriptive_method_language_rollup",
        },
    )
    method_language_summary.sort(key=lambda row: (str(row["language"]), _method_rank(row["method"]), -float(row[HEADLINE_SCORE_FIELD])))

    method_attack_summary = _group_rows(
        by_method_attack_rows,
        by_method_attack_runs,
        extra_fields=lambda key, rows, _runs: {
            "method": key[0],
            "attack": key[1],
            "table_role": "repo_descriptive_method_attack_rollup",
            "aggregation_view": "descriptive_method_attack_rollup",
            "attacked_detect_rate": _round(sum(1.0 for row in rows if bool(row.attacked_detected)) / max(len(rows), 1)),
            "mean_attacked_score": _round(mean(row.attacked_score for row in rows) if rows else 0.0),
            "mean_quality_score": _round(mean(row.quality_score for row in rows) if rows else 0.0),
        },
    )
    method_attack_summary.sort(key=lambda row: (str(row["attack"]), -float(row["attacked_detect_rate"]), str(row["method"])))

    timing_summary = [
        {
            "method": row["method"],
            "table_role": "repo_descriptive_timing_rollup",
            "aggregation_view": "descriptive_method_rollup",
            "score_semantics": _TIMING_ONLY_SCORE_SEMANTICS,
            "task_count": row["task_count"],
            "attack_row_count": row["attack_row_count"],
            "duration_hours_total": row["duration_hours_total"],
            "duration_hours_mean": row["duration_hours_mean"],
            "clean_generation_hours_total": _timing_hours(row["clean_generation_seconds_total"]),
            "watermarked_generation_hours_total": _timing_hours(row["watermarked_generation_seconds_total"]),
            "attack_hours_total": _timing_hours(row["attack_seconds_total"]),
            "validation_hours_total": _timing_hours(row["validation_seconds_total"]),
            "detection_hours_total": _timing_hours(row["detection_seconds_total"]),
            "total_example_hours_total": _timing_hours(row["total_example_seconds_total"]),
            "total_example_seconds_mean_per_task": row["total_example_seconds_mean_per_task"],
            "clean_generation_seconds_per_1k_token": row["clean_generation_seconds_per_1k_token"],
            "watermarked_generation_seconds_per_1k_token": row["watermarked_generation_seconds_per_1k_token"],
            "report_count": row["report_count"],
        }
        for row in method_summary
    ]
    timing_summary.sort(key=lambda row: (-float(row["duration_hours_total"]), str(row["method"])))
    model_method_timing = _paper_timing_rows(model_method_summary)

    run_inventory.sort(key=lambda row: (_model_rank(row["model"]), _method_rank(row["method"]), str(row["source_group"])))

    suite_method_master = build_suite_method_master_leaderboard(
        report_payloads,
        track=GENERATION_TIME_TRACK,
        allowed_methods=OFFICIAL_RUNTIME_BASELINES,
    )
    suite_method_model = build_suite_method_model_leaderboard(
        report_payloads,
        track=GENERATION_TIME_TRACK,
        allowed_methods=OFFICIAL_RUNTIME_BASELINES,
    )
    _require_complete_suite_atomic_roster(suite_method_master, label="suite_all_models_methods_method_master_leaderboard")
    _require_complete_suite_atomic_roster(suite_method_model, label="suite_all_models_methods_method_model_leaderboard")
    suite_method_master = _annotate_release_rows(
        suite_method_master,
        table_role="paper_method_master_leaderboard",
        aggregation_view="suite_method_master_leaderboard",
        score_semantics=_LEADERBOARD_SCORE_SEMANTICS,
    )
    suite_method_model = _annotate_release_rows(
        suite_method_model,
        table_role="paper_method_model_leaderboard",
        aggregation_view="suite_method_model_leaderboard",
        score_semantics=_LEADERBOARD_SCORE_SEMANTICS,
    )
    suite_method_model.sort(key=lambda row: (_model_rank(row.get("model", "")), _method_rank(row.get("method", ""))))
    suite_upstream_only = _annotate_release_rows(
        [row for row in suite_method_master if str(row.get("origin", "")).strip() == "upstream"],
        table_role="paper_upstream_only_leaderboard",
        aggregation_view="suite_upstream_only_leaderboard",
        score_semantics=_LEADERBOARD_SCORE_SEMANTICS,
    )
    utility_robustness_summary = _paper_utility_robustness_rows(suite_method_master)
    model_method_functional_quality = _paper_functional_quality_rows(model_method_summary)
    per_attack_robustness_breakdown = _per_attack_robustness_breakdown(suite_method_master)
    core_vs_stress_robustness_summary = _core_vs_stress_robustness_summary_rows(suite_method_master)
    robustness_factor_decomposition = _robustness_factor_decomposition_rows(suite_method_master)
    utility_factor_decomposition = _utility_factor_decomposition_rows(suite_method_master)
    generalization_axis_breakdown = _generalization_axis_breakdown_rows(suite_method_master)
    descriptive_method_rows = {str(row.get("method", "")).strip(): row for row in method_summary}
    gate_decomposition = _gate_decomposition_rows(
        suite_method_master,
        descriptive_method_rows=descriptive_method_rows,
    )

    _write_table(output_dir, "method_summary", method_summary)
    _write_table(output_dir, "model_summary", model_summary)
    _write_table(output_dir, "model_method_summary", model_method_summary)
    _write_table(output_dir, "method_source_summary", method_source_summary)
    _write_table(output_dir, "method_language_summary", method_language_summary)
    _write_table(output_dir, "method_attack_summary", method_attack_summary)
    _write_table(output_dir, "timing_summary", timing_summary)
    _write_table(output_dir, "suite_all_models_methods_model_method_timing", model_method_timing)
    _write_table(output_dir, "suite_all_models_methods_run_inventory", run_inventory)
    _write_table(output_dir, "suite_all_models_methods_method_master_leaderboard", suite_method_master)
    _write_table(output_dir, "suite_all_models_methods_method_model_leaderboard", suite_method_model)
    _write_table(output_dir, "suite_all_models_methods_upstream_only_leaderboard", suite_upstream_only)
    _write_table(output_dir, "suite_all_models_methods_utility_robustness_summary", utility_robustness_summary)
    _write_table(output_dir, "suite_all_models_methods_model_method_functional_quality", model_method_functional_quality)
    _write_table(output_dir, "per_attack_robustness_breakdown", per_attack_robustness_breakdown)
    _write_table(output_dir, "core_vs_stress_robustness_summary", core_vs_stress_robustness_summary)
    _write_table(output_dir, "robustness_factor_decomposition", robustness_factor_decomposition)
    _write_table(output_dir, "utility_factor_decomposition", utility_factor_decomposition)
    _write_table(output_dir, "generalization_axis_breakdown", generalization_axis_breakdown)
    _write_table(output_dir, "gate_decomposition", gate_decomposition)
    score_version = str(method_summary[0].get("score_version", "")).strip() if method_summary else ""
    _write_summary_export_identity(
        output_dir,
        matrix_index_path=matrix_index_path,
        matrix_index=matrix_index,
        score_version=score_version,
        observed_models=[str(row.get("model", "")).strip() for row in model_summary],
    )
    print(f"wrote tables to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
