from __future__ import annotations

from dataclasses import replace
import json
import sys
from pathlib import Path

import pytest

from posteditbench.models import BenchmarkRow, ExperimentConfig
from posteditbench.report import build_report
from posteditbench.suite import SUITE_AGGREGATE_SOURCE_GROUPS, SUITE_MODEL_ROSTER
from scripts import export_full_run_tables


def _suite_row(
    *,
    example_id: str,
    method: str,
    model_label: str,
    source_group: str,
) -> BenchmarkRow:
    return BenchmarkRow(
        example_id=example_id,
        task_id=example_id,
        attack_name="budgeted_adaptive",
        dataset=source_group,
        source_group=source_group,
        language="python" if "crafted" not in source_group else "java",
        task_category="strings/parsing" if "public" in source_group else "graphs/search",
        reference_kind="canonical",
        evaluation_track="generation_time",
        method_origin="upstream",
        watermark_scheme=method,
        model_label=model_label,
        clean_score=0.92,
        attacked_score=0.83,
        clean_detected=True,
        attacked_detected=True,
        quality_score=0.88,
        stealth_score=0.9,
        mutation_distance=0.12,
        watermark_retention=0.9022,
        robustness_score=0.73,
        semantic_validation_available=True,
        semantic_preserving=True,
        metadata={
            "stage_timing": {
                "clean_generation_seconds": 1.25,
                "watermarked_generation_seconds": 0.75,
                "attack_seconds": 0.4,
                "validation_seconds": 0.3,
                "detection_seconds": 0.2,
                "total_example_seconds": 2.9,
                "clean_generation_standardized_token_count": 11,
                "watermarked_generation_standardized_token_count": 13,
                "attacked_standardized_token_count": 15,
                "clean_generation_line_count": 4,
                "watermarked_line_count": 5,
                "attacked_line_count": 6,
                "shared_stage_components": {
                    "clean_generation_seconds": 1.25,
                    "watermarked_generation_seconds": 0.75,
                    "clean_validation_seconds": 0.12,
                    "watermarked_validation_seconds": 0.09,
                    "negative_control_detection_seconds": 0.05,
                    "watermarked_detection_seconds": 0.04,
                },
                "row_stage_components": {
                    "attack_seconds": 0.4,
                    "attacked_validation_seconds": 0.09,
                    "attacked_detection_seconds": 0.11,
                },
            },
            "negative_controls": {
                "human_reference": {"available": True, "score": 0.08, "detected": False, "threshold": 0.5, "metadata": {}},
                "clean_generation": {"available": True, "score": 0.14, "detected": False, "threshold": 0.5, "metadata": {}},
            },
            "clean_functional_trials": [
                {"sample_index": 0, "compile_success": True, "passed": True, "error_kind": ""},
            ],
            "clean_functional_summary": {"compile_success_rate": 1.0, "test_pass_rate": 1.0},
            "clean_validation": {"available": True, "passed": True, "metadata": {"compile_success": True, "error_kind": ""}},
            "watermarked_validation": {"available": True, "passed": True, "metadata": {"compile_success": True, "error_kind": ""}},
            "attacked_validation": {"available": True, "passed": True, "metadata": {"compile_success": True, "error_kind": ""}},
            "example_metadata": {
                "validation_supported": True,
                "category": "strings/parsing" if "public" in source_group else "graphs/search",
                "source_group": source_group,
                "difficulty": "medium",
                "reference_kind": "canonical",
            },
        },
    )


def _write_report(path: Path, *, method: str, model_label: str) -> None:
    rows = [
        _suite_row(
            example_id=f"{model_label}-{method}-{index}",
            method=method,
            model_label=model_label,
            source_group=source_group,
        )
        for index, source_group in enumerate(SUITE_AGGREGATE_SOURCE_GROUPS, start=1)
    ]
    report = build_report(ExperimentConfig(provider_mode="local_hf", watermark_name=method), rows, benchmark_manifest={})
    path.write_text(report.to_json(), encoding="utf-8")


def _write_shared_example_report(
    path: Path,
    *,
    method: str,
    model_label: str,
    clean_score: float,
    attacked_score: float,
    quality_score: float,
) -> None:
    rows = [
        _suite_row(
            example_id=f"shared-{index}",
            method=method,
            model_label=model_label,
            source_group=source_group,
        )
        for index, source_group in enumerate(SUITE_AGGREGATE_SOURCE_GROUPS, start=1)
    ]
    adjusted_rows: list[BenchmarkRow] = []
    for row in rows:
        adjusted_rows.append(
            replace(
                row,
                clean_score=clean_score,
                attacked_score=attacked_score,
                quality_score=quality_score,
                mutation_distance=round(1.0 - quality_score, 4),
                watermark_retention=round(attacked_score / clean_score, 4),
                robustness_score=round(attacked_score * quality_score, 4),
            )
        )
    report = build_report(ExperimentConfig(provider_mode="local_hf", watermark_name=method), adjusted_rows, benchmark_manifest={})
    path.write_text(report.to_json(), encoding="utf-8")


def test_export_full_run_tables_writes_leaderboards_and_functional_quality_table(tmp_path: Path, monkeypatch) -> None:
    reports: list[Path] = []
    methods = ["stone_runtime", "sweet_runtime", "stone_runtime", "sweet_runtime", "stone_runtime"]
    for index, model_name in enumerate(SUITE_MODEL_ROSTER, start=1):
        report_path = tmp_path / f"report_{index}.json"
        _write_report(report_path, method=methods[index - 1], model_label=model_name)
        reports.append(report_path)

    matrix_index = tmp_path / "matrix_index.json"
    matrix_index.write_text(
        json.dumps(
                {
                    "code_snapshot_digest": "snapshot-digest",
                    "execution_environment_fingerprint": "env-fingerprint",
                    "runs": [
                        {"status": "success", "report_path": str(report_path), "duration_seconds": 10.0 + index}
                        for index, report_path in enumerate(reports, start=1)
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "tables"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_full_run_tables.py",
            "--matrix-index",
            str(matrix_index),
            "--output-dir",
            str(output_dir),
        ],
    )
    monkeypatch.setattr(export_full_run_tables, "_require_canonical_suite_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(export_full_run_tables, "_require_formal_single_host_execution_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(export_full_run_tables, "_require_complete_suite_atomic_roster", lambda *args, **kwargs: None)

    assert export_full_run_tables.main() == 0

    expected = {
        "suite_all_models_methods_method_master_leaderboard.json",
        "suite_all_models_methods_method_model_leaderboard.json",
        "suite_all_models_methods_upstream_only_leaderboard.json",
        "suite_all_models_methods_model_method_functional_quality.json",
        "suite_all_models_methods_utility_robustness_summary.json",
        "suite_all_models_methods_model_method_timing.json",
        "method_language_summary.json",
        "per_attack_robustness_breakdown.json",
        "core_vs_stress_robustness_summary.json",
        "robustness_factor_decomposition.json",
        "utility_factor_decomposition.json",
        "generalization_axis_breakdown.json",
        "gate_decomposition.json",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})

    functional_rows = json.loads((output_dir / "suite_all_models_methods_model_method_functional_quality.json").read_text(encoding="utf-8"))
    assert functional_rows
    assert functional_rows[0]["table_role"] == "paper_functional_quality"
    assert {row["method"] for row in functional_rows} == {"STONE", "SWEET"}
    assert "clean_pass@1" in functional_rows[0]
    assert "watermarked_pass_preservation" in functional_rows[0]
    assert "attacked_pass_preservation" in functional_rows[0]

    timing_rows = json.loads((output_dir / "suite_all_models_methods_model_method_timing.json").read_text(encoding="utf-8"))
    assert timing_rows
    assert timing_rows[0]["table_role"] == "paper_timing"
    assert timing_rows[0]["score_semantics"] == "generation_stage_timing_descriptive_only"
    assert "clean_generation_hours_total" in timing_rows[0]
    assert "total_example_seconds_mean_per_task" in timing_rows[0]
    assert "clean_generation_seconds_per_1k_token" in timing_rows[0]
    assert "watermarked_generation_seconds_per_1k_token" in timing_rows[0]

    language_rows = json.loads((output_dir / "method_language_summary.json").read_text(encoding="utf-8"))
    assert language_rows
    assert language_rows[0]["table_role"] == "repo_descriptive_method_language_rollup"
    assert language_rows[0]["score_semantics"] == "scorecard_recomputed_from_grouped_benchmark_rows"
    assert {row["method"] for row in language_rows} == {"STONE", "SWEET"}
    assert "language" in language_rows[0]
    assert "PostEditScore" in language_rows[0]
    assert "headline_core_score" in language_rows[0]
    assert "generalization_status" in language_rows[0]
    assert "detection_separability" in language_rows[0]
    assert "efficiency" in language_rows[0]

    descriptive_rows = json.loads((output_dir / "method_summary.json").read_text(encoding="utf-8"))
    assert descriptive_rows
    assert descriptive_rows[0]["table_role"] == "repo_descriptive_method_rollup"
    assert descriptive_rows[0]["aggregation_view"] == "descriptive_method_rollup"
    assert descriptive_rows[0]["score_semantics"] == "scorecard_recomputed_from_grouped_benchmark_rows"
    assert "headline_generalization" in descriptive_rows[0]
    assert "PostEditScore" in descriptive_rows[0]

    export_identity = json.loads(
        (output_dir / "suite_all_models_methods_export_identity.json").read_text(encoding="utf-8")
    )
    assert export_identity["artifact_role"] == "suite_all_models_methods_release_summary_export_identity"
    assert [entry["model"] for entry in export_identity["model_roster"]] == list(SUITE_MODEL_ROSTER)
    revision_by_model = {
        entry["model"]: entry["model_revision"]
        for entry in export_identity["model_roster"]
    }
    assert revision_by_model["Qwen/Qwen2.5-Coder-7B-Instruct"] == "c03e6d358207e414f1eca0bb1891e29f1db0e242"
    assert export_identity["matrix_code_snapshot_digest"] == "snapshot-digest"
    assert isinstance(export_identity["code_snapshot_digest"], str)
    assert len(export_identity["code_snapshot_digest"]) == 64
    assert export_identity["required_figure_hashes"] == {}

    master_rows = json.loads(
        (output_dir / "suite_all_models_methods_method_master_leaderboard.json").read_text(encoding="utf-8")
    )
    assert master_rows
    assert master_rows[0]["table_role"] == "paper_method_master_leaderboard"
    assert master_rows[0]["aggregation_view"] == "suite_method_master_leaderboard"
    assert master_rows[0]["score_semantics"] == "scorecard_recomputed_from_grouped_benchmark_rows"

    model_leaderboard_rows = json.loads(
        (output_dir / "suite_all_models_methods_method_model_leaderboard.json").read_text(encoding="utf-8")
    )
    assert model_leaderboard_rows
    assert model_leaderboard_rows[0]["table_role"] == "paper_method_model_leaderboard"
    assert model_leaderboard_rows[0]["aggregation_view"] == "suite_method_model_leaderboard"
    assert model_leaderboard_rows[0]["score_semantics"] == "scorecard_recomputed_from_grouped_benchmark_rows"

    upstream_rows = json.loads(
        (output_dir / "suite_all_models_methods_upstream_only_leaderboard.json").read_text(encoding="utf-8")
    )
    assert upstream_rows
    assert upstream_rows[0]["table_role"] == "paper_upstream_only_leaderboard"
    assert upstream_rows[0]["aggregation_view"] == "suite_upstream_only_leaderboard"
    assert upstream_rows[0]["score_semantics"] == "scorecard_recomputed_from_grouped_benchmark_rows"

    model_summary_rows = json.loads((output_dir / "model_summary.json").read_text(encoding="utf-8"))
    assert model_summary_rows
    assert model_summary_rows[0]["score_semantics"] == "descriptive_mean_of_model_method_scorecard_rollups"

    utility_robustness_rows = json.loads(
        (output_dir / "suite_all_models_methods_utility_robustness_summary.json").read_text(encoding="utf-8")
    )
    assert utility_robustness_rows
    assert utility_robustness_rows[0]["table_role"] == "paper_utility_robustness_summary"
    assert utility_robustness_rows[0]["aggregation_view"] == "suite_method_utility_robustness_summary"
    assert "utility_robustness_drop" in utility_robustness_rows[0]
    assert "robustness" in utility_robustness_rows[0]
    assert "utility" in utility_robustness_rows[0]

    timing_summary_rows = json.loads((output_dir / "timing_summary.json").read_text(encoding="utf-8"))
    assert timing_summary_rows
    assert timing_summary_rows[0]["score_semantics"] == "generation_stage_timing_descriptive_only"
    assert "clean_generation_seconds_per_1k_token" in timing_summary_rows[0]
    assert "watermarked_generation_seconds_per_1k_token" in timing_summary_rows[0]

    attack_breakdown_rows = json.loads(
        (output_dir / "per_attack_robustness_breakdown.json").read_text(encoding="utf-8")
    )
    assert attack_breakdown_rows
    assert attack_breakdown_rows[0]["table_role"] == "repo_attack_robustness_breakdown"
    assert "attack_tier" in attack_breakdown_rows[0]
    assert "attack_robustness" in attack_breakdown_rows[0]
    assert "raw_attack_robustness_strict" in attack_breakdown_rows[0]

    core_vs_stress_rows = json.loads(
        (output_dir / "core_vs_stress_robustness_summary.json").read_text(encoding="utf-8")
    )
    assert core_vs_stress_rows
    assert core_vs_stress_rows[0]["table_role"] == "repo_core_vs_stress_robustness_summary"
    assert "stress_robustness" in core_vs_stress_rows[0]

    gate_rows = json.loads((output_dir / "gate_decomposition.json").read_text(encoding="utf-8"))
    assert gate_rows
    assert gate_rows[0]["table_role"] == "repo_gate_decomposition"
    assert "negative_control_fpr" in gate_rows[0]


def test_summary_export_identity_uses_release_environment_fingerprint(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    environment_dir = repo_root / "results" / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    (environment_dir / "runtime_environment.json").write_text(
        json.dumps({"execution": {"execution_environment_fingerprint": "release-env-fingerprint"}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "tables"
    output_dir.mkdir()
    for filename in export_full_run_tables._SUMMARY_EXPORT_IDENTITY_TABLES:
        (output_dir / filename).write_text("[]\n", encoding="utf-8")
    matrix_index_path = tmp_path / "matrix_index.json"
    matrix_index_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(export_full_run_tables, "ROOT", repo_root)
    monkeypatch.setattr(export_full_run_tables, "_repo_snapshot", None)

    export_full_run_tables._write_summary_export_identity(
        output_dir,
        matrix_index_path=matrix_index_path,
        matrix_index={
            "code_snapshot_digest": "matrix-code-digest",
            "execution_environment_fingerprint": "matrix-env-fingerprint",
            "manifest": "configs/matrices/suite_all_models_methods.json",
            "profile": "suite_all_models_methods",
            "canonical_manifest_digest": export_full_run_tables._CANONICAL_SUITE_MANIFEST_DIGEST,
            "execution_mode": "single_host_canonical",
            "run_count": 140,
            "success_count": 140,
            "failed_count": 0,
        },
        score_version="test-score-v1",
        observed_models=list(SUITE_MODEL_ROSTER),
    )

    export_identity = json.loads(
        (output_dir / "suite_all_models_methods_export_identity.json").read_text(encoding="utf-8")
    )
    assert export_identity["execution_environment_fingerprint"] == "release-env-fingerprint"
    assert export_identity["matrix_code_snapshot_digest"] == "matrix-code-digest"


def test_export_full_run_tables_rejects_success_run_without_report_path(tmp_path: Path, monkeypatch) -> None:
    matrix_index = tmp_path / "matrix_index.json"
    matrix_index.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_id": "broken_success",
                        "status": "success",
                        "report_path": str(tmp_path / "missing_report.json"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "tables"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_full_run_tables.py",
            "--matrix-index",
            str(matrix_index),
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(SystemExit, match="success_without_report"):
        export_full_run_tables.main()


def test_export_full_run_tables_rejects_resume_existing_report_runs(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "report.json"
    _write_report(report_path, method="stone_runtime", model_label="Qwen/Qwen2.5-Coder-7B-Instruct")
    matrix_index = tmp_path / "matrix_index.json"
    matrix_index.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_id": "resumed",
                        "status": "skipped",
                        "reason": "resume_existing_report",
                        "report_path": str(report_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "tables"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_full_run_tables.py",
            "--matrix-index",
            str(matrix_index),
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(SystemExit, match="only success runs"):
        export_full_run_tables.main()


def test_export_full_run_tables_rejects_success_resume_existing_report_runs(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "report.json"
    _write_report(report_path, method="stone_runtime", model_label="Qwen/Qwen2.5-Coder-7B-Instruct")
    matrix_index = tmp_path / "matrix_index.json"
    matrix_index.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_id": "resumed-success",
                        "status": "success",
                        "reason": "resume_existing_report",
                        "report_path": str(report_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "tables"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_full_run_tables.py",
            "--matrix-index",
            str(matrix_index),
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(SystemExit, match="only success runs"):
        export_full_run_tables.main()


def test_export_full_run_tables_stays_canonical_only_even_with_multiple_successful_reports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_a = tmp_path / "report_a.json"
    report_b = tmp_path / "report_b.json"
    _write_report(report_a, method="stone_runtime", model_label="bigcode/starcoder2-7b")
    _write_report(report_b, method="sweet_runtime", model_label="Qwen/Qwen2.5-Coder-7B-Instruct")

    matrix_index = tmp_path / "matrix_index.json"
    matrix_index.write_text(
        json.dumps(
            {
                "runs": [
                    {"status": "success", "report_path": str(report_a), "duration_seconds": 12.0},
                    {"status": "success", "report_path": str(report_b), "duration_seconds": 14.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "tables"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_full_run_tables.py",
            "--matrix-index",
            str(matrix_index),
            "--output-dir",
            str(output_dir),
        ],
    )
    monkeypatch.setattr(export_full_run_tables, "_require_canonical_suite_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(export_full_run_tables, "_require_formal_single_host_execution_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(export_full_run_tables, "_require_complete_suite_atomic_roster", lambda *args, **kwargs: None)
    monkeypatch.setattr(export_full_run_tables, "_write_summary_export_identity", lambda *args, **kwargs: None)

    assert export_full_run_tables.main() == 0

    canonical_rows = json.loads(
        (output_dir / "suite_all_models_methods_method_master_leaderboard.json").read_text(encoding="utf-8")
    )
    assert {row["method"] for row in canonical_rows} == {"STONE", "SWEET"}
    assert not any("common_support" in path.name.lower() for path in output_dir.iterdir())


def test_export_full_run_tables_model_summary_is_order_invariant_across_methods(tmp_path: Path, monkeypatch) -> None:
    report_a = tmp_path / "report_a.json"
    report_b = tmp_path / "report_b.json"
    _write_shared_example_report(
        report_a,
        method="stone_runtime",
        model_label="Qwen/Qwen2.5-Coder-7B-Instruct",
        clean_score=0.95,
        attacked_score=0.9,
        quality_score=0.9,
    )
    _write_shared_example_report(
        report_b,
        method="sweet_runtime",
        model_label="Qwen/Qwen2.5-Coder-7B-Instruct",
        clean_score=0.65,
        attacked_score=0.5,
        quality_score=0.55,
    )

    matrix_a = tmp_path / "matrix_a.json"
    matrix_b = tmp_path / "matrix_b.json"
    matrix_a.write_text(
        json.dumps(
            {
                "runs": [
                    {"status": "success", "report_path": str(report_a), "duration_seconds": 10.0},
                    {"status": "success", "report_path": str(report_b), "duration_seconds": 14.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    matrix_b.write_text(
        json.dumps(
            {
                "runs": [
                    {"status": "success", "report_path": str(report_b), "duration_seconds": 14.0},
                    {"status": "success", "report_path": str(report_a), "duration_seconds": 10.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    output_a = tmp_path / "tables_a"
    output_b = tmp_path / "tables_b"
    monkeypatch.setattr(export_full_run_tables, "_require_canonical_suite_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(export_full_run_tables, "_require_formal_single_host_execution_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(export_full_run_tables, "_require_complete_suite_atomic_roster", lambda *args, **kwargs: None)
    monkeypatch.setattr(export_full_run_tables, "_write_summary_export_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["export_full_run_tables.py", "--matrix-index", str(matrix_a), "--output-dir", str(output_a)])
    assert export_full_run_tables.main() == 0
    monkeypatch.setattr(sys, "argv", ["export_full_run_tables.py", "--matrix-index", str(matrix_b), "--output-dir", str(output_b)])
    assert export_full_run_tables.main() == 0

    model_summary_a = json.loads((output_a / "model_summary.json").read_text(encoding="utf-8"))
    model_summary_b = json.loads((output_b / "model_summary.json").read_text(encoding="utf-8"))
    assert model_summary_a == model_summary_b
    assert model_summary_a[0]["method_count"] == 2
    assert set(model_summary_a[0]["methods"]) == {"STONE", "SWEET"}


def test_model_summary_marks_mixed_generalization_status_as_descriptive_mixed() -> None:
    rows = [
        {
            "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "method": "STONE",
            "row_count": 7,
            "report_count": 1,
            "duration_seconds_total": 10.0,
            "task_count": 7,
            "attack_row_count": 7,
            "clean_generation_seconds_total": 1.0,
            "watermarked_generation_seconds_total": 1.0,
            "attack_seconds_total": 1.0,
            "validation_seconds_total": 1.0,
            "detection_seconds_total": 1.0,
            "total_example_seconds_total": 5.0,
            "clean_generation_seconds_per_1k_token": 1.0,
            "watermarked_generation_seconds_per_1k_token": 1.0,
            "PostEditScore": 0.2,
            "detection_separability": 0.5,
            "robustness": 0.1,
            "utility": 0.4,
            "stealth": 0.6,
            "efficiency": 0.7,
            "stealth_conditioned": 0.3,
            "efficiency_conditioned": 0.4,
            "core_score": 0.3,
            "headline_core_score": 0.3,
            "generalization": None,
            "headline_generalization": 0.5,
            "generalization_supported": False,
            "generalization_status": "unsupported",
            "generalization_available_axes": [],
            "scale_supported_families": [],
            "scale_supported_family_count": 0,
            "negative_control_fpr": 0.0,
            "negative_control_support_rate": 1.0,
            "clean_compile_success_rate": 1.0,
            "clean_test_pass_rate": 1.0,
            "clean_pass@1": 1.0,
            "watermarked_test_pass_rate": 1.0,
            "watermarked_pass@1": 1.0,
            "attacked_test_pass_rate": 1.0,
            "watermarked_pass_preservation": 1.0,
            "attacked_pass_preservation": 1.0,
            "gate": 1.0,
            "score_version": "v-test",
        },
        {
            "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "method": "SWEET",
            "row_count": 7,
            "report_count": 1,
            "duration_seconds_total": 12.0,
            "task_count": 7,
            "attack_row_count": 7,
            "clean_generation_seconds_total": 1.0,
            "watermarked_generation_seconds_total": 1.0,
            "attack_seconds_total": 1.0,
            "validation_seconds_total": 1.0,
            "detection_seconds_total": 1.0,
            "total_example_seconds_total": 5.0,
            "clean_generation_seconds_per_1k_token": 1.0,
            "watermarked_generation_seconds_per_1k_token": 1.0,
            "PostEditScore": 0.1,
            "detection_separability": 0.4,
            "robustness": 0.0,
            "utility": 0.2,
            "stealth": 0.5,
            "efficiency": 0.6,
            "stealth_conditioned": 0.2,
            "efficiency_conditioned": 0.3,
            "core_score": 0.2,
            "headline_core_score": 0.2,
            "generalization": 0.0,
            "headline_generalization": 0.05,
            "generalization_supported": True,
            "generalization_status": "supported_zero",
            "generalization_available_axes": ["source"],
            "scale_supported_families": [],
            "scale_supported_family_count": 0,
            "negative_control_fpr": 0.0,
            "negative_control_support_rate": 1.0,
            "clean_compile_success_rate": 1.0,
            "clean_test_pass_rate": 1.0,
            "clean_pass@1": 1.0,
            "watermarked_test_pass_rate": 1.0,
            "watermarked_pass@1": 1.0,
            "attacked_test_pass_rate": 1.0,
            "watermarked_pass_preservation": 1.0,
            "attacked_pass_preservation": 1.0,
            "gate": 1.0,
            "score_version": "v-test",
        },
    ]

    aggregated = export_full_run_tables._aggregate_model_summary(rows)

    assert aggregated[0]["generalization_status"] == "descriptive_mixed"
    assert aggregated[0]["generalization_supported"] is True
    assert aggregated[0]["headline_generalization"] == pytest.approx(0.275)


def test_export_full_run_tables_rejects_incomplete_suite_atomic_roster(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "partial_report.json"
    partial_rows = [
        _suite_row(
            example_id="partial-1",
            method="stone_runtime",
            model_label="Qwen/Qwen2.5-Coder-7B-Instruct",
            source_group="humaneval_plus",
        )
    ]
    report = build_report(ExperimentConfig(provider_mode="local_hf", watermark_name="stone_runtime"), partial_rows, benchmark_manifest={})
    report_path.write_text(report.to_json(), encoding="utf-8")

    matrix_index = tmp_path / "matrix_index.json"
    matrix_index.write_text(
        json.dumps(
            {
                "runs": [
                    {"run_id": "partial", "status": "success", "report_path": str(report_path), "duration_seconds": 12.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "tables"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_full_run_tables.py",
            "--matrix-index",
            str(matrix_index),
            "--output-dir",
            str(output_dir),
        ],
    )
    monkeypatch.setattr(export_full_run_tables, "_require_canonical_suite_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(export_full_run_tables, "_require_formal_single_host_execution_mode", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit, match="requires a complete result-backed suite matrix|requires complete seven-source suite coverage"):
        export_full_run_tables.main()


def test_require_canonical_suite_identity_rejects_sharded_index_without_digest() -> None:
    with pytest.raises(SystemExit, match="missing canonical_manifest_digest"):
        export_full_run_tables._require_canonical_suite_identity(
            {
                "manifest": "configs/matrices/suite_all_models_methods.json",
                "profile": "suite_all_models_methods",
                "run_count": 140,
                "execution_mode": "sharded_identical_execution_class",
                "shard_profiles": ["suite_all_models_methods_shard_01_of_02"],
            },
            label="exact-value exports",
        )


def test_require_canonical_suite_identity_rejects_canonical_index_without_execution_mode() -> None:
    with pytest.raises(SystemExit, match="missing execution_mode"):
        export_full_run_tables._require_canonical_suite_identity(
            {
                "manifest": "configs/matrices/suite_all_models_methods.json",
                "profile": "suite_all_models_methods",
                "run_count": 140,
                "canonical_manifest_digest": export_full_run_tables._CANONICAL_SUITE_MANIFEST_DIGEST,
            },
            label="exact-value exports",
        )


def test_require_canonical_suite_identity_rejects_sharded_index_without_code_snapshot_digest() -> None:
    with pytest.raises(SystemExit, match="missing code_snapshot_digest"):
        export_full_run_tables._require_canonical_suite_identity(
            {
                "manifest": "configs/matrices/suite_all_models_methods.json",
                "profile": "suite_all_models_methods",
                "run_count": 140,
                "canonical_manifest_digest": export_full_run_tables._CANONICAL_SUITE_MANIFEST_DIGEST,
                "execution_mode": "sharded_identical_execution_class",
                "shard_profiles": ["suite_all_models_methods_shard_01_of_02"],
                "execution_environment_fingerprint": "fp",
            },
            label="exact-value exports",
        )


def test_require_canonical_suite_identity_rejects_sharded_index_without_execution_environment_fingerprint() -> None:
    with pytest.raises(SystemExit, match="missing execution_environment_fingerprint"):
        export_full_run_tables._require_canonical_suite_identity(
            {
                "manifest": "configs/matrices/suite_all_models_methods.json",
                "profile": "suite_all_models_methods",
                "run_count": 140,
                "canonical_manifest_digest": export_full_run_tables._CANONICAL_SUITE_MANIFEST_DIGEST,
                "execution_mode": "sharded_identical_execution_class",
                "shard_profiles": ["suite_all_models_methods_shard_01_of_02"],
                "code_snapshot_digest": "snapshot-digest",
            },
            label="exact-value exports",
        )


def test_require_generation_time_release_contract_rejects_row_level_track_drift() -> None:
    report = build_report(
        ExperimentConfig(provider_mode="local_hf", watermark_name="stone_runtime"),
        [
            _suite_row(
                example_id="row-drift",
                method="stone_runtime",
                model_label="Qwen/Qwen2.5-Coder-7B-Instruct",
                source_group="humaneval_plus",
            )
        ],
        benchmark_manifest={},
    ).as_dict()
    report["summary"]["evaluation_tracks"] = [export_full_run_tables.GENERATION_TIME_TRACK]
    report["summary"]["paper_primary_track"] = export_full_run_tables.GENERATION_TIME_TRACK
    report["summary"]["paper_track_ready"] = True
    report["rows"][0]["evaluation_track"] = "functional_quality"

    with pytest.raises(SystemExit, match=r"row\[1\]\.evaluation_track=functional_quality"):
        export_full_run_tables._require_generation_time_release_contract(
            [report],
            label="suite_all_models_methods exact-value exports",
        )


def test_require_formal_single_host_execution_mode_rejects_incomplete_assembly_provenance() -> None:
    with pytest.raises(SystemExit, match="must provide both assembly_source_execution_modes and assembly_source_indexes"):
        export_full_run_tables._require_formal_single_host_execution_mode(
            {
                "execution_mode": "single_host_canonical",
                "code_snapshot_digest": "snapshot-digest",
                "execution_environment_fingerprint": "env-fp",
                "assembly_source_execution_modes": ["single_host_canonical", "sharded_identical_execution_class"],
            },
            label="suite_all_models_methods exact-value exports",
        )


def test_require_formal_single_host_execution_mode_rejects_missing_assembly_arrays() -> None:
    with pytest.raises(SystemExit, match="must retain non-empty assembly_source_execution_modes and assembly_source_indexes"):
        export_full_run_tables._require_formal_single_host_execution_mode(
            {
                "execution_mode": "single_host_canonical",
                "code_snapshot_digest": "snapshot-digest",
                "execution_environment_fingerprint": "env-fp",
            },
            label="suite_all_models_methods exact-value exports",
        )


def test_require_formal_single_host_execution_mode_rejects_inconsistent_assembly_provenance() -> None:
    with pytest.raises(SystemExit, match="assembly provenance is inconsistent"):
        export_full_run_tables._require_formal_single_host_execution_mode(
            {
                "execution_mode": "single_host_canonical",
                "code_snapshot_digest": "snapshot-digest",
                "execution_environment_fingerprint": "env-fp",
                "assembly_source_execution_modes": ["single_host_canonical", "sharded_identical_execution_class"],
                "assembly_source_indexes": [
                    {"execution_mode": "single_host_canonical"},
                ],
            },
            label="suite_all_models_methods exact-value exports",
        )


def test_require_formal_single_host_execution_mode_rejects_non_single_host_source_modes() -> None:
    with pytest.raises(SystemExit, match="must be backed only by single_host_canonical source indexes"):
        export_full_run_tables._require_formal_single_host_execution_mode(
            {
                "execution_mode": "single_host_canonical",
                "code_snapshot_digest": "snapshot-digest",
                "execution_environment_fingerprint": "env-fp",
                "assembly_source_execution_modes": ["sharded_identical_execution_class"],
                "assembly_source_indexes": [
                    {"execution_mode": "sharded_identical_execution_class"},
                ],
            },
            label="suite_all_models_methods exact-value exports",
        )


def test_require_formal_single_host_execution_mode_rejects_multi_segment_single_host_assembly() -> None:
    with pytest.raises(SystemExit, match="exactly one source index"):
        export_full_run_tables._require_formal_single_host_execution_mode(
            {
                "execution_mode": "single_host_canonical",
                "code_snapshot_digest": "snapshot-digest",
                "execution_environment_fingerprint": "env-fp",
                "assembly_source_execution_modes": ["single_host_canonical"],
                "assembly_source_indexes": [
                    {"execution_mode": "single_host_canonical"},
                    {"execution_mode": "single_host_canonical"},
                ],
            },
            label="suite_all_models_methods exact-value exports",
        )
