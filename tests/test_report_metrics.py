from __future__ import annotations

import pytest

from posteditbench.models import BenchmarkRow, ExperimentConfig
from posteditbench.report import _clean_functional_metrics, _stage_timing_metrics, _watermarked_functional_metrics, build_report


def _row(*, example_id: str, model_label: str, passed: bool) -> BenchmarkRow:
    return BenchmarkRow(
        example_id=example_id,
        attack_name="comment_strip",
        task_id=example_id,
        dataset="public_humaneval_plus",
        language="python",
        task_category="strings/parsing",
        reference_kind="canonical",
        method_origin="upstream",
        evaluation_track="generation_time",
        model_label=model_label,
        source_group="public_humaneval_plus",
        origin_type="public",
        difficulty="easy",
        watermark_scheme="sweet_runtime",
        clean_score=0.9,
        attacked_score=0.9,
        clean_detected=True,
        attacked_detected=passed,
        quality_score=1.0 if passed else 0.2,
        stealth_score=1.0,
        mutation_distance=0.0 if passed else 0.8,
        watermark_retention=1.0 if passed else 0.2,
        robustness_score=1.0 if passed else 0.04,
        semantic_validation_available=True,
        semantic_preserving=passed,
        metadata={
            "stage_timing": {
                "clean_generation_seconds": 1.2 if model_label == "model-a" else 0.8,
                "watermarked_generation_seconds": 0.4 if model_label == "model-a" else 0.6,
                "attack_seconds": 0.3 if passed else 0.5,
                "validation_seconds": 0.25 if passed else 0.35,
                "detection_seconds": 0.15 if passed else 0.2,
                "total_example_seconds": 2.3 if passed else 2.45,
                "clean_generation_standardized_token_count": 10,
                "watermarked_generation_standardized_token_count": 12,
                "attacked_standardized_token_count": 14 if passed else 16,
                "clean_generation_line_count": 3,
                "watermarked_line_count": 4,
                "attacked_line_count": 5 if passed else 6,
                "shared_stage_components": {
                    "clean_generation_seconds": 1.2 if model_label == "model-a" else 0.8,
                    "watermarked_generation_seconds": 0.4 if model_label == "model-a" else 0.6,
                    "clean_validation_seconds": 0.1,
                    "watermarked_validation_seconds": 0.08,
                    "negative_control_detection_seconds": 0.04,
                    "watermarked_detection_seconds": 0.05,
                },
                "row_stage_components": {
                    "attack_seconds": 0.3 if passed else 0.5,
                    "attacked_validation_seconds": 0.07 if passed else 0.11,
                    "attacked_detection_seconds": 0.06 if passed else 0.09,
                },
            },
            "clean_functional_trials": [
                {
                    "sample_index": 0,
                    "compile_success": passed,
                    "passed": passed,
                    "error_kind": "" if passed else "compile",
                }
            ],
            "clean_validation": {
                "available": True,
                "passed": passed,
                "metadata": {"compile_success": passed, "error_kind": "" if passed else "compile"},
            },
            "example_metadata": {
                "validation_supported": True,
                "category": "strings/parsing",
                "source_group": "public_humaneval_plus",
            },
        },
    )


def test_report_functional_metrics_keep_same_example_ids_from_different_models() -> None:
    rows = [
        _row(example_id="shared-example", model_label="model-a", passed=True),
        _row(example_id="shared-example", model_label="model-b", passed=False),
    ]

    clean = _clean_functional_metrics(rows)
    watermarked = _watermarked_functional_metrics(rows)

    assert clean["task_count"] == 2
    assert clean["pass@1"] == pytest.approx(0.5)
    assert watermarked["validated_tasks"] == 2
    assert watermarked["pass@1"] == pytest.approx(0.5)


def test_stage_timing_metrics_keep_same_example_ids_from_different_models() -> None:
    rows = [
        _row(example_id="shared-example", model_label="model-a", passed=True),
        _row(example_id="shared-example", model_label="model-b", passed=False),
    ]

    timing = _stage_timing_metrics(rows)

    assert timing["task_count"] == 2
    assert timing["attack_row_count"] == 2
    assert timing["clean_generation_seconds_total"] == pytest.approx(2.0)
    assert timing["watermarked_generation_seconds_total"] == pytest.approx(1.0)
    assert timing["attack_seconds_total"] == pytest.approx(0.8)
    assert timing["validation_seconds_total"] == pytest.approx(0.54)
    assert timing["detection_seconds_total"] == pytest.approx(0.33)
    assert timing["total_example_seconds_total"] == pytest.approx(4.67)


def test_summary_excludes_unsupported_rows_from_attacked_side_metrics() -> None:
    supported = _row(example_id="supported", model_label="model-a", passed=False)
    unsupported_base = _row(example_id="unsupported", model_label="model-a", passed=True)
    unsupported = BenchmarkRow(
        **{
            **unsupported_base.as_dict(),
            "metadata": {
                **dict(unsupported_base.metadata),
                "attack_metadata": {"supported": False, "unsupported_reason": "python_comment_strip_parse_failed"},
            },
        }
    )

    summary = build_report(
        ExperimentConfig(provider_mode="offline_mock", watermark_name="stone_runtime"),
        [supported, unsupported],
        benchmark_manifest={"claimed_languages": ["python"], "coverage": {}},
    ).summary
    supported_only = build_report(
        ExperimentConfig(provider_mode="offline_mock", watermark_name="stone_runtime"),
        [supported],
        benchmark_manifest={"claimed_languages": ["python"], "coverage": {}},
    ).summary

    assert summary["mean_detection_score"] == pytest.approx(supported_only["mean_detection_score"], rel=1e-3)
    assert summary["attacked_detect_rate"] == pytest.approx(supported_only["attacked_detect_rate"], rel=1e-3)
    assert summary["mean_quality_score"] == pytest.approx(supported_only["mean_quality_score"], rel=1e-3)
    assert summary["mean_watermark_retention"] == pytest.approx(supported_only["mean_watermark_retention"], rel=1e-3)
    assert summary["mean_robustness_score"] == pytest.approx(supported_only["mean_robustness_score"], rel=1e-3)
    assert summary["semantic_attack_robustness"] == supported_only["semantic_attack_robustness"]
    assert summary["by_attack"]["comment_strip"]["count"] == pytest.approx(2.0)
    assert summary["by_attack"]["comment_strip"]["supported_count"] == pytest.approx(1.0)
    assert summary["supported_attack_row_count"] == 1
    assert summary["unsupported_attack_row_count"] == 1
    assert summary["attack_support_rate"] == pytest.approx(0.5)
    assert summary["detection_calibration"]["attacked_score_count"] == 1
