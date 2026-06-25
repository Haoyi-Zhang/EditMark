from __future__ import annotations

from posteditbench.baselines import baseline_family_names
from posteditbench.models import BenchmarkRow, ExperimentConfig
from posteditbench.report import build_report
from posteditbench.watermarks.registry import available_watermarks


def test_official_runtime_baselines_are_registered_once():
    expected = ("stone_runtime", "sweet_runtime", "ewd_runtime", "kgw_runtime")
    assert baseline_family_names() == expected
    assert all(name in available_watermarks() for name in expected)


def test_report_includes_baseline_family_and_watermarked_metrics():
    row = BenchmarkRow(
        example_id="example-1",
        attack_name="comment_strip",
        language="python",
        baseline_family="runtime_official",
        baseline_origin="external_checkout",
        baseline_upstream_commit="deadbeef",
        clean_score=0.9,
        attacked_score=0.8,
        clean_detected=True,
        attacked_detected=True,
        quality_score=0.95,
        stealth_score=0.9,
        mutation_distance=0.05,
        watermark_retention=0.88,
        robustness_score=0.83,
        semantic_validation_available=True,
        semantic_preserving=True,
        metadata={
            "clean_validation": {
                "available": True,
                "passed": True,
                "metadata": {"compile_success": True, "error_kind": ""},
            }
        },
    )

    report = build_report(ExperimentConfig(), [row], benchmark_manifest={})

    assert report.summary["by_baseline_family"]["runtime_official"]["count"] == 1.0
    assert report.summary["baseline_families"] == ["runtime_official"]
    assert report.summary["baseline_origins"] == ["external_checkout"]
    assert report.summary["baseline_upstream_commits"] == ["deadbeef"]
    assert report.summary["watermarked_functional_metrics"]["test_pass_rate"] == 1.0
