from __future__ import annotations

import pytest

from posteditbench.leaderboards import (
    build_method_master_leaderboard,
    build_method_model_leaderboard,
    build_reference_track_master_leaderboard,
    build_suite_method_master_leaderboard,
    build_upstream_only_leaderboard,
)
from posteditbench.models import BenchmarkRow


def _row(
    *,
    example_id: str,
    method: str,
    origin: str,
    evaluation_track: str,
    model_label: str,
    source_group: str,
    attack_name: str = "comment_strip",
) -> BenchmarkRow:
    return BenchmarkRow(
        example_id=example_id,
        attack_name=attack_name,
        task_id=example_id,
        dataset=source_group,
        language="python",
        task_category="strings/parsing" if "humaneval" in source_group else "graphs/search",
        reference_kind="canonical",
        method_origin=origin,
        evaluation_track=evaluation_track,
        model_label=model_label,
        source_group=source_group,
        origin_type="public",
        difficulty="medium",
        watermark_scheme=method,
        clean_score=0.9,
        attacked_score=0.8,
        clean_detected=True,
        attacked_detected=True,
        quality_score=0.9,
        stealth_score=0.9,
        mutation_distance=0.1,
        watermark_retention=0.8889,
        robustness_score=0.72,
        semantic_validation_available=True,
        semantic_preserving=True,
        metadata={
            "negative_controls": {
                "human_reference": {"available": True, "score": 0.1, "detected": False, "threshold": 0.5, "metadata": {}},
                "clean_generation": {"available": True, "score": 0.15, "detected": False, "threshold": 0.5, "metadata": {}},
            },
            "clean_functional_trials": [{"sample_index": 0, "compile_success": True, "passed": True, "error_kind": ""}],
            "clean_validation": {"available": True, "passed": True, "metadata": {"compile_success": True, "error_kind": ""}},
            "attacked_validation": {"available": True, "passed": True, "metadata": {"compile_success": True, "error_kind": ""}},
            "example_metadata": {
                "validation_supported": True,
                "category": "strings/parsing" if "humaneval" in source_group else "graphs/search",
                "source_group": source_group,
                "difficulty": "medium",
                "reference_kind": "canonical",
            },
        },
    )


def _report(*rows: BenchmarkRow, name: str) -> dict:
    return {
        "output_path": name,
        "rows": [row.as_dict() for row in rows],
    }


def test_generation_time_master_leaderboard_uses_common_support_and_dedupes_duplicates() -> None:
    reports = [
        _report(
            _row(
                example_id="he-1",
                method="kgw",
                origin="native",
                evaluation_track="generation_time",
                model_label="bigcode/starcoder2-7b",
                source_group="public_humaneval_plus",
            ),
            _row(
                example_id="he-2",
                method="kgw",
                origin="native",
                evaluation_track="generation_time",
                model_label="bigcode/starcoder2-7b",
                source_group="public_humaneval_plus",
            ),
            _row(
                example_id="mbpp-1",
                method="kgw",
                origin="native",
                evaluation_track="generation_time",
                model_label="bigcode/starcoder2-7b",
                source_group="public_mbpp_plus",
                attack_name="identifier_rename",
            ),
            name="kgw_starcoder_primary",
        ),
        _report(
            _row(
                example_id="he-1",
                method="kgw",
                origin="native",
                evaluation_track="generation_time",
                model_label="bigcode/starcoder2-7b",
                source_group="public_humaneval_plus",
            ),
            _row(
                example_id="mbpp-1",
                method="kgw",
                origin="native",
                evaluation_track="generation_time",
                model_label="bigcode/starcoder2-7b",
                source_group="public_mbpp_plus",
                attack_name="identifier_rename",
            ),
            name="kgw_starcoder_duplicate_overlap",
        ),
        _report(
            _row(
                example_id="he-1",
                method="kgw",
                origin="native",
                evaluation_track="generation_time",
                model_label="Qwen/Qwen2.5-Coder-7B-Instruct",
                source_group="public_humaneval_plus",
            ),
            _row(
                example_id="he-2",
                method="kgw",
                origin="native",
                evaluation_track="generation_time",
                model_label="Qwen/Qwen2.5-Coder-7B-Instruct",
                source_group="public_humaneval_plus",
            ),
            _row(
                example_id="mbpp-1",
                method="kgw",
                origin="native",
                evaluation_track="generation_time",
                model_label="Qwen/Qwen2.5-Coder-7B-Instruct",
                source_group="public_mbpp_plus",
                attack_name="identifier_rename",
            ),
            name="kgw_qwen_extra_model",
        ),
        _report(
            _row(
                example_id="he-1",
                method="stone_runtime",
                origin="upstream",
                evaluation_track="generation_time",
                model_label="bigcode/starcoder2-7b",
                source_group="public_humaneval_plus",
            ),
            _row(
                example_id="mbpp-1",
                method="stone_runtime",
                origin="upstream",
                evaluation_track="generation_time",
                model_label="bigcode/starcoder2-7b",
                source_group="public_mbpp_plus",
                attack_name="identifier_rename",
            ),
            name="stone_runtime_starcoder",
        ),
    ]

    leaderboard = build_method_master_leaderboard(reports)
    by_method = {entry["method"]: entry for entry in leaderboard}

    assert set(by_method) == {"kgw", "stone_runtime"}
    assert by_method["kgw"]["evaluation_track"] == "generation_time"
    assert by_method["kgw"]["comparable_models"] == ["bigcode/starcoder2-7b"]
    assert by_method["kgw"]["comparable_source_groups"] == ["public_humaneval_plus", "public_mbpp_plus"]
    assert by_method["kgw"]["comparable_languages"] == ["python"]
    assert by_method["kgw"]["comparable_task_count"] == 2
    assert by_method["kgw"]["row_count"] == 2
    assert by_method["kgw"]["duplicate_rows_removed"] == 2
    assert by_method["kgw"]["noncomparable_rows_removed"] == 4


def test_method_model_leaderboard_intersects_common_support_on_language_and_task_identity() -> None:
    reports = [
        _report(
            _row(
                example_id="shared-python",
                method="kgw",
                origin="native",
                evaluation_track="generation_time",
                model_label="bigcode/starcoder2-7b",
                source_group="public_humaneval_plus",
            ),
            _row(
                example_id="shared-go",
                method="kgw",
                origin="native",
                evaluation_track="generation_time",
                model_label="bigcode/starcoder2-7b",
                source_group="public_humaneval_plus",
            ),
            name="kgw_model_slice",
        ),
        _report(
            _row(
                example_id="shared-python",
                method="stone_runtime",
                origin="upstream",
                evaluation_track="generation_time",
                model_label="bigcode/starcoder2-7b",
                source_group="public_humaneval_plus",
            ),
            name="stone_model_slice",
        ),
    ]
    reports[0]["rows"][1]["language"] = "go"

    leaderboard = build_method_model_leaderboard(reports)
    by_method = {entry["method"]: entry for entry in leaderboard}

    assert by_method["kgw"]["row_count"] == 1
    assert by_method["kgw"]["noncomparable_rows_removed"] == 1
    assert by_method["kgw"]["comparable_languages"] == ["python"]
    assert by_method["kgw"]["comparable_task_count"] == 1


def test_reference_and_generation_tracks_are_reported_separately() -> None:
    reports = [
        _report(
            _row(
                example_id="he-1",
                method="kgw",
                origin="native",
                evaluation_track="reference_code",
                model_label="reference_oracle",
                source_group="public_humaneval_plus",
            ),
            _row(
                example_id="mbpp-1",
                method="kgw",
                origin="native",
                evaluation_track="reference_code",
                model_label="reference_oracle",
                source_group="public_mbpp_plus",
                attack_name="identifier_rename",
            ),
            name="kgw_reference",
        ),
        _report(
            _row(
                example_id="he-1",
                method="comment",
                origin="native",
                evaluation_track="reference_code",
                model_label="reference_oracle",
                source_group="public_humaneval_plus",
            ),
            _row(
                example_id="mbpp-1",
                method="comment",
                origin="native",
                evaluation_track="reference_code",
                model_label="reference_oracle",
                source_group="public_mbpp_plus",
                attack_name="identifier_rename",
            ),
            name="comment_reference",
        ),
        _report(
            _row(
                example_id="he-1",
                method="stone_runtime",
                origin="upstream",
                evaluation_track="generation_time",
                model_label="bigcode/starcoder2-7b",
                source_group="public_humaneval_plus",
            ),
            _row(
                example_id="mbpp-1",
                method="stone_runtime",
                origin="upstream",
                evaluation_track="generation_time",
                model_label="bigcode/starcoder2-7b",
                source_group="public_mbpp_plus",
                attack_name="identifier_rename",
            ),
            name="stone_generation",
        ),
    ]

    reference_master = build_reference_track_master_leaderboard(reports)
    generation_master = build_method_master_leaderboard(reports)
    upstream_only = build_upstream_only_leaderboard(reports)
    method_model = build_method_model_leaderboard(reports)

    assert {entry["method"] for entry in reference_master} == {"kgw", "comment"}
    assert all(entry["evaluation_track"] == "reference_code" for entry in reference_master)
    assert all(entry["comparable_models"] == ["reference_oracle"] for entry in reference_master)

    assert {entry["method"] for entry in generation_master} == {"stone_runtime"}
    assert all(entry["evaluation_track"] == "generation_time" for entry in generation_master)
    assert [entry["method"] for entry in upstream_only] == ["stone_runtime"]

    tracks_by_method_model = {(entry["method"], entry["model"]): entry["evaluation_track"] for entry in method_model}
    assert tracks_by_method_model[("comment", "reference_oracle")] == "reference_code"
    assert tracks_by_method_model[("stone_runtime", "bigcode/starcoder2-7b")] == "generation_time"


def test_suite_master_leaderboard_balances_atomic_source_groups_equally() -> None:
    reports = [
        _report(
            _row(
                example_id="he-good",
                method="stone_runtime",
                origin="upstream",
                evaluation_track="generation_time",
                model_label="bigcode/starcoder2-7b",
                source_group="public_humaneval_plus",
            ),
            *[
                BenchmarkRow(
                    **{
                        **_row(
                            example_id=f"crafted-bad-{index}",
                            method="stone_runtime",
                            origin="upstream",
                            evaluation_track="generation_time",
                            model_label="bigcode/starcoder2-7b",
                            source_group="crafted_translation",
                        ).as_dict(),
                        "attacked_detected": False,
                        "quality_score": 0.1,
                        "stealth_score": 0.1,
                        "watermark_retention": 0.0,
                        "metadata": {
                            **dict(
                                _row(
                                    example_id=f"crafted-bad-{index}",
                                    method="stone_runtime",
                                    origin="upstream",
                                    evaluation_track="generation_time",
                                    model_label="bigcode/starcoder2-7b",
                                    source_group="crafted_translation",
                                ).metadata
                            ),
                            "negative_controls": {
                                "human_reference": {
                                    "available": True,
                                    "score": 0.9,
                                    "detected": True,
                                    "threshold": 0.5,
                                    "metadata": {},
                                },
                                "clean_generation": {
                                    "available": True,
                                    "score": 0.8,
                                    "detected": True,
                                    "threshold": 0.5,
                                    "metadata": {},
                                },
                            },
                            "attacked_validation": {
                                "available": True,
                                "passed": False,
                                "metadata": {"compile_success": False, "error_kind": "assertion"},
                            },
                        },
                    }
                )
                for index in range(6)
            ],
            name="stone_suite_weighting",
        )
    ]

    leaderboard = build_suite_method_master_leaderboard(reports)

    assert len(leaderboard) == 1
    entry = leaderboard[0]
    coverage = dict(entry["score_coverage"])
    assert coverage["aggregation_mode"] == "source_balanced"
    assert coverage["aggregated_source_groups"] == ["crafted_translation", "public_humaneval_plus"]
    assert entry["suite_atomic_source_complete"] is False
    assert set(entry["missing_source_groups"]) == {
        "crafted_original",
        "crafted_stress",
        "public_humaneval_x",
        "public_mbpp_plus",
        "public_mbxp_5lang",
    }
    assert entry["core_score"] == 0.0
    assert entry["headline_core_score"] > 0.0
    assert entry["PostEditScore"] == pytest.approx(
        entry["gate"] * (entry["headline_core_score"] * entry["headline_generalization"]) ** 0.5,
        rel=1e-3,
    )


def test_suite_master_leaderboard_dedupes_known_public_python_overlaps_across_sources() -> None:
    reports = [
        _report(
            _row(
                example_id="he-plus-python",
                method="stone_runtime",
                origin="upstream",
                evaluation_track="generation_time",
                model_label="Qwen/Qwen2.5-Coder-14B-Instruct",
                source_group="public_humaneval_plus",
            ),
            BenchmarkRow(
                **{
                    **_row(
                        example_id="he-x-python",
                        method="stone_runtime",
                        origin="upstream",
                        evaluation_track="generation_time",
                        model_label="Qwen/Qwen2.5-Coder-14B-Instruct",
                        source_group="public_humaneval_x",
                    ).as_dict(),
                    "task_id": "Python/0",
                    "prompt_digest": "shared-humaneval-python",
                    "source_group": "public_humaneval_x",
                    "language": "python",
                }
            ),
            BenchmarkRow(
                **{
                    **_row(
                        example_id="he-x-cpp",
                        method="stone_runtime",
                        origin="upstream",
                        evaluation_track="generation_time",
                        model_label="Qwen/Qwen2.5-Coder-14B-Instruct",
                        source_group="public_humaneval_x",
                    ).as_dict(),
                    "task_id": "CPP/0",
                    "prompt_digest": "humaneval-cpp-0",
                    "source_group": "public_humaneval_x",
                    "language": "cpp",
                }
            ),
            BenchmarkRow(
                **{
                    **_row(
                        example_id="mbpp-plus-python",
                        method="stone_runtime",
                        origin="upstream",
                        evaluation_track="generation_time",
                        model_label="Qwen/Qwen2.5-Coder-14B-Instruct",
                        source_group="public_mbpp_plus",
                    ).as_dict(),
                    "task_id": "Mbpp/2",
                    "prompt_digest": "mbpp-plus-2",
                    "source_group": "public_mbpp_plus",
                    "language": "python",
                }
            ),
            BenchmarkRow(
                **{
                    **_row(
                        example_id="mbxp-python",
                        method="stone_runtime",
                        origin="upstream",
                        evaluation_track="generation_time",
                        model_label="Qwen/Qwen2.5-Coder-14B-Instruct",
                        source_group="public_mbxp_5lang",
                    ).as_dict(),
                    "task_id": "MBPP/2",
                    "prompt_digest": "mbxp-python-2",
                    "source_group": "public_mbxp_5lang",
                    "language": "python",
                }
            ),
            BenchmarkRow(
                **{
                    **_row(
                        example_id="mbxp-cpp",
                        method="stone_runtime",
                        origin="upstream",
                        evaluation_track="generation_time",
                        model_label="Qwen/Qwen2.5-Coder-14B-Instruct",
                        source_group="public_mbxp_5lang",
                    ).as_dict(),
                    "task_id": "CPP/2",
                    "prompt_digest": "mbxp-cpp-2",
                    "source_group": "public_mbxp_5lang",
                    "language": "cpp",
                }
            ),
            _row(
                example_id="crafted-original",
                method="stone_runtime",
                origin="upstream",
                evaluation_track="generation_time",
                model_label="Qwen/Qwen2.5-Coder-14B-Instruct",
                source_group="crafted_original",
            ),
            _row(
                example_id="crafted-translation",
                method="stone_runtime",
                origin="upstream",
                evaluation_track="generation_time",
                model_label="Qwen/Qwen2.5-Coder-14B-Instruct",
                source_group="crafted_translation",
            ),
            _row(
                example_id="crafted-stress",
                method="stone_runtime",
                origin="upstream",
                evaluation_track="generation_time",
                model_label="Qwen/Qwen2.5-Coder-14B-Instruct",
                source_group="crafted_stress",
            ),
            name="suite_overlap_probe",
        )
    ]
    # Force a humaneval-style prompt digest match across HE+ and HumanEval-X python.
    reports[0]["rows"][0]["prompt_digest"] = "shared-humaneval-python"
    leaderboard = build_suite_method_master_leaderboard(reports)

    assert len(leaderboard) == 1
    entry = leaderboard[0]
    assert entry["row_count"] == 7
    assert entry["duplicate_rows_removed"] == 2
    assert entry["suite_atomic_source_complete"] is True
    assert set(entry["comparable_source_groups"]) == {
        "crafted_original",
        "crafted_stress",
        "crafted_translation",
        "public_humaneval_plus",
        "public_humaneval_x",
        "public_mbpp_plus",
        "public_mbxp_5lang",
    }
