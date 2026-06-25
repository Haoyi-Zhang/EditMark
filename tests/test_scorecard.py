from __future__ import annotations

from statistics import mean

import pytest

from posteditbench.models import BenchmarkRow
from posteditbench.scorecard import _clean_functional_metrics, _watermarked_functional_metrics, scorecard_for_rows


def _score_row(
    *,
    example_id: str,
    attack_name: str = "comment_strip",
    model_label: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    source_group: str,
    task_category: str,
    language: str = "python",
    human_detected: bool = False,
    semantic_validation_available: bool = True,
    semantic_preserving: bool | None = True,
    quality_score: float = 1.0,
    stealth_score: float = 1.0,
    watermark_retention: float = 1.0,
    attacked_detected: bool = True,
    attacked_passed: bool = True,
    clean_generation_seconds: float = 1.0,
    watermarked_generation_seconds: float = 1.0,
) -> BenchmarkRow:
    return BenchmarkRow(
        example_id=example_id,
        attack_name=attack_name,
        task_id=example_id,
        dataset=source_group,
        language=language,
        task_category=task_category,
        reference_kind="canonical",
        method_origin="native",
        model_label=model_label,
        source_group=source_group,
        origin_type="public",
        difficulty="easy",
        watermark_scheme="kgw",
        clean_score=0.9,
        watermarked_score=0.9,
        attacked_score=0.9,
        clean_detected=True,
        watermarked_detected=True,
        attacked_detected=attacked_detected,
        quality_score=quality_score,
        stealth_score=stealth_score,
        mutation_distance=0.0,
        watermark_retention=watermark_retention,
        robustness_score=1.0,
        semantic_validation_available=semantic_validation_available,
        semantic_preserving=semantic_preserving,
        metadata={
            "negative_controls": {
                "human_reference": {"available": True, "score": 0.8 if human_detected else 0.1, "detected": human_detected, "threshold": 0.5, "metadata": {}},
                "clean_generation": {"available": True, "score": 0.2, "detected": False, "threshold": 0.5, "metadata": {}},
            },
            "clean_functional_trials": [{"sample_index": 0, "compile_success": True, "passed": True, "error_kind": ""}],
            "clean_validation": {"available": True, "passed": True, "metadata": {"compile_success": True, "error_kind": ""}},
            "watermarked_validation": {"available": True, "passed": True, "metadata": {"compile_success": True, "error_kind": ""}},
            "attacked_validation": {"available": True, "passed": attacked_passed, "metadata": {"compile_success": True, "error_kind": ""}},
            "example_metadata": {"validation_supported": True, "category": task_category},
            "stage_timing": {
                "clean_generation_standardized_token_count": 10,
                "watermarked_generation_standardized_token_count": 10,
                "shared_stage_components": {
                    "clean_generation_seconds": clean_generation_seconds,
                    "watermarked_generation_seconds": watermarked_generation_seconds,
                }
            },
        },
    )


def test_scorecard_hits_perfect_normalized_score_for_perfect_rows() -> None:
    base_examples = [
        dict(
            example_id="e1",
            model_label="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            source_group="public_humaneval_x",
            task_category="strings/parsing",
            language="python",
        ),
        dict(
            example_id="e2",
            model_label="Qwen/Qwen2.5-Coder-14B-Instruct",
            source_group="public_mbxp_5lang",
            task_category="graphs/search",
            language="cpp",
        ),
        dict(
            example_id="e3",
            model_label="bigcode/starcoder2-7b",
            source_group="crafted_original",
            task_category="logic/reasoning",
            language="java",
        ),
        dict(
            example_id="e4",
            model_label="deepseek-ai/deepseek-coder-6.7b-instruct",
            source_group="crafted_translation",
            task_category="math/synthesis",
            language="javascript",
        ),
    ]
    rows = [
        _score_row(attack_name=attack_name, **example)
        for example in base_examples
        for attack_name in ("comment_strip", "identifier_rename", "whitespace_normalize", "noise_insert")
    ]

    scorecard = scorecard_for_rows(rows)

    assert scorecard["detection_separability"] == pytest.approx(1.0)
    assert scorecard["robustness"] == pytest.approx(1.0)
    assert scorecard["utility"] == pytest.approx(1.0)
    assert scorecard["stealth"] == pytest.approx(1.0)
    assert scorecard["efficiency"] == pytest.approx(1.0)
    assert scorecard["core_score"] == pytest.approx(1.0)
    assert scorecard["headline_core_score"] == pytest.approx(1.0)
    assert scorecard["gate"] == pytest.approx(1.0)
    assert scorecard["generalization"] == pytest.approx(1.0)
    assert scorecard["headline_generalization"] == pytest.approx(1.0)
    assert scorecard["generalization_status"] == "supported_nonzero"
    assert scorecard["cross_family_transfer"] == pytest.approx(1.0)
    assert scorecard["scale_consistency"] == pytest.approx(1.0)
    assert scorecard["scale_supported_families"] == ["qwen25"]
    assert scorecard["PostEditScore"] == pytest.approx(1.0)
    assert "slice_core_by_language" not in scorecard
    assert "slice_core_by_family" not in scorecard
    assert "scale_consistency_by_family" not in scorecard
    assert scorecard["score_coverage"]["scale_consistency_by_family"] == {"qwen25": 1.0}


def test_scorecard_gate_penalizes_high_negative_control_false_positive_rate() -> None:
    rows = [
        _score_row(example_id="e1", model_label="model-a", source_group="public_humaneval_plus", task_category="strings/parsing", human_detected=True),
        _score_row(example_id="e2", model_label="model-b", source_group="crafted_translation", task_category="graphs/search", human_detected=False),
    ]

    scorecard = scorecard_for_rows(rows)

    assert scorecard["negative_control_fpr"] == pytest.approx(0.25)
    assert scorecard["gate"] == pytest.approx(0.75)
    assert 0.0 < scorecard["detection_separability"] <= 1.0
    assert scorecard["PostEditScore"] == pytest.approx(
        scorecard["gate"] * (scorecard["headline_core_score"] * scorecard["headline_generalization"]) ** 0.5,
        rel=1e-3,
    )
    assert scorecard["PostEditScore"] < 1.0


def test_scorecard_uses_available_generalization_axes_only() -> None:
    rows = [
        _score_row(
            example_id="e1",
            model_label="Qwen/Qwen2.5-Coder-7B-Instruct",
            source_group="public_humaneval_plus",
            task_category="strings/parsing",
        ),
        _score_row(
            example_id="e2",
            model_label="Qwen/Qwen2.5-Coder-7B-Instruct",
            source_group="crafted_translation",
            task_category="strings/parsing",
        ),
    ]

    scorecard = scorecard_for_rows(rows)

    assert scorecard["source_stability"] == pytest.approx(1.0)
    assert scorecard["task_stability"] is None
    assert scorecard["language_stability"] is None
    assert scorecard["cross_family_transfer"] is None
    assert scorecard["generalization"] == pytest.approx(1.0)
    assert scorecard["headline_generalization"] == pytest.approx(1.0)
    assert scorecard["generalization_status"] == "supported_nonzero"
    assert scorecard["generalization_supported"] is True
    assert scorecard["generalization_available_axes"] == ["source_stability"]


def test_scorecard_exposes_local_view_without_generalization_penalty() -> None:
    row = _score_row(
        example_id="e1",
        model_label="model-a",
        source_group="public_humaneval_plus",
        task_category="strings/parsing",
        semantic_validation_available=False,
        semantic_preserving=None,
    )

    scorecard = scorecard_for_rows([row], include_generalization=False)

    assert scorecard["generalization"] is None
    assert scorecard["generalization_supported"] is False
    assert scorecard["headline_generalization"] == pytest.approx(0.5)
    assert scorecard["generalization_status"] == "unsupported"
    assert scorecard["generalization_available_axes"] == []
    assert scorecard["semantic_validation_rate"] == pytest.approx(0.0)
    assert scorecard["declared_semantic_validation_rate"] == pytest.approx(1.0)
    assert scorecard["utility"] < 1.0
    assert scorecard["PostEditScore"] == pytest.approx(
        scorecard["gate"] * (scorecard["headline_core_score"] * scorecard["headline_generalization"]) ** 0.5,
        rel=1e-3,
    )


def test_scorecard_penalizes_missing_clean_generation_negatives_for_generation_time_rows() -> None:
    row = _score_row(
        example_id="e1",
        model_label="model-a",
        source_group="public_humaneval_plus",
        task_category="strings/parsing",
    )
    row = BenchmarkRow(
        **{
            **row.as_dict(),
            "evaluation_track": "generation_time",
            "metadata": {
                **dict(row.metadata),
                "provider_mode": "local_hf",
                "negative_controls": {
                    "human_reference": {"available": True, "score": 0.1, "detected": False, "threshold": 0.5, "metadata": {}},
                    "clean_generation": {"available": False, "score": 0.0, "detected": False, "threshold": 0.5, "metadata": {}},
                },
                "example_metadata": {
                    **dict(dict(row.metadata).get("example_metadata", {})),
                    "provider_generation_succeeded": True,
                },
            },
        }
    )

    scorecard = scorecard_for_rows([row])

    assert scorecard["negative_control_support_rate"] == pytest.approx(0.5)
    assert scorecard["gate"] == pytest.approx(0.5)
    assert scorecard["PostEditScore"] < scorecard["core_score"]


def test_scorecard_does_not_award_detection_credit_without_available_negative_controls() -> None:
    row = _score_row(
        example_id="e1",
        model_label="model-a",
        source_group="public_humaneval_plus",
        task_category="strings/parsing",
    )
    row = BenchmarkRow(
        **{
            **row.as_dict(),
            "metadata": {
                **dict(row.metadata),
                "negative_controls": {
                    "human_reference": {"applicable": True, "available": False, "score": 0.0, "detected": False, "threshold": 0.5, "metadata": {}},
                    "clean_generation": {"applicable": True, "available": False, "score": 0.0, "detected": False, "threshold": 0.5, "metadata": {}},
                },
            },
        }
    )

    scorecard = scorecard_for_rows([row])

    assert scorecard["negative_control_support_rate"] == pytest.approx(0.0)
    assert scorecard["negative_control_fpr"] == pytest.approx(1.0)
    assert scorecard["detection_separability"] == pytest.approx(0.0)
    assert scorecard["gate"] == pytest.approx(0.0)
    assert scorecard["PostEditScore"] == pytest.approx(0.0)


def test_scorecard_preserves_nonzero_public_robustness_when_one_attack_factor_collapses() -> None:
    row = _score_row(
        example_id="e1",
        model_label="model-a",
        source_group="public_humaneval_plus",
        task_category="strings/parsing",
        attacked_detected=False,
    )

    scorecard = scorecard_for_rows([row], include_generalization=False)

    assert scorecard["robustness"] == pytest.approx(0.6667, rel=1e-3)
    assert scorecard["raw_robustness_strict"] == pytest.approx(0.0)
    assert scorecard["robustness_status"] == "supported_nonzero"
    assert scorecard["core_score"] > 0.0
    assert scorecard["headline_core_score"] > 0.0
    assert scorecard["generalization"] is None
    assert scorecard["headline_generalization"] == pytest.approx(0.5)
    assert scorecard["generalization_status"] == "unsupported"
    assert scorecard["PostEditScore"] == pytest.approx(
        scorecard["gate"] * (scorecard["headline_core_score"] * scorecard["headline_generalization"]) ** 0.5,
        rel=1e-3,
    )


def test_scorecard_keeps_public_utility_scalar_separate_from_support_rate() -> None:
    row = _score_row(
        example_id="e1",
        model_label="model-a",
        source_group="public_humaneval_plus",
        task_category="strings/parsing",
        semantic_validation_available=False,
        semantic_preserving=None,
    )

    scorecard = scorecard_for_rows([row], include_generalization=False)

    assert scorecard["utility"] == pytest.approx(0.5, rel=1e-3)
    assert scorecard["utility_support_rate"] == pytest.approx(2.0 / 3.0, rel=1e-3)
    assert scorecard["utility_status"] == "supported_nonzero"


def test_scorecard_excludes_unsupported_slices_from_public_generalization_and_source_balancing() -> None:
    supported = _score_row(
        example_id="supported",
        model_label="Qwen/Qwen2.5-Coder-7B-Instruct",
        source_group="public_humaneval_plus",
        task_category="strings/parsing",
    )
    unsupported_base = _score_row(
        example_id="unsupported",
        model_label="Qwen/Qwen2.5-Coder-7B-Instruct",
        source_group="crafted_translation",
        task_category="strings/parsing",
    )
    unsupported = BenchmarkRow(
        **{
            **unsupported_base.as_dict(),
            "metadata": {
                **dict(unsupported_base.metadata),
                "attack_metadata": {"supported": False},
            },
        }
    )

    scorecard = scorecard_for_rows([supported, unsupported])
    balanced = scorecard_for_rows([supported, unsupported], include_generalization=False, balance_by_source_group=True)
    supported_only = scorecard_for_rows([supported], include_generalization=False, balance_by_source_group=True)

    assert scorecard["robustness"] == pytest.approx(1.0)
    assert scorecard["source_stability"] is None
    assert scorecard["generalization"] is None
    assert scorecard["generalization_status"] == "unsupported"
    assert scorecard["headline_generalization"] == pytest.approx(0.5)
    assert balanced["core_score"] == pytest.approx(supported_only["core_score"], rel=1e-3)


def test_source_balanced_headline_ignores_unsupported_sources_for_public_core_pillars() -> None:
    supported_zero = _score_row(
        example_id="supported-zero",
        model_label="Qwen/Qwen2.5-Coder-7B-Instruct",
        source_group="public_humaneval_plus",
        task_category="strings/parsing",
        quality_score=0.72,
        watermark_retention=0.0,
        attacked_detected=False,
        attacked_passed=False,
    )
    unsupported_strong_base = _score_row(
        example_id="unsupported-strong",
        model_label="Qwen/Qwen2.5-Coder-7B-Instruct",
        source_group="crafted_translation",
        task_category="strings/parsing",
        quality_score=0.95,
    )
    unsupported_strong = BenchmarkRow(
        **{
            **unsupported_strong_base.as_dict(),
            "metadata": {
                **dict(unsupported_strong_base.metadata),
                "attack_metadata": {"supported": False},
            },
        }
    )

    balanced = scorecard_for_rows(
        [supported_zero, unsupported_strong],
        include_generalization=False,
        balance_by_source_group=True,
    )
    supported_only = scorecard_for_rows(
        [supported_zero],
        include_generalization=False,
        balance_by_source_group=True,
    )

    assert balanced["robustness"] == pytest.approx(supported_only["robustness"], rel=1e-3)
    assert balanced["utility"] == pytest.approx(supported_only["utility"], rel=1e-3)
    assert balanced["stealth"] == pytest.approx(supported_only["stealth"], rel=1e-3)
    assert balanced["efficiency"] == pytest.approx(supported_only["efficiency"], rel=1e-3)
    assert balanced["headline_core_score"] == pytest.approx(supported_only["headline_core_score"], rel=1e-3)
    assert balanced["PostEditScore"] == pytest.approx(supported_only["PostEditScore"], rel=1e-3)


def test_scorecard_excludes_unsupported_rows_from_default_utility_path() -> None:
    supported = _score_row(
        example_id="supported",
        model_label="Qwen/Qwen2.5-Coder-7B-Instruct",
        source_group="public_humaneval_plus",
        task_category="strings/parsing",
        quality_score=0.2,
        attacked_detected=False,
        attacked_passed=False,
        watermark_retention=0.0,
    )
    unsupported_base = _score_row(
        example_id="unsupported",
        model_label="Qwen/Qwen2.5-Coder-7B-Instruct",
        source_group="crafted_translation",
        task_category="strings/parsing",
        quality_score=1.0,
        attacked_detected=True,
        attacked_passed=True,
        watermark_retention=1.0,
    )
    unsupported = BenchmarkRow(
        **{
            **unsupported_base.as_dict(),
            "metadata": {
                **dict(unsupported_base.metadata),
                "attack_metadata": {"supported": False, "unsupported_reason": "python_comment_strip_parse_failed"},
            },
        }
    )

    scorecard = scorecard_for_rows([supported, unsupported], include_generalization=False)
    supported_only = scorecard_for_rows([supported], include_generalization=False)

    assert scorecard["utility"] == pytest.approx(supported_only["utility"], rel=1e-3)
    assert scorecard["quality_score_mean"] == pytest.approx(supported_only["quality_score_mean"], rel=1e-3)
    assert scorecard["semantic_preservation_rate"] == pytest.approx(supported_only["semantic_preservation_rate"], rel=1e-3)
    assert scorecard["attacked_detected_semantic_rate"] == pytest.approx(
        supported_only["attacked_detected_semantic_rate"],
        rel=1e-3,
    )
    assert scorecard["attack_supported_row_count"] == 1
    assert scorecard["attack_unsupported_row_count"] == 1
    assert scorecard["attack_support_rate"] == pytest.approx(0.5)
    assert scorecard["PostEditScore"] == pytest.approx(supported_only["PostEditScore"], rel=1e-3)


def test_scorecard_exposes_supported_zero_generalization_status_and_softened_headline() -> None:
    rows = [
        _score_row(
            example_id="e1",
            model_label="model-a",
            source_group="public_humaneval_plus",
            task_category="strings/parsing",
        ),
        _score_row(
            example_id="e2",
            model_label="model-a",
            source_group="crafted_translation",
            task_category="graphs/search",
            quality_score=0.0,
            attacked_detected=False,
            attacked_passed=False,
            watermark_retention=0.0,
        ),
    ]

    scorecard = scorecard_for_rows(rows)

    assert scorecard["generalization_supported"] is True
    assert scorecard["generalization"] == pytest.approx(0.0)
    assert scorecard["generalization_status"] == "supported_zero"
    assert scorecard["headline_generalization"] == pytest.approx(0.05)
    assert scorecard["raw_generalization_strict"] == pytest.approx(0.0)
    assert scorecard["raw_composite_strict"] == pytest.approx(0.0)
    assert scorecard["PostEditScore"] > 0.0


def test_scorecard_treats_all_zero_supported_axes_as_supported_zero_not_unsupported() -> None:
    rows = []
    for task_category in ("strings/parsing", "graphs/search"):
        for index in range(20):
            rows.append(
                _score_row(
                    example_id=f"{task_category}-{index}",
                    model_label="Qwen/Qwen2.5-Coder-7B-Instruct" if index % 2 == 0 else "bigcode/starcoder2-7b",
                    source_group="public_humaneval_x" if index % 4 < 2 else "crafted_translation",
                    task_category=task_category,
                    language="python" if index % 2 == 0 else "java",
                    quality_score=0.0,
                    attacked_detected=False,
                    attacked_passed=False,
                    watermark_retention=0.0,
                )
            )

    scorecard = scorecard_for_rows(rows)

    assert scorecard["generalization_supported"] is True
    assert scorecard["generalization_status"] == "supported_zero"
    assert scorecard["generalization"] == pytest.approx(0.0)
    assert scorecard["headline_generalization"] == pytest.approx(0.05)
    assert scorecard["generalization_available_axes"] == [
        "source_stability",
        "task_stability",
        "language_stability",
        "cross_family_transfer",
    ]


def test_scorecard_efficiency_uses_generation_overhead_and_is_conditioned_by_utility() -> None:
    row = _score_row(
        example_id="e1",
        model_label="model-a",
        source_group="public_humaneval_plus",
        task_category="strings/parsing",
        quality_score=0.64,
        clean_generation_seconds=1.0,
        watermarked_generation_seconds=2.0,
    )

    scorecard = scorecard_for_rows([row], include_generalization=False)

    assert scorecard["efficiency"] == pytest.approx(0.5)
    assert scorecard["utility"] == pytest.approx(0.88, rel=1e-3)
    assert scorecard["raw_utility_strict"] == pytest.approx(0.8618, rel=1e-3)
    assert scorecard["efficiency_conditioned"] == pytest.approx((0.5 * scorecard["utility"]) ** 0.5, rel=1e-3)
    assert scorecard["PostEditScore"] < scorecard["gate"]


def test_language_stability_ignores_python_only_sources() -> None:
    rows = [
        _score_row(
            example_id="multi-python",
            source_group="public_humaneval_x",
            task_category="strings/parsing",
            language="python",
            quality_score=1.0,
        ),
        _score_row(
            example_id="multi-cpp",
            source_group="public_humaneval_x",
            task_category="strings/parsing",
            language="cpp",
            quality_score=0.25,
        ),
        _score_row(
            example_id="public-python",
            source_group="public_mbpp_plus",
            task_category="strings/parsing",
            language="python",
            quality_score=0.05,
            stealth_score=0.2,
            watermark_retention=0.2,
            attacked_detected=False,
            attacked_passed=False,
        ),
    ]

    scorecard = scorecard_for_rows(rows)

    assert scorecard["language_stability"] is not None
    assert scorecard["language_stability"] < 1.0


def test_cross_family_transfer_equal_weights_families_instead_of_models() -> None:
    qwen_1 = _score_row(
        example_id="q1",
        model_label="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        source_group="crafted_original",
        task_category="strings/parsing",
    )
    qwen_2 = _score_row(
        example_id="q2",
        model_label="Qwen/Qwen2.5-Coder-7B-Instruct",
        source_group="crafted_original",
        task_category="strings/parsing",
    )
    qwen_3 = _score_row(
        example_id="q3",
        model_label="Qwen/Qwen2.5-Coder-14B-Instruct",
        source_group="crafted_original",
        task_category="strings/parsing",
    )
    starcoder = _score_row(
        example_id="s1",
        model_label="bigcode/starcoder2-7b",
        source_group="crafted_original",
        task_category="strings/parsing",
        quality_score=0.015625,
    )
    deepseek = _score_row(
        example_id="d1",
        model_label="deepseek-ai/deepseek-coder-6.7b-instruct",
        source_group="crafted_original",
        task_category="strings/parsing",
        quality_score=0.015625,
    )

    scorecard = scorecard_for_rows([qwen_1, qwen_2, qwen_3, starcoder, deepseek])

    qwen_family = mean(
        [
            scorecard_for_rows([qwen_1], include_generalization=False)["core_score"],
            scorecard_for_rows([qwen_2], include_generalization=False)["core_score"],
            scorecard_for_rows([qwen_3], include_generalization=False)["core_score"],
        ]
    )
    star_family = scorecard_for_rows([starcoder], include_generalization=False)["core_score"]
    deep_family = scorecard_for_rows([deepseek], include_generalization=False)["core_score"]
    expected = ((qwen_family * star_family * deep_family) ** (1.0 / 3.0)) / (
        (qwen_family + star_family + deep_family) / 3.0
    )
    assert scorecard["cross_family_transfer"] == pytest.approx(expected, rel=1e-3)
    assert scorecard["generalization_available_axes"] == ["cross_family_transfer"]
    assert scorecard["scale_consistency"] == pytest.approx(1.0)


def test_scale_consistency_is_released_but_not_counted_as_a_generalization_axis() -> None:
    base_rows = [
        _score_row(
            example_id="q1",
            model_label="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            source_group="public_humaneval_x",
            task_category="graphs/search",
            language="python",
        ),
        _score_row(
            example_id="q2",
            model_label="Qwen/Qwen2.5-Coder-14B-Instruct",
            source_group="public_mbxp_5lang",
            task_category="graphs/search",
            language="cpp",
        ),
        _score_row(
            example_id="s1",
            model_label="bigcode/starcoder2-7b",
            source_group="crafted_translation",
            task_category="graphs/search",
            language="java",
        ),
        _score_row(
            example_id="d1",
            model_label="deepseek-ai/deepseek-coder-6.7b-instruct",
            source_group="crafted_stress",
            task_category="graphs/search",
            language="javascript",
        ),
    ]
    scaled_variant_rows = [
        base_rows[0],
        _score_row(
            example_id="q2-low",
            model_label="Qwen/Qwen2.5-Coder-14B-Instruct",
            source_group="public_mbxp_5lang",
            task_category="graphs/search",
            language="cpp",
            quality_score=0.0625,
        ),
        base_rows[2],
        base_rows[3],
    ]

    base_scorecard = scorecard_for_rows(base_rows)
    variant_scorecard = scorecard_for_rows(scaled_variant_rows)

    assert "scale" not in base_scorecard["generalization_available_axes"]
    assert "scale" not in variant_scorecard["generalization_available_axes"]
    assert base_scorecard["scale_supported_family_count"] == 1
    assert variant_scorecard["scale_supported_family_count"] == 1
    assert variant_scorecard["scale_consistency"] < base_scorecard["scale_consistency"]


def test_scorecard_can_balance_atomic_sources_without_row_count_domination() -> None:
    good = _score_row(
        example_id="good",
        model_label="model-a",
        source_group="public_humaneval_plus",
        task_category="strings/parsing",
    )
    bad_rows = []
    for index in range(6):
        bad_rows.append(
            _score_row(
                example_id=f"bad-{index}",
                model_label="model-a",
                source_group="crafted_translation",
                task_category="graphs/search",
                human_detected=True,
                attacked_detected=False,
                quality_score=0.1,
                stealth_score=0.1,
                watermark_retention=0.0,
                attacked_passed=False,
                watermarked_generation_seconds=4.0,
            )
        )

    unbalanced = scorecard_for_rows([good, *bad_rows])
    balanced = scorecard_for_rows([good, *bad_rows], balance_by_source_group=True)

    assert balanced["score_coverage"]["aggregation_mode"] == "source_balanced"
    assert balanced["score_coverage"]["aggregated_source_groups"] == ["crafted_translation", "public_humaneval_plus"]
    assert balanced["score_coverage"]["source_balanced_sources"]["crafted_translation"]["core_score"] >= 0.0
    assert balanced["score_coverage"]["source_balanced_sources"]["public_humaneval_plus"]["core_score"] >= 0.0
    expected_core = (
        balanced["score_coverage"]["source_balanced_sources"]["crafted_translation"]["core_score"]
        + balanced["score_coverage"]["source_balanced_sources"]["public_humaneval_plus"]["core_score"]
    ) / 2.0
    expected_gate = min(
        balanced["watermarked_pass_preservation"],
        1.0 - balanced["negative_control_fpr"],
        balanced["negative_control_support_rate"],
    )
    assert balanced["core_score"] == pytest.approx(expected_core, rel=1e-3)
    assert balanced["gate"] == pytest.approx(expected_gate, rel=1e-3)
    assert balanced["core_score"] > unbalanced["core_score"]


def test_functional_helpers_report_expected_per_task_pass_rates() -> None:
    rows = [
        _score_row(example_id="e1", model_label="model-a", source_group="public_humaneval_plus", task_category="strings/parsing"),
        _score_row(example_id="e2", model_label="model-a", source_group="public_humaneval_plus", task_category="strings/parsing"),
    ]

    clean = _clean_functional_metrics(rows)
    watermarked = _watermarked_functional_metrics(rows)

    assert clean["compile_success_rate"] == pytest.approx(1.0)
    assert clean["test_pass_rate"] == pytest.approx(1.0)
    assert clean["pass@1"] == pytest.approx(1.0)
    assert watermarked["test_pass_rate"] == pytest.approx(1.0)
    assert watermarked["pass@1"] == pytest.approx(1.0)
