from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from posteditbench.suite import (
    ACTIVE_SUITE_LIMITS,
    OFFICIAL_RUNTIME_BASELINES,
    OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES,
    SUITE_ATOMIC_SOURCE_ORDER,
    SUITE_MODEL_REVISIONS,
    SUITE_MODEL_ROSTER,
)
from scripts import build_suite_manifests


def _manifest(path_name: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs" / "matrices" / path_name).read_text(encoding="utf-8"))


def _normalize_language(row: dict) -> str:
    return str(row.get("language", "")).strip().lower()


def test_suite_full_manifest_matches_models_methods_and_atomic_sources() -> None:
    manifest = _manifest("suite_all_models_methods.json")

    assert manifest["profile"] == "suite_all_models_methods"
    assert manifest["schema_version"] == 1
    assert manifest["model_roster"] == list(SUITE_MODEL_ROSTER)
    assert manifest["model_revisions"] == dict(SUITE_MODEL_REVISIONS)
    assert manifest["method_roster"] == list(OFFICIAL_RUNTIME_BASELINES)
    assert manifest["atomic_benchmark_sources"] == list(SUITE_ATOMIC_SOURCE_ORDER)
    assert len(manifest["runs"]) == len(SUITE_MODEL_ROSTER) * len(OFFICIAL_RUNTIME_BASELINES) * len(SUITE_ATOMIC_SOURCE_ORDER)
    assert all(run["resource"] == "gpu" for run in manifest["runs"])
    assert all(run["gpu_pool"] == "runtime" for run in manifest["runs"])
    assert all(run["baseline_eval"] is True for run in manifest["runs"])


def test_suite_full_manifest_uses_complete_public_sources_and_balanced_crafted_subsets() -> None:
    manifest = _manifest("suite_all_models_methods.json")

    expected_limits = {
        "HumanEval+": 164,
        "MBPP+": ACTIVE_SUITE_LIMITS["mbpp_plus"],
        "HumanEval-X (5-language balanced slice)": ACTIVE_SUITE_LIMITS["humaneval_x"],
        "MBXP-5lang (5-language balanced slice)": ACTIVE_SUITE_LIMITS["mbxp_5lang"],
        "Crafted Original": ACTIVE_SUITE_LIMITS["crafted_original"],
        "Crafted Translation": ACTIVE_SUITE_LIMITS["crafted_translation"],
        "Crafted Stress": ACTIVE_SUITE_LIMITS["crafted_stress"],
    }
    for run in manifest["runs"]:
        benchmark = dict(run["config_overrides"]["benchmark"])
        dataset_label = str(benchmark["dataset_label"])
        assert int(benchmark["limit"]) == expected_limits[dataset_label]


def test_suite_stage_a_manifest_uses_heavy_model_across_atomic_sources() -> None:
    manifest = _manifest("suite_canary_heavy.json")

    assert manifest["profile"] == "suite_canary_heavy"
    assert len(manifest["model_roster"]) == 1
    assert manifest["method_roster"] == list(OFFICIAL_RUNTIME_BASELINES)
    assert manifest["atomic_benchmark_sources"] == list(SUITE_ATOMIC_SOURCE_ORDER)
    assert len(manifest["runs"]) == len(OFFICIAL_RUNTIME_BASELINES) * len(SUITE_ATOMIC_SOURCE_ORDER)
    for run in manifest["runs"]:
        benchmark = dict(run["config_overrides"]["benchmark"])
        dataset_label = str(benchmark["dataset_label"])
        if dataset_label in {"HumanEval+", "MBPP+"}:
            assert int(benchmark["limit"]) == 12
        else:
            assert int(benchmark["limit"]) == 15
        if dataset_label in {"HumanEval-X (5-language balanced slice)", "MBXP-5lang (5-language balanced slice)", "Crafted Original", "Crafted Translation", "Crafted Stress"}:
            assert set(benchmark["languages"]) == set(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES)
        assert int(run["baseline_eval_sample_limit"]) == 16


def test_suite_stage_b_manifest_smokes_remaining_models_on_full_atomic_roster() -> None:
    manifest = _manifest("model_invocation_smoke.json")

    assert manifest["profile"] == "model_invocation_smoke"
    assert len(manifest["model_roster"]) == len(SUITE_MODEL_ROSTER) - 1
    assert manifest["benchmark_roster"] == [
        "HumanEval+",
        "MBPP+",
        "HumanEval-X (5-language balanced slice)",
        "MBXP-5lang (5-language balanced slice)",
        "Crafted Original",
        "Crafted Translation",
        "Crafted Stress",
    ]
    assert manifest["atomic_benchmark_sources"] == list(SUITE_ATOMIC_SOURCE_ORDER)
    assert len(manifest["runs"]) == (len(SUITE_MODEL_ROSTER) - 1) * len(OFFICIAL_RUNTIME_BASELINES) * len(SUITE_ATOMIC_SOURCE_ORDER)
    for run in manifest["runs"]:
        benchmark = dict(run["config_overrides"]["benchmark"])
        assert int(benchmark["limit"]) == 2
        dataset_label = str(benchmark["dataset_label"])
        if dataset_label in {"HumanEval-X (5-language balanced slice)", "MBXP-5lang (5-language balanced slice)", "Crafted Original", "Crafted Translation", "Crafted Stress"}:
            assert set(benchmark["languages"]) == set(OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES)
        assert int(run["baseline_eval_sample_limit"]) == 4


def test_release_prepared_sources_match_active_release_sizes_and_multilingual_language_roster() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "data/release/sources/suite_humaneval_plus_release.normalized.jsonl": (164, {"python"}),
        "data/release/sources/suite_mbpp_plus_release.normalized.jsonl": (378, {"python"}),
        "data/release/sources/suite_humanevalx_release.normalized.jsonl": (200, {"python", "cpp", "java", "javascript", "go"}),
        "data/release/sources/suite_mbxp_release.normalized.jsonl": (200, {"python", "cpp", "java", "javascript", "go"}),
        "data/release/sources/crafted_original_release.normalized.jsonl": (240, {"python", "cpp", "java", "javascript", "go"}),
        "data/release/sources/crafted_translation_release.normalized.jsonl": (240, {"python", "cpp", "java", "javascript", "go"}),
        "data/release/sources/crafted_stress_release.normalized.jsonl": (240, {"python", "cpp", "java", "javascript", "go"}),
    }
    for relative_path, (expected_count, expected_languages) in expected.items():
        path = root / relative_path
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == expected_count
        languages = {_normalize_language(row) for row in rows}
        assert languages == expected_languages
        if "crafted_original_release" in relative_path:
            assert {str(row.get("dataset", "")).strip() for row in rows} == {"Crafted Original"}
        if "crafted_translation_release" in relative_path:
            assert {str(row.get("dataset", "")).strip() for row in rows} == {"Crafted Translation"}
        if "crafted_stress_release" in relative_path:
            assert {str(row.get("dataset", "")).strip() for row in rows} == {"Crafted Stress"}
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        assert int(manifest["record_count"]) == expected_count
        assert int(manifest["canonical_reference_count"]) + int(manifest.get("smoke_overlay_reference_count", 0)) == expected_count
        if "suite_mbxp_release" in relative_path:
            assert int(manifest["canonical_reference_count"]) < expected_count
            assert int(manifest.get("smoke_overlay_reference_count", 0)) > 0
        else:
            assert int(manifest["canonical_reference_count"]) == expected_count
        if len(expected_languages) > 1:
            language_counts = Counter(_normalize_language(row) for row in rows)
            assert set(language_counts.values()) == {expected_count // len(expected_languages)}


def test_crafted_release_sources_keep_balanced_family_slices() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "data/release/sources/crafted_original_release.normalized.jsonl",
        "data/release/sources/crafted_translation_release.normalized.jsonl",
        "data/release/sources/crafted_stress_release.normalized.jsonl",
    ):
        path = root / relative_path
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        assert int(manifest["family_count"]) == 48
        assert manifest["suite_selection_policy"]["type"] == "category_balanced_canonical_release_source"


def test_crafted_release_manifests_use_public_labels_and_shared_provenance_note() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "crafted_original_release.normalized.manifest.json": "Crafted Original",
        "crafted_translation_release.normalized.manifest.json": "Crafted Translation",
        "crafted_stress_release.normalized.manifest.json": "Crafted Stress",
    }
    expected_note = (
        "Crafted sources use project-authored curated benchmark families together with cross-language review, "
        "deterministic release checks, and manually finalized public release records."
    )

    for file_name, expected_label in expected.items():
        manifest_path = root / "data" / "release" / "sources" / file_name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["dataset_label"] == expected_label
        assert manifest["construction_note"] == expected_note
        assert all(source["dataset_label"] == expected_label for source in manifest["source_manifests"])
        if "datasets" in manifest:
            assert manifest["datasets"] == [expected_label]


def test_crafted_release_manifests_share_the_schema_v2_family_level_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    for file_name in (
        "crafted_original_release.normalized.manifest.json",
        "crafted_translation_release.normalized.manifest.json",
        "crafted_stress_release.normalized.manifest.json",
    ):
        manifest_path = root / "data" / "release" / "sources" / file_name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_manifest = manifest["source_manifests"][0]

        assert manifest["schema_version"] == 2
        assert manifest["task_count_per_family"] == 5
        assert manifest["family_language_coverage_rate"] == pytest.approx(1.0)
        assert manifest["contract_drift_families"] == []
        assert manifest["category_counts"] == source_manifest["category_counts"]
        assert manifest["template_family_counts"] == source_manifest["template_family_counts"]
        assert manifest["inputs"] == [manifest["normalized_path"]]
        assert manifest["input_filtered_counts"] == {manifest["normalized_path"]: manifest["record_count"]}
        assert set(manifest["observed_languages"]) == set(manifest["languages"])
        assert manifest["quota_per_language"] == manifest["language_counts"]
        assert manifest["quota_per_source_group"] == {}
        assert manifest["reference_kind_total"] == manifest["record_count"]
        assert manifest["sample_ids_path"].endswith(".sample_ids.json")
        assert manifest["coverage"]["observed_language_count"] == len(manifest["claimed_languages"])
        assert sum(manifest["validation_backend_counts"].values()) == manifest["record_count"]


def test_suite_manifests_align_sampling_seed_per_model_source_within_stage() -> None:
    for manifest_name in ("suite_all_models_methods.json", "suite_canary_heavy.json", "model_invocation_smoke.json"):
        manifest = _manifest(manifest_name)
        seeds_by_slice: dict[tuple[str, str], set[int]] = {}
        for run in manifest["runs"]:
            project = dict(run["config_overrides"]["project"])
            benchmark = dict(run["config_overrides"]["benchmark"])
            watermark = dict(run["config_overrides"]["watermark"])
            model_name = str(watermark["model_name"])
            source_group = str(benchmark["source_group"])
            key = (model_name, source_group)
            seeds_by_slice.setdefault(key, set()).add(int(project["seed"]))
        assert all(len(seeds) == 1 for seeds in seeds_by_slice.values())


def test_canonical_suite_manifests_pin_run_level_model_revisions() -> None:
    for manifest_name in ("suite_all_models_methods.json", "suite_canary_heavy.json", "model_invocation_smoke.json"):
        manifest = _manifest(manifest_name)
        for run in manifest["runs"]:
            model_name = str(run["model"])
            watermark = dict(run["config_overrides"]["watermark"])
            assert run["model_revision"] == SUITE_MODEL_REVISIONS[model_name]
            assert watermark["revision"] == SUITE_MODEL_REVISIONS[model_name]


def test_suite_full_manifest_uses_release_paths_without_stale_collection_sources() -> None:
    manifest = _manifest("suite_all_models_methods.json")

    release_runs = [
        run
        for run in manifest["runs"]
        if "data/release/sources/" in str(run["config_overrides"]["benchmark"].get("prepared_output", ""))
    ]

    assert release_runs
    assert all("collection_sources" not in dict(run["config_overrides"]["benchmark"]) for run in release_runs)


def test_suite_full_manifest_prioritizes_heavier_model_method_source_combinations() -> None:
    manifest = _manifest("suite_all_models_methods.json")
    by_run_id = {run["run_id"]: run for run in manifest["runs"]}

    qwen14_ewd_crafted_original = by_run_id["suite_qwen25_14b_crafted_original_ewd_runtime"]
    qwen14_stone_humaneval = by_run_id["suite_qwen25_14b_heplus_stone_runtime"]
    starcoder_stone_humaneval = by_run_id["suite_starcoder2_7b_heplus_stone_runtime"]

    assert int(qwen14_ewd_crafted_original["priority"]) > int(qwen14_stone_humaneval["priority"])
    assert int(qwen14_stone_humaneval["priority"]) > int(starcoder_stone_humaneval["priority"])


def test_canonical_suite_manifests_keep_runtime_roster_canonical_only() -> None:
    for manifest_name in ("suite_all_models_methods.json", "suite_canary_heavy.json", "model_invocation_smoke.json"):
        manifest = _manifest(manifest_name)
        assert manifest["method_roster"] == list(OFFICIAL_RUNTIME_BASELINES)
        assert all(run["method"] in OFFICIAL_RUNTIME_BASELINES for run in manifest["runs"])


def test_subset_manifest_payload_uses_subset_profile_and_subset_required_methods() -> None:
    selected_runs = build_suite_manifests._filter_run_items(
        build_suite_manifests._suite_run_items(),
        models=["Qwen/Qwen2.5-Coder-14B-Instruct"],
        methods=["sweet_runtime"],
        sources=["crafted_original"],
    )

    payload = build_suite_manifests._subset_manifest_payload(
        profile="suite_reviewer_subset",
        description="subset smoke",
        runs=selected_runs,
    )

    assert payload["profile"] == "suite_reviewer_subset"
    assert payload["model_roster"] == ["Qwen/Qwen2.5-Coder-14B-Instruct"]
    assert payload["model_revisions"] == {
        "Qwen/Qwen2.5-Coder-14B-Instruct": SUITE_MODEL_REVISIONS["Qwen/Qwen2.5-Coder-14B-Instruct"]
    }
    assert payload["benchmark_roster"] == ["Crafted Original"]
    assert payload["atomic_benchmark_sources"] == ["crafted_original"]
    assert payload["method_roster"] == ["sweet_runtime"]
    assert payload["required_watermark_methods"] == ["sweet_runtime"]
    assert len(payload["runs"]) == 1
    assert all(run["profile"] == "suite_reviewer_subset" for run in payload["runs"])
    assert payload["runs"][0]["config_overrides"]["watermark"]["revision"] == SUITE_MODEL_REVISIONS["Qwen/Qwen2.5-Coder-14B-Instruct"]


def test_subset_manifest_payload_applies_limit_override_without_expanding_scope() -> None:
    selected_runs = build_suite_manifests._filter_run_items(
        build_suite_manifests._suite_run_items(),
        models=["Qwen/Qwen2.5-Coder-14B-Instruct"],
        methods=["sweet_runtime"],
        sources=["crafted_original"],
    )

    payload = build_suite_manifests._subset_manifest_payload(
        profile="suite_reviewer_subset_limit",
        description="subset smoke",
        runs=selected_runs,
        limit=8,
    )

    assert payload["profile"] == "suite_reviewer_subset_limit"
    assert payload["method_roster"] == ["sweet_runtime"]
    assert len(payload["runs"]) == 1
    run = payload["runs"][0]
    assert int(run["config_overrides"]["benchmark"]["limit"]) == 8
    assert int(run["baseline_eval_sample_limit"]) == 8


def test_subset_manifest_cli_accepts_relative_output_manifest(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    output_manifest = root / "configs" / "matrices" / "test_relative_subset_manifest.json"
    if output_manifest.exists():
        output_manifest.unlink()
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_suite_manifests.py",
            "--output-manifest",
            "configs/matrices/test_relative_subset_manifest.json",
            "--profile",
            "suite_reviewer_subset_relative",
            "--models",
            "Qwen/Qwen2.5-Coder-14B-Instruct",
            "--methods",
            "sweet_runtime",
            "--sources",
            "crafted_original",
        ],
    )

    try:
        assert build_suite_manifests.main() == 0
        payload = json.loads(output_manifest.read_text(encoding="utf-8"))
        assert payload["profile"] == "suite_reviewer_subset_relative"
        assert payload["required_watermark_methods"] == ["sweet_runtime"]
    finally:
        if output_manifest.exists():
            output_manifest.unlink()


def test_subset_manifest_cli_accepts_limit_override(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    output_manifest = root / "configs" / "matrices" / "test_relative_subset_manifest.json"
    if output_manifest.exists():
        output_manifest.unlink()
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_suite_manifests.py",
            "--output-manifest",
            "configs/matrices/test_relative_subset_manifest.json",
            "--profile",
            "suite_reviewer_subset_relative",
            "--models",
            "Qwen/Qwen2.5-Coder-14B-Instruct",
            "--methods",
            "sweet_runtime",
            "--sources",
            "crafted_original",
            "--limit",
            "8",
        ],
    )

    try:
        assert build_suite_manifests.main() == 0
        payload = json.loads(output_manifest.read_text(encoding="utf-8"))
        assert int(payload["runs"][0]["config_overrides"]["benchmark"]["limit"]) == 8
    finally:
        if output_manifest.exists():
            output_manifest.unlink()
