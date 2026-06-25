from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from scripts import build_matrix_shards


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MANIFEST_PATH = ROOT / "configs" / "matrices" / "suite_all_models_methods.json"


def _load_manifest() -> dict:
    return json.loads(CANONICAL_MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("shard_count", "expected_profiles", "expected_runs_per_shard"),
    [
        (
            2,
            [
                "suite_all_models_methods_shard_01_of_02",
                "suite_all_models_methods_shard_02_of_02",
            ],
            70,
        ),
        (
            5,
            [
                "suite_all_models_methods_shard_01_of_05",
                "suite_all_models_methods_shard_02_of_05",
                "suite_all_models_methods_shard_03_of_05",
                "suite_all_models_methods_shard_04_of_05",
                "suite_all_models_methods_shard_05_of_05",
            ],
            28,
        ),
    ],
)
def test_build_matrix_shards_is_deterministic_and_weight_balanced(
    shard_count: int,
    expected_profiles: list[str],
    expected_runs_per_shard: int,
) -> None:
    manifest = _load_manifest()
    digest = build_matrix_shards._canonical_manifest_digest(manifest, manifest_path=CANONICAL_MANIFEST_PATH)

    shards_first = build_matrix_shards.build_matrix_shards(
        manifest,
        manifest_path=CANONICAL_MANIFEST_PATH,
        profile="suite_all_models_methods",
        shard_count=shard_count,
    )
    shards_second = build_matrix_shards.build_matrix_shards(
        manifest,
        manifest_path=CANONICAL_MANIFEST_PATH,
        profile="suite_all_models_methods",
        shard_count=shard_count,
    )

    assert shards_first == shards_second
    assert [shard["shard_profile"] for shard in shards_first] == expected_profiles

    all_run_ids: list[str] = []
    all_indices: list[int] = []
    shard_weights: list[int] = []
    shard_model_counts: list[Counter[str]] = []
    for shard in shards_first:
        assert shard["profile"] == shard["shard_profile"]
        assert shard["canonical_profile"] == "suite_all_models_methods"
        assert shard["canonical_manifest"] == "configs/matrices/suite_all_models_methods.json"
        assert shard["canonical_manifest_digest"] == digest
        assert shard["canonical_manifest_digest_algorithm"] == "sha256"
        assert shard["execution_mode"] == "sharded_identical_execution_class"
        assert shard["canonical_model_revisions"] == manifest["model_revisions"]
        assert shard["model_revisions"] == manifest["model_revisions"]
        assert shard["shard_strategy"] == "weighted_greedy_by_priority"
        assert 1 <= shard["shard_index"] <= shard_count
        assert shard["shard_count"] == shard_count
        assert shard["shard_run_count"] == expected_runs_per_shard
        assert len(shard["shard_run_ids"]) == expected_runs_per_shard
        assert len(shard["shard_canonical_run_indices"]) == expected_runs_per_shard
        assert [run["run_id"] for run in shard["runs"]] == shard["shard_run_ids"]
        assert [int(run["canonical_run_index"]) for run in shard["runs"]] == shard["shard_canonical_run_indices"]
        assert shard["shard_canonical_run_indices"] == sorted(shard["shard_canonical_run_indices"])
        all_run_ids.extend(shard["shard_run_ids"])
        all_indices.extend(shard["shard_canonical_run_indices"])
        shard_weights.append(int(shard["shard_weight_total"]))
        shard_model_counts.append(Counter(run["model"] for run in shard["runs"]))
        for run in shard["runs"]:
            assert run["profile"] == shard["shard_profile"]
            assert run["canonical_profile"] == "suite_all_models_methods"
            assert run["canonical_manifest"] == "configs/matrices/suite_all_models_methods.json"
            assert run["canonical_manifest_digest"] == digest
            assert run["canonical_manifest_digest_algorithm"] == "sha256"
            assert run["shard_profile"] == shard["shard_profile"]
            assert run["shard_index"] == shard["shard_index"]
            assert run["shard_count"] == shard_count
            assert "canonical_run_index" in run
            assert "shard_run_cost" in run

    assert len(all_run_ids) == 140
    assert len(set(all_run_ids)) == 140
    assert len(all_indices) == 140
    assert len(set(all_indices)) == 140
    assert min(all_indices) == 0
    assert max(all_indices) == 139
    assert Counter(all_run_ids).most_common(1)[0][1] == 1
    assert max(shard_weights) - min(shard_weights) <= 100_000_000
    if shard_count == 2:
        assert shard_model_counts[0] == shard_model_counts[1]
        assert shard_model_counts[0]["Qwen/Qwen2.5-Coder-14B-Instruct"] == 14


def test_write_matrix_shards_writes_predictable_files(tmp_path: Path) -> None:
    manifest = _load_manifest()
    output_dir = tmp_path / "matrix_shards"

    output_paths = build_matrix_shards.write_matrix_shards(
        manifest,
        manifest_path=CANONICAL_MANIFEST_PATH,
        output_dir=output_dir,
        profile="suite_all_models_methods",
        shard_count=2,
    )

    assert [path.name for path in output_paths] == [
        "suite_all_models_methods_shard_01_of_02.json",
        "suite_all_models_methods_shard_02_of_02.json",
    ]
    assert all(path.exists() for path in output_paths)

    written = [json.loads(path.read_text(encoding="utf-8")) for path in output_paths]
    assert [payload["shard_profile"] for payload in written] == [path.stem for path in output_paths]
    assert all(payload["profile"] == payload["shard_profile"] for payload in written)
    assert all(payload["execution_mode"] == "sharded_identical_execution_class" for payload in written)
    assert all(payload["canonical_manifest"] == "configs/matrices/suite_all_models_methods.json" for payload in written)
    assert all(payload["canonical_model_revisions"] == manifest["model_revisions"] for payload in written)
    assert all(payload["model_revisions"] == manifest["model_revisions"] for payload in written)


def test_build_matrix_shards_rejects_wrong_profile() -> None:
    manifest = _load_manifest()

    with pytest.raises(ValueError, match="profile mismatch"):
        build_matrix_shards.build_matrix_shards(
            manifest,
            manifest_path=CANONICAL_MANIFEST_PATH,
            profile="suite_not_canonical",
            shard_count=2,
        )
