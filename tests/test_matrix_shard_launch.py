from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import _matrix_shard_launch
from scripts._hf_readiness import HFModelRequirement


def _write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "profile": "suite_all_models_methods_shard_01_of_02",
                "canonical_model_revisions": {"Qwen/Qwen2.5-Coder-7B-Instruct": "c03e6d358207e414f1eca0bb1891e29f1db0e242"},
                "runs": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_receipt(path: Path, *, root: Path, manifest_rel: str, canonical_manifest_rel: str, code_snapshot_digest: str) -> None:
    payload = {
        "receipt_type": "matrix_shard_readiness",
        "status": "passed",
        "execution_mode": "sharded_identical_execution_class",
        "profile": "suite_all_models_methods_shard_01_of_02",
        "manifest": manifest_rel,
        "canonical_manifest": canonical_manifest_rel,
        "shard_index": 1,
        "shard_count": 2,
        "gpu_slots": 8,
        "gpu_pool_mode": "shared",
        "cpu_workers": 9,
        "retry_count": 1,
        "manifest_digests": {
            "manifest": "manifest_digest",
            "canonical_manifest": "canonical_digest",
        },
        "suite_model_revisions": {"Qwen/Qwen2.5-Coder-7B-Instruct": "c03e6d358207e414f1eca0bb1891e29f1db0e242"},
        "code_snapshot_digest": code_snapshot_digest,
        "environment_receipt": {
            "execution_environment_fingerprint": "fp",
            "cuda_visible_devices": "0,1,2,3,4,5,6,7",
            "visible_gpu_count": 8,
        },
        "audits": {
            "full_matrix_audit": {
                "required_hf_models": ["Qwen/Qwen2.5-Coder-7B-Instruct"],
                "hf_cache_validation": [{"model": "Qwen/Qwen2.5-Coder-7B-Instruct", "status": "ok"}],
                "hf_model_smoke": [{"model": "Qwen/Qwen2.5-Coder-7B-Instruct", "status": "ok"}],
                "hf_evaluator_smoke": [{"model": "Qwen/Qwen2.5-Coder-7B-Instruct", "status": "ok"}],
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_prepare_clean_launch_tree_removes_extras_and_leaves_output_empty(tmp_path: Path) -> None:
    output_dir = tmp_path / "results" / "matrix" / "suite_all_models_methods_shard_01_of_02"
    audit_dir = tmp_path / "results" / "audits" / "suite_all_models_methods_shard_01_of_02"
    figure_dir = tmp_path / "results" / "figures" / "suite_all_models_methods_shard_01_of_02"
    table_dir = tmp_path / "results" / "tables" / "suite_all_models_methods_shard_01_of_02"
    for path in (output_dir / "stale.txt", audit_dir / "stale.json", figure_dir / "fig.png", table_dir / "tbl.csv"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale", encoding="utf-8")

    payload = _matrix_shard_launch.prepare_clean_launch_tree(tmp_path, output_dir, [audit_dir, figure_dir, table_dir])

    assert payload["status"] == "clean"
    assert output_dir.exists()
    assert list(output_dir.iterdir()) == []
    assert not audit_dir.exists()
    assert not figure_dir.exists()
    assert not table_dir.exists()


def test_prepare_clean_launch_tree_rejects_paths_outside_results_boundary(tmp_path: Path) -> None:
    output_dir = tmp_path / "results" / "matrix" / "suite_all_models_methods_shard_01_of_02"
    outside_dir = tmp_path / "docs"
    outside_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(SystemExit, match="clean launch targets must stay under one of"):
        _matrix_shard_launch.prepare_clean_launch_tree(tmp_path, output_dir, [outside_dir])


def test_validate_existing_readiness_receipt_rejects_code_snapshot_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path
    manifest_path = root / "results" / "matrix_shards" / "suite_all_models_methods" / "suite_all_models_methods_shard_01_of_02.json"
    canonical_manifest_path = root / "configs" / "matrices" / "suite_all_models_methods.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_path)
    _write_manifest(canonical_manifest_path)
    receipt_path = root / "results" / "certifications" / "suite_all_models_methods_shard_01_of_02" / "matrix_shard_readiness.json"
    _write_receipt(
        receipt_path,
        root=root,
        manifest_rel="results/matrix_shards/suite_all_models_methods/suite_all_models_methods_shard_01_of_02.json",
        canonical_manifest_rel="configs/matrices/suite_all_models_methods.json",
        code_snapshot_digest="stale",
    )

    monkeypatch.setattr(_matrix_shard_launch, "_manifest_digests", lambda *_args: ("manifest_digest", "canonical_digest"))
    monkeypatch.setattr(_matrix_shard_launch, "_current_suite_model_revisions", lambda *_args: {"Qwen/Qwen2.5-Coder-7B-Instruct": "c03e6d358207e414f1eca0bb1891e29f1db0e242"})
    monkeypatch.setattr(_matrix_shard_launch._repo_snapshot, "repo_snapshot_sha256", lambda *_args: "current")
    monkeypatch.setattr(_matrix_shard_launch.capture_environment, "_collect", lambda: {"gpu": []})
    monkeypatch.setattr(_matrix_shard_launch.capture_environment, "execution_environment_fingerprint_sha256", lambda *_args, **_kwargs: "fp")
    monkeypatch.setattr(_matrix_shard_launch.capture_environment, "execution_class_gpu_devices", lambda *_args, **_kwargs: ["0", "1", "2", "3", "4", "5", "6", "7"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setattr(_matrix_shard_launch, "collect_current_hf_requirements", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(_matrix_shard_launch, "validate_current_hf_requirements", lambda *_args, **_kwargs: ([], []))

    with pytest.raises(SystemExit, match="code_snapshot_digest mismatch at launch time"):
        _matrix_shard_launch.validate_existing_readiness_receipt(
            root=root,
            receipt_path=receipt_path,
            profile="suite_all_models_methods_shard_01_of_02",
            manifest_rel="results/matrix_shards/suite_all_models_methods/suite_all_models_methods_shard_01_of_02.json",
            manifest_path=manifest_path,
            canonical_manifest_path=canonical_manifest_path,
            shard_index=1,
            shard_count=2,
            gpu_slots=8,
            gpu_pool_mode="shared",
            cpu_workers=9,
            retry_count=1,
        )


def test_validate_existing_readiness_receipt_rejects_current_hf_cache_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path
    manifest_path = root / "results" / "matrix_shards" / "suite_all_models_methods" / "suite_all_models_methods_shard_01_of_02.json"
    canonical_manifest_path = root / "configs" / "matrices" / "suite_all_models_methods.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_path)
    _write_manifest(canonical_manifest_path)
    receipt_path = root / "results" / "certifications" / "suite_all_models_methods_shard_01_of_02" / "matrix_shard_readiness.json"
    _write_receipt(
        receipt_path,
        root=root,
        manifest_rel="results/matrix_shards/suite_all_models_methods/suite_all_models_methods_shard_01_of_02.json",
        canonical_manifest_rel="configs/matrices/suite_all_models_methods.json",
        code_snapshot_digest="current",
    )

    monkeypatch.setattr(_matrix_shard_launch, "_manifest_digests", lambda *_args: ("manifest_digest", "canonical_digest"))
    monkeypatch.setattr(_matrix_shard_launch, "_current_suite_model_revisions", lambda *_args: {"Qwen/Qwen2.5-Coder-7B-Instruct": "c03e6d358207e414f1eca0bb1891e29f1db0e242"})
    monkeypatch.setattr(_matrix_shard_launch._repo_snapshot, "repo_snapshot_sha256", lambda *_args: "current")
    monkeypatch.setattr(_matrix_shard_launch.capture_environment, "_collect", lambda: {"gpu": []})
    monkeypatch.setattr(_matrix_shard_launch.capture_environment, "execution_environment_fingerprint_sha256", lambda *_args, **_kwargs: "fp")
    monkeypatch.setattr(_matrix_shard_launch.capture_environment, "execution_class_gpu_devices", lambda *_args, **_kwargs: ["0", "1", "2", "3", "4", "5", "6", "7"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setattr(
        _matrix_shard_launch,
        "collect_current_hf_requirements",
        lambda *_args, **_kwargs: {
            "Qwen/Qwen2.5-Coder-7B-Instruct": HFModelRequirement(
                model="Qwen/Qwen2.5-Coder-7B-Instruct",
                cache_dir="/example/model_cache/huggingface",
                local_files_only=True,
                revision="c03e6d358207e414f1eca0bb1891e29f1db0e242",
            )
        },
    )
    monkeypatch.setattr(
        _matrix_shard_launch,
        "validate_current_hf_requirements",
        lambda *_args, **_kwargs: (
            [{"model": "Qwen/Qwen2.5-Coder-7B-Instruct", "status": "failed", "issues": ["missing shard"]}],
            ["current HF cache validation failed for Qwen/Qwen2.5-Coder-7B-Instruct: ['missing shard']"],
        ),
    )

    with pytest.raises(SystemExit, match="current HF cache validation failed for Qwen/Qwen2.5-Coder-7B-Instruct"):
        _matrix_shard_launch.validate_existing_readiness_receipt(
            root=root,
            receipt_path=receipt_path,
            profile="suite_all_models_methods_shard_01_of_02",
            manifest_rel="results/matrix_shards/suite_all_models_methods/suite_all_models_methods_shard_01_of_02.json",
            manifest_path=manifest_path,
            canonical_manifest_path=canonical_manifest_path,
            shard_index=1,
            shard_count=2,
            gpu_slots=8,
            gpu_pool_mode="shared",
            cpu_workers=9,
            retry_count=1,
        )


def test_validate_current_hf_requirements_rechecks_model_and_evaluator_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirement = HFModelRequirement(
        model="Qwen/Qwen2.5-Coder-7B-Instruct",
        cache_dir="/example/model_cache/huggingface",
        local_files_only=True,
        revision="c03e6d358207e414f1eca0bb1891e29f1db0e242",
        usage=("baseline_eval", "runtime"),
        config_paths=("configs/test.yaml",),
    )
    monkeypatch.setattr(
        _matrix_shard_launch,
        "validate_local_hf_cache",
        lambda _requirement, require_root_entry=True: {
            "model": _requirement.model,
            "status": "ok",
            "issues": [],
        },
    )
    monkeypatch.setattr(
        _matrix_shard_launch,
        "smoke_load_local_hf_model",
        lambda _requirement: {"model": _requirement.model, "status": "failed", "issues": ["model drift"]},
    )
    monkeypatch.setattr(
        _matrix_shard_launch,
        "smoke_load_local_hf_evaluator",
        lambda _requirement: {"model": _requirement.model, "status": "failed", "issues": ["evaluator drift"]},
    )

    payloads, issues = _matrix_shard_launch.validate_current_hf_requirements(
        {"Qwen/Qwen2.5-Coder-7B-Instruct": requirement}
    )

    assert payloads[0]["model_smoke"]["status"] == "failed"
    assert payloads[0]["evaluator_smoke"]["status"] == "failed"
    assert "current HF model smoke failed for Qwen/Qwen2.5-Coder-7B-Instruct: ['model drift']" in issues
    assert "current HF evaluator smoke failed for Qwen/Qwen2.5-Coder-7B-Instruct: ['evaluator drift']" in issues
