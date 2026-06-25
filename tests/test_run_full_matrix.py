from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts import run_full_matrix


ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(path: Path, *, config: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "suite_all_models_methods",
                "runs": [
                    {
                        "run_id": "planner_smoke",
                        "profile": "suite_all_models_methods",
                        "config": str(config),
                        "resource": "cpu",
                        "baseline_eval": False,
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_full_matrix_dry_run_writes_sidecar_index(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, config=ROOT / "configs" / "debug.yaml")
    output_root = tmp_path / "matrix"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--profile",
            "suite_all_models_methods",
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
    )

    assert run_full_matrix.main() == 0
    assert (output_root / "suite_all_models_methods" / "matrix_index.dry_run.json").exists()
    assert not (output_root / "suite_all_models_methods" / "matrix_index.json").exists()


def test_run_full_matrix_records_invalid_metadata_in_matrix_index(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, config=tmp_path / "missing-config.yaml")
    output_root = tmp_path / "matrix"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--profile",
            "suite_all_models_methods",
            "--output-root",
            str(output_root),
        ],
    )

    assert run_full_matrix.main() == 1
    payload = json.loads((output_root / "suite_all_models_methods" / "matrix_index.json").read_text(encoding="utf-8"))
    assert payload["failed_count"] == 1
    assert payload["runs"][0]["reason"] == "invalid_run_metadata"
    assert payload["canonical_manifest"] == str(manifest_path)
    assert payload["canonical_manifest_digest"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert payload["execution_mode"] == "single_host_canonical"


def test_run_full_matrix_dry_run_writes_canonical_identity_metadata(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, config=ROOT / "configs" / "debug.yaml")
    output_root = tmp_path / "matrix"
    monkeypatch.setattr(run_full_matrix._repo_snapshot, "repo_snapshot_sha256", lambda _root: "snapshot-digest")
    monkeypatch.setattr(
        run_full_matrix.capture_environment,
        "_collect",
        lambda: {
            "host": {"hostname": "example-host", "fqdn": "example.invalid"},
            "gpu": {
                "count": 8,
                "visible_gpu_count": 8,
                "cuda_visible_devices": "0,1,2,3,4,5,6,7",
                "devices": [{"name": "NVIDIA A800-SXM4-40GB", "driver_version": "550.163.01", "memory_total": "40960 MiB"}] * 8,
            },
            "platform": {"system": "Linux", "release": "6.8.0", "version": "#1 SMP", "machine": "x86_64", "processor": "Xeon"},
            "python": {"executable": "/venv/bin/python", "version": "3.10.12"},
            "packages": {"torch": "2.6.0+cu124", "transformers": "4.57.6", "numpy": "2.2.6", "pandas": None},
            "cuda": {"torch_cuda_version": "12.4", "nvidia_smi_cuda_version": "12.4"},
            "tools": {},
        },
    )
    monkeypatch.setattr(
        run_full_matrix.capture_environment,
        "execution_environment_fingerprint_sha256",
        lambda *_args, **_kwargs: "env-fingerprint",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--profile",
            "suite_all_models_methods",
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
    )

    assert run_full_matrix.main() == 0
    payload = json.loads((output_root / "suite_all_models_methods" / "matrix_index.dry_run.json").read_text(encoding="utf-8"))
    assert payload["canonical_manifest"] == str(manifest_path)
    assert payload["canonical_manifest_digest"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert payload["execution_mode"] == "single_host_canonical"


def test_visible_gpu_tokens_rejects_masked_ordinals_missing_from_host_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Completed:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setattr(
        run_full_matrix.subprocess,
        "run",
        lambda *args, **kwargs: _Completed("0\n1\n2\n3\n"),
    )

    with pytest.raises(SystemExit, match="requested gpu-slots references masked GPU ordinals"):
        run_full_matrix._visible_gpu_tokens(requested_slots=8)


def test_visible_gpu_tokens_rejects_duplicate_masked_ordinals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,0,1,2,3,4,5,6")

    with pytest.raises(SystemExit, match="must expose distinct GPU ordinals"):
        run_full_matrix._visible_gpu_tokens(requested_slots=8)


def test_run_full_matrix_uses_repo_relative_manifest_identity_for_in_repo_manifest(tmp_path: Path, monkeypatch) -> None:
    class _Completed:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    output_root = tmp_path / "matrix"
    monkeypatch.setattr(run_full_matrix._repo_snapshot, "repo_snapshot_sha256", lambda _root: "snapshot-digest")
    monkeypatch.setattr(
        run_full_matrix.capture_environment,
        "_collect",
        lambda: {
            "host": {"hostname": "example-host", "fqdn": "example.invalid"},
            "gpu": {"count": 8, "visible_gpu_count": 8, "cuda_visible_devices": "0,1,2,3,4,5,6,7", "devices": []},
            "platform": {"system": "Linux", "release": "6.8.0", "version": "#1 SMP", "machine": "x86_64", "processor": "Xeon"},
            "python": {"executable": "/venv/bin/python", "version": "3.10.12"},
            "packages": {"torch": "2.6.0+cu124", "transformers": "4.57.6", "numpy": "2.2.6", "pandas": None},
            "cuda": {"torch_cuda_version": "12.4", "nvidia_smi_cuda_version": "12.4"},
            "tools": {},
        },
    )
    monkeypatch.setattr(
        run_full_matrix.capture_environment,
        "execution_environment_fingerprint_sha256",
        lambda *_args, **_kwargs: "env-fingerprint",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setattr(
        run_full_matrix.subprocess,
        "run",
        lambda *args, **kwargs: _Completed("0\n1\n2\n3\n4\n5\n6\n7\n"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(ROOT / "configs" / "matrices" / "suite_all_models_methods.json"),
            "--profile",
            "suite_all_models_methods",
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
    )

    assert run_full_matrix.main() == 0
    payload = json.loads((output_root / "suite_all_models_methods" / "matrix_index.dry_run.json").read_text(encoding="utf-8"))
    assert payload["manifest"] == "configs/matrices/suite_all_models_methods.json"
    assert payload["canonical_manifest"] == "configs/matrices/suite_all_models_methods.json"


def test_run_full_matrix_rejects_direct_canonical_release_path_without_wrapper(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(ROOT / "configs" / "matrices" / "suite_all_models_methods.json"),
            "--profile",
            "suite_all_models_methods",
            "--output-root",
            str(ROOT / "results" / "matrix"),
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit, match="reserved for the formal release-facing full run"):
        run_full_matrix.main()


def test_run_full_matrix_rejects_resume_on_formal_release_path(tmp_path: Path, monkeypatch) -> None:
    output_root = ROOT / "results" / "matrix"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(ROOT / "configs" / "matrices" / "suite_all_models_methods.json"),
            "--profile",
            "suite_all_models_methods",
            "--output-root",
            str(output_root),
            "--resume",
        ],
    )

    with pytest.raises(SystemExit, match="not allowed for the formal single-host suite_all_models_methods release path"):
        run_full_matrix.main()


def test_run_full_matrix_rejects_resume_on_formal_release_path_when_profile_is_implicit(monkeypatch) -> None:
    output_root = ROOT / "results" / "matrix"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(ROOT / "configs" / "matrices" / "suite_all_models_methods.json"),
            "--output-root",
            str(output_root),
            "--resume",
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit, match="not allowed for the formal single-host suite_all_models_methods release path"):
        run_full_matrix.main()


def test_run_full_matrix_cleans_stale_final_outputs_before_rerun(tmp_path: Path) -> None:
    run = run_full_matrix.MatrixRun(
        run_id="cleanup_smoke",
        config_path=ROOT / "configs" / "debug.yaml",
        resource="cpu",
        output_dir=tmp_path / "cleanup_smoke",
        report_path=tmp_path / "cleanup_smoke" / "report.json",
        log_path=tmp_path / "cleanup_smoke" / "run.log",
    )
    for name in (
        "report.json",
        "baseline_eval.json",
        "analysis.json",
        "progress.json",
        "run.log",
        "_resolved_config.yaml",
        "baseline_eval_records.jsonl",
        "baseline_eval_payloads.private.jsonl",
        "partial_rows.jsonl",
        "partial_report.json",
    ):
        path = run.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale\n", encoding="utf-8")

    run_full_matrix._cleanup_previous_final_outputs(run)

    assert run.output_dir.exists()
    assert list(run.output_dir.iterdir()) == []
