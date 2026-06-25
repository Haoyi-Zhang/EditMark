from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import certify_suite_precheck


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _stable_environment_receipt(*, code_snapshot_digest: str = "snapshot-digest") -> dict[str, object]:
    return {
        "python_bin": "python",
        "python_executable": "python",
        "environment_fingerprint": "stable-env",
        "execution_environment_fingerprint": "stable-exec",
        "cuda_visible_devices": "",
        "visible_gpu_count": 0,
        "preflight_gpu_slots": 0,
        "code_snapshot_digest": code_snapshot_digest,
        "host_identity": {"hostname": "example-host", "fqdn": "example.invalid"},
    }


def _stable_launch_environment_receipt(*, code_snapshot_digest: str = "snapshot-digest") -> dict[str, object]:
    return {
        "python_bin": "python",
        "python_executable": "python",
        "execution_environment_fingerprint": "stable-exec",
        "cuda_visible_devices": "",
        "visible_gpu_count": 0,
        "preflight_gpu_slots": 0,
        "code_snapshot_digest": code_snapshot_digest,
        "host_identity": {"hostname": "example-host", "fqdn": "example.invalid"},
    }


def _stable_runtime_checkout_receipt() -> dict[str, dict[str, object]]:
    return {
        method: {
            "checkout_present": True,
            "checkout_valid": True,
            "origin": "external_checkout",
            "repo_root": f"/example/workspace/posteditbench_clean/external_checkout/{method}",
            "source_root": f"/example/workspace/posteditbench_clean/external_checkout/{method}",
            "source_relative": ".",
            "external_path": f"external_checkout/{method}",
            "remote_url": f"https://example.com/{method}.git",
            "manifest_pinned_commit": f"{method}-manifest",
            "upstream_commit": f"{method}-upstream",
            "dirty": False,
        }
        for method in ("stone_runtime", "sweet_runtime", "ewd_runtime", "kgw_runtime")
    }


def _launch_receipt_payload(*, output_root: Path, environment_receipt: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "receipt_type": "post_precheck_launch",
        "status": "ready",
        "full_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/suite_all_models_methods.json")),
        "full_profile": "suite_all_models_methods",
        "stage_a_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/suite_canary_heavy.json")),
        "stage_a_profile": "suite_canary_heavy",
        "stage_b_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/model_invocation_smoke.json")),
        "stage_b_profile": "model_invocation_smoke",
        "output_root": str(output_root),
        "skip_hf_access": True,
        "manifest_digests": {
            "full_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/suite_all_models_methods.json"),
            "stage_a_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/suite_canary_heavy.json"),
            "stage_b_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/model_invocation_smoke.json"),
        },
        "suite_model_revisions": dict(certify_suite_precheck.SUITE_MODEL_REVISIONS),
        "runtime_checkout_receipt": _stable_runtime_checkout_receipt(),
        "environment_receipt": environment_receipt or _stable_launch_environment_receipt(),
    }


@pytest.fixture(autouse=True)
def _stable_runtime_checkout_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        certify_suite_precheck,
        "_current_runtime_checkout_receipt",
        lambda: _stable_runtime_checkout_receipt(),
    )


def test_first_report_path_reads_matrix_index(tmp_path: Path) -> None:
    report_path = tmp_path / "run" / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("{}", encoding="utf-8")
    matrix_index = tmp_path / "matrix_index.json"
    matrix_index.write_text(
        json.dumps(
            {
                "runs": [
                    {"report_path": ""},
                    {"report_path": str(report_path)},
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = certify_suite_precheck._first_report_path(matrix_index)

    assert resolved == report_path


def test_render_command_uses_precheck_safe_basic_suite(tmp_path: Path) -> None:
    matrix_index = tmp_path / "matrix_index.json"
    anchor_report = tmp_path / "report.json"
    output_dir = tmp_path / "figures"

    command = certify_suite_precheck._render_command(
        python_bin="python",
        matrix_index_path=matrix_index,
        anchor_report_path=anchor_report,
        figure_output_dir=output_dir,
    )

    assert "--matrix-index" not in command
    assert str(matrix_index) not in command
    assert "--report" in command
    assert str(anchor_report) in command
    assert "--anchor-report" not in command
    assert "--allow-font-fallback" in command
    assert "--require-times-new-roman" not in command
    assert "--suite" in command
    assert "basic" in command
    assert "all" not in command
    assert "--include-reference-artifacts" not in command


def test_main_rejects_resume_for_suite_precheck(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "gate.json"
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=tmp_path / "results" / "matrix",
        figure_output_dir=tmp_path / "results" / "figures",
        output=output_path,
        preflight_receipt=tmp_path / "preflight.json",
        gpu_slots=8,
        gpu_pool_mode="shared",
        cpu_workers=12,
        retry_count=1,
        command_timeout_seconds=259200,
        step_timeout_seconds=60,
        skip_hf_access=True,
        resume=True,
        fail_fast=False,
    )
    monkeypatch.setattr(certify_suite_precheck, "parse_args", lambda: args)

    assert certify_suite_precheck.main() == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "not supported for suite precheck" in payload["error"]


def test_main_fails_when_render_step_fails(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "gate.json"
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=tmp_path / "results" / "matrix",
        figure_output_dir=tmp_path / "results" / "figures",
        output=output_path,
        preflight_receipt=tmp_path / "preflight.json",
        gpu_slots=8,
        gpu_pool_mode="shared",
        cpu_workers=12,
        retry_count=1,
        command_timeout_seconds=259200,
        step_timeout_seconds=60,
        skip_hf_access=True,
        resume=False,
    )
    monkeypatch.setattr(certify_suite_precheck, "parse_args", lambda: args)
    monkeypatch.setattr(certify_suite_precheck, "_first_report_path", lambda _: tmp_path / "anchor_report.json")

    def fake_run_step(command, *, timeout_seconds, label):
        return {
            "name": label,
            "command": command,
            "returncode": 1 if label == "render_suite_figures" else 0,
            "duration_seconds": 0.01,
            "stdout_tail": "",
            "stderr_tail": "",
            "status": "failed" if label == "render_suite_figures" else "passed",
        }

    monkeypatch.setattr(certify_suite_precheck, "_run_step", fake_run_step)
    monkeypatch.setattr(
        certify_suite_precheck,
        "_run_matrix_step",
        lambda command, *, timeout_seconds, label, gate_output_path, steps, matrix_index_path: fake_run_step(
            command,
            timeout_seconds=timeout_seconds,
            label=label,
        ),
    )

    exit_code = certify_suite_precheck.main()

    assert exit_code == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["current_step"] == "render_suite_figures"
    assert payload["steps"][-1]["name"] == "render_suite_figures"


def test_precheck_uses_suite_benchmark_audit_profile(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "gate.json"
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=tmp_path / "results" / "matrix",
        figure_output_dir=tmp_path / "results" / "figures",
        output=output_path,
        preflight_receipt=tmp_path / "preflight.json",
        gpu_slots=8,
        gpu_pool_mode="shared",
        cpu_workers=12,
        retry_count=1,
        command_timeout_seconds=259200,
        step_timeout_seconds=60,
        skip_hf_access=True,
        resume=False,
        fail_fast=True,
    )
    monkeypatch.setattr(certify_suite_precheck, "parse_args", lambda: args)
    monkeypatch.setattr(certify_suite_precheck, "_first_report_path", lambda _: tmp_path / "anchor_report.json")

    seen_commands: list[tuple[str, list[str]]] = []

    def fake_run_step(command, *, timeout_seconds, label):
        seen_commands.append((label, list(command)))
        return {
            "name": label,
            "command": command,
            "returncode": 0,
            "duration_seconds": 0.01,
            "stdout_tail": "",
            "stderr_tail": "",
            "status": "passed",
        }

    monkeypatch.setattr(certify_suite_precheck, "_run_step", fake_run_step)
    monkeypatch.setattr(
        certify_suite_precheck,
        "_run_matrix_step",
        lambda command, *, timeout_seconds, label, gate_output_path, steps, matrix_index_path: fake_run_step(
            command,
            timeout_seconds=timeout_seconds,
            label=label,
        ),
    )

    exit_code = certify_suite_precheck.main()

    assert exit_code == 0
    audit_command = next(command for label, command in seen_commands if label == "audit_benchmarks")
    stage_a_command = next(command for label, command in seen_commands if label == "stage_a_suite_canary_heavy")
    stage_b_command = next(command for label, command in seen_commands if label == "stage_b_model_invocation_smoke")
    assert "--profile" in audit_command
    assert "suite_all_models_methods" in audit_command
    assert "--manifest" in audit_command
    assert "--matrix-profile" in audit_command
    assert "--command-timeout-seconds" in stage_a_command
    assert "259200" in stage_a_command
    assert "--command-timeout-seconds" in stage_b_command
    assert "259200" in stage_b_command


def test_precheck_reuses_matching_preflight_receipt(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "gate.json"
    receipt_path = tmp_path / "preflight.json"
    output_root = tmp_path / "results" / "matrix"
    figure_output_dir = tmp_path / "results" / "figures"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt_type": "remote_preflight",
                "status": "passed",
                "full_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/suite_all_models_methods.json")),
                "full_profile": "suite_all_models_methods",
                "stage_a_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/suite_canary_heavy.json")),
                "stage_a_profile": "suite_canary_heavy",
                "stage_b_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/model_invocation_smoke.json")),
                "stage_b_profile": "model_invocation_smoke",
                "output_root": str(output_root),
                "skip_hf_access": True,
                "manifest_digests": {
                    "full_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/suite_all_models_methods.json"),
                    "stage_a_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/suite_canary_heavy.json"),
                    "stage_b_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/model_invocation_smoke.json"),
                },
                "suite_model_revisions": dict(certify_suite_precheck.SUITE_MODEL_REVISIONS),
                "runtime_checkout_receipt": _stable_runtime_checkout_receipt(),
                "environment_receipt": {
                    "python_bin": "python",
                    "python_executable": "python",
                    "environment_fingerprint": "stable-env",
                    "execution_environment_fingerprint": "stable-exec",
                    "cuda_visible_devices": "",
                    "visible_gpu_count": 0,
                    "preflight_gpu_slots": 0,
                    "code_snapshot_digest": "snapshot-digest",
                    "host_identity": {"hostname": "example-host", "fqdn": "example.invalid"},
                },
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=output_root,
        figure_output_dir=figure_output_dir,
        output=output_path,
        preflight_receipt=receipt_path,
        gpu_slots=8,
        gpu_pool_mode="shared",
        cpu_workers=12,
        retry_count=1,
        command_timeout_seconds=259200,
        step_timeout_seconds=60,
        skip_hf_access=True,
        resume=False,
        fail_fast=False,
    )
    monkeypatch.setattr(certify_suite_precheck, "parse_args", lambda: args)
    monkeypatch.setattr(certify_suite_precheck, "_first_report_path", lambda _: tmp_path / "anchor_report.json")
    monkeypatch.setattr(certify_suite_precheck, "_current_environment_receipt_payload", lambda current_args: _stable_environment_receipt())

    run_step_labels: list[str] = []

    def fake_run_step(command, *, timeout_seconds, label):
        run_step_labels.append(label)
        return {
            "name": label,
            "command": command,
            "returncode": 0,
            "duration_seconds": 0.01,
            "stdout_tail": "",
            "stderr_tail": "",
            "status": "passed",
        }

    monkeypatch.setattr(certify_suite_precheck, "_run_step", fake_run_step)
    monkeypatch.setattr(
        certify_suite_precheck,
        "_run_matrix_step",
        lambda command, *, timeout_seconds, label, gate_output_path, steps, matrix_index_path: fake_run_step(
            command,
            timeout_seconds=timeout_seconds,
            label=label,
        ),
    )

    assert certify_suite_precheck.main() == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    reused = payload["steps"][:3]
    assert [step["name"] for step in reused] == ["build_suite_manifests", "audit_benchmarks", "audit_suite_matrix"]
    assert all(step["reason"] == "matching_clean_preflight_receipt" for step in reused)
    assert "build_suite_manifests" not in run_step_labels
    assert "audit_benchmarks" not in run_step_labels
    assert "audit_suite_matrix" not in run_step_labels


def test_precheck_rejects_receipt_when_manifest_digest_changes(tmp_path: Path) -> None:
    receipt_path = tmp_path / "preflight.json"
    output_root = tmp_path / "results" / "matrix"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "receipt_type": "remote_preflight",
                "status": "passed",
                "full_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/suite_all_models_methods.json")),
                "full_profile": "suite_all_models_methods",
                "stage_a_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/suite_canary_heavy.json")),
                "stage_a_profile": "suite_canary_heavy",
                "stage_b_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/model_invocation_smoke.json")),
                "stage_b_profile": "model_invocation_smoke",
                "output_root": str(output_root),
                "skip_hf_access": True,
                "manifest_digests": {
                    "full_manifest": "stale",
                    "stage_a_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/suite_canary_heavy.json"),
                    "stage_b_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/model_invocation_smoke.json"),
                },
                "suite_model_revisions": dict(certify_suite_precheck.SUITE_MODEL_REVISIONS),
                "runtime_checkout_receipt": _stable_runtime_checkout_receipt(),
                "environment_receipt": {
                    "python_bin": "python",
                    "python_executable": "python",
                    "environment_fingerprint": "stable-env",
                    "execution_environment_fingerprint": "stable-exec",
                    "cuda_visible_devices": "",
                    "visible_gpu_count": 0,
                    "preflight_gpu_slots": 0,
                    "code_snapshot_digest": "snapshot-digest",
                    "host_identity": {"hostname": "example-host", "fqdn": "example.invalid"},
                },
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=output_root,
        preflight_receipt=receipt_path,
        skip_hf_access=True,
    )

    assert certify_suite_precheck._load_matching_preflight_receipt(args, output_root=output_root) is None


def test_precheck_rejects_receipt_when_environment_fingerprint_changes(tmp_path: Path, monkeypatch) -> None:
    receipt_path = tmp_path / "preflight.json"
    output_root = tmp_path / "results" / "matrix"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "receipt_type": "remote_preflight",
                "status": "passed",
                "full_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/suite_all_models_methods.json")),
                "full_profile": "suite_all_models_methods",
                "stage_a_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/suite_canary_heavy.json")),
                "stage_a_profile": "suite_canary_heavy",
                "stage_b_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/model_invocation_smoke.json")),
                "stage_b_profile": "model_invocation_smoke",
                "output_root": str(output_root),
                "skip_hf_access": True,
                "manifest_digests": {
                    "full_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/suite_all_models_methods.json"),
                    "stage_a_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/suite_canary_heavy.json"),
                    "stage_b_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/model_invocation_smoke.json"),
                },
                "suite_model_revisions": dict(certify_suite_precheck.SUITE_MODEL_REVISIONS),
                "runtime_checkout_receipt": _stable_runtime_checkout_receipt(),
                "environment_receipt": {
                    "python_bin": "python",
                    "python_executable": "python",
                    "environment_fingerprint": "stale-env",
                    "execution_environment_fingerprint": "stable-exec",
                    "cuda_visible_devices": "",
                    "visible_gpu_count": 0,
                    "preflight_gpu_slots": 0,
                    "code_snapshot_digest": "snapshot-digest",
                    "host_identity": {"hostname": "example-host", "fqdn": "example.invalid"},
                },
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=output_root,
        preflight_receipt=receipt_path,
        skip_hf_access=True,
    )
    monkeypatch.setattr(
        certify_suite_precheck,
        "_current_environment_receipt_payload",
        lambda current_args: {
            **_stable_environment_receipt(),
            "environment_fingerprint": "fresh-env",
        },
    )

    assert certify_suite_precheck._load_matching_preflight_receipt(args, output_root=output_root) is None


def test_precheck_rejects_receipt_when_cuda_visible_devices_change(tmp_path: Path, monkeypatch) -> None:
    receipt_path = tmp_path / "preflight.json"
    output_root = tmp_path / "results" / "matrix"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "receipt_type": "remote_preflight",
                "status": "passed",
                "full_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/suite_all_models_methods.json")),
                "full_profile": "suite_all_models_methods",
                "stage_a_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/suite_canary_heavy.json")),
                "stage_a_profile": "suite_canary_heavy",
                "stage_b_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/model_invocation_smoke.json")),
                "stage_b_profile": "model_invocation_smoke",
                "output_root": str(output_root),
                "skip_hf_access": True,
                "manifest_digests": {
                    "full_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/suite_all_models_methods.json"),
                    "stage_a_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/suite_canary_heavy.json"),
                    "stage_b_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/model_invocation_smoke.json"),
                },
                "suite_model_revisions": dict(certify_suite_precheck.SUITE_MODEL_REVISIONS),
                "runtime_checkout_receipt": _stable_runtime_checkout_receipt(),
                "environment_receipt": {
                    "python_bin": "python",
                    "python_executable": "python",
                    "environment_fingerprint": "stable-env",
                    "execution_environment_fingerprint": "stable-exec",
                    "cuda_visible_devices": "0,1",
                    "visible_gpu_count": 2,
                    "preflight_gpu_slots": 2,
                    "code_snapshot_digest": "snapshot-digest",
                    "host_identity": {"hostname": "example-host", "fqdn": "example.invalid"},
                },
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=output_root,
        preflight_receipt=receipt_path,
        skip_hf_access=True,
    )
    monkeypatch.setattr(
        certify_suite_precheck,
        "_current_environment_receipt_payload",
        lambda current_args: {
            **_stable_environment_receipt(),
            "cuda_visible_devices": "0",
            "visible_gpu_count": 1,
            "preflight_gpu_slots": 1,
        },
    )

    assert certify_suite_precheck._load_matching_preflight_receipt(args, output_root=output_root) is None


def test_precheck_rejects_receipt_when_runtime_checkout_changes(tmp_path: Path) -> None:
    receipt_path = tmp_path / "preflight.json"
    output_root = tmp_path / "results" / "matrix"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "receipt_type": "remote_preflight",
                "status": "passed",
                "full_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/suite_all_models_methods.json")),
                "full_profile": "suite_all_models_methods",
                "stage_a_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/suite_canary_heavy.json")),
                "stage_a_profile": "suite_canary_heavy",
                "stage_b_manifest": str((certify_suite_precheck.ROOT / "configs/matrices/model_invocation_smoke.json")),
                "stage_b_profile": "model_invocation_smoke",
                "output_root": str(output_root),
                "skip_hf_access": True,
                "manifest_digests": {
                    "full_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/suite_all_models_methods.json"),
                    "stage_a_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/suite_canary_heavy.json"),
                    "stage_b_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/model_invocation_smoke.json"),
                },
                "suite_model_revisions": dict(certify_suite_precheck.SUITE_MODEL_REVISIONS),
                "runtime_checkout_receipt": {
                    **_stable_runtime_checkout_receipt(),
                    "stone_runtime": {
                        **_stable_runtime_checkout_receipt()["stone_runtime"],
                        "upstream_commit": "stale-upstream",
                    },
                },
                "environment_receipt": _stable_environment_receipt(),
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=output_root,
        preflight_receipt=receipt_path,
        skip_hf_access=True,
    )

    assert certify_suite_precheck._load_matching_preflight_receipt(args, output_root=output_root) is None


def test_precheck_writes_post_precheck_launch_receipt(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "gate.json"
    output_root = tmp_path / "results" / "matrix"
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=output_root,
        figure_output_dir=tmp_path / "results" / "figures",
        output=output_path,
        preflight_receipt=tmp_path / "preflight.json",
        gpu_slots=8,
        gpu_pool_mode="shared",
        cpu_workers=12,
        retry_count=1,
        command_timeout_seconds=259200,
        step_timeout_seconds=60,
        skip_hf_access=True,
        resume=False,
        fail_fast=False,
    )
    monkeypatch.setattr(certify_suite_precheck, "parse_args", lambda: args)
    monkeypatch.setattr(certify_suite_precheck, "_first_report_path", lambda _: tmp_path / "anchor_report.json")
    monkeypatch.setattr(certify_suite_precheck, "_current_environment_receipt_payload", lambda current_args: _stable_environment_receipt(code_snapshot_digest="post-precheck"))

    def fake_run_step(command, *, timeout_seconds, label):
        return {
            "name": label,
            "command": command,
            "returncode": 0,
            "duration_seconds": 0.01,
            "stdout_tail": "",
            "stderr_tail": "",
            "status": "passed",
        }

    monkeypatch.setattr(certify_suite_precheck, "_run_step", fake_run_step)
    monkeypatch.setattr(
        certify_suite_precheck,
        "_run_matrix_step",
        lambda command, *, timeout_seconds, label, gate_output_path, steps, matrix_index_path: fake_run_step(
            command,
            timeout_seconds=timeout_seconds,
            label=label,
        ),
    )

    assert certify_suite_precheck.main() == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    launch_receipt = payload["launch_receipt"]
    assert launch_receipt["receipt_type"] == "post_precheck_launch"
    assert launch_receipt["status"] == "ready"
    assert launch_receipt["full_profile"] == "suite_all_models_methods"
    assert launch_receipt["stage_a_profile"] == "suite_canary_heavy"
    assert launch_receipt["stage_b_profile"] == "model_invocation_smoke"
    assert launch_receipt["output_root"] == str(output_root)
    assert launch_receipt["skip_hf_access"] is True
    assert launch_receipt["manifest_digests"] == {
        "full_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/suite_all_models_methods.json"),
        "stage_a_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/suite_canary_heavy.json"),
        "stage_b_manifest": _file_sha256(certify_suite_precheck.ROOT / "configs/matrices/model_invocation_smoke.json"),
    }
    assert launch_receipt["suite_model_revisions"] == dict(certify_suite_precheck.SUITE_MODEL_REVISIONS)
    assert launch_receipt["runtime_checkout_receipt"] == _stable_runtime_checkout_receipt()
    assert launch_receipt["environment_receipt"] == _stable_launch_environment_receipt(code_snapshot_digest="post-precheck")


def test_precheck_launch_receipt_excludes_non_contract_host_surface_but_keeps_host_identity(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "results" / "matrix"
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        skip_hf_access=True,
    )
    monkeypatch.setattr(
        certify_suite_precheck,
        "_current_environment_receipt_payload",
        lambda current_args: _stable_environment_receipt(code_snapshot_digest="post-precheck"),
    )
    monkeypatch.setattr(
        certify_suite_precheck,
        "_current_runtime_checkout_receipt",
        lambda: _stable_runtime_checkout_receipt(),
    )

    launch_receipt = certify_suite_precheck._build_launch_receipt(args, output_root=output_root)

    assert launch_receipt["environment_receipt"] == _stable_launch_environment_receipt(code_snapshot_digest="post-precheck")
    assert launch_receipt["runtime_checkout_receipt"] == _stable_runtime_checkout_receipt()
    assert "environment_fingerprint" not in launch_receipt["environment_receipt"]
    assert launch_receipt["environment_receipt"]["host_identity"] == {"hostname": "example-host", "fqdn": "example.invalid"}


def test_launch_receipt_loader_accepts_matching_gate(tmp_path: Path, monkeypatch) -> None:
    gate_path = tmp_path / "suite_precheck_gate.json"
    output_root = tmp_path / "results" / "matrix"
    launch_receipt = _launch_receipt_payload(
        output_root=output_root,
        environment_receipt=_stable_launch_environment_receipt(code_snapshot_digest="post-precheck"),
    )
    gate_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "current_step": "complete",
                "steps": [],
                "launch_receipt": launch_receipt,
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=output_root,
        precheck_gate=gate_path,
        skip_hf_access=True,
    )
    monkeypatch.setattr(
        certify_suite_precheck,
        "_current_environment_receipt_payload",
        lambda current_args: _stable_environment_receipt(code_snapshot_digest="post-precheck"),
    )

    matched = certify_suite_precheck._load_matching_launch_receipt(args, output_root=output_root, gate_path=gate_path)

    assert matched == launch_receipt


def test_launch_receipt_loader_rejects_snapshot_mismatch(tmp_path: Path, monkeypatch) -> None:
    gate_path = tmp_path / "suite_precheck_gate.json"
    output_root = tmp_path / "results" / "matrix"
    gate_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "current_step": "complete",
                "steps": [],
                "launch_receipt": _launch_receipt_payload(
                    output_root=output_root,
                    environment_receipt=_stable_launch_environment_receipt(code_snapshot_digest="old-snapshot"),
                ),
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=output_root,
        precheck_gate=gate_path,
        skip_hf_access=True,
    )
    monkeypatch.setattr(certify_suite_precheck, "_current_environment_receipt_payload", lambda current_args: _stable_environment_receipt(code_snapshot_digest="new-snapshot"))

    matched = certify_suite_precheck._load_matching_launch_receipt(args, output_root=output_root, gate_path=gate_path)

    assert matched is None


def test_launch_receipt_loader_accepts_non_contract_environment_fingerprint_drift(tmp_path: Path, monkeypatch) -> None:
    gate_path = tmp_path / "suite_precheck_gate.json"
    output_root = tmp_path / "results" / "matrix"
    gate_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "current_step": "complete",
                "steps": [],
                "launch_receipt": _launch_receipt_payload(
                    output_root=output_root,
                    environment_receipt=_stable_launch_environment_receipt(code_snapshot_digest="post-precheck"),
                ),
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=output_root,
        precheck_gate=gate_path,
        skip_hf_access=True,
    )
    monkeypatch.setattr(
        certify_suite_precheck,
        "_current_environment_receipt_payload",
        lambda current_args: {
            **_stable_environment_receipt(code_snapshot_digest="post-precheck"),
            "environment_fingerprint": "different-host-surface",
        },
    )

    matched = certify_suite_precheck._load_matching_launch_receipt(args, output_root=output_root, gate_path=gate_path)

    assert matched is not None


def test_launch_receipt_loader_rejects_host_identity_mismatch(tmp_path: Path, monkeypatch) -> None:
    gate_path = tmp_path / "suite_precheck_gate.json"
    output_root = tmp_path / "results" / "matrix"
    gate_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "current_step": "complete",
                "steps": [],
                "launch_receipt": _launch_receipt_payload(
                    output_root=output_root,
                    environment_receipt=_stable_launch_environment_receipt(code_snapshot_digest="post-precheck"),
                ),
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=output_root,
        precheck_gate=gate_path,
        skip_hf_access=True,
    )
    monkeypatch.setattr(
        certify_suite_precheck,
        "_current_environment_receipt_payload",
        lambda current_args: {
            **_stable_environment_receipt(code_snapshot_digest="post-precheck"),
            "host_identity": {"hostname": "other-host", "fqdn": "other.example.invalid"},
        },
    )

    matched = certify_suite_precheck._load_matching_launch_receipt(args, output_root=output_root, gate_path=gate_path)

    assert matched is None


def test_launch_receipt_loader_rejects_incomplete_gate(tmp_path: Path, monkeypatch) -> None:
    gate_path = tmp_path / "suite_precheck_gate.json"
    output_root = tmp_path / "results" / "matrix"
    gate_path.write_text(
        json.dumps(
            {
                "status": "running",
                "current_step": "stage_b_model_invocation_smoke",
                "steps": [],
                "launch_receipt": _launch_receipt_payload(output_root=output_root),
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=output_root,
        precheck_gate=gate_path,
        skip_hf_access=True,
    )
    monkeypatch.setattr(certify_suite_precheck, "_current_environment_receipt_payload", lambda current_args: _stable_environment_receipt())

    matched = certify_suite_precheck._load_matching_launch_receipt(args, output_root=output_root, gate_path=gate_path)

    assert matched is None


def test_launch_receipt_loader_rejects_non_ready_launch_receipt(tmp_path: Path, monkeypatch) -> None:
    gate_path = tmp_path / "suite_precheck_gate.json"
    output_root = tmp_path / "results" / "matrix"
    launch_receipt = _launch_receipt_payload(output_root=output_root)
    launch_receipt["status"] = "failed"
    gate_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "current_step": "complete",
                "steps": [],
                "launch_receipt": launch_receipt,
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=output_root,
        precheck_gate=gate_path,
        skip_hf_access=True,
    )
    monkeypatch.setattr(certify_suite_precheck, "_current_environment_receipt_payload", lambda current_args: _stable_environment_receipt())

    matched = certify_suite_precheck._load_matching_launch_receipt(args, output_root=output_root, gate_path=gate_path)

    assert matched is None


def test_launch_receipt_loader_rejects_manifest_digest_mismatch(tmp_path: Path, monkeypatch) -> None:
    gate_path = tmp_path / "suite_precheck_gate.json"
    output_root = tmp_path / "results" / "matrix"
    launch_receipt = _launch_receipt_payload(output_root=output_root)
    launch_receipt["manifest_digests"] = {
        **launch_receipt["manifest_digests"],
        "full_manifest": "stale-manifest-digest",
    }
    gate_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "current_step": "complete",
                "steps": [],
                "launch_receipt": launch_receipt,
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=output_root,
        precheck_gate=gate_path,
        skip_hf_access=True,
    )
    monkeypatch.setattr(certify_suite_precheck, "_current_environment_receipt_payload", lambda current_args: _stable_environment_receipt())

    matched = certify_suite_precheck._load_matching_launch_receipt(args, output_root=output_root, gate_path=gate_path)

    assert matched is None


def test_launch_receipt_loader_rejects_execution_environment_fingerprint_mismatch(tmp_path: Path, monkeypatch) -> None:
    gate_path = tmp_path / "suite_precheck_gate.json"
    output_root = tmp_path / "results" / "matrix"
    gate_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "current_step": "complete",
                "steps": [],
                "launch_receipt": _launch_receipt_payload(output_root=output_root),
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=output_root,
        precheck_gate=gate_path,
        skip_hf_access=True,
    )
    monkeypatch.setattr(
        certify_suite_precheck,
        "_current_environment_receipt_payload",
        lambda current_args: {
            **_stable_environment_receipt(),
            "execution_environment_fingerprint": "different-execution-class",
        },
    )

    matched = certify_suite_precheck._load_matching_launch_receipt(args, output_root=output_root, gate_path=gate_path)

    assert matched is None


def test_launch_receipt_loader_rejects_runtime_checkout_drift(tmp_path: Path, monkeypatch) -> None:
    gate_path = tmp_path / "suite_precheck_gate.json"
    output_root = tmp_path / "results" / "matrix"
    gate_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "current_step": "complete",
                "steps": [],
                "launch_receipt": _launch_receipt_payload(output_root=output_root),
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=output_root,
        precheck_gate=gate_path,
        skip_hf_access=True,
    )
    monkeypatch.setattr(
        certify_suite_precheck,
        "_current_runtime_checkout_receipt",
        lambda: {
            **_stable_runtime_checkout_receipt(),
            "stone_runtime": {
                **_stable_runtime_checkout_receipt()["stone_runtime"],
                "upstream_commit": "different-upstream",
            },
        },
    )

    matched = certify_suite_precheck._load_matching_launch_receipt(args, output_root=output_root, gate_path=gate_path)

    assert matched is None


def test_validate_stage_manifest_against_full_accepts_matching_subset_with_tighter_limit(tmp_path: Path) -> None:
    full_manifest = tmp_path / "suite_all_models_methods.json"
    stage_manifest = tmp_path / "suite_canary_heavy.json"
    full_manifest.write_text(
        json.dumps(
            {
                "profile": "suite_all_models_methods",
                "runs": [
                    {
                        "run_id": "suite_qwen_crafted_stone",
                        "profile": "suite_all_models_methods",
                        "model": "Qwen/Qwen2.5-Coder-14B-Instruct",
                        "method": "stone_runtime",
                        "source_slug": "crafted_original",
                        "config": "crafted_original",
                        "baseline_eval": "crafted_eval",
                        "baseline_eval_sample_limit": 64,
                        "resource": "gpu",
                        "gpu_pool": "shared",
                        "model_revision": "rev-a",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    stage_manifest.write_text(
        json.dumps(
            {
                "profile": "suite_canary_heavy",
                "runs": [
                    {
                        "run_id": "stage_a_qwen_crafted_stone",
                        "profile": "suite_canary_heavy",
                        "model": "Qwen/Qwen2.5-Coder-14B-Instruct",
                        "method": "stone_runtime",
                        "source_slug": "crafted_original",
                        "config": "crafted_original",
                        "baseline_eval": "crafted_eval",
                        "baseline_eval_sample_limit": 16,
                        "resource": "gpu",
                        "gpu_pool": "shared",
                        "model_revision": "rev-a",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    certify_suite_precheck.validate_stage_manifest_against_full(
        full_manifest_path=full_manifest,
        full_profile="suite_all_models_methods",
        stage_manifest_path=stage_manifest,
        stage_profile="suite_canary_heavy",
    )


def test_validate_stage_manifest_against_full_rejects_mixed_roster(tmp_path: Path) -> None:
    full_manifest = tmp_path / "suite_all_models_methods.json"
    stage_manifest = tmp_path / "suite_canary_heavy.json"
    full_manifest.write_text(
        json.dumps(
            {
                "profile": "suite_all_models_methods",
                "runs": [
                    {
                        "run_id": "suite_qwen_crafted_stone",
                        "profile": "suite_all_models_methods",
                        "model": "Qwen/Qwen2.5-Coder-14B-Instruct",
                        "method": "stone_runtime",
                        "source_slug": "crafted_original",
                        "config": "crafted_original",
                        "baseline_eval": "crafted_eval",
                        "baseline_eval_sample_limit": 64,
                        "resource": "gpu",
                        "gpu_pool": "shared",
                        "model_revision": "rev-a",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    stage_manifest.write_text(
        json.dumps(
            {
                "profile": "suite_canary_heavy",
                "runs": [
                    {
                        "run_id": "stage_a_qwen_other_stone",
                        "profile": "suite_canary_heavy",
                        "model": "Qwen/Qwen2.5-Coder-14B-Instruct",
                        "method": "stone_runtime",
                        "source_slug": "humaneval_plus",
                        "config": "humaneval_plus",
                        "baseline_eval": "crafted_eval",
                        "baseline_eval_sample_limit": 16,
                        "resource": "gpu",
                        "gpu_pool": "shared",
                        "model_revision": "rev-a",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not align with full manifest"):
        certify_suite_precheck.validate_stage_manifest_against_full(
            full_manifest_path=full_manifest,
            full_profile="suite_all_models_methods",
            stage_manifest_path=stage_manifest,
            stage_profile="suite_canary_heavy",
        )


def test_precheck_rejects_resume_instead_of_emitting_next_command(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "gate.json"
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=tmp_path / "results" / "matrix",
        figure_output_dir=tmp_path / "results" / "figures",
        output=output_path,
        preflight_receipt=tmp_path / "preflight.json",
        gpu_slots=8,
        gpu_pool_mode="shared",
        cpu_workers=12,
        retry_count=1,
        command_timeout_seconds=259200,
        step_timeout_seconds=60,
        skip_hf_access=True,
        resume=True,
        fail_fast=False,
    )
    monkeypatch.setattr(certify_suite_precheck, "parse_args", lambda: args)
    assert certify_suite_precheck.main() == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "not supported for suite precheck" in payload["error"]


def test_run_matrix_step_terminates_process_tree_on_timeout(tmp_path: Path, monkeypatch) -> None:
    class FakeProcess:
        pid = 4321

        def poll(self):
            return None

        def wait(self, timeout=None):
            return None

    gate_output = tmp_path / "gate.json"
    matrix_index = tmp_path / "matrix_index.json"
    matrix_index.write_text('{"runs":[]}\n', encoding="utf-8")
    killed: list[int] = []
    time_values = iter([0.0, 10.0])

    monkeypatch.setattr(certify_suite_precheck.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(certify_suite_precheck, "_matrix_progress", lambda _path: None)
    monkeypatch.setattr(certify_suite_precheck, "_terminate_process_tree", lambda process: killed.append(process.pid))
    monkeypatch.setattr(certify_suite_precheck.time, "time", lambda: next(time_values))
    monkeypatch.setattr(certify_suite_precheck.time, "sleep", lambda _seconds: None)

    with pytest.raises(subprocess.TimeoutExpired):
        certify_suite_precheck._run_matrix_step(
            ["python", "fake.py"],
            timeout_seconds=1,
            label="stage_a_suite_canary_heavy",
            gate_output_path=gate_output,
            steps=[],
            matrix_index_path=matrix_index,
        )

    assert killed == [4321]


def test_terminate_process_tree_kills_descendants_on_posix(monkeypatch) -> None:
    class FakeProcess:
        pid = 4321

        def poll(self):
            return None

    killed: list[object] = []
    monkeypatch.setattr(certify_suite_precheck.os, "name", "posix", raising=False)
    monkeypatch.setattr(
        certify_suite_precheck.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="5000 4321\n6000 5000\n", returncode=0),
    )
    monkeypatch.setattr(certify_suite_precheck.os, "kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(certify_suite_precheck.os, "killpg", lambda pid, sig: killed.append(("pg", pid)), raising=False)

    certify_suite_precheck._terminate_process_tree(FakeProcess())

    assert 6000 in killed
    assert 5000 in killed
    assert ("pg", 4321) in killed


def test_main_writes_failed_payload_when_step_times_out(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "gate.json"
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=tmp_path / "results" / "matrix",
        figure_output_dir=tmp_path / "results" / "figures",
        output=output_path,
        preflight_receipt=tmp_path / "preflight.json",
        gpu_slots=8,
        gpu_pool_mode="shared",
        cpu_workers=12,
        retry_count=1,
        command_timeout_seconds=259200,
        step_timeout_seconds=60,
        skip_hf_access=True,
        resume=False,
        fail_fast=False,
    )
    monkeypatch.setattr(certify_suite_precheck, "parse_args", lambda: args)
    monkeypatch.setattr(
        certify_suite_precheck,
        "_run_step",
        lambda command, *, timeout_seconds, label: (_ for _ in ()).throw(subprocess.TimeoutExpired(command, timeout_seconds)),
    )

    assert certify_suite_precheck.main() == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["steps"][-1]["name"] == "build_suite_manifests"
    assert "timed out" in payload["steps"][-1]["error"]
