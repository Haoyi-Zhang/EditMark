from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from posteditbench.baselines.stone_family.common import runtime_watermark_names, stone_family_checkout_status
from posteditbench.suite import SUITE_MODEL_REVISIONS
from scripts import _repo_snapshot, capture_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the two-stage PostEditBench suite precheck.")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--full-manifest", type=Path, default=Path("configs/matrices/suite_all_models_methods.json"))
    parser.add_argument("--full-profile", type=str, default="suite_all_models_methods")
    parser.add_argument("--stage-a-manifest", type=Path, default=Path("configs/matrices/suite_canary_heavy.json"))
    parser.add_argument("--stage-a-profile", type=str, default="suite_canary_heavy")
    parser.add_argument("--stage-b-manifest", type=Path, default=Path("configs/matrices/model_invocation_smoke.json"))
    parser.add_argument("--stage-b-profile", type=str, default="model_invocation_smoke")
    parser.add_argument("--output-root", type=Path, default=Path("results/matrix"))
    parser.add_argument("--figure-output-dir", type=Path, default=Path("results/figures/suite_precheck"))
    parser.add_argument("--output", type=Path, default=Path("results/certifications/suite_precheck_gate.json"))
    parser.add_argument("--preflight-receipt", type=Path, default=Path("results/certifications/remote_preflight_receipt.json"))
    parser.add_argument("--gpu-slots", type=int, default=8)
    parser.add_argument("--gpu-pool-mode", choices=("split", "shared"), default="shared")
    parser.add_argument("--cpu-workers", type=int, default=9)
    parser.add_argument("--retry-count", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--command-timeout-seconds", type=int, default=259200)
    parser.add_argument("--step-timeout-seconds", type=int, default=259200)
    parser.add_argument("--skip-hf-access", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_manifest_digests(args: argparse.Namespace) -> dict[str, str]:
    return {
        "full_manifest": _file_sha256(_resolve(args.full_manifest)),
        "stage_a_manifest": _file_sha256(_resolve(args.stage_a_manifest)),
        "stage_b_manifest": _file_sha256(_resolve(args.stage_b_manifest)),
    }


def _current_model_revision_payload() -> dict[str, str]:
    return dict(SUITE_MODEL_REVISIONS)


def _normalize_python_bin(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    candidate = Path(raw)
    if candidate.is_absolute() or any(separator in raw for separator in ("/", "\\")):
        return str(candidate.resolve(strict=False))
    return raw


def _normalized_cuda_visible_devices(raw: str | None = None) -> str:
    source = os.environ.get("CUDA_VISIBLE_DEVICES", "") if raw is None else raw
    tokens = [token.strip() for token in str(source).split(",") if token.strip()]
    return ",".join(tokens)


def _environment_receipt_from_payload(
    *,
    python_bin: str,
    environment_payload: Mapping[str, Any],
    code_snapshot_digest: str | None = None,
    cuda_visible_devices: str | None = None,
) -> dict[str, Any]:
    python_payload = environment_payload.get("python", {}) if isinstance(environment_payload, Mapping) else {}
    host_payload = environment_payload.get("host", {}) if isinstance(environment_payload, Mapping) else {}
    execution_payload = environment_payload.get("execution", {}) if isinstance(environment_payload, Mapping) else {}
    normalized_visible_devices = _normalized_cuda_visible_devices(cuda_visible_devices)
    visible_gpu_count = len(
        capture_environment.execution_class_gpu_devices(
            environment_payload,
            cuda_visible_devices=normalized_visible_devices,
        )
    )
    execution_environment_fingerprint = str(
        execution_payload.get("execution_environment_fingerprint", "")
        if isinstance(execution_payload, Mapping)
        else ""
    ).strip()
    if not execution_environment_fingerprint:
        execution_environment_fingerprint = capture_environment.execution_environment_fingerprint_sha256(
            environment_payload,
            cuda_visible_devices=normalized_visible_devices,
        )
    normalized_snapshot_digest = str(code_snapshot_digest or "").strip()
    if not normalized_snapshot_digest:
        normalized_snapshot_digest = str(
            execution_payload.get("code_snapshot_digest", "")
            if isinstance(execution_payload, Mapping)
            else ""
        ).strip()
    return {
        "python_bin": _normalize_python_bin(str(python_bin or "")),
        "python_executable": _normalize_python_bin(
            str(python_payload.get("executable", "") if isinstance(python_payload, Mapping) else "")
        ),
        "environment_fingerprint": capture_environment.environment_fingerprint_sha256(environment_payload),
        "execution_environment_fingerprint": execution_environment_fingerprint,
        "cuda_visible_devices": normalized_visible_devices,
        "visible_gpu_count": visible_gpu_count,
        "preflight_gpu_slots": visible_gpu_count,
        "code_snapshot_digest": normalized_snapshot_digest,
        "host_identity": {
            "hostname": str(host_payload.get("hostname", "") if isinstance(host_payload, Mapping) else "").strip(),
            "fqdn": str(host_payload.get("fqdn", "") if isinstance(host_payload, Mapping) else "").strip(),
        },
    }


def _current_environment_receipt_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = capture_environment._collect()
    return _environment_receipt_from_payload(
        python_bin=str(getattr(args, "python_bin", "") or ""),
        environment_payload=payload,
        code_snapshot_digest=_repo_snapshot.repo_snapshot_sha256(ROOT),
        cuda_visible_devices=_normalized_cuda_visible_devices(),
    )


def _launch_environment_receipt_payload(environment_receipt: Mapping[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(environment_receipt, Mapping):
        return {}
    host_identity = environment_receipt.get("host_identity")
    return {
        "python_bin": _normalize_python_bin(str(environment_receipt.get("python_bin", ""))),
        "python_executable": _normalize_python_bin(str(environment_receipt.get("python_executable", ""))),
        "execution_environment_fingerprint": str(
            environment_receipt.get("execution_environment_fingerprint", "")
        ).strip(),
        "cuda_visible_devices": _normalized_cuda_visible_devices(str(environment_receipt.get("cuda_visible_devices", ""))),
        "visible_gpu_count": int(environment_receipt.get("visible_gpu_count", 0) or 0),
        "preflight_gpu_slots": int(environment_receipt.get("preflight_gpu_slots", 0) or 0),
        "code_snapshot_digest": str(environment_receipt.get("code_snapshot_digest", "")).strip(),
        "host_identity": {
            "hostname": str(host_identity.get("hostname", "") if isinstance(host_identity, Mapping) else "").strip(),
            "fqdn": str(host_identity.get("fqdn", "") if isinstance(host_identity, Mapping) else "").strip(),
        },
    }


def _launch_environment_receipt_matches_contract(
    observed: Mapping[str, Any] | Any,
    expected: Mapping[str, Any],
) -> bool:
    return _launch_environment_receipt_payload(observed) == _launch_environment_receipt_payload(expected)


def _runtime_checkout_receipt_entry(method: str) -> dict[str, Any]:
    checkout = stone_family_checkout_status(method)
    if checkout is None:
        return {
            "checkout_present": False,
            "checkout_valid": False,
            "origin": "missing",
            "repo_root": "",
            "source_root": "",
            "source_relative": "",
            "external_path": "",
            "remote_url": "",
            "manifest_pinned_commit": "",
            "upstream_commit": "",
            "dirty": False,
        }
    payload = checkout.as_dict()
    return {
        "checkout_present": bool(payload.get("checkout_present", False)),
        "checkout_valid": bool(payload.get("checkout_valid", False)),
        "origin": str(payload.get("origin", "")).strip(),
        "repo_root": str(payload.get("repo_root", "")).strip(),
        "source_root": str(payload.get("source_root", "")).strip(),
        "source_relative": str(payload.get("source_relative", "")).strip(),
        "external_path": str(payload.get("external_path", "")).strip(),
        "remote_url": str(payload.get("remote_url", "")).strip(),
        "manifest_pinned_commit": str(payload.get("manifest_pinned_commit", "")).strip(),
        "upstream_commit": str(payload.get("upstream_commit", "")).strip(),
        "dirty": bool(payload.get("dirty", False)),
    }


def _current_runtime_checkout_receipt() -> dict[str, dict[str, Any]]:
    return {
        method: _runtime_checkout_receipt_entry(method)
        for method in runtime_watermark_names()
    }


def _receipt_matches_expected_contract(
    payload: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    output_root: Path,
    environment_receipt: dict[str, Any],
    runtime_checkout_receipt: dict[str, dict[str, Any]],
    launch_receipt: bool = False,
) -> bool:
    if not isinstance(payload, Mapping):
        return False
    expected_pairs = {
        "full_manifest": str(_resolve(args.full_manifest)),
        "full_profile": str(args.full_profile),
        "stage_a_manifest": str(_resolve(args.stage_a_manifest)),
        "stage_a_profile": str(args.stage_a_profile),
        "stage_b_manifest": str(_resolve(args.stage_b_manifest)),
        "stage_b_profile": str(args.stage_b_profile),
        "output_root": str(output_root),
    }
    for key, expected in expected_pairs.items():
        observed = str(payload.get(key, "")).strip()
        if observed != expected:
            return False
    if bool(payload.get("skip_hf_access", False)) != bool(args.skip_hf_access):
        return False
    if payload.get("manifest_digests") != _current_manifest_digests(args):
        return False
    if payload.get("suite_model_revisions") != _current_model_revision_payload():
        return False
    if payload.get("runtime_checkout_receipt") != runtime_checkout_receipt:
        return False
    observed_environment_receipt = payload.get("environment_receipt")
    if launch_receipt:
        if not _launch_environment_receipt_matches_contract(observed_environment_receipt, environment_receipt):
            return False
    elif observed_environment_receipt != environment_receipt:
        return False
    return True


def _stage_manifest_alignment_key(run: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(run.get("model", "")).strip(),
        str(run.get("method", "")).strip(),
        str(run.get("source_slug", "")).strip(),
        str(run.get("config", "")).strip(),
        str(run.get("baseline_eval", "")).strip(),
        str(run.get("resource", "")).strip(),
        str(run.get("gpu_pool", "")).strip(),
        str(run.get("model_revision", "")).strip(),
    )


def _optional_int(value: Any) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def validate_stage_manifest_against_full(
    *,
    full_manifest_path: Path,
    full_profile: str,
    stage_manifest_path: Path,
    stage_profile: str,
) -> None:
    full_payload = json.loads(full_manifest_path.read_text(encoding="utf-8"))
    stage_payload = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    if str(full_payload.get("profile", "")).strip() != str(full_profile).strip():
        raise ValueError(f"full manifest profile mismatch: expected {full_profile}, found {full_payload.get('profile')}")
    if str(stage_payload.get("profile", "")).strip() != str(stage_profile).strip():
        raise ValueError(f"stage manifest profile mismatch: expected {stage_profile}, found {stage_payload.get('profile')}")
    full_runs = full_payload.get("runs", [])
    stage_runs = stage_payload.get("runs", [])
    if not isinstance(full_runs, list) or not isinstance(stage_runs, list):
        raise ValueError("manifest payloads must contain list-valued 'runs'")
    full_by_key: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for run in full_runs:
        if not isinstance(run, Mapping):
            raise ValueError(f"full manifest {full_manifest_path} contains a non-object run entry")
        key = _stage_manifest_alignment_key(run)
        if key in full_by_key:
            raise ValueError(f"full manifest {full_manifest_path} contains duplicate alignment key {key}")
        full_by_key[key] = run
    for run in stage_runs:
        if not isinstance(run, Mapping):
            raise ValueError(f"stage manifest {stage_manifest_path} contains a non-object run entry")
        key = _stage_manifest_alignment_key(run)
        full_run = full_by_key.get(key)
        stage_run_id = str(run.get("run_id", "")).strip() or "<unknown>"
        if full_run is None:
            raise ValueError(
                f"stage manifest {stage_manifest_path} contains run {stage_run_id} that does not align with full manifest {full_manifest_path}"
            )
        stage_limit = _optional_int(run.get("baseline_eval_sample_limit"))
        full_limit = _optional_int(full_run.get("baseline_eval_sample_limit"))
        if full_limit is not None and (stage_limit is None or stage_limit > full_limit):
            raise ValueError(
                f"stage manifest {stage_manifest_path} contains run {stage_run_id} with baseline_eval_sample_limit={stage_limit} "
                f"outside the full-manifest bound {full_limit}"
            )


def _run_step(command: list[str], *, timeout_seconds: int, label: str) -> dict[str, Any]:
    print(f"[suite_precheck] start {label}: {' '.join(command)}", flush=True)
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=max(1, int(timeout_seconds)),
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")},
    )
    result = {
        "name": label,
        "command": command,
        "returncode": int(completed.returncode),
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": (completed.stdout or "")[-4000:],
        "stderr_tail": (completed.stderr or "")[-4000:],
        "status": "passed" if int(completed.returncode) == 0 else "failed",
    }
    print(
        f"[suite_precheck] finish {label}: status={result['status']} returncode={result['returncode']} duration_seconds={result['duration_seconds']}",
        flush=True,
    )
    return result


def _tail_text(path: Path, *, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    payload = path.read_text(encoding="utf-8", errors="replace")
    return payload[-limit:]


def _matrix_progress(matrix_index_path: Path) -> dict[str, Any] | None:
    if not matrix_index_path.exists():
        return None
    try:
        from monitor_matrix import build_dashboard_data

        dashboard = build_dashboard_data(matrix_index_path)
    except Exception:
        return None
    overall = dict(dashboard.get("overall", {}))
    longest_tail = dict(dashboard.get("longest_tail") or {})
    return {
        "matrix_index": str(matrix_index_path),
        "success_count": int(overall.get("success_count", 0) or 0),
        "running_count": int(overall.get("running_count", 0) or 0),
        "failed_count": int(overall.get("failed_count", 0) or 0),
        "pending_count": int(overall.get("pending_count", 0) or 0),
        "progress_fraction": float(overall.get("progress_fraction", 0.0) or 0.0),
        "eta_seconds": float(overall.get("eta_seconds", 0.0) or 0.0) if overall.get("eta_seconds") is not None else None,
        "active_models": list(dashboard.get("active_models", [])),
        "completed_models": list(dashboard.get("completed_models", [])),
        "longest_tail": {
            "run_id": str(longest_tail.get("run_id", "")),
            "model_name": str(longest_tail.get("model_name", "")),
            "benchmark_name": str(longest_tail.get("benchmark_name", "")),
            "method_name": str(longest_tail.get("method_name", "")),
            "elapsed_seconds": float(longest_tail.get("elapsed_seconds", 0.0) or 0.0),
        }
        if longest_tail
        else {},
    }


def _descendant_pids(pid: int) -> list[int]:
    if os.name == "nt" or pid <= 0:
        return []
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,ppid="],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return []
    children: dict[int, list[int]] = {}
    for line in completed.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        try:
            child_pid = int(parts[0])
            parent_pid = int(parts[1])
        except ValueError:
            continue
        children.setdefault(parent_pid, []).append(child_pid)
    descendants: list[int] = []
    stack = list(children.get(pid, []))
    while stack:
        child = stack.pop()
        descendants.append(child)
        stack.extend(children.get(child, []))
    descendants.sort(reverse=True)
    return descendants


def _kill_signal() -> int:
    return int(getattr(signal, "SIGKILL", signal.SIGTERM))


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except Exception:
            pass
    else:
        for child_pid in _descendant_pids(int(process.pid or 0)):
            try:
                os.kill(child_pid, _kill_signal())
            except OSError:
                pass
        try:
            os.killpg(process.pid, _kill_signal())
            return
        except Exception:
            pass
    try:
        process.kill()
    except Exception:
        pass


def _run_matrix_step(
    command: list[str],
    *,
    timeout_seconds: int,
    label: str,
    gate_output_path: Path,
    steps: list[dict[str, Any]],
    matrix_index_path: Path,
) -> dict[str, Any]:
    print(f"[suite_precheck] start {label}: {' '.join(command)}", flush=True)
    started = time.time()
    stdout_handle = tempfile.NamedTemporaryFile("w+", encoding="utf-8", newline="\n", delete=False)
    stderr_handle = tempfile.NamedTemporaryFile("w+", encoding="utf-8", newline="\n", delete=False)
    stdout_path = Path(stdout_handle.name)
    stderr_path = Path(stderr_handle.name)
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")},
        **(
            {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
            if os.name == "nt"
            else {"start_new_session": True}
        ),
    )
    try:
        while True:
            returncode = process.poll()
            progress = _matrix_progress(matrix_index_path)
            _write_payload(
                gate_output_path,
                {
                    "status": "running",
                    "current_step": label,
                    "steps": steps,
                    "matrix_progress": progress,
                },
            )
            if returncode is not None:
                break
            if time.time() - started > max(1, int(timeout_seconds)):
                _terminate_process_tree(process)
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            time.sleep(2.0)
    finally:
        try:
            process.wait(timeout=5)
        except Exception:
            pass
        try:
            stdout_handle.close()
        except Exception:
            pass
        try:
            stderr_handle.close()
        except Exception:
            pass
    result = {
        "name": label,
        "command": command,
        "returncode": int(process.returncode or 0),
        "duration_seconds": round(time.time() - started, 3),
        "stdout_tail": _tail_text(stdout_path),
        "stderr_tail": _tail_text(stderr_path),
        "status": "passed" if int(process.returncode or 0) == 0 else "failed",
        "matrix_progress": _matrix_progress(matrix_index_path),
    }
    try:
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)
    except Exception:
        pass
    print(
        f"[suite_precheck] finish {label}: status={result['status']} returncode={result['returncode']} duration_seconds={result['duration_seconds']}",
        flush=True,
    )
    return result


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def _matrix_index_path(output_root: Path, profile: str) -> Path:
    return output_root / profile / "matrix_index.json"


def _load_matching_preflight_receipt(args: argparse.Namespace, *, output_root: Path) -> dict[str, Any] | None:
    receipt_path = _resolve(args.preflight_receipt)
    if not receipt_path.exists():
        return None
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if str(payload.get("receipt_type", "")).strip() != "remote_preflight":
        return None
    if str(payload.get("status", "")).strip().lower() != "passed":
        return None
    if not _receipt_matches_expected_contract(
        payload,
        args=args,
        output_root=output_root,
        environment_receipt=_current_environment_receipt_payload(args),
        runtime_checkout_receipt=_current_runtime_checkout_receipt(),
    ):
        return None
    return payload


def _build_launch_receipt(args: argparse.Namespace, *, output_root: Path) -> dict[str, Any]:
    environment_receipt = _current_environment_receipt_payload(args)
    return {
        "schema_version": 1,
        "receipt_type": "post_precheck_launch",
        "status": "ready",
        "created_at_epoch_seconds": int(time.time()),
        "full_manifest": str(_resolve(args.full_manifest)),
        "full_profile": str(args.full_profile),
        "stage_a_manifest": str(_resolve(args.stage_a_manifest)),
        "stage_a_profile": str(args.stage_a_profile),
        "stage_b_manifest": str(_resolve(args.stage_b_manifest)),
        "stage_b_profile": str(args.stage_b_profile),
        "output_root": str(output_root),
        "skip_hf_access": bool(args.skip_hf_access),
        "manifest_digests": _current_manifest_digests(args),
        "suite_model_revisions": _current_model_revision_payload(),
        "runtime_checkout_receipt": _current_runtime_checkout_receipt(),
        "environment_receipt": _launch_environment_receipt_payload(environment_receipt),
    }


def _load_matching_launch_receipt(
    args: argparse.Namespace,
    *,
    output_root: Path,
    gate_path: Path,
) -> dict[str, Any] | None:
    resolved_gate_path = _resolve(gate_path)
    if not resolved_gate_path.exists():
        return None
    try:
        gate_payload = json.loads(resolved_gate_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if str(gate_payload.get("status", "")).strip().lower() != "passed":
        return None
    if str(gate_payload.get("current_step", "")).strip() != "complete":
        return None
    launch_receipt = gate_payload.get("launch_receipt")
    if not isinstance(launch_receipt, Mapping):
        return None
    if str(launch_receipt.get("receipt_type", "")).strip() != "post_precheck_launch":
        return None
    if str(launch_receipt.get("status", "")).strip().lower() != "ready":
        return None
    if not _receipt_matches_expected_contract(
        launch_receipt,
        args=args,
        output_root=output_root,
        environment_receipt=_current_environment_receipt_payload(args),
        runtime_checkout_receipt=_current_runtime_checkout_receipt(),
        launch_receipt=True,
    ):
        return None
    return dict(launch_receipt)


def _reused_step(command: list[str], *, label: str, receipt_path: Path) -> dict[str, Any]:
    return {
        "name": label,
        "command": command,
        "returncode": 0,
        "duration_seconds": 0.0,
        "stdout_tail": "",
        "stderr_tail": "",
        "status": "passed",
        "reason": "matching_clean_preflight_receipt",
        "preflight_receipt": str(receipt_path),
    }


def _failed_step(
    command: list[str],
    *,
    label: str,
    error: str,
    duration_seconds: float = 0.0,
) -> dict[str, Any]:
    return {
        "name": label,
        "command": command,
        "returncode": None,
        "duration_seconds": round(float(duration_seconds or 0.0), 3),
        "stdout_tail": "",
        "stderr_tail": "",
        "status": "failed",
        "error": error,
    }


def _first_report_path(matrix_index_path: Path) -> Path:
    payload = json.loads(matrix_index_path.read_text(encoding="utf-8"))
    for run in payload.get("runs", []):
        report_path = str(run.get("report_path", "")).strip()
        if report_path:
            candidate = Path(report_path)
            if candidate.exists():
                return candidate
    raise RuntimeError(f"no report_path found in {matrix_index_path}")


def _render_command(
    *,
    python_bin: str,
    matrix_index_path: Path,
    anchor_report_path: Path,
    figure_output_dir: Path,
    require_times_new_roman: bool = False,
) -> list[str]:
    command = [
        python_bin,
        str(ROOT / "scripts" / "render_paper_figures.py"),
        "--report",
        str(anchor_report_path),
        "--suite",
        "basic",
        "--paper-track",
        "generation_time",
        "--output-dir",
        str(figure_output_dir),
        "--prefix",
        "suite_precheck",
    ]
    command.append("--require-times-new-roman" if require_times_new_roman else "--allow-font-fallback")
    return command


def main() -> int:
    args = parse_args()
    output_path = _resolve(args.output)
    if bool(getattr(args, "resume", False)):
        payload = {
            "status": "failed",
            "current_step": "initial_validation",
            "steps": [],
            "error": "--resume is not supported for suite precheck; rerun precheck cleanly and reserve --resume for run_full_matrix.py",
        }
        _write_payload(output_path, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1
    fail_fast = bool(getattr(args, "fail_fast", False))
    output_root = _resolve(args.output_root)
    figure_output_dir = _resolve(args.figure_output_dir)
    preflight_receipt = _load_matching_preflight_receipt(args, output_root=output_root)
    preflight_receipt_path = _resolve(args.preflight_receipt)
    steps: list[dict[str, Any]] = []

    stage_commands = [
        (
            "build_suite_manifests",
            [
                args.python_bin,
                str(ROOT / "scripts" / "build_suite_manifests.py"),
            ],
        ),
        (
            "audit_benchmarks",
            [
                args.python_bin,
                str(ROOT / "scripts" / "audit_benchmarks.py"),
                "--manifest",
                str(_resolve(args.full_manifest)),
                "--matrix-profile",
                args.full_profile,
                "--profile",
                args.full_profile,
            ],
        ),
        (
            "audit_suite_matrix",
            [
                args.python_bin,
                str(ROOT / "scripts" / "audit_full_matrix.py"),
                "--manifest",
                str(_resolve(args.full_manifest)),
                "--profile",
                args.full_profile,
                "--strict-hf-cache",
                "--model-load-smoke",
                "--runtime-smoke",
                *(["--skip-hf-access"] if args.skip_hf_access else []),
            ],
        ),
        (
            "stage_a_suite_canary_heavy",
            [
                args.python_bin,
                str(ROOT / "scripts" / "run_full_matrix.py"),
                "--manifest",
                str(_resolve(args.stage_a_manifest)),
                "--profile",
                args.stage_a_profile,
                "--output-root",
                str(output_root),
                "--gpu-slots",
                str(args.gpu_slots),
                "--gpu-pool-mode",
                args.gpu_pool_mode,
                "--cpu-workers",
                str(args.cpu_workers),
                "--retry-count",
                str(args.retry_count),
                "--command-timeout-seconds",
                str(getattr(args, "command_timeout_seconds", 259200)),
                *(["--fail-fast"] if fail_fast else []),
                *(["--resume"] if args.resume else []),
            ],
        ),
        (
            "stage_b_model_invocation_smoke",
            [
                args.python_bin,
                str(ROOT / "scripts" / "run_full_matrix.py"),
                "--manifest",
                str(_resolve(args.stage_b_manifest)),
                "--profile",
                args.stage_b_profile,
                "--output-root",
                str(output_root),
                "--gpu-slots",
                str(args.gpu_slots),
                "--gpu-pool-mode",
                args.gpu_pool_mode,
                "--cpu-workers",
                str(args.cpu_workers),
                "--retry-count",
                str(args.retry_count),
                "--command-timeout-seconds",
                str(getattr(args, "command_timeout_seconds", 259200)),
                *(["--fail-fast"] if fail_fast else []),
                *(["--resume"] if args.resume else []),
            ],
        ),
    ]

    status = "passed"
    for label, command in stage_commands:
        _write_payload(
            output_path,
            {
                "status": "running",
                "current_step": label,
                "steps": steps,
            },
        )
        step_started = time.time()
        try:
            if preflight_receipt is not None and label in {"build_suite_manifests", "audit_benchmarks", "audit_suite_matrix"}:
                result = _reused_step(command, label=label, receipt_path=preflight_receipt_path)
            elif label == "stage_a_suite_canary_heavy":
                result = _run_matrix_step(
                    command,
                    timeout_seconds=args.step_timeout_seconds,
                    label=label,
                    gate_output_path=output_path,
                    steps=steps,
                    matrix_index_path=_matrix_index_path(output_root, args.stage_a_profile),
                )
            elif label == "stage_b_model_invocation_smoke":
                result = _run_matrix_step(
                    command,
                    timeout_seconds=args.step_timeout_seconds,
                    label=label,
                    gate_output_path=output_path,
                    steps=steps,
                    matrix_index_path=_matrix_index_path(output_root, args.stage_b_profile),
                )
            else:
                result = _run_step(command, timeout_seconds=args.step_timeout_seconds, label=label)
        except subprocess.TimeoutExpired:
            result = _failed_step(
                command,
                label=label,
                error=f"step timed out after {int(args.step_timeout_seconds)} seconds",
                duration_seconds=time.time() - step_started,
            )
        except Exception as exc:
            result = _failed_step(
                command,
                label=label,
                error=f"{exc.__class__.__name__}: {exc}",
                duration_seconds=time.time() - step_started,
            )
        steps.append(result)
        payload = {
            "status": "running" if result["status"] == "passed" else "failed",
            "current_step": label,
            "steps": steps,
        }
        if result.get("matrix_progress") is not None:
            payload["matrix_progress"] = result["matrix_progress"]
        if result["status"] != "passed":
            status = "failed"
            payload["status"] = status
            _write_payload(output_path, payload)
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 1
        _write_payload(output_path, payload)

    try:
        stage_a_index = _matrix_index_path(output_root, args.stage_a_profile)
        anchor_report_path = _first_report_path(stage_a_index)
    except Exception as exc:  # pragma: no cover - defensive gate failure path
        payload = {
            "status": "failed",
            "current_step": "render_suite_figures",
            "steps": steps,
            "error": str(exc),
        }
        _write_payload(output_path, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1

    _write_payload(
        output_path,
        {
            "status": "running",
            "current_step": "render_suite_figures",
            "steps": steps,
        },
    )
    render_command = _render_command(
        python_bin=args.python_bin,
        matrix_index_path=stage_a_index,
        anchor_report_path=anchor_report_path,
        figure_output_dir=figure_output_dir,
        require_times_new_roman=False,
    )
    render_started = time.time()
    try:
        render_step = _run_step(
            render_command,
            timeout_seconds=args.step_timeout_seconds,
            label="render_suite_figures",
        )
    except subprocess.TimeoutExpired:
        render_step = _failed_step(
            render_command,
            label="render_suite_figures",
            error=f"step timed out after {int(args.step_timeout_seconds)} seconds",
            duration_seconds=time.time() - render_started,
        )
    except Exception as exc:
        render_step = _failed_step(
            render_command,
            label="render_suite_figures",
            error=f"{exc.__class__.__name__}: {exc}",
            duration_seconds=time.time() - render_started,
        )
    steps.append(render_step)
    if render_step["status"] != "passed":
        payload = {
            "status": "failed",
            "current_step": "render_suite_figures",
            "steps": steps,
        }
        _write_payload(output_path, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1
    _write_payload(
        output_path,
        {
            "status": "running",
            "current_step": "render_suite_figures",
            "steps": steps,
        },
    )

    canonical_full = (
        _resolve(args.full_manifest) == (ROOT / "configs" / "matrices" / "suite_all_models_methods.json").resolve(strict=False)
        and str(args.full_profile).strip() == "suite_all_models_methods"
    )
    if canonical_full:
        next_command = [
            "bash",
            str(ROOT / "scripts" / "remote" / "run_suite_matrix.sh"),
            "--python",
            args.python_bin,
            "--output-root",
            str(output_root),
            "--gpu-slots",
            str(args.gpu_slots),
            "--gpu-pool-mode",
            args.gpu_pool_mode,
            "--cpu-workers",
            str(args.cpu_workers),
            "--retry-count",
            str(args.retry_count),
            "--command-timeout-seconds",
            str(getattr(args, "command_timeout_seconds", 259200)),
            "--run-full",
            *(["--skip-hf-access"] if args.skip_hf_access else []),
            *(["--full-fail-fast"] if fail_fast else []),
        ]
    else:
        next_command = [
            args.python_bin,
            str(ROOT / "scripts" / "run_full_matrix.py"),
            "--manifest",
            str(_resolve(args.full_manifest)),
            "--profile",
            args.full_profile,
            "--output-root",
            str(output_root),
            "--gpu-slots",
            str(args.gpu_slots),
            "--gpu-pool-mode",
            args.gpu_pool_mode,
            "--cpu-workers",
            str(args.cpu_workers),
            "--retry-count",
            str(args.retry_count),
            "--command-timeout-seconds",
            str(getattr(args, "command_timeout_seconds", 259200)),
            *(["--resume"] if args.resume else []),
            *(["--fail-fast"] if fail_fast else []),
        ]

    launch_receipt = _build_launch_receipt(args, output_root=output_root)
    final_payload = {
        "status": status,
        "current_step": "complete",
        "steps": steps,
        "stage_a_profile": args.stage_a_profile,
        "stage_b_profile": args.stage_b_profile,
        "full_profile": args.full_profile,
        "figure_output_dir": str(figure_output_dir),
        "preflight_receipt": str(preflight_receipt_path) if preflight_receipt is not None else "",
        "launch_receipt": launch_receipt,
        "next_command": next_command,
    }
    _write_payload(output_path, final_payload)
    print(json.dumps(final_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
