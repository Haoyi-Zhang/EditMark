from __future__ import annotations

from pathlib import Path
import sys

import importlib.util
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "capture_environment.py"
SPEC = importlib.util.spec_from_file_location("capture_environment", MODULE_PATH)
assert SPEC and SPEC.loader
capture_environment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture_environment)


def test_parse_cuda_version_from_nvidia_smi_block() -> None:
    stdout = """
    NVIDIA-SMI 546.33                 Driver Version: 546.33       CUDA Version: 12.3
    """
    assert capture_environment._parse_cuda_version(stdout) == "12.3"


def test_run_normalizes_missing_command_error_to_english() -> None:
    result = capture_environment._run(["definitely-not-a-real-command-for-posteditbench-tests"])
    assert result["error"] in {"command not found", "process launch failed (OSError)"}
    assert result["returncode"] is None
    assert result["stdout"] == ""
    assert result["stderr"] == ""


def test_render_markdown_includes_normalized_tool_error() -> None:
    payload = {
        "label": "local",
        "host": {"hostname": "example-host", "fqdn": "example.invalid"},
        "platform": {"system": "Windows", "release": "11", "version": "10.0", "machine": "AMD64"},
        "python": {"executable": "python", "version": "3.14"},
        "packages": {"torch": "2.10.0+cpu", "transformers": "5.2.0", "numpy": "2.4.4", "pandas": "3.0.1"},
        "cuda": {"torch_cuda_version": "unknown", "nvidia_smi_cuda_version": "12.3"},
        "gpu": {"count": 0, "visible_gpu_count": 0, "cuda_visible_devices": "", "devices": [], "driver_version": None},
        "execution": {
            "execution_mode": "single_host_canonical",
            "cuda_visible_devices": "",
            "visible_gpu_count": 0,
            "code_snapshot_digest": "digest-123",
            "execution_environment_fingerprint": "fingerprint-456",
        },
        "tools": {
            "node": {
                "command": ["node", "--version"],
                "error": "command not found",
                "returncode": None,
                "stdout": "",
                "stderr": "",
            }
        },
    }
    rendered = capture_environment._render_markdown(payload)
    assert "- Hostname: `example-host`" in rendered
    assert "- FQDN: `example.invalid`" in rendered
    assert "- Execution mode: `single_host_canonical`" in rendered
    assert "- GPU count (physical): `0`" in rendered
    assert "- GPU count (visible execution class): `0`" in rendered
    assert "- Code snapshot digest: `digest-123`" in rendered
    assert "- Execution environment fingerprint: `fingerprint-456`" in rendered
    assert "- `node`: `error`" in rendered
    assert "error: `command not found`" in rendered


def test_public_safe_paths_remove_host_aliases_and_absolute_python_path() -> None:
    payload = {
        "host": {"hostname": "private-host", "fqdn": "private-host.example"},
        "python": {"executable": "/example/private/posteditbench_env/posteditbench_release_clean/bin/python"},
    }

    capture_environment._apply_public_safe_paths(payload)

    assert payload["host"] == {"hostname": "execution-host", "fqdn": "execution-host"}
    assert payload["python"]["executable"] == "<release-python>/python"
    rendered = capture_environment._render_markdown(
        {
            "label": "public",
            **payload,
            "platform": {},
            "packages": {},
            "cuda": {},
            "gpu": {},
            "execution": {},
            "tools": {},
        }
    )
    assert "/example/private" not in rendered
    assert "private-host" not in rendered


def test_render_markdown_prefers_visible_execution_class_devices_when_present() -> None:
    payload = {
        "label": "single_host_release_of_record",
        "platform": {"system": "Linux", "release": "6.8.0", "version": "#1 SMP", "machine": "x86_64"},
        "python": {"executable": "/venv/bin/python", "version": "3.10.12"},
        "packages": {"torch": "2.6.0+cu124", "transformers": "4.57.6", "numpy": "2.2.6", "pandas": None},
        "cuda": {"torch_cuda_version": "12.4", "nvidia_smi_cuda_version": "12.4"},
        "gpu": {
            "count": 8,
            "visible_gpu_count": 6,
            "cuda_visible_devices": "0,1,2,3,4,5",
            "devices": [
                {"name": "NVIDIA A800-SXM4-40GB", "driver_version": "550.163.01", "memory_total": "40960 MiB"}
                for _ in range(8)
            ],
            "visible_devices": [
                {"name": "NVIDIA A800-SXM4-40GB", "driver_version": "550.163.01", "memory_total": "40960 MiB"}
                for _ in range(6)
            ],
            "driver_version": "550.163.01",
        },
        "execution": {
            "execution_mode": "single_host_canonical",
            "cuda_visible_devices": "0,1,2,3,4,5",
            "visible_gpu_count": 6,
            "code_snapshot_digest": "snapshot-digest",
            "execution_environment_fingerprint": "env-fingerprint",
        },
        "tools": {},
    }

    rendered = capture_environment._render_markdown(payload)
    assert "- GPU count (physical): `8`" in rendered
    assert "- GPU count (visible execution class): `6`" in rendered
    assert "- CUDA_VISIBLE_DEVICES: `0,1,2,3,4,5`" in rendered
    assert "Listing the execution-class-visible devices selected by CUDA_VISIBLE_DEVICES" in rendered


def test_render_markdown_adds_code_identity_note_when_git_metadata_is_unavailable() -> None:
    payload = {
        "label": "single_host_release_of_record",
        "platform": {"system": "Linux", "release": "6.8.0", "version": "#1 SMP", "machine": "x86_64"},
        "python": {"executable": "/venv/bin/python", "version": "3.10.12"},
        "packages": {"torch": "2.6.0+cu124", "transformers": "4.57.6", "numpy": "2.2.6", "pandas": None},
        "cuda": {"torch_cuda_version": "12.4", "nvidia_smi_cuda_version": "12.4"},
        "gpu": {"count": 6, "visible_gpu_count": 6, "cuda_visible_devices": "0,1,2,3,4,5", "devices": []},
        "execution": {
            "execution_mode": "single_host_canonical",
            "cuda_visible_devices": "0,1,2,3,4,5",
            "visible_gpu_count": 6,
            "code_snapshot_digest": "snapshot-digest",
            "execution_environment_fingerprint": "env-fingerprint",
        },
        "tools": {
            "git": {
                "command": ["git", "rev-parse", "HEAD"],
                "error": "command returned non-zero exit status 128",
                "returncode": 128,
                "stdout": "",
                "stderr": "fatal: not a git repository",
            }
        },
    }

    rendered = capture_environment._render_markdown(payload)
    assert "code snapshot digest as the release code-identity anchor" in rendered


def test_environment_fingerprint_includes_platform_version_processor_and_gpu_devices() -> None:
    payload = {
        "platform": {
            "system": "Linux",
            "release": "6.8.0",
            "version": "#1 SMP PREEMPT_DYNAMIC",
            "machine": "x86_64",
            "processor": "Intel(R) Xeon(R)",
        },
        "python": {"executable": "/venv/bin/python", "version": "3.11.9"},
        "packages": {"torch": "2.7.0", "transformers": "4.52.0", "numpy": "2.2.0", "pandas": "2.2.3"},
        "cuda": {"torch_cuda_version": "12.4", "nvidia_smi_cuda_version": "12.4"},
        "gpu": {
            "count": 6,
            "driver_version": "550.54.15",
            "devices": [
                {"name": "NVIDIA A800-SXM4-40GB", "driver_version": "550.54.15", "memory_total": "40960 MiB"},
                {"name": "NVIDIA A800-SXM4-40GB", "driver_version": "550.54.15", "memory_total": "40960 MiB"},
            ],
        },
        "tools": {},
    }

    fingerprint_payload = capture_environment.environment_fingerprint_payload(payload)
    assert fingerprint_payload["platform"]["version"] == "#1 SMP PREEMPT_DYNAMIC"
    assert fingerprint_payload["platform"]["processor"] == "Intel(R) Xeon(R)"
    assert fingerprint_payload["gpu"]["devices"][0]["name"] == "NVIDIA A800-SXM4-40GB"
    assert fingerprint_payload["gpu"]["devices"][0]["memory_total"] == "40960 MiB"

    changed = {
        **payload,
        "gpu": {
            **payload["gpu"],
            "devices": [
                {"name": "NVIDIA A800-SXM4-80GB", "driver_version": "550.54.15", "memory_total": "81920 MiB"},
            ],
        },
    }
    assert capture_environment.environment_fingerprint_sha256(payload) != capture_environment.environment_fingerprint_sha256(changed)


def test_environment_fingerprint_ignores_git_head() -> None:
    base = {
        "platform": {"system": "Linux", "release": "6.8.0", "version": "#1 SMP", "machine": "x86_64", "processor": "Xeon"},
        "python": {"executable": "/venv/bin/python", "version": "3.10.12"},
        "packages": {"torch": "2.6.0+cu124", "transformers": "4.57.6", "numpy": "2.2.6", "pandas": None},
        "cuda": {"torch_cuda_version": "12.4", "nvidia_smi_cuda_version": "12.4"},
        "gpu": {"devices": []},
        "tools": {"git": {"stdout": "old-head\n", "stderr": "", "returncode": 0}},
    }
    changed = {**base, "tools": {"git": {"stdout": "new-head\n", "stderr": "", "returncode": 0}}}

    assert "git" not in capture_environment.environment_fingerprint_payload(base)["tools"]
    assert capture_environment.environment_fingerprint_sha256(base) == capture_environment.environment_fingerprint_sha256(changed)


def test_execution_environment_fingerprint_tracks_visible_gpu_selection() -> None:
    payload = {
        "platform": {
            "system": "Linux",
            "release": "6.8.0",
            "version": "#1 SMP PREEMPT_DYNAMIC",
            "machine": "x86_64",
            "processor": "Intel(R) Xeon(R)",
        },
        "python": {"executable": "/venv/bin/python", "version": "3.11.9"},
        "packages": {"torch": "2.7.0", "transformers": "4.52.0", "numpy": "2.2.0", "pandas": "2.2.3"},
        "cuda": {"torch_cuda_version": "12.4", "nvidia_smi_cuda_version": "12.4"},
        "gpu": {
            "count": 8,
            "driver_version": "550.54.15",
            "cuda_visible_devices": "0,1,2,3,4,5",
            "devices": [
                {"name": "NVIDIA A800-SXM4-40GB", "driver_version": "550.54.15", "memory_total": "40960 MiB"}
                for _ in range(8)
            ],
        },
        "tools": {},
    }

    fingerprint_payload = capture_environment.execution_environment_fingerprint_payload(payload)
    assert fingerprint_payload["execution"]["cuda_visible_devices"] == "0,1,2,3,4,5"
    assert fingerprint_payload["execution"]["visible_gpu_count"] == 6
    assert fingerprint_payload["gpu"]["count"] == 6
    assert len(fingerprint_payload["gpu"]["devices"]) == 6

    changed = {**payload, "gpu": {**payload["gpu"], "cuda_visible_devices": "0,1,2,3"}}
    assert capture_environment.execution_environment_fingerprint_sha256(
        payload
    ) != capture_environment.execution_environment_fingerprint_sha256(changed)


def test_execution_environment_fingerprint_accepts_different_host_gpu_totals_with_same_visible_class() -> None:
    base_device = {"name": "NVIDIA A800-SXM4-40GB", "driver_version": "550.54.15", "memory_total": "40960 MiB"}
    common = {
        "platform": {
            "system": "Linux",
            "release": "6.8.0",
            "version": "#1 SMP PREEMPT_DYNAMIC",
            "machine": "x86_64",
            "processor": "Intel(R) Xeon(R)",
        },
        "python": {"executable": "/venv/bin/python", "version": "3.11.9"},
        "packages": {"torch": "2.7.0", "transformers": "4.52.0", "numpy": "2.2.0", "pandas": "2.2.3"},
        "cuda": {"torch_cuda_version": "12.4", "nvidia_smi_cuda_version": "12.4"},
        "tools": {},
    }
    eight_gpu_host = {
        **common,
        "gpu": {
            "count": 8,
            "driver_version": "550.54.15",
            "cuda_visible_devices": "0,1,2,3,4,5",
            "devices": [base_device.copy() for _ in range(8)],
        },
    }
    six_gpu_host = {
        **common,
        "gpu": {
            "count": 6,
            "driver_version": "550.54.15",
            "cuda_visible_devices": "0,1,2,3,4,5",
            "devices": [base_device.copy() for _ in range(6)],
        },
    }

    assert capture_environment.environment_fingerprint_sha256(
        eight_gpu_host
    ) != capture_environment.environment_fingerprint_sha256(six_gpu_host)
    assert capture_environment.execution_environment_fingerprint_sha256(
        eight_gpu_host
    ) == capture_environment.execution_environment_fingerprint_sha256(six_gpu_host)


def test_execution_class_gpu_devices_rejects_out_of_range_mask() -> None:
    payload = {
        "gpu": {
            "devices": [
                {"name": "GPU0", "driver_version": "550.54.15", "memory_total": "40960 MiB"},
                {"name": "GPU1", "driver_version": "550.54.15", "memory_total": "40960 MiB"},
            ],
            "cuda_visible_devices": "0,7",
        }
    }

    assert capture_environment.execution_class_gpu_devices(payload) == []


def test_environment_fingerprint_normalizes_python_build_metadata() -> None:
    base = {
        "platform": {
            "system": "Linux",
            "release": "6.8.0",
            "version": "#1 SMP PREEMPT_DYNAMIC",
            "machine": "x86_64",
            "processor": "Intel(R) Xeon(R)",
        },
        "python": {"executable": "/venv/bin/python", "version": "3.10.12 (main, Mar  3 2026, 11:56:32) [GCC 11.4.0]"},
        "packages": {"torch": "2.6.0+cu124", "transformers": "4.57.6", "numpy": "2.2.6", "pandas": None},
        "cuda": {"torch_cuda_version": "12.4", "nvidia_smi_cuda_version": "12.4"},
        "gpu": {
            "count": 6,
            "driver_version": "550.163.01",
            "cuda_visible_devices": "0,1,2,3,4,5",
            "devices": [
                {"name": "NVIDIA A800-SXM4-40GB", "driver_version": "550.163.01", "memory_total": "40960 MiB"}
                for _ in range(6)
            ],
        },
        "tools": {},
    }
    changed = {
        **base,
        "python": {"executable": "/venv/bin/python", "version": "3.10.12 (main, May 27 2025, 17:12:29) [GCC 11.4.0]"},
    }

    assert capture_environment.environment_fingerprint_sha256(base) == capture_environment.environment_fingerprint_sha256(
        changed
    )
    assert capture_environment.execution_environment_fingerprint_sha256(
        base
    ) == capture_environment.execution_environment_fingerprint_sha256(changed)


def test_main_autofills_release_identity_fields_when_not_provided(tmp_path: Path, monkeypatch) -> None:
    json_path = tmp_path / "runtime_environment.json"
    md_path = tmp_path / "runtime_environment.md"
    monkeypatch.setattr(
        capture_environment,
        "_collect",
        lambda: {
            "platform": {"system": "Linux", "release": "6.8.0", "version": "#1 SMP", "machine": "x86_64", "processor": "Xeon"},
            "python": {"executable": "/venv/bin/python", "version": "3.10.12"},
            "packages": {"torch": "2.6.0+cu124", "transformers": "4.57.6", "numpy": "2.2.6", "pandas": None},
            "cuda": {"torch_cuda_version": "12.4", "nvidia_smi_cuda_version": "12.4"},
            "gpu": {
                "count": 8,
                "visible_gpu_count": 8,
                "cuda_visible_devices": "0,1,2,3,4,5,6,7",
                "devices": [{"name": "NVIDIA A800-SXM4-40GB", "driver_version": "550.163.01", "memory_total": "40960 MiB"}] * 8,
            },
            "tools": {},
        },
    )
    monkeypatch.setattr(
        capture_environment,
        "_repo_snapshot",
        type("RepoSnapshot", (), {"repo_snapshot_sha256": staticmethod(lambda _root: "snapshot-digest")})(),
    )
    monkeypatch.setattr(
        capture_environment,
        "execution_environment_fingerprint_sha256",
        lambda *_args, **_kwargs: "b" * 64,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture_environment.py",
            "--label",
            "release",
            "--execution-mode",
            "single_host_canonical",
            "--output-json",
            str(json_path),
            "--output-md",
            str(md_path),
        ],
    )

    assert capture_environment.main() == 0
    payload = __import__("json").loads(json_path.read_text(encoding="utf-8"))
    assert payload["execution"]["execution_mode"] == "single_host_canonical"
    assert payload["execution"]["code_snapshot_digest"] == "snapshot-digest"
    assert payload["execution"]["execution_environment_fingerprint"] == "b" * 64


def test_safe_write_text_rejects_symlinked_leaf(tmp_path: Path) -> None:
    target = tmp_path / "runtime_environment.json"
    outside = tmp_path / "outside.json"
    outside.write_text("outside\n", encoding="utf-8")
    try:
        target.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available in this environment")

    with pytest.raises(RuntimeError, match="symlinked path components"):
        capture_environment._safe_write_text(target, "{}\n")
