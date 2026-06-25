from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from posteditbench.toolchains import TOOLCHAIN_REQUIREMENTS, _version_at_least, _version_matches
except Exception:  # pragma: no cover - keep capture_environment robust when repo imports fail unexpectedly
    TOOLCHAIN_REQUIREMENTS = {}

    def _version_matches(actual: str, expected_prefix: str) -> bool:
        return bool(actual and expected_prefix and actual.startswith(expected_prefix))

    def _version_at_least(actual: str, minimum_version: str) -> bool:
        actual_parts = tuple(int(part) for part in re.findall(r"\d+", str(actual)))
        minimum_parts = tuple(int(part) for part in re.findall(r"\d+", str(minimum_version)))
        if not actual_parts or not minimum_parts:
            return False
        width = max(len(actual_parts), len(minimum_parts))
        actual_parts = actual_parts + (0,) * (width - len(actual_parts))
        minimum_parts = minimum_parts + (0,) * (width - len(minimum_parts))
        return actual_parts >= minimum_parts

try:
    import _repo_snapshot
except Exception:  # pragma: no cover - keep capture_environment robust when repo imports fail unexpectedly
    _repo_snapshot = None

_FINGERPRINT_TOOL_KEYS = ("g++", "javac", "java", "node", "go", "nvidia_smi")
_TOOLCHAIN_FINGERPRINT_REQUIREMENTS: dict[str, dict[str, str]] = {}
for _requirements in TOOLCHAIN_REQUIREMENTS.values():
    for _requirement in _requirements:
        _tool_name = str(_requirement.get("display_name") or _requirement.get("tool") or "").strip()
        if not _tool_name or _tool_name in _TOOLCHAIN_FINGERPRINT_REQUIREMENTS:
            continue
        _TOOLCHAIN_FINGERPRINT_REQUIREMENTS[_tool_name] = {
            "pattern": str(_requirement.get("pattern", "") or "").strip(),
            "minimum_version": str(_requirement.get("minimum_version", "") or "").strip(),
            "expected_prefix": str(_requirement.get("expected_prefix", "") or "").strip(),
        }


def _normalized_python_version(value: str | None) -> str:
    text = str(value or "").strip()
    match = re.search(r"(?P<version>\d+\.\d+\.\d+)", text)
    if match is None:
        return text
    return str(match.group("version")).strip()


def _tool_fingerprint_signature(tool_name: str, tool_result: Any) -> Any:
    if not isinstance(tool_result, Mapping):
        return tool_result
    error = str(tool_result.get("error", "") or "").strip()
    stdout = str(tool_result.get("stdout", "") or "").strip()
    stderr = str(tool_result.get("stderr", "") or "").strip()
    signature: dict[str, Any] = {
        "available": not bool(error),
        "error": error or None,
    }
    if error:
        return signature
    requirement = _TOOLCHAIN_FINGERPRINT_REQUIREMENTS.get(tool_name, {})
    output = "\n".join(part for part in (stdout, stderr) if part).strip()
    pattern = str(requirement.get("pattern", "") or "").strip()
    if pattern:
        match = re.search(pattern, output, flags=re.IGNORECASE | re.MULTILINE)
        version = str(match.group("version")).strip() if match is not None else ""
    else:
        version = ""
    if tool_name == "git":
        version = stdout or stderr
    signature["version"] = version or None
    minimum_version = str(requirement.get("minimum_version", "") or "").strip()
    expected_prefix = str(requirement.get("expected_prefix", "") or "").strip()
    if minimum_version:
        signature["minimum_version"] = minimum_version
        signature["minimum_satisfied"] = _version_at_least(version, minimum_version)
    if expected_prefix:
        signature["recommended_prefix"] = expected_prefix
        signature["recommended_match"] = _version_matches(version, expected_prefix)
    return signature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture the exact runtime environment used for a reviewer-visible rerun.")
    parser.add_argument("--label", type=str, default="runtime_environment", help="Human-readable label for the capture.")
    parser.add_argument("--output-json", type=Path, required=True, help="Path for the machine-readable JSON summary.")
    parser.add_argument("--output-md", type=Path, required=True, help="Path for the reviewer-facing Markdown summary.")
    parser.add_argument(
        "--execution-mode",
        type=str,
        default="",
        help="Optional release-facing execution mode to persist alongside the environment capture.",
    )
    parser.add_argument(
        "--code-snapshot-digest",
        type=str,
        default="",
        help="Optional release-facing code snapshot digest to persist alongside the environment capture.",
    )
    parser.add_argument(
        "--execution-environment-fingerprint",
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--public-safe-paths",
        action="store_true",
        help="Suppress execution-host aliases and absolute interpreter paths in reviewer-facing captures.",
    )
    return parser.parse_args()


def _first_symlink_component(path: Path) -> Path | None:
    absolute = Path(os.path.abspath(str(path)))
    anchor = Path(absolute.anchor) if absolute.anchor else Path("/")
    current = anchor
    if current.is_symlink():
        return current
    for part in absolute.parts[len(anchor.parts) :]:
        current = current / part
        if current.is_symlink():
            return current
    return None


def _safe_write_text(path: Path, payload: str) -> None:
    symlink_component = _first_symlink_component(path)
    if symlink_component is not None:
        raise RuntimeError(f"refusing to write environment capture through symlinked path components: {symlink_component}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def _run(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    except FileNotFoundError:  # pragma: no cover - platform dependent
        return {"command": command, "error": "command not found", "returncode": None, "stdout": "", "stderr": ""}
    except OSError as exc:  # pragma: no cover - platform dependent
        if getattr(exc, "errno", None) == 2 or getattr(exc, "winerror", None) == 2:
            error = "command not found"
        else:
            error = f"process launch failed ({exc.__class__.__name__})"
        return {"command": command, "error": error, "returncode": None, "stdout": "", "stderr": ""}
    except subprocess.CalledProcessError as exc:  # pragma: no cover - platform dependent
        return {
            "command": command,
            "error": f"command returned non-zero exit status {exc.returncode}",
            "returncode": exc.returncode,
            "stdout": (exc.stdout or "").strip(),
            "stderr": (exc.stderr or "").strip(),
        }
    except Exception as exc:  # pragma: no cover - platform dependent
        return {
            "command": command,
            "error": f"{exc.__class__.__name__}: {exc}",
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }
    return {
        "command": command,
        "error": None,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _import_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", None) or getattr(module, "version", None) or "unknown")


def _torch_cuda_version() -> str | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return str(getattr(getattr(torch, "version", None), "cuda", None) or "unknown")


def _parse_gpu_devices(csv_stdout: str) -> list[dict[str, str]]:
    devices: list[dict[str, str]] = []
    if not csv_stdout.strip():
        return devices
    reader = csv.reader(io.StringIO(csv_stdout))
    for row in reader:
        if len(row) < 3:
            continue
        name, driver_version, memory_total = [value.strip() for value in row[:3]]
        devices.append(
            {
                "name": name,
                "driver_version": driver_version,
                "memory_total": memory_total,
            }
        )
    return devices


def _parse_cuda_version(nvidia_smi_stdout: str) -> str | None:
    match = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_smi_stdout)
    if not match:
        return None
    return match.group(1)


def _normalized_cuda_visible_devices(value: str | None) -> str:
    if value is None:
        return ""
    tokens = [token.strip() for token in str(value).split(",") if token.strip()]
    return ",".join(tokens)


def _normalized_gpu_devices(gpu_payload: Mapping[str, Any]) -> list[dict[str, str]]:
    devices = gpu_payload.get("devices", []) if isinstance(gpu_payload, Mapping) else []
    normalized: list[dict[str, str]] = []
    for device in devices:
        if not isinstance(device, Mapping):
            continue
        normalized.append(
            {
                "name": str(device.get("name", "")).strip(),
                "driver_version": str(device.get("driver_version", "")).strip(),
                "memory_total": str(device.get("memory_total", "")).strip(),
            }
        )
    return normalized


def execution_class_gpu_devices(
    payload: Mapping[str, Any],
    *,
    cuda_visible_devices: str | None = None,
) -> list[dict[str, str]]:
    gpu_payload = payload.get("gpu", {})
    devices = _normalized_gpu_devices(gpu_payload if isinstance(gpu_payload, Mapping) else {})
    if not devices:
        return []
    normalized_visible = _normalized_cuda_visible_devices(
        cuda_visible_devices
        if cuda_visible_devices is not None
        else (
            gpu_payload.get("cuda_visible_devices", "")
            if isinstance(gpu_payload, Mapping)
            else ""
        )
    )
    if not normalized_visible:
        return devices
    ordinals: list[int] = []
    for token in normalized_visible.split(","):
        if not token.isdigit():
            return []
        ordinal = int(token)
        if ordinal < 0 or ordinal >= len(devices):
            return []
        ordinals.append(ordinal)
    return [devices[ordinal] for ordinal in ordinals]


def _collect() -> dict[str, Any]:
    gpu_summary = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    nvidia_smi_full = _run(["nvidia-smi"])
    if gpu_summary.get("error") and not nvidia_smi_full.get("error"):
        gpu_summary = nvidia_smi_full
    gpu_stdout = str(gpu_summary.get("stdout", "") or "")
    gpu_devices = _parse_gpu_devices(gpu_stdout) if not gpu_summary.get("error") else []
    gpu_count = len(gpu_devices) if gpu_devices else len([line for line in gpu_stdout.splitlines() if line.strip()]) if gpu_stdout else 0
    normalized_visible = _normalized_cuda_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    visible_devices = execution_class_gpu_devices(
        {
            "gpu": {
                "devices": gpu_devices,
                "cuda_visible_devices": normalized_visible,
            }
        }
    )
    driver_versions = sorted({device["driver_version"] for device in gpu_devices if device.get("driver_version")})
    try:
        hostname = str(socket.gethostname() or "").strip()
    except Exception:
        hostname = ""
    try:
        fqdn = str(socket.getfqdn() or "").strip()
    except Exception:
        fqdn = ""
    return {
        "host": {
            "hostname": hostname,
            "fqdn": fqdn,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
        },
        "packages": {
            "torch": _import_version("torch"),
            "transformers": _import_version("transformers"),
            "numpy": _import_version("numpy"),
            "pandas": _import_version("pandas"),
        },
        "cuda": {
            "torch_cuda_version": _torch_cuda_version(),
            "nvidia_smi_cuda_version": _parse_cuda_version(str(nvidia_smi_full.get("stdout", "") or "")),
        },
        "tools": {
            "g++": _run(["g++", "--version"]),
            "javac": _run(["javac", "-version"]),
            "java": _run(["java", "-version"]),
            "node": _run(["node", "--version"]),
            "go": _run(["go", "version"]),
            "nvidia_smi": nvidia_smi_full if not nvidia_smi_full.get("error") else gpu_summary,
            "git": _run(["git", "rev-parse", "HEAD"]),
        },
        "gpu": {
            "count": gpu_count,
            "visible_gpu_count": len(visible_devices),
            "devices": gpu_devices,
            "visible_devices": visible_devices,
            "driver_version": driver_versions[0] if len(driver_versions) == 1 else driver_versions,
            "summary": gpu_stdout,
            "cuda_visible_devices": normalized_visible,
        },
    }


def _public_safe_python_executable(executable: str) -> str:
    value = str(executable or "").strip()
    if not value:
        return ""
    suffix = "python.exe" if value.lower().endswith("python.exe") else "python"
    return f"<release-python>/{suffix}"


def _apply_public_safe_paths(payload: dict[str, Any]) -> None:
    host_payload = payload.get("host", {})
    if isinstance(host_payload, dict):
        host_payload["hostname"] = "execution-host"
        host_payload["fqdn"] = "execution-host"
    python_payload = payload.get("python", {})
    if isinstance(python_payload, dict):
        python_payload["executable"] = _public_safe_python_executable(
            str(python_payload.get("executable", ""))
        )


def environment_fingerprint_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    python_payload = payload.get("python", {})
    packages_payload = payload.get("packages", {})
    cuda_payload = payload.get("cuda", {})
    gpu_payload = payload.get("gpu", {})
    tools_payload = payload.get("tools", {})
    fingerprint_tools: dict[str, Any] = {}
    for tool_name in _FINGERPRINT_TOOL_KEYS:
        tool_result = tools_payload.get(tool_name, {})
        fingerprint_tools[tool_name] = _tool_fingerprint_signature(tool_name, tool_result)
    return {
        "platform": {
            "system": str(payload.get("platform", {}).get("system", "") if isinstance(payload.get("platform", {}), Mapping) else ""),
            "release": str(payload.get("platform", {}).get("release", "") if isinstance(payload.get("platform", {}), Mapping) else ""),
            "version": str(payload.get("platform", {}).get("version", "") if isinstance(payload.get("platform", {}), Mapping) else ""),
            "machine": str(payload.get("platform", {}).get("machine", "") if isinstance(payload.get("platform", {}), Mapping) else ""),
            "processor": str(payload.get("platform", {}).get("processor", "") if isinstance(payload.get("platform", {}), Mapping) else ""),
        },
        "python": {
            "executable": str(python_payload.get("executable", "") if isinstance(python_payload, Mapping) else ""),
            "version": _normalized_python_version(
                python_payload.get("version", "") if isinstance(python_payload, Mapping) else ""
            ),
        },
        "packages": {
            "torch": packages_payload.get("torch") if isinstance(packages_payload, Mapping) else None,
            "transformers": packages_payload.get("transformers") if isinstance(packages_payload, Mapping) else None,
            "numpy": packages_payload.get("numpy") if isinstance(packages_payload, Mapping) else None,
            "pandas": packages_payload.get("pandas") if isinstance(packages_payload, Mapping) else None,
        },
        "cuda": {
            "torch_cuda_version": cuda_payload.get("torch_cuda_version") if isinstance(cuda_payload, Mapping) else None,
            "nvidia_smi_cuda_version": cuda_payload.get("nvidia_smi_cuda_version") if isinstance(cuda_payload, Mapping) else None,
        },
        "gpu": {
            "count": gpu_payload.get("count") if isinstance(gpu_payload, Mapping) else None,
            "driver_version": gpu_payload.get("driver_version") if isinstance(gpu_payload, Mapping) else None,
            "devices": _normalized_gpu_devices(gpu_payload if isinstance(gpu_payload, Mapping) else {}),
        },
        "tools": fingerprint_tools,
    }


def execution_environment_fingerprint_payload(
    payload: Mapping[str, Any],
    *,
    cuda_visible_devices: str | None = None,
) -> dict[str, Any]:
    fingerprint_payload = environment_fingerprint_payload(payload)
    visible_devices = execution_class_gpu_devices(payload, cuda_visible_devices=cuda_visible_devices)
    normalized_visible = _normalized_cuda_visible_devices(
        cuda_visible_devices
        if cuda_visible_devices is not None
        else (
            payload.get("gpu", {}).get("cuda_visible_devices", "")
            if isinstance(payload.get("gpu", {}), Mapping)
            else ""
        )
    )
    driver_versions = sorted({device["driver_version"] for device in visible_devices if device.get("driver_version")})
    fingerprint_payload["gpu"] = {
        "count": len(visible_devices),
        "driver_version": driver_versions[0] if len(driver_versions) == 1 else driver_versions,
        "devices": visible_devices,
    }
    fingerprint_payload["execution"] = {
        "cuda_visible_devices": normalized_visible,
        "visible_gpu_count": len(visible_devices),
    }
    return fingerprint_payload


def environment_fingerprint_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        environment_fingerprint_payload(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def execution_environment_fingerprint_sha256(
    payload: Mapping[str, Any],
    *,
    cuda_visible_devices: str | None = None,
) -> str:
    canonical = json.dumps(
        execution_environment_fingerprint_payload(payload, cuda_visible_devices=cuda_visible_devices),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _render_markdown(payload: dict[str, Any]) -> str:
    host_block = payload.get("host", {})
    platform_block = payload.get("platform", {})
    python_block = payload.get("python", {})
    packages = payload.get("packages", {})
    cuda = payload.get("cuda", {})
    tools = payload.get("tools", {})
    gpu = payload.get("gpu", {})
    execution = payload.get("execution", {})
    lines = [
        f"# Environment Capture",
        "",
        f"- Label: `{payload.get('label', '')}`",
        f"- Hostname: `{host_block.get('hostname', '')}`",
        f"- FQDN: `{host_block.get('fqdn', '')}`",
        f"- System: `{platform_block.get('system', '')}`",
        f"- Release: `{platform_block.get('release', '')}`",
        f"- Version: `{platform_block.get('version', '')}`",
        f"- Machine: `{platform_block.get('machine', '')}`",
        f"- Python executable: `{python_block.get('executable', '')}`",
        f"- Python version: `{python_block.get('version', '')}`",
        f"- Execution mode: `{execution.get('execution_mode', '')}`",
        f"- GPU count (physical): `{gpu.get('count', 0)}`",
        f"- GPU count (visible execution class): `{gpu.get('visible_gpu_count', gpu.get('count', 0))}`",
        f"- CUDA_VISIBLE_DEVICES: `{gpu.get('cuda_visible_devices', '')}`",
        f"- Code snapshot digest: `{execution.get('code_snapshot_digest', '')}`",
        f"- Execution environment fingerprint: `{execution.get('execution_environment_fingerprint', '')}`",
        f"- GPU driver version: `{gpu.get('driver_version')}`",
        f"- CUDA version (torch build): `{cuda.get('torch_cuda_version')}`",
        f"- CUDA version (nvidia-smi): `{cuda.get('nvidia_smi_cuda_version')}`",
        "",
        "## Package Versions",
        f"- `torch`: `{packages.get('torch')}`",
        f"- `transformers`: `{packages.get('transformers')}`",
        f"- `numpy`: `{packages.get('numpy')}`",
        f"- `pandas`: `{packages.get('pandas')}`",
        "",
        "## GPU Devices",
    ]
    devices = gpu.get("devices", []) or []
    visible_devices = gpu.get("visible_devices", []) or []
    git_result = tools.get("git", {}) if isinstance(tools, Mapping) else {}
    if (
        isinstance(git_result, Mapping)
        and git_result.get("error")
        and str(execution.get("code_snapshot_digest", "")).strip()
    ):
        lines.append(
            "_Git metadata is unavailable in this execution-host work copy; use the recorded code snapshot digest as the release code-identity anchor._"
        )
    if visible_devices and len(visible_devices) != len(devices):
        lines.append(
            f"_Listing the execution-class-visible devices selected by CUDA_VISIBLE_DEVICES. Physical inventory remains `{gpu.get('count', 0)}` devices._"
        )
    devices_to_render = visible_devices or devices
    if devices_to_render:
        for device in devices_to_render:
            lines.append(
                f"- `{device.get('name', 'unknown')}` | driver `{device.get('driver_version', 'unknown')}` | memory `{device.get('memory_total', 'unknown')}`"
            )
    else:
        lines.append("- `nvidia-smi` device query unavailable")
    lines.extend(
        [
            "",
        "## Toolchain Checks",
        ]
    )
    for name, result in tools.items():
        if isinstance(result, dict):
            status = "ok" if not result.get("error") else "error"
            lines.append(f"- `{name}`: `{status}`")
            if result.get("error"):
                lines.append(f"  - error: `{result['error']}`")
            if result.get("stdout"):
                lines.append(f"  - stdout: `{result['stdout']}`")
            if result.get("stderr"):
                lines.append(f"  - stderr: `{result['stderr']}`")
        else:
            lines.append(f"- `{name}`: `{result}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    payload = _collect()
    if bool(args.public_safe_paths):
        _apply_public_safe_paths(payload)
    execution_mode = str(args.execution_mode).strip()
    cuda_visible_devices = str(payload.get("gpu", {}).get("cuda_visible_devices", "")).strip()
    code_snapshot_digest = str(args.code_snapshot_digest).strip()
    if not code_snapshot_digest and _repo_snapshot is not None:
        try:
            code_snapshot_digest = str(_repo_snapshot.repo_snapshot_sha256(ROOT)).strip()
        except Exception:
            code_snapshot_digest = ""
    try:
        execution_environment_fingerprint = str(
            execution_environment_fingerprint_sha256(
                payload,
                cuda_visible_devices=cuda_visible_devices,
            )
        ).strip()
    except Exception:
        execution_environment_fingerprint = ""
    payload["label"] = str(args.label)
    payload["execution"] = {
        "execution_mode": execution_mode,
        "cuda_visible_devices": cuda_visible_devices,
        "visible_gpu_count": int(payload.get("gpu", {}).get("visible_gpu_count", 0) or 0),
        "code_snapshot_digest": code_snapshot_digest,
        "execution_environment_fingerprint": execution_environment_fingerprint,
    }
    _safe_write_text(args.output_json, json.dumps(payload, indent=2) + "\n")
    _safe_write_text(args.output_md, _render_markdown(payload))
    print(json.dumps({"json": str(args.output_json), "markdown": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
