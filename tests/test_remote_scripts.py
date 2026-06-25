from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH_EXE = shutil.which("bash")


def _wsl_path(path: Path) -> str:
    if BASH_EXE is None:  # pragma: no cover
        raise RuntimeError("bash is required for shell-wrapper integration tests")
    resolved = path.resolve(strict=False)
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix()[2:] if resolved.drive else resolved.as_posix()
    if drive:
        return f"/mnt/{drive}{tail}"
    return resolved.as_posix()


def _write_shim_bundle(tmp_path: Path) -> tuple[Path, Path]:
    py_shim = tmp_path / "matrix_shard_python_shim.py"
    py_shim.write_text(
        (
            "#!/usr/bin/env python3\n"
            "from __future__ import annotations\n"
            "import json\n"
            "import os\n"
            "import pathlib\n"
            "import sys\n"
            "log_path = pathlib.Path(os.environ['MATRIX_SHARD_SHIM_LOG'])\n"
            "log_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "with log_path.open('a', encoding='utf-8') as handle:\n"
            "    handle.write(json.dumps({'argv': sys.argv[1:]}, ensure_ascii=False) + '\\n')\n"
            "mode = os.environ.get('MATRIX_SHARD_SHIM_MODE', '').strip()\n"
            "if len(sys.argv) > 1 and sys.argv[1] == '-':\n"
            "    _ = sys.stdin.read()\n"
            "    print('{}')\n"
            "    raise SystemExit(0)\n"
            "script_name = pathlib.Path(sys.argv[1]).name if len(sys.argv) > 1 else ''\n"
            "command = sys.argv[2] if len(sys.argv) > 2 else ''\n"
            "if script_name == '_matrix_shard_launch.py' and command == 'validate-existing-receipt':\n"
            "    if mode == 'stale_receipt':\n"
            "        print('code_snapshot_digest mismatch at launch time', file=sys.stderr)\n"
            "        raise SystemExit(1)\n"
            "    print(json.dumps({'status': 'launch_ready'}))\n"
            "    raise SystemExit(0)\n"
            "if script_name == '_matrix_shard_launch.py' and command == 'prepare-clean-launch-tree':\n"
            "    if mode == 'dirty_output':\n"
            "        print('Shard output tree is not clean before launch', file=sys.stderr)\n"
            "        raise SystemExit(1)\n"
            "    output_dir = sys.argv[sys.argv.index('--output-dir') + 1]\n"
            "    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)\n"
            "    print(json.dumps({'status': 'clean', 'output_dir': output_dir, 'deleted': []}))\n"
            "    raise SystemExit(0)\n"
            "print(json.dumps({'argv': sys.argv[1:]}, ensure_ascii=False))\n"
            "raise SystemExit(0)\n"
        ),
        encoding="utf-8",
        newline="\n",
    )
    shell_shim = tmp_path / "matrix_shard_python_shim.sh"
    shell_shim.write_text(
        "#!/usr/bin/env bash\nexec /usr/bin/python3 " + shlex.quote(_wsl_path(py_shim)) + ' "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    chmod_command = f"chmod +x {shlex.quote(_wsl_path(py_shim))} {shlex.quote(_wsl_path(shell_shim))}"
    subprocess.run([BASH_EXE, "-lc", chmod_command], check=True)
    return py_shim, shell_shim


def _prepare_shell_wrapper_fixture(tmp_path: Path) -> dict[str, Path]:
    manifest_path = tmp_path / "suite_all_models_methods_shard_01_of_02.json"
    canonical_manifest_path = tmp_path / "suite_all_models_methods.json"
    fixture_tag = f"_pytest_matrix_shard_{tmp_path.name}"
    certifications_root = ROOT / "results" / "certifications" / fixture_tag
    receipt_path = (
        certifications_root
        / "suite_all_models_methods_shard_01_of_02"
        / "matrix_shard_readiness.json"
    )
    output_root = ROOT / "results" / "matrix" / fixture_tag
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "profile": "suite_all_models_methods_shard_01_of_02",
        "canonical_profile": "suite_all_models_methods",
        "canonical_manifest": str(canonical_manifest_path),
        "shard_index": 1,
        "shard_count": 2,
        "runs": [{"run_id": "suite_qwen25_7b_crafted_original_ewd_runtime"}],
    }
    manifest_path.write_text(json.dumps(manifest_payload) + "\n", encoding="utf-8")
    canonical_manifest_path.write_text(json.dumps({"profile": "suite_all_models_methods", "runs": []}) + "\n", encoding="utf-8")
    receipt_path.write_text(json.dumps({"status": "passed"}) + "\n", encoding="utf-8")
    return {
        "manifest": manifest_path,
        "canonical_manifest": canonical_manifest_path,
        "receipt": receipt_path,
        "certifications_root": certifications_root,
        "output_root": output_root,
        "cleanup_roots": [output_root, certifications_root],
    }


def _write_formal_suite_python_shim(tmp_path: Path) -> tuple[Path, Path]:
    py_shim = tmp_path / "formal_suite_python_shim.py"
    py_shim.write_text(
        (
            "#!/usr/bin/env python3\n"
            "from __future__ import annotations\n"
            "import json\n"
            "import os\n"
            "import pathlib\n"
            "import subprocess\n"
            "import sys\n"
            "\n"
            "def _log(payload):\n"
            "    log_path = pathlib.Path(os.environ['FORMAL_SUITE_SHIM_LOG'])\n"
            "    log_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    with log_path.open('a', encoding='utf-8') as handle:\n"
            "        handle.write(json.dumps(payload, ensure_ascii=False) + '\\n')\n"
            "\n"
            "actual_python = os.environ.get('FORMAL_SUITE_ACTUAL_PYTHON', '/usr/bin/python3')\n"
            "argv = sys.argv[1:]\n"
            "if argv and argv[0] == '-V':\n"
            "    print('Python 3.10.12')\n"
            "    raise SystemExit(0)\n"
            "if argv and argv[0] == '-':\n"
            "    script_text = sys.stdin.read()\n"
            "    _log({'stdin_script': True, 'argv': argv[1:], 'snippet': script_text[:120]})\n"
            "    if 'validate_checkout' in script_text:\n"
            "        raise SystemExit(0)\n"
            "    completed = subprocess.run(\n"
            "        [actual_python, '-', *argv[1:]],\n"
            "        input=script_text,\n"
            "        text=True,\n"
            "        capture_output=True,\n"
            "        check=False,\n"
            "        env=os.environ.copy(),\n"
            "    )\n"
            "    sys.stdout.write(completed.stdout)\n"
            "    sys.stderr.write(completed.stderr)\n"
            "    raise SystemExit(completed.returncode)\n"
            "script_path = pathlib.Path(argv[0]) if argv else pathlib.Path('')\n"
            "script_name = script_path.name\n"
            "script_args = argv[1:]\n"
            "mode = os.environ.get('FORMAL_SUITE_SHIM_MODE', '').strip()\n"
            "payload = {'script': script_name, 'argv': script_args}\n"
            "if mode:\n"
            "    payload['mode'] = mode\n"
            "_log(payload)\n"
            "if script_name == 'capture_environment.py':\n"
            "    completed = subprocess.run([actual_python, str(script_path), *script_args], check=False, capture_output=True, text=True, env=os.environ.copy())\n"
            "    sys.stdout.write(completed.stdout)\n"
            "    sys.stderr.write(completed.stderr)\n"
            "    raise SystemExit(completed.returncode)\n"
            "if script_name in {'check_zero_compatibility_name.py', 'build_suite_manifests.py', 'audit_benchmarks.py', 'audit_full_matrix.py', 'certify_suite_precheck.py', 'validate_single_host_launch_receipt.py', 'clean_suite_outputs.py'}:\n"
            "    if script_name == 'audit_full_matrix.py' and mode == 'preflight_failure':\n"
            "        print('preflight audit failed', file=sys.stderr)\n"
            "        raise SystemExit(1)\n"
            "    raise SystemExit(0)\n"
            "if script_name == 'run_full_matrix.py':\n"
            "    if '--dry-run' in script_args:\n"
            "        print('{}')\n"
            "        raise SystemExit(0)\n"
            "    output_root = pathlib.Path(script_args[script_args.index('--output-root') + 1])\n"
            "    profile = script_args[script_args.index('--profile') + 1]\n"
            "    matrix_index = output_root / profile / 'matrix_index.json'\n"
            "    matrix_index.parent.mkdir(parents=True, exist_ok=True)\n"
            "    matrix_index.write_text(json.dumps({\n"
            "        'run_count': 140,\n"
            "        'success_count': 140,\n"
            "        'failed_count': 0,\n"
            "        'execution_mode': 'single_host_canonical',\n"
            "        'gpu_pool_mode': 'shared',\n"
            "        'code_snapshot_digest': 'a' * 64,\n"
            "        'execution_environment_fingerprint': 'b' * 64,\n"
            "        'assembly_source_execution_modes': ['single_host_canonical'],\n"
            "        'assembly_source_indexes': [{'execution_mode': 'single_host_canonical'}],\n"
            "        'runs': [],\n"
            "    }, indent=2) + '\\n', encoding='utf-8')\n"
            "    raise SystemExit(0)\n"
            "completed = subprocess.run([actual_python, *argv], check=False, capture_output=True, text=True, env=os.environ.copy())\n"
            "sys.stdout.write(completed.stdout)\n"
            "sys.stderr.write(completed.stderr)\n"
            "raise SystemExit(completed.returncode)\n"
        ),
        encoding="utf-8",
        newline="\n",
    )
    shell_shim = tmp_path / "formal_suite_python_shim.sh"
    shell_shim.write_text(
        "#!/usr/bin/env bash\nexec /usr/bin/python3 " + shlex.quote(_wsl_path(py_shim)) + ' "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    subprocess.run([BASH_EXE, "-lc", f"chmod +x {shlex.quote(_wsl_path(py_shim))} {shlex.quote(_wsl_path(shell_shim))}"], check=True)
    return py_shim, shell_shim


def _write_fake_toolchain_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    tools = {
        "g++": "#!/usr/bin/env bash\necho 'g++ (Ubuntu) 13.4.0'\n",
        "javac": "#!/usr/bin/env bash\necho 'javac 21.0.10'\n",
        "java": "#!/usr/bin/env bash\nif [[ \"$1\" == '-version' ]]; then echo 'openjdk version \"21.0.10\"' 1>&2; exit 0; fi\necho 'java smoke ok'\n",
        "node": "#!/usr/bin/env bash\necho 'node smoke ok'\n",
        "go": "#!/usr/bin/env bash\nif [[ \"$1\" == 'version' ]]; then echo 'go version go1.22.12 linux/amd64'; exit 0; fi\necho 'go smoke ok'\n",
        "nvidia-smi": (
            "#!/usr/bin/env bash\n"
            "if [[ \"$1\" == '--query-gpu=index' ]]; then\n"
            "  printf '0\\n1\\n2\\n3\\n4\\n5\\n6\\n7\\n'\n"
            "  exit 0\n"
            "fi\n"
            "echo 'NVIDIA-SMI 550.54.15'\n"
        ),
    }
    for name, content in tools.items():
        path = bin_dir / name
        path.write_text(content, encoding="utf-8", newline="\n")
    chmod_targets = " ".join(shlex.quote(_wsl_path(path)) for path in bin_dir.iterdir())
    subprocess.run([BASH_EXE, "-lc", f"chmod +x {chmod_targets}"], check=True)
    return bin_dir


def _backup_repo_file(path: Path) -> tuple[bool, str]:
    if path.exists() or path.is_symlink():
        return True, path.read_text(encoding="utf-8")
    return False, ""


def _restore_repo_file(path: Path, state: tuple[bool, str]) -> None:
    existed, payload = state
    if path.exists() or path.is_symlink():
        path.unlink()
    if existed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        return


def _cleanup_stale_formal_suite_test_processes() -> None:
    if BASH_EXE is None:  # pragma: no cover
        return
    cleanup_script = "\n".join(
        [
            f"python3 - {shlex.quote(_wsl_path(ROOT))} <<'PY'",
            "from __future__ import annotations",
            "",
            "import os",
            "import signal",
            "import subprocess",
            "import sys",
            "",
            "current_pid = os.getpid()",
            "parent_pid = os.getppid()",
            "root = sys.argv[1]",
            "completed = subprocess.run(",
            "    ['ps', '-eo', 'pid=,ppid=,args='],",
            "    check=True,",
            "    capture_output=True,",
            "    text=True,",
            "    encoding='utf-8',",
            "    errors='replace',",
            ")",
            "children = {}",
            "targets = []",
            "for raw_line in completed.stdout.splitlines():",
            "    line = raw_line.rstrip()",
            "    if not line:",
            "        continue",
            "    parts = line.strip().split(None, 2)",
            "    if len(parts) != 3:",
            "        continue",
            "    try:",
            "        pid = int(parts[0])",
            "        ppid = int(parts[1])",
            "    except ValueError:",
            "        continue",
            "    args = parts[2]",
            "    children.setdefault(ppid, []).append(pid)",
            "    if pid in {current_pid, parent_pid}:",
            "        continue",
            "    if root in args and 'pytest-of-' in args and 'run_formal_suite.sh' in args:",
            "        targets.append(pid)",
            "seen = set()",
            "kill_order = []",
            "stack = list(targets)",
            "while stack:",
            "    pid = stack.pop()",
            "    if pid in seen:",
            "        continue",
            "    seen.add(pid)",
            "    kill_order.append(pid)",
            "    stack.extend(children.get(pid, []))",
            "for pid in sorted(kill_order, reverse=True):",
            "    try:",
            "        os.kill(pid, signal.SIGKILL)",
            "    except OSError:",
            "        pass",
            "PY",
        ]
    )
    subprocess.run([BASH_EXE, "-lc", cleanup_script], check=True)


def _run_bash_command(command: str, *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    launcher = command
    if cwd is not None:
        launcher = " && ".join([f"cd {shlex.quote(_wsl_path(cwd))}", command])
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        stdout_path = tmpdir_path / "stdout.log"
        stderr_path = tmpdir_path / "stderr.log"
        completed = subprocess.run(
            [
                BASH_EXE,
                "-lc",
                f"{launcher} > {shlex.quote(_wsl_path(stdout_path))} 2> {shlex.quote(_wsl_path(stderr_path))}",
            ],
            check=False,
        )
        return subprocess.CompletedProcess(
            args=list(completed.args),
            returncode=completed.returncode,
            stdout=stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else "",
            stderr=stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else "",
        )


def _run_formal_suite_wrapper(
    tmp_path: Path,
    *,
    extra_args: list[str] | None = None,
    mode: str = "",
    detached: bool = True,
    manage_repo_artifacts: bool = True,
) -> subprocess.CompletedProcess[str]:
    _cleanup_stale_formal_suite_test_processes()
    _, shell_shim = _write_formal_suite_python_shim(tmp_path)
    fake_bin = _write_fake_toolchain_bin(tmp_path)
    log_path = tmp_path / "formal_suite_shim.log"
    output_root = ROOT / "results" / "matrix"
    env_json = ROOT / "results" / "environment" / "runtime_environment.json"
    env_md = ROOT / "results" / "environment" / "runtime_environment.md"
    preflight_receipt = ROOT / "results" / "certifications" / "remote_preflight_receipt.json"
    matrix_index = ROOT / "results" / "matrix" / "suite_all_models_methods" / "matrix_index.json"
    backups = (
        {
            env_json: _backup_repo_file(env_json),
            env_md: _backup_repo_file(env_md),
            preflight_receipt: _backup_repo_file(preflight_receipt),
            matrix_index: _backup_repo_file(matrix_index),
        }
        if manage_repo_artifacts
        else {}
    )
    command_parts = [
        shlex.quote(_wsl_path(ROOT / "scripts" / "remote" / "run_formal_single_host_full.sh")),
        "--python",
        shlex.quote(_wsl_path(shell_shim)),
        "--output-root",
        shlex.quote(_wsl_path(output_root)),
    ]
    if extra_args:
        command_parts.extend(extra_args)
    launcher = tmp_path / "run_formal_suite.sh"
    launcher.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(_wsl_path(ROOT))}",
                f"export PATH={shlex.quote(_wsl_path(fake_bin))}:$PATH",
                "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
                f"export FORMAL_SUITE_SHIM_LOG={shlex.quote(_wsl_path(log_path))}",
                f"export FORMAL_SUITE_SHIM_MODE={shlex.quote(mode)}",
                "export FORMAL_SUITE_ACTUAL_PYTHON=\"$(python3.10 -c 'import sys; print(sys.executable)')\"",
                *(["export POSTEDITBENCH_FORMAL_FULL_DETACHED=1"] if detached else []),
                " ".join(command_parts),
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    subprocess.run([BASH_EXE, "-lc", f"chmod +x {shlex.quote(_wsl_path(launcher))}"], check=True)
    stdout_path = tmp_path / "formal_wrapper.stdout.log"
    stderr_path = tmp_path / "formal_wrapper.stderr.log"
    try:
        completed = subprocess.run(
            [BASH_EXE, "-lc", f"bash {shlex.quote(_wsl_path(launcher))} > {shlex.quote(_wsl_path(stdout_path))} 2> {shlex.quote(_wsl_path(stderr_path))}"],
            check=False,
        )
        return subprocess.CompletedProcess(
            args=list(completed.args),
            returncode=completed.returncode,
            stdout=stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else "",
            stderr=stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else "",
        )
    finally:
        if manage_repo_artifacts:
            for path, state in backups.items():
                _restore_repo_file(path, state)
        _cleanup_stale_formal_suite_test_processes()


def _run_matrix_shard_wrapper(tmp_path: Path, *, mode: str) -> subprocess.CompletedProcess[str]:
    fixture = _prepare_shell_wrapper_fixture(tmp_path)
    _, shell_shim = _write_shim_bundle(tmp_path)
    log_path = tmp_path / "shim.log"
    stdout_path = tmp_path / "wrapper.stdout.log"
    stderr_path = tmp_path / "wrapper.stderr.log"
    command = " ".join(
        [
            shlex.quote(_wsl_path(ROOT / "scripts" / "remote" / "run_matrix_shard.sh")),
            "--manifest",
            shlex.quote(_wsl_path(fixture["manifest"])),
            "--canonical-manifest",
            shlex.quote(_wsl_path(fixture["canonical_manifest"])),
            "--profile",
            "suite_all_models_methods_shard_01_of_02",
            "--canonical-profile",
            "suite_all_models_methods",
            "--shard-index",
            "1",
            "--shard-count",
            "2",
            "--output-root",
            shlex.quote(_wsl_path(fixture["output_root"])),
            "--certifications-root",
            shlex.quote(_wsl_path(fixture["certifications_root"])),
            "--python",
            shlex.quote(_wsl_path(shell_shim)),
            "--skip-readiness",
            "--no-clean",
        ]
    )
    launcher_path = tmp_path / "invoke_wrapper.sh"
    launcher_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(_wsl_path(ROOT))}",
                f"export MATRIX_SHARD_SHIM_MODE={shlex.quote(mode)}",
                f"export MATRIX_SHARD_SHIM_LOG={shlex.quote(_wsl_path(log_path))}",
                f"{command} > {shlex.quote(_wsl_path(stdout_path))} 2> {shlex.quote(_wsl_path(stderr_path))}",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    subprocess.run([BASH_EXE, "-lc", f"chmod +x {shlex.quote(_wsl_path(launcher_path))}"], check=True)
    try:
        completed = subprocess.run([BASH_EXE, "-lc", f"bash {shlex.quote(_wsl_path(launcher_path))}"], check=False)
        time.sleep(0.2)
        return subprocess.CompletedProcess(
            args=list(completed.args),
            returncode=completed.returncode,
            stdout=stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else "",
            stderr=stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else "",
        )
    finally:
        for cleanup_root in fixture["cleanup_roots"]:
            shutil.rmtree(cleanup_root, ignore_errors=True)


def test_remote_preflight_dry_run_declares_prepare_step():
    script = (ROOT / "scripts" / "remote" / "run_preflight.sh").read_text(encoding="utf-8")
    assert '"zero_compatibility_name"' in script
    assert '"build_suite_manifests"' in script
    assert '"audit_benchmarks"' in script
    assert '"audit_suite_matrix"' in script
    assert 'FULL_MANIFEST="configs/matrices/suite_all_models_methods.json"' in script
    assert 'STAGE_A_MANIFEST="configs/matrices/suite_canary_heavy.json"' in script
    assert 'STAGE_B_MANIFEST="configs/matrices/model_invocation_smoke.json"' in script
    assert 'DEFAULT_REMOTE_VENV="${POSTEDITBENCH_REMOTE_VENV:-}"' in script
    assert '--preflight-receipt PATH' in script
    assert '--venv PATH' in script
    assert '"require_hf_token": $REQUIRE_HF_TOKEN' in script
    assert '"skip_hf_access": $SKIP_HF_ACCESS' in script
    assert '"preflight_receipt": "$PREFLIGHT_RECEIPT_PATH"' in script


def test_remote_preflight_audits_suite_before_matrix_dry_runs():
    script = (ROOT / "scripts" / "remote" / "run_preflight.sh").read_text(encoding="utf-8")
    assert '"$PYTHON_BIN" "$ROOT/scripts/check_zero_compatibility_name.py" --root "$ROOT"' in script
    assert "build_suite_manifests.py" in script
    assert "validate_stage_roster_alignment" in script
    assert "audit_benchmarks.py" in script
    assert "audit_full_matrix.py" in script
    assert script.index("build_suite_manifests.py") < script.index("audit_full_matrix.py")
    assert script.index("validate_stage_roster_alignment") < script.index("audit_benchmarks.py")
    assert '--manifest "$FULL_MANIFEST_PATH"' in script
    assert '--matrix-profile "$FULL_PROFILE"' in script
    assert '--strict-hf-cache' in script
    assert '--model-load-smoke' in script
    assert '--runtime-smoke' in script
    assert 'PYTHON_BIN="$VENV_DIR/bin/python"' in script
    assert 'Create the venv first with:' in script
    assert "runtime_checkouts_ready()" in script
    assert "Pinned runtime upstream checkouts already validate cleanly; skipping network refresh." in script
    assert "Run bash $ROOT/scripts/fetch_runtime_upstreams.sh all explicitly before preflight." in script
    assert 'PYTHON_BIN="$PYTHON_BIN" bash "$ROOT/scripts/fetch_runtime_upstreams.sh" all' not in script
    assert 'Using caller-provided CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES for controlled preflight.' in script
    assert 'PRECHECK_IDLE_GPU_MAX_MEMORY_MB="${PRECHECK_IDLE_GPU_MAX_MEMORY_MB:-512}"' in script
    assert 'export PRECHECK_IDLE_GPU_MAX_MEMORY_MB' in script
    assert '($3 + 0) <= max_mem' in script
    assert 'if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then' in script
    assert 'visible_gpu_slot_count()' in script
    assert 'PREFLIGHT_GPU_SLOTS="$(visible_gpu_slot_count)"' in script
    assert 'Using PREFLIGHT_GPU_SLOTS=$PREFLIGHT_GPU_SLOTS for controlled preflight.' in script
    assert '--gpu-slots "$PREFLIGHT_GPU_SLOTS" --dry-run' in script
    assert '--skip-provider-credentials' in script
    assert 'node -e "const add = (a, b) => a + b;' in script
    assert 'command -v java >/dev/null 2>&1' in script
    assert 'java -version' in script
    assert 'javac "$JAVA_SMOKE_DIR/Smoke.java"' in script
    assert 'java -cp "$JAVA_SMOKE_DIR" Smoke' in script
    assert 'go run "$GO_SMOKE_DIR/main.go"' in script
    assert "Run bash $ROOT/scripts/fetch_runtime_upstreams.sh all explicitly before preflight." in script
    assert 'PYTHON_BIN="$PYTHON_BIN" bash "$ROOT/scripts/fetch_runtime_upstreams.sh" all' not in script


def test_remote_preflight_only_rebuilds_canonical_full_manifest():
    script = (ROOT / "scripts" / "remote" / "run_preflight.sh").read_text(encoding="utf-8")
    assert 'validate_results_control_path()' in script
    assert 'ensure_preflight_control_surfaces_safe()' in script
    assert 'validate_matrix_output_surface()' in script
    assert 'normalize_abs_path()' in script
    assert 'CANONICAL_FULL_MANIFEST_PATH="$(normalize_abs_path "configs/matrices/suite_all_models_methods.json")"' in script
    assert 'CANONICAL_STAGE_A_MANIFEST_PATH="$(normalize_abs_path "configs/matrices/suite_canary_heavy.json")"' in script
    assert 'CANONICAL_STAGE_B_MANIFEST_PATH="$(normalize_abs_path "configs/matrices/model_invocation_smoke.json")"' in script
    assert 'FULL_MANIFEST_IS_CANONICAL=0' in script
    assert 'if [[ "$FULL_MANIFEST_PATH" == "$CANONICAL_FULL_MANIFEST_PATH" && "$FULL_PROFILE" == "suite_all_models_methods" ]]; then' in script
    assert 'if [[ $FULL_MANIFEST_IS_CANONICAL -eq 1 ]]; then' in script
    assert 'Canonical --full-manifest/--full-profile requires the canonical stage A/B manifests and profiles so preflight certifies one canonical manifest set.' in script
    assert '"$PYTHON_BIN" "$ROOT/scripts/build_suite_manifests.py"\n' in script
    assert '"--output-manifest" "$FULL_MANIFEST_PATH"' not in script
    assert 'if [[ $FULL_MANIFEST_IS_CANONICAL -ne 1 ]]; then' not in script
    assert 'rm -f "$PREFLIGHT_RECEIPT_PATH"' in script
    assert 'PREFLIGHT_LOCK_DIR="$ROOT/results/launchers/.remote_preflight.lock"' in script
    assert 'acquire_preflight_lock()' in script
    assert 'write_preflight_lock_metadata()' in script
    assert 'find_active_repo_blockers()' in script
    assert 'run_full_matrix.py|certify_suite_precheck.py|clean_suite_outputs.py' in script
    assert 'Another remote preflight is already active for this repository root' in script
    assert 'lock_probe_attempts=0' in script
    assert 'if [[ -z "$existing_pid" && $lock_probe_attempts -lt 3 ]]; then' in script
    assert 'sleep 1' in script
    assert 'trap cleanup_preflight EXIT' in script
    assert 'remote preflight environment json' in script
    assert 'remote preflight output root' in script
    assert 'Canonical --formal-full-only preflight requires repo-local output root results/matrix.' in script
    assert 'remote preflight receipt' in script
    assert '"receipt_type": "remote_preflight"' in script
    assert 'certify_suite_precheck._environment_receipt_from_payload(' in script
    assert 'code_snapshot_digest=str(environment_payload.get("execution", {}).get("code_snapshot_digest", "")).strip(),' in script
    assert 'cuda_visible_devices=normalized_cuda_visible_devices()' in script


@pytest.mark.skipif(BASH_EXE is None, reason="bash is required for shell-wrapper integration tests")
def test_remote_preflight_dry_run_accepts_formal_full_only_without_stage_fields() -> None:
    script_path = _wsl_path(ROOT / "scripts" / "remote" / "run_preflight.sh")
    manifest_path = _wsl_path(ROOT / "configs" / "matrices" / "suite_all_models_methods.json")
    completed = _run_bash_command(
        " ".join(
            [
                "bash",
                shlex.quote(script_path),
                "--dry-run",
                "--formal-full-only",
                "--full-manifest",
                shlex.quote(manifest_path),
                "--full-profile",
                "suite_all_models_methods",
            ]
        ),
        cwd=ROOT,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["full_manifest"] == manifest_path
    assert payload["formal_full_only"] == 1
    assert "stage_a_manifest" not in payload
    assert "stage_b_manifest" not in payload


@pytest.mark.skipif(BASH_EXE is None, reason="bash is required for shell-wrapper integration tests")
def test_remote_preflight_rejects_noncanonical_output_root_for_formal_full_only(tmp_path: Path) -> None:
    script_path = _wsl_path(ROOT / "scripts" / "remote" / "run_preflight.sh")
    manifest_path = _wsl_path(ROOT / "configs" / "matrices" / "suite_all_models_methods.json")
    completed = _run_bash_command(
        " ".join(
            [
                "bash",
                shlex.quote(script_path),
                "--dry-run",
                "--formal-full-only",
                "--full-manifest",
                shlex.quote(manifest_path),
                "--full-profile",
                "suite_all_models_methods",
                "--output-root",
                shlex.quote(_wsl_path(tmp_path / "alternate-matrix")),
            ]
        ),
        cwd=ROOT,
    )

    assert completed.returncode != 0
    assert "remote preflight output root" in (completed.stdout + completed.stderr)


def test_remote_transfer_scripts_support_remote_port():
    upload = (ROOT / "scripts" / "remote" / "upload_bundle.sh").read_text(encoding="utf-8")
    fetch = (ROOT / "scripts" / "remote" / "fetch_results.sh").read_text(encoding="utf-8")
    assert 'REMOTE_PORT="${REMOTE_PORT:-22}"' in upload
    assert '--remote-port PORT' in upload
    assert 'rsync -av -e "ssh -p $REMOTE_PORT"' in upload
    assert 'scp -P "$REMOTE_PORT"' in upload
    assert 'REMOTE_PORT="${REMOTE_PORT:-22}"' in fetch
    assert '--remote-port PORT' in fetch
    assert 'rsync -av -e "ssh -p $REMOTE_PORT"' in fetch
    assert 'scp -P "$REMOTE_PORT"' in fetch
    assert 'RUN_DIR="${RUN_DIR:-results}"' in fetch


def test_remote_bootstrap_uses_public_safe_configurable_venv():
    script = (ROOT / "scripts" / "remote" / "bootstrap_linux_gpu.sh").read_text(encoding="utf-8")
    assert 'DEFAULT_REMOTE_VENV="${POSTEDITBENCH_REMOTE_VENV:-}"' in script
    assert 'MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-}"' in script
    assert "--model-cache-root PATH" in script
    assert 'ln -s "$MODEL_CACHE_ROOT" "$ROOT/model_cache"' in script
    assert "torch.cuda.is_available()" in script
    assert "torch.version.cuda" in script
    assert 'constraints-release-cu124.txt' in script
    assert 'https://download.pytorch.org/whl/cu124' in script
    assert 'torch torchvision torchaudio' not in script


def test_remote_suite_matrix_uses_bootstrapped_venv_python():
    script = (ROOT / "scripts" / "remote" / "run_suite_matrix.sh").read_text(encoding="utf-8")
    formal_script = (ROOT / "scripts" / "remote" / "run_formal_single_host_full.sh").read_text(encoding="utf-8")
    assert 'MANIFEST="configs/matrices/suite_all_models_methods.json"' in script
    assert 'STAGE_A_MANIFEST="configs/matrices/suite_canary_heavy.json"' in script
    assert 'STAGE_B_MANIFEST="configs/matrices/model_invocation_smoke.json"' in script
    assert 'OUTPUT_ROOT="$ROOT/results/matrix"' in script
    assert 'DEFAULT_REMOTE_VENV="${POSTEDITBENCH_REMOTE_VENV:-}"' in script
    assert '--stage-a-manifest PATH' in script
    assert '--stage-a-profile NAME' in script
    assert '--stage-b-manifest PATH' in script
    assert '--stage-b-profile NAME' in script
    assert '--venv PATH' in script
    assert '--bootstrap' in script
    assert 'BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"' in script
    assert 'bash "$ROOT/scripts/remote/run_preflight.sh"' in script
    assert 'scripts/certify_suite_precheck.py' in script
    assert 'scripts/run_full_matrix.py' in script
    assert 'RUN_FULL=0' in script
    assert 'FULL_FAIL_FAST=0' in script
    assert 'RESUME=0' in script
    assert 'REQUIRE_HF_TOKEN=0' in script
    assert 'SKIP_HF_ACCESS=0' in script
    assert '--run-full' in script
    assert '--full-fail-fast' in script
    assert '--resume' in script
    assert '--require-hf-token' in script
    assert '--skip-hf-access' in script
    assert 'This wrapper exists for engineering smoke and compatibility A/B precheck coverage only; it is not the publication-facing entrypoint.' in script
    assert 'use scripts/remote/run_formal_single_host_full.sh for the formal release rerun' in script
    assert 'FULL_MATRIX_ARGS=(' in script
    assert 'FULL_MATRIX_ARGS+=(--fail-fast)' in script
    assert 'FULL_MATRIX_ARGS+=(--resume)' in script
    assert 'if [[ ! -x "$PYTHON_BIN" ]]; then' in script
    assert 'Create the venv first with:' in script
    assert 'PREFLIGHT_ARGS=(' in script
    assert '--stage-a-manifest "$STAGE_A_MANIFEST_PATH"' in script
    assert '--stage-b-manifest "$STAGE_B_MANIFEST_PATH"' in script
    assert 'PREFLIGHT_ARGS+=(--require-hf-token)' in script
    assert 'PREFLIGHT_ARGS+=(--skip-hf-access)' in script
    assert 'CERTIFY_ARGS=(' in script
    assert 'CERTIFY_ARGS+=(--skip-hf-access)' in script
    assert 'scripts/validate_single_host_launch_receipt.py' in script
    assert 'LAUNCH_VALIDATE_ARGS=(' in script
    assert 'find_active_certification_processes()' in script
    assert 'find_active_full_matrix_processes()' in script
    assert 'find_active_full_matrix_lock_owner()' in script
    assert 'find_active_cleanup_processes()' in script
    assert 'Detected active cleanup process for $ROOT' in script
    assert 'Detected active full-matrix run for $ROOT; refusing to start a preflight-only wrapper' in script
    assert 'FULL_LAUNCH_LOCK_DIR="$LAUNCHER_RUN_DIR/full_run.launch.lock"' in script
    assert 'cleanup_detached_launcher_lock()' in script
    assert 'mkdir -p $FULL_LAUNCH_LOCK_DIR_ESCAPED' in script
    assert 'if [[ -n "\\$lock_pid" && "\\$lock_pid" == "\\$\\$" ]]; then' in script
    assert 'effective_gpu_slots()' in script
    assert 'EFFECTIVE_GPU_SLOTS="$(effective_gpu_slots)"' in script
    assert 'CANONICAL_STAGE_A_MANIFEST_DEFAULT_PATH="$(normalize_abs_path "configs/matrices/suite_canary_heavy.json")"' in script
    assert 'CANONICAL_STAGE_B_MANIFEST_DEFAULT_PATH="$(normalize_abs_path "configs/matrices/model_invocation_smoke.json")"' in script
    assert 'CANONICAL_SUITE_PROFILE=0' in script
    assert 'Canonical --manifest/--profile requires the canonical stage A/B manifests and profiles so precheck certifies one canonical manifest set.' in script
    assert 'CANONICAL_FORMAL_FULL=0' in script
    assert 'run_suite_matrix.sh no longer owns the canonical publication-facing full launch.' in script
    assert 'Use scripts/remote/run_formal_single_host_full.sh for the A/B-free standalone-preflight direct-full contract.' in script
    assert script.index('CANONICAL_FORMAL_FULL=0') < script.index('if [[ $DRY_RUN -eq 1 ]]')
    assert 'Adjusting gpu-slots from $GPU_SLOTS to $EFFECTIVE_GPU_SLOTS to match CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES.' in script
    assert 'Detected active certification processes for $ROOT; wait for them to finish before starting another preflight/precheck stack.' in script
    assert 'Detected active full-matrix run for $ROOT; refusing to clean outputs or start another full launch.' in script
    assert 'Full-run resume requested: precheck stages will rerun cleanly; --resume is reserved for the final full matrix launch.' in script
    assert 'POSTEDITBENCH_SUITE_LAUNCHER_DETACHED=1' in script
    assert 'Detached engineering suite launcher started.' in script
    assert 'full_run.launch.log' in script
    assert 'full_run.launch.status' in script
    assert 'full_run.launch.pid' in script
    assert 'LAUNCHER_RUN_DIR="$ROOT/results/launchers/$PROFILE"' in script
    assert 'CERTIFICATION_RUN_DIR="$ROOT/results/certifications/$PROFILE"' in script
    assert 'mkdir -p "$FULL_RUN_DIR" "$CERTIFICATION_RUN_DIR" "$LAUNCHER_RUN_DIR"' in script
    assert 'Custom --manifest/--profile requires explicit --stage-a-manifest/--stage-a-profile/--stage-b-manifest/--stage-b-profile so precheck certifies the same roster.' in script
    assert script.index('scripts/certify_suite_precheck.py') < script.index('scripts/clean_suite_outputs.py')
    assert '--preserve-precheck-artifacts' in script
    assert '--preserve-launcher-artifacts' in script
    assert '--allow-formal-release-path' not in script
    assert 'standalone preflight -> clean -> direct canonical full launch.' in formal_script
    assert '--formal-full-only' in formal_script
    assert 'Detached formal direct-full launcher started.' in formal_script
    assert 'POSTEDITBENCH_FORMAL_FULL_DETACHED=1' in formal_script
    assert 'validate_results_control_path()' in formal_script
    assert 'ensure_formal_matrix_output_surface_safe()' in formal_script
    assert 'ensure_formal_launcher_control_surfaces_safe()' in formal_script
    assert 'formal output root' in formal_script
    assert 'formal canonical output root' in formal_script
    assert 'The formal direct-full helper is reserved for repo-local output root results/matrix.' in formal_script


@pytest.mark.skipif(BASH_EXE is None, reason="bash is required for shell-wrapper integration tests")
def test_remote_suite_matrix_rejects_canonical_run_full_and_points_to_formal_helper() -> None:
    command = " ".join(
        [
            shlex.quote(_wsl_path(ROOT / "scripts" / "remote" / "run_suite_matrix.sh")),
            "--dry-run",
            "--run-full",
        ]
    )
    completed = _run_bash_command(command)

    assert completed.returncode != 0
    message = completed.stdout + completed.stderr
    assert "run_suite_matrix.sh no longer owns the canonical publication-facing full launch." in message
    assert "Use scripts/remote/run_formal_single_host_full.sh for the A/B-free standalone-preflight direct-full contract." in message


@pytest.mark.skipif(BASH_EXE is None, reason="bash is required for shell-wrapper integration tests")
def test_remote_formal_single_host_full_dry_run_describes_direct_full_contract(tmp_path: Path) -> None:
    completed = _run_formal_suite_wrapper(tmp_path, extra_args=["--dry-run"])

    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["steps"] == ["standalone_preflight", "clean_suite_outputs", "run_full_matrix"]


@pytest.mark.skipif(BASH_EXE is None, reason="bash is required for shell-wrapper integration tests")
def test_remote_formal_single_host_full_rejects_noncanonical_output_root(tmp_path: Path) -> None:
    command = " ".join(
        [
            shlex.quote(_wsl_path(ROOT / "scripts" / "remote" / "run_formal_single_host_full.sh")),
            "--dry-run",
            "--output-root",
            shlex.quote(_wsl_path(tmp_path / "alternate-matrix")),
        ]
    )
    completed = _run_bash_command(command, cwd=ROOT)

    assert completed.returncode != 0
    assert "formal output root" in (completed.stdout + completed.stderr)


@pytest.mark.skipif(BASH_EXE is None, reason="bash is required for shell-wrapper integration tests")
def test_remote_formal_single_host_full_executes_preflight_clean_and_launch_in_order(tmp_path: Path) -> None:
    env_json = ROOT / "results" / "environment" / "runtime_environment.json"
    env_md = ROOT / "results" / "environment" / "runtime_environment.md"
    preflight_receipt = ROOT / "results" / "certifications" / "remote_preflight_receipt.json"
    backups = {
        env_json: _backup_repo_file(env_json),
        env_md: _backup_repo_file(env_md),
        preflight_receipt: _backup_repo_file(preflight_receipt),
    }
    try:
        completed = _run_formal_suite_wrapper(tmp_path)
        log_path = tmp_path / "formal_suite_shim.log"
        assert log_path.exists(), completed.stderr + completed.stdout
        shim_log = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert completed.returncode == 0, completed.stderr + completed.stdout
        ordered_scripts = [entry.get("script", "") for entry in shim_log if entry.get("script")]
        assert "check_zero_compatibility_name.py" in ordered_scripts
        assert "audit_full_matrix.py" in ordered_scripts
        assert "capture_environment.py" in ordered_scripts
        assert "clean_suite_outputs.py" in ordered_scripts
        run_full_indices = [index for index, script_name in enumerate(ordered_scripts) if script_name == "run_full_matrix.py"]
        assert len(run_full_indices) == 2
        clean_entry = next(entry for entry in shim_log if entry.get("script") == "clean_suite_outputs.py")
        assert "--preserve-precheck-artifacts" in clean_entry["argv"]
        assert "--preserve-launcher-artifacts" in clean_entry["argv"]
        preflight_indices = [
            ordered_scripts.index("check_zero_compatibility_name.py"),
            ordered_scripts.index("audit_full_matrix.py"),
            ordered_scripts.index("capture_environment.py"),
        ]
        clean_index = ordered_scripts.index("clean_suite_outputs.py")
        launch_index = run_full_indices[-1]
        assert run_full_indices[0] < clean_index
        assert max(preflight_indices) < clean_index < launch_index
    finally:
        for path, state in backups.items():
            _restore_repo_file(path, state)


@pytest.mark.skipif(BASH_EXE is None, reason="bash is required for shell-wrapper integration tests")
def test_remote_formal_single_host_full_stops_before_cleanup_when_preflight_fails(tmp_path: Path) -> None:
    completed = _run_formal_suite_wrapper(tmp_path, mode="preflight_failure")
    log_path = tmp_path / "formal_suite_shim.log"
    assert log_path.exists(), completed.stderr + completed.stdout
    shim_log = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ordered_scripts = [entry.get("script", "") for entry in shim_log if entry.get("script")]

    assert completed.returncode != 0
    assert "preflight audit failed" in (completed.stdout + completed.stderr)
    assert "audit_full_matrix.py" in ordered_scripts
    audit_index = ordered_scripts.index("audit_full_matrix.py")
    run_full_indices = [index for index, script_name in enumerate(ordered_scripts) if script_name == "run_full_matrix.py"]
    assert len(run_full_indices) == 0
    assert "clean_suite_outputs.py" not in ordered_scripts
    assert ordered_scripts[audit_index + 1 :] == []


@pytest.mark.skipif(BASH_EXE is None, reason="bash is required for shell-wrapper integration tests")
def test_remote_formal_single_host_full_rejects_symlinked_preflight_environment_surface(tmp_path: Path) -> None:
    env_json = ROOT / "results" / "environment" / "runtime_environment.json"
    env_md = ROOT / "results" / "environment" / "runtime_environment.md"
    preflight_receipt = ROOT / "results" / "certifications" / "remote_preflight_receipt.json"
    backups = {
        env_json: _backup_repo_file(env_json),
        env_md: _backup_repo_file(env_md),
        preflight_receipt: _backup_repo_file(preflight_receipt),
    }
    outside = tmp_path / "outside-environment.json"
    outside.write_text("outside\n", encoding="utf-8")
    try:
        env_json.parent.mkdir(parents=True, exist_ok=True)
        if env_json.exists() or env_json.is_symlink():
            env_json.unlink()
        try:
            env_json.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation is not available in this environment")

        completed = _run_formal_suite_wrapper(tmp_path, manage_repo_artifacts=False)
        log_path = tmp_path / "formal_suite_shim.log"
        shim_log = (
            [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if log_path.exists()
            else []
        )
        ordered_scripts = [entry.get("script", "") for entry in shim_log if entry.get("script")]

        assert completed.returncode != 0
        assert "remote preflight environment json" in (completed.stdout + completed.stderr)
        assert "capture_environment.py" not in ordered_scripts
        assert "clean_suite_outputs.py" not in ordered_scripts
    finally:
        for path, state in backups.items():
            _restore_repo_file(path, state)


@pytest.mark.skipif(BASH_EXE is None, reason="bash is required for shell-wrapper integration tests")
def test_remote_formal_single_host_full_rejects_symlinked_launcher_surface_before_launch(tmp_path: Path) -> None:
    launcher_status = ROOT / "results" / "launchers" / "suite_all_models_methods" / "full_run.launch.status"
    backup = _backup_repo_file(launcher_status)
    outside = tmp_path / "outside-launch-status.txt"
    outside.write_text("outside\n", encoding="utf-8")
    try:
        launcher_status.parent.mkdir(parents=True, exist_ok=True)
        if launcher_status.exists() or launcher_status.is_symlink():
            launcher_status.unlink()
        try:
            launcher_status.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation is not available in this environment")

        completed = _run_formal_suite_wrapper(
            tmp_path,
            detached=False,
            manage_repo_artifacts=False,
        )

        assert completed.returncode != 0
        assert "formal launcher status" in (completed.stdout + completed.stderr)
        assert "Detached formal direct-full launcher started." not in (completed.stdout + completed.stderr)
    finally:
        _restore_repo_file(launcher_status, backup)


@pytest.mark.skipif(BASH_EXE is None, reason="bash is required for shell-wrapper integration tests")
def test_remote_preflight_rejects_noncanonical_stage_pair_for_canonical_full_manifest(tmp_path: Path) -> None:
    custom_stage_a = tmp_path / "custom_stage_a.json"
    custom_stage_b = tmp_path / "custom_stage_b.json"
    command = " ".join(
        [
            shlex.quote(_wsl_path(ROOT / "scripts" / "remote" / "run_preflight.sh")),
            "--dry-run",
            "--full-manifest",
            shlex.quote(_wsl_path(ROOT / "configs" / "matrices" / "suite_all_models_methods.json")),
            "--full-profile",
            "suite_all_models_methods",
            "--stage-a-manifest",
            shlex.quote(_wsl_path(custom_stage_a)),
            "--stage-a-profile",
            "custom_stage_a",
            "--stage-b-manifest",
            shlex.quote(_wsl_path(custom_stage_b)),
            "--stage-b-profile",
            "custom_stage_b",
        ]
    )
    completed = _run_bash_command(command)

    assert completed.returncode != 0
    assert "Canonical --full-manifest/--full-profile requires the canonical stage A/B manifests and profiles" in (
        completed.stdout + completed.stderr
    )


@pytest.mark.skipif(BASH_EXE is None, reason="bash is required for shell-wrapper integration tests")
def test_remote_suite_matrix_rejects_noncanonical_stage_pair_for_canonical_manifest(tmp_path: Path) -> None:
    custom_stage_a = tmp_path / "custom_stage_a.json"
    custom_stage_b = tmp_path / "custom_stage_b.json"
    command = " ".join(
        [
            shlex.quote(_wsl_path(ROOT / "scripts" / "remote" / "run_suite_matrix.sh")),
            "--dry-run",
            "--manifest",
            shlex.quote(_wsl_path(ROOT / "configs" / "matrices" / "suite_all_models_methods.json")),
            "--profile",
            "suite_all_models_methods",
            "--stage-a-manifest",
            shlex.quote(_wsl_path(custom_stage_a)),
            "--stage-a-profile",
            "custom_stage_a",
            "--stage-b-manifest",
            shlex.quote(_wsl_path(custom_stage_b)),
            "--stage-b-profile",
            "custom_stage_b",
        ]
    )
    completed = _run_bash_command(command)

    assert completed.returncode != 0
    assert "Canonical --manifest/--profile requires the canonical stage A/B manifests and profiles" in (
        completed.stdout + completed.stderr
    )


def test_remote_subset_pair_uses_isolated_profiles_and_distinct_gpus():
    script = (ROOT / "scripts" / "remote" / "run_reviewer_subset_pair.sh").read_text(encoding="utf-8")
    assert 'DEFAULT_REMOTE_VENV="${POSTEDITBENCH_REMOTE_VENV:-}"' in script
    assert '--gpu-a ID' in script
    assert '--gpu-b ID' in script
    assert '--profile suite_reviewer_subset_a' in script
    assert '--profile suite_reviewer_subset_b' in script
    assert '--models Qwen/Qwen2.5-Coder-14B-Instruct' in script
    assert '--methods sweet_runtime' in script
    assert '--sources crafted_original' in script
    assert '--models Qwen/Qwen2.5-Coder-7B-Instruct' in script
    assert '--methods kgw_runtime' in script
    assert '--sources humaneval_plus' in script
    assert 'CUDA_VISIBLE_DEVICES=$GPU_A' in script
    assert 'CUDA_VISIBLE_DEVICES=$GPU_B' in script
    assert 'subset_a.log' in script
    assert 'subset_b.log' in script


def test_package_zenodo_requires_pinned_python_or_repo_local_venv():
    script = (ROOT / "scripts" / "package_zenodo.sh").read_text(encoding="utf-8")
    assert 'PYTHON_BIN="${PYTHON_BIN:-}"' in script
    assert 'if [[ -x "$ROOT/.venv/bin/python" ]]; then' in script
    assert 'Missing Python interpreter. Set PYTHON_BIN or create $ROOT/.venv/bin/python.' in script
    assert 'if str(scripts_dir) not in sys.path:' in script
    assert "_ensure_real_environment_capture()" in script
    assert "placeholder runtime environment capture" in script


def test_export_publish_repo_defaults_to_sibling_release_repo():
    script = (ROOT / "scripts" / "export_publish_repo.py").read_text(encoding="utf-8")
    assert 'DEFAULT_OUTPUT = ROOT.parent / "PostEditBench_release"' in script
    assert 'compatibility_project_findings_for_text(' in script
    assert 'publish export root name' in script
    assert 'publish export output must be outside the active repository root' in script
    assert '_ensure_real_environment_capture(ROOT)' in script


def test_remote_matrix_shard_wrapper_is_host_local_and_deterministic():
    script = (ROOT / "scripts" / "remote" / "run_matrix_shard.sh").read_text(encoding="utf-8")
    assert "optional identical-execution-class sharded reproduction" in script
    assert 'Use scripts/remote/run_formal_single_host_full.sh for the formal single-host full suite' in script
    assert '--readiness-only' in script
    assert '--skip-readiness' in script
    assert 'Use --skip-readiness together with --no-clean' in script
    assert 'is_safe_profile_name()' in script
    assert 'Invalid --profile:' in script
    assert 'Invalid --canonical-profile:' in script
    assert 'require_repo_results_root()' in script
    assert '--output-root' in script
    assert '--certifications-root' in script
    assert 'must stay under' in script
    assert '$ROOT/results/matrix' in script
    assert '$ROOT/results/certifications' in script
    assert 'validate_visible_device_ordinals()' in script
    assert 'requires distinct CUDA_VISIBLE_DEVICES ordinals' in script
    assert 'results/certifications/<shard_profile>/' in script
    assert 'rm -rf' in script
    assert 'SHARD_OUTPUT_DIR' in script
    assert 'SHARD_CERT_DIR' in script
    assert 'validate_shard_manifest()' in script
    assert 'runtime_checkouts_ready()' in script
    assert 'ensure_runtime_checkouts()' in script
    assert 'fetch_runtime_upstreams.sh' in script
    assert 'Run bash $ROOT/scripts/fetch_runtime_upstreams.sh all explicitly before readiness or shard launch.' in script
    assert 'runtime_checkouts.log' in script
    assert 'validate_shard_manifest.log' in script
    assert 'capture_environment.py' in script
    assert 'toolchain_smoke()' in script
    assert 'toolchain_smoke.log' in script
    assert '"toolchain_smoke",' in script
    assert 'command -v g++ >/dev/null 2>&1 || { echo "Missing g++"' in script
    assert 'command -v javac >/dev/null 2>&1 || { echo "Missing javac"' in script
    assert 'command -v java >/dev/null 2>&1 || { echo "Missing java"' in script
    assert 'command -v node >/dev/null 2>&1 || { echo "Missing node"' in script
    assert 'command -v go >/dev/null 2>&1 || { echo "Missing go"' in script
    assert 'node -e "const add = (a, b) => a + b;' in script
    assert 'javac "$java_smoke_dir/Smoke.java"' in script
    assert 'java -cp "$java_smoke_dir" Smoke' in script
    assert 'go run "$go_smoke_dir/main.go"' in script
    assert 'audit_benchmarks.py' in script
    assert 'audit_full_matrix.py' in script
    assert '--strict-hf-cache' in script
    assert '--model-load-smoke' in script
    assert '--runtime-smoke' in script
    assert '--skip-provider-credentials' in script
    assert '--skip-hf-access' in script
    assert 'matrix_shard_readiness.json' in script
    assert 'host_environment.json' in script
    assert 'host_environment.md' in script
    assert 'benchmark_audit.json' in script
    assert 'full_matrix_audit.json' in script
    assert '"validate_shard_manifest",' in script
    assert '"ensure_runtime_checkouts",' in script
    assert '"manifest": manifest_rel' in script
    assert '"canonical_manifest": canonical_manifest_rel' in script
    assert '"manifest_digests": {' in script
    assert '"code_snapshot_digest": _repo_snapshot.repo_snapshot_sha256(root),' in script
    assert '"suite_model_revisions": suite_model_revisions' in script
    assert '"environment_receipt": {' in script
    assert '"execution_mode": "sharded_identical_execution_class",' in script
    assert '"cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip(),' in script
    assert '"environment_fingerprint": execution_environment_fingerprint' in script
    assert '"host_environment_fingerprint": host_environment_fingerprint' in script
    assert '"execution_environment_fingerprint": execution_environment_fingerprint' in script
    assert '"visible_gpu_count": visible_gpu_count' in script
    assert 'validate_existing_readiness_receipt()' in script
    assert '"$ROOT/scripts/_matrix_shard_launch.py" validate-existing-receipt' in script
    assert 'validate_shard_full_matrix_audit()' in script
    assert 'prepare_clean_launch_tree()' in script
    assert 'prepare_clean_launch_tree before launch' in script
    assert 'prepare_clean_launch_tree.log' in script
    assert '"$ROOT/scripts/_matrix_shard_launch.py" prepare-clean-launch-tree' in script
    assert '--root "$ROOT"' in script
    assert 'run_audit_full_matrix_step()' in script
    assert 'merge_safe_has_issues=true' in script
    assert 'GPU_SLOTS="${GPU_SLOTS:-8}"' in script
    assert 'GPU_POOL_MODE="${GPU_POOL_MODE:-shared}"' in script
    assert 'CPU_WORKERS="${CPU_WORKERS:-9}"' in script
    assert 'RETRY_COUNT="${RETRY_COUNT:-1}"' in script
    assert 'if [[ -z "$SHARD_COUNT" ]]; then' in script
    assert '--manifest "$MANIFEST_PATH"' in script
    assert '--output-root "$OUTPUT_ROOT_REL"' in script
    assert '--gpu-slots "$GPU_SLOTS"' in script
    assert '--gpu-pool-mode "$GPU_POOL_MODE"' in script
    assert '--cpu-workers "$CPU_WORKERS"' in script
    assert '--retry-count "$RETRY_COUNT"' in script
    assert 'readiness-only complete' in script
    assert 'shard_index' in script
    assert 'shard_count' in script
    assert 'matrix_shard_${SHARD_INDEX}_of_${SHARD_COUNT}' in script
    assert script.count('run_step "ensure_runtime_checkouts" "$SHARD_CERT_DIR/runtime_checkouts.log" ensure_runtime_checkouts') == 2


@pytest.mark.skipif(BASH_EXE is None, reason="bash is required for shell-wrapper integration tests")
def test_remote_matrix_shard_skip_readiness_aborts_at_shell_boundary_on_stale_receipt(tmp_path: Path) -> None:
    completed = _run_matrix_shard_wrapper(tmp_path, mode="stale_receipt")
    shim_log = (tmp_path / "shim.log").read_text(encoding="utf-8")

    assert completed.returncode != 0
    assert "code_snapshot_digest mismatch at launch time" in (completed.stderr + completed.stdout)
    assert "_matrix_shard_launch.py" in shim_log
    assert "validate-existing-receipt" in shim_log
    assert "run_full_matrix.py" not in shim_log


@pytest.mark.skipif(BASH_EXE is None, reason="bash is required for shell-wrapper integration tests")
def test_remote_matrix_shard_skip_readiness_aborts_at_shell_boundary_on_dirty_output_tree(tmp_path: Path) -> None:
    completed = _run_matrix_shard_wrapper(tmp_path, mode="dirty_output")
    shim_log = (tmp_path / "shim.log").read_text(encoding="utf-8")

    assert completed.returncode != 0
    assert "Shard output tree is not clean before launch" in (completed.stderr + completed.stdout)
    assert "_matrix_shard_launch.py" in shim_log
    assert "validate-existing-receipt" in shim_log
    assert "prepare-clean-launch-tree" in shim_log
    assert "run_full_matrix.py" not in shim_log
