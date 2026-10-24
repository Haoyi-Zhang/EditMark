from __future__ import annotations

import hashlib
import contextlib
import io
import os
import re
import subprocess
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ValidationTrace:
    compile_ok: bool
    tests_ok: bool
    executed_tests: tuple[str, ...]
    failure_reason: str | None
    validator: str


def stable_code_digest(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:24]


def _normalize_language(language: str) -> str:
    lowered = language.strip().lower()
    if lowered in {"js", "javascript", "node"}:
        return "javascript"
    if lowered in {"c++", "cpp"}:
        return "cpp"
    return lowered


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int = 30,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
    except (FileNotFoundError, PermissionError) as exc:
        return subprocess.CompletedProcess(
            command,
            127,
            stdout="",
            stderr=f"launch_failed:{exc.__class__.__name__}:{exc}",
        )


def _temp_root() -> Path:
    default_tmp = Path(__file__).resolve().parents[3] / ".cache" / "shared" / "tmp" / "semcodebook_validator"
    base = Path(os.environ.get("SEMCODEBOOK_TMPDIR", str(default_tmp)))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _failure_detail(result: subprocess.CompletedProcess[str], *, fallback: str) -> str:
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    message = stderr or stdout or fallback
    return message[:500]


def _infer_java_primary_class(code: str, metadata: dict[str, str]) -> str:
    explicit = metadata.get("java_primary_class", "").strip()
    if explicit:
        return explicit
    match = re.search(r"\b(?:public\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)", code)
    return match.group(1) if match else "Solution"


def _infer_java_test_class(test: str) -> str:
    match = re.search(r"\bpublic\s+class\s+([A-Za-z_][A-Za-z0-9_]*)", test)
    if match:
        return match.group(1)
    match = re.search(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)", test)
    return match.group(1) if match else "Main"


def _ensure_go_package(code: str) -> str:
    stripped = code.lstrip()
    if stripped.startswith("package "):
        return code
    return "package main\n\n" + code


def _python_trace(code: str, tests: tuple[str, ...]) -> ValidationTrace:
    namespace: dict[str, object] = {}
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        compile(code, "semcodebook_candidate.py", "exec")
    except SyntaxError as exc:
        return ValidationTrace(
            compile_ok=False,
            tests_ok=False,
            executed_tests=(),
            failure_reason=f"syntax_error:{exc.msg}",
            validator="python_compile_exec",
        )
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(code, namespace, namespace)
    except Exception:
        return ValidationTrace(
            compile_ok=True,
            tests_ok=False,
            executed_tests=(),
            failure_reason=traceback.format_exc(limit=1).strip(),
            validator="python_compile_exec",
        )
    executed_tests: list[str] = []
    for index, test in enumerate(tests):
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                exec(test, namespace, namespace)
            executed_tests.append(f"test_{index}")
        except Exception:
            executed_tests.append(f"test_{index}")
            return ValidationTrace(
                compile_ok=True,
                tests_ok=False,
                executed_tests=tuple(executed_tests),
                failure_reason=traceback.format_exc(limit=1).strip(),
                validator="python_compile_exec",
            )
    return ValidationTrace(
        compile_ok=True,
        tests_ok=True,
        executed_tests=tuple(executed_tests),
        failure_reason=None,
        validator="python_compile_exec",
    )


def _javascript_trace(code: str, tests: tuple[str, ...]) -> ValidationTrace:
    with tempfile.TemporaryDirectory(prefix="semcodebook_js_", dir=str(_temp_root())) as temp_dir:
        root = Path(temp_dir)
        candidate_path = root / "candidate.js"
        candidate_path.write_text(code, encoding="utf-8")
        compile_result = _run_command(["node", "--check", candidate_path.name], cwd=root)
        if compile_result.returncode != 0:
            return ValidationTrace(
                compile_ok=False,
                tests_ok=False,
                executed_tests=(),
                failure_reason=_failure_detail(compile_result, fallback="node_check_failed"),
                validator="javascript_node",
            )
        executed_tests: list[str] = []
        for index, test in enumerate(tests):
            harness = root / f"test_{index}.js"
            harness.write_text(code.rstrip() + "\n\n" + test.rstrip() + "\n", encoding="utf-8")
            run_result = _run_command(["node", harness.name], cwd=root)
            executed_tests.append(f"test_{index}")
            if run_result.returncode != 0:
                return ValidationTrace(
                    compile_ok=True,
                    tests_ok=False,
                    executed_tests=tuple(executed_tests),
                    failure_reason=_failure_detail(run_result, fallback="node_execution_failed"),
                    validator="javascript_node",
                )
        return ValidationTrace(
            compile_ok=True,
            tests_ok=True,
            executed_tests=tuple(executed_tests),
            failure_reason=None,
            validator="javascript_node",
        )


def _java_trace(code: str, tests: tuple[str, ...], metadata: dict[str, str]) -> ValidationTrace:
    with tempfile.TemporaryDirectory(prefix="semcodebook_java_", dir=str(_temp_root())) as temp_dir:
        root = Path(temp_dir)
        primary_class = _infer_java_primary_class(code, metadata)
        candidate_path = root / f"{primary_class}.java"
        candidate_path.write_text(code, encoding="utf-8")
        compile_result = _run_command(["javac", candidate_path.name], cwd=root)
        if compile_result.returncode != 0:
            return ValidationTrace(
                compile_ok=False,
                tests_ok=False,
                executed_tests=(),
                failure_reason=_failure_detail(compile_result, fallback="javac_failed"),
                validator="java_javac",
            )
        executed_tests: list[str] = []
        for index, test in enumerate(tests):
            test_class = _infer_java_test_class(test)
            test_path = root / f"{test_class}.java"
            test_path.write_text(test, encoding="utf-8")
            compile_all = _run_command(["javac", candidate_path.name, test_path.name], cwd=root)
            executed_tests.append(f"test_{index}")
            if compile_all.returncode != 0:
                return ValidationTrace(
                    compile_ok=True,
                    tests_ok=False,
                    executed_tests=tuple(executed_tests),
                    failure_reason=_failure_detail(compile_all, fallback="javac_test_failed"),
                    validator="java_javac",
                )
            run_result = _run_command(["java", test_class], cwd=root)
            if run_result.returncode != 0:
                return ValidationTrace(
                    compile_ok=True,
                    tests_ok=False,
                    executed_tests=tuple(executed_tests),
                    failure_reason=_failure_detail(run_result, fallback="java_execution_failed"),
                    validator="java_javac",
                )
        return ValidationTrace(
            compile_ok=True,
            tests_ok=True,
            executed_tests=tuple(executed_tests),
            failure_reason=None,
            validator="java_javac",
        )


def _go_trace(code: str, tests: tuple[str, ...], metadata: dict[str, str]) -> ValidationTrace:
    with tempfile.TemporaryDirectory(prefix="semcodebook_go_", dir=str(_temp_root())) as temp_dir:
        root = Path(temp_dir)
        (root / "go.mod").write_text("module semcodebooktmp\n\ngo 1.22\n", encoding="utf-8")
        candidate_path = root / "candidate.go"
        candidate_path.write_text(_ensure_go_package(code), encoding="utf-8")
        base_env = dict(os.environ)
        base_env["GO111MODULE"] = "on"
        base_env["TMPDIR"] = str(_temp_root())
        compile_result = _run_command(["go", "test", "-run", "^$"], cwd=root, env=base_env)
        if compile_result.returncode != 0:
            return ValidationTrace(
                compile_ok=False,
                tests_ok=False,
                executed_tests=(),
                failure_reason=_failure_detail(compile_result, fallback="go_compile_failed"),
                validator="go_test",
            )
        executed_tests: list[str] = []
        setup = metadata.get("go_test_setup", "").strip()
        for index, test in enumerate(tests):
            test_source = test if test.lstrip().startswith("package ") else "\n\n".join(part for part in (setup, test) if part)
            test_path = root / "candidate_test.go"
            test_path.write_text(test_source + "\n", encoding="utf-8")
            run_result = _run_command(["go", "test"], cwd=root, env=base_env)
            executed_tests.append(f"test_{index}")
            if run_result.returncode != 0:
                return ValidationTrace(
                    compile_ok=True,
                    tests_ok=False,
                    executed_tests=tuple(executed_tests),
                    failure_reason=_failure_detail(run_result, fallback="go_test_failed"),
                    validator="go_test",
                )
        return ValidationTrace(
            compile_ok=True,
            tests_ok=True,
            executed_tests=tuple(executed_tests),
            failure_reason=None,
            validator="go_test",
        )


def _cpp_trace(code: str, tests: tuple[str, ...]) -> ValidationTrace:
    with tempfile.TemporaryDirectory(prefix="semcodebook_cpp_", dir=str(_temp_root())) as temp_dir:
        root = Path(temp_dir)
        base_env = dict(os.environ)
        base_env["TMPDIR"] = str(_temp_root())
        candidate_path = root / "candidate.cpp"
        candidate_path.write_text(code, encoding="utf-8")
        compile_result = _run_command(
            ["g++", "-std=c++17", "-c", candidate_path.name, "-o", "candidate.o"],
            cwd=root,
            env=base_env,
        )
        if compile_result.returncode != 0:
            return ValidationTrace(
                compile_ok=False,
                tests_ok=False,
                executed_tests=(),
                failure_reason=_failure_detail(compile_result, fallback="cpp_compile_failed"),
                validator="cpp_gpp",
            )
        executed_tests: list[str] = []
        for index, test in enumerate(tests):
            harness = root / f"test_{index}.cpp"
            harness.write_text(code.rstrip() + "\n\n" + test.rstrip() + "\n", encoding="utf-8")
            exe_path = root / f"test_{index}"
            run_build = _run_command(
                ["g++", "-std=c++17", harness.name, "-o", exe_path.name],
                cwd=root,
                env=base_env,
            )
            executed_tests.append(f"test_{index}")
            if run_build.returncode != 0:
                return ValidationTrace(
                    compile_ok=True,
                    tests_ok=False,
                    executed_tests=tuple(executed_tests),
                    failure_reason=_failure_detail(run_build, fallback="cpp_test_compile_failed"),
                    validator="cpp_gpp",
                )
            run_result = _run_command([str(exe_path)], cwd=root)
            if run_result.returncode != 0:
                return ValidationTrace(
                    compile_ok=True,
                    tests_ok=False,
                    executed_tests=tuple(executed_tests),
                    failure_reason=_failure_detail(run_result, fallback="cpp_execution_failed"),
                    validator="cpp_gpp",
                )
        return ValidationTrace(
            compile_ok=True,
            tests_ok=True,
            executed_tests=tuple(executed_tests),
            failure_reason=None,
            validator="cpp_gpp",
        )


def validate_semantics(
    code: str,
    language: str,
    tests: tuple[str, ...],
    *,
    task_id: str = "candidate",
    metadata: dict[str, str] | None = None,
) -> ValidationTrace:
    del task_id
    resolved_metadata = dict(metadata or {})
    resolved_language = _normalize_language(language)
    if resolved_language == "python":
        return _python_trace(code, tests)
    if resolved_language == "javascript":
        return _javascript_trace(code, tests)
    if resolved_language == "java":
        return _java_trace(code, tests, resolved_metadata)
    if resolved_language == "go":
        return _go_trace(code, tests, resolved_metadata)
    if resolved_language == "cpp":
        return _cpp_trace(code, tests)
    return ValidationTrace(
        compile_ok=False,
        tests_ok=False,
        executed_tests=(),
        failure_reason=f"unsupported_language:{resolved_language}",
        validator="unsupported_language",
    )
