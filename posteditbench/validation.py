from __future__ import annotations

import os
import re
import signal
import shutil
import subprocess
import tempfile
import math
import itertools
import functools
import collections
import heapq
import bisect
import operator
import random
import string
import statistics
import fractions
import decimal
import datetime
import pathlib
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .models import BenchmarkExample
from .toolchains import inspect_local_toolchain


SAFE_BUILTINS = MappingProxyType(
    {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "range": range,
        "reversed": reversed,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "__import__": __import__,
    }
)

COMPLETION_SURFACE_PUBLIC_SOURCES = {
    "human_eval",
    "humaneval_plus",
    "humaneval_x",
    "mbxp_5lang",
}

COMPLETION_SURFACE_ADAPTERS = {
    "human-eval",
    "human-eval-plus",
    "humaneval-x",
    "mbxp-5lang",
}

STANDALONE_CODE_PREFIXES = (
    "def ",
    "class ",
    "from ",
    "import ",
    "@",
    "package ",
    "public ",
    "private ",
    "protected ",
    "func ",
    "#include",
    "using ",
    "fn ",
)

FENCED_CODE_BLOCK_RE = re.compile(r"```(?P<label>[A-Za-z0-9_+#-]*)\s*\n(?P<body>.*?)```", re.DOTALL)
LANGUAGE_FENCE_LABELS = {
    "python": {"python", "py"},
    "cpp": {"cpp", "c++", "cc", "cxx"},
    "java": {"java"},
    "javascript": {"javascript", "js", "node", "nodejs"},
    "go": {"go", "golang"},
}


@dataclass(frozen=True, slots=True)
class SemanticValidationResult:
    example_id: str
    language: str
    available: bool
    passed: bool | None
    failures: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "language": self.language,
            "available": self.available,
            "passed": self.passed,
            "failures": list(self.failures),
            "metadata": dict(self.metadata),
        }


def _python_namespace() -> dict[str, Any]:
    return {
        "__builtins__": dict(SAFE_BUILTINS),
        "math": math,
        "re": re,
        "itertools": itertools,
        "functools": functools,
        "collections": collections,
        "heapq": heapq,
        "bisect": bisect,
        "operator": operator,
        "random": random,
        "string": string,
        "statistics": statistics,
        "fractions": fractions,
        "decimal": decimal,
        "datetime": datetime,
        "pathlib": pathlib,
    }


def _python_timeout_seconds() -> float:
    raw = str(os.environ.get("POSTEDITBENCH_PYTHON_VALIDATION_TIMEOUT_SECONDS", "5.0")).strip()
    try:
        return max(0.1, float(raw))
    except Exception:
        return 5.0


def _python_exec_inproc(source: str, tests: tuple[str, ...]) -> tuple[bool, bool, tuple[str, ...], str]:
    namespace = _python_namespace()
    try:
        exec(compile(source, "<posteditbench>", "exec"), namespace, namespace)
    except _PythonValidationTimeout:
        raise
    except Exception as exc:
        return False, False, (f"source:{exc.__class__.__name__}:{exc}",), "compile"
    failures: list[str] = []
    for index, test in enumerate(tests):
        try:
            exec(compile(test, f"<posteditbench-test-{index}>", "exec"), namespace, namespace)
        except AssertionError as exc:
            failures.append(f"test_{index}:AssertionError:{exc}")
        except _PythonValidationTimeout:
            raise
        except Exception as exc:
            failures.append(f"test_{index}:{exc.__class__.__name__}:{exc}")
    if failures:
        kind = "assertion" if any("AssertionError" in item for item in failures) else "runtime"
        return True, False, tuple(failures), kind
    return True, True, (), ""


class _PythonValidationTimeout(Exception):
    pass


def _python_exec(source: str, tests: tuple[str, ...]) -> tuple[bool, bool, tuple[str, ...], str]:
    timeout_seconds = _python_timeout_seconds()
    if not hasattr(signal, "SIGALRM"):
        return _python_exec_inproc(source, tests)

    def _handle_timeout(signum, frame):  # pragma: no cover - exercised on Linux remote more than local Windows
        raise _PythonValidationTimeout()

    try:
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, _handle_timeout)
    except Exception:
        return _python_exec_inproc(source, tests)
    try:
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        return _python_exec_inproc(source, tests)
    except _PythonValidationTimeout:
        return True, False, (f"phase_python:TimeoutExpired:{timeout_seconds}",), "timeout"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _tool(name: str) -> str | None:
    return shutil.which(name)


def _classify_output(text: str, *, phase: str) -> str:
    lowered = text.lower()
    if phase == "compile":
        return "compile"
    if "assert" in lowered or "exception -- test case" in lowered or "runtimeerror(\"case_" in lowered:
        return "assertion"
    if "timeout" in lowered:
        return "timeout"
    return "runtime"


def _sanitize_js_test(test: str) -> str:
    cleaned = test.replace('const _ = require("lodash")', "")
    cleaned = cleaned.replace("return _.isEqual(object1, object2)", "return JSON.stringify(object1) === JSON.stringify(object2)")
    return cleaned


def _sanitize_go_test(test: str) -> tuple[str, str]:
    if "func Test" not in test:
        return "main", test
    body = test
    body = re.sub(r"import\s*\((?:.|\n)*?\)", "", body, count=1)
    body = re.sub(r"func\s+Test\w+\s*\(t\s+\*testing\.T\)\s*\{", "func main() {", body, count=1)
    body = re.sub(r"\s*assert\s*:=\s*assert\.New\(t\)\s*", "\n", body)

    def replace_equal(match: re.Match[str]) -> str:
        expected = match.group(1).strip()
        actual = match.group(2).strip()
        return f'if {actual} != {expected} {{ panic("assert.Equal failed") }}'

    body = re.sub(r"assert\.Equal\((.+?),\s*(.+?)\)", replace_equal, body)
    body = body.replace("testing.T", "")
    return "main", "package main\n" + body.strip()


def _run_subprocess(commands: list[list[str]], *, files: dict[str, str], timeout: float = 20.0) -> tuple[bool, bool, tuple[str, ...], str]:
    with tempfile.TemporaryDirectory(prefix="posteditbench-") as temp_dir:
        root = Path(temp_dir)
        for name, content in files.items():
            (root / name).write_text(content, encoding="utf-8", newline="\n")
        compile_success = True
        for index, command in enumerate(commands):
            resolved_command = list(command)
            candidate = root / resolved_command[0]
            if candidate.exists():
                resolved_command[0] = str(candidate)
            try:
                completed = subprocess.run(
                    resolved_command,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return compile_success, False, (f"phase_{index}:TimeoutExpired",), "timeout"
            output = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
            if completed.returncode != 0:
                if index < len(commands) - 1:
                    return False, False, (output or f"phase_{index}:compile_failed",), _classify_output(output, phase="compile")
                return compile_success, False, (output or f"phase_{index}:runtime_failed",), _classify_output(output, phase="run")
        return compile_success, True, (), ""


def _compiled_exec(language: str, source: str, tests: tuple[str, ...]) -> tuple[bool, bool, tuple[str, ...], str] | None:
    normalized = language.lower()
    if not tests:
        return None
    snippet = "\n".join(test for test in tests if test.strip())
    if normalized == "javascript":
        node = _tool("node")
        if not node:
            return None
        return _run_subprocess([[node, "solution.js"]], files={"solution.js": source + "\n" + _sanitize_js_test(snippet)})
    if normalized == "cpp":
        gpp = _tool("g++")
        if not gpp:
            return None
        main_source = '#include "solution.cpp"\n' + snippet
        return _run_subprocess(
            [[gpp, "-std=c++17", "-O2", "main.cpp", "-o", "run.exe"], ["./run.exe"]],
            files={"solution.cpp": source, "main.cpp": main_source},
        )
    if normalized == "java":
        javac = _tool("javac")
        java = _tool("java")
        if not javac or not java:
            return None
        return _run_subprocess(
            [[javac, "Solution.java", "Main.java"], [java, "Main"]],
            files={"Solution.java": source, "Main.java": snippet},
        )
    if normalized == "go":
        go = _tool("go")
        if not go:
            return None
        mode, body = _sanitize_go_test(snippet)
        if mode == "main":
            return _run_subprocess([[go, "run", "solution.go", "main.go"]], files={"solution.go": source, "main.go": body})
    return None


def _toolchain_metadata(language: str) -> dict[str, Any]:
    inspection = inspect_local_toolchain(language)
    return {
        "toolchain_verified": bool(inspection.get("verified", False)),
        "toolchain_runner_image": str(inspection.get("runner_image", "")),
        "toolchain_language_version": str(inspection.get("language_version", "")),
        "toolchain_tools": list(inspection.get("tools", [])),
        "toolchain_issues": list(inspection.get("issues", [])),
    }


def uses_completion_surface(example: BenchmarkExample) -> bool:
    metadata = dict(example.metadata)
    adapter_name = str(metadata.get("adapter_name", "")).strip().lower()
    public_source = str(metadata.get("public_source", "")).strip().lower()
    if adapter_name in COMPLETION_SURFACE_ADAPTERS or public_source in COMPLETION_SURFACE_PUBLIC_SOURCES:
        return True
    return False


def _split_source_lines(text: str) -> list[str]:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _dedent_source(text: str) -> str:
    return textwrap.dedent(str(text or "")).lstrip("\n")


def _strip_prompt_prefix_once(prompt: str, source: str) -> str:
    prompt_lines = _split_source_lines(prompt)
    source_lines = _split_source_lines(source)
    prompt_index = 0
    source_index = 0

    while prompt_index < len(prompt_lines) and source_index < len(source_lines):
        prompt_line = prompt_lines[prompt_index].rstrip()
        source_line = source_lines[source_index].rstrip()
        if not prompt_line.strip():
            prompt_index += 1
            continue
        if not source_line.strip():
            source_index += 1
            continue
        if prompt_line != source_line:
            return source
        prompt_index += 1
        source_index += 1

    while prompt_index < len(prompt_lines) and not prompt_lines[prompt_index].strip():
        prompt_index += 1
    if prompt_index != len(prompt_lines):
        return source
    if source_index == 0:
        return source
    return "\n".join(source_lines[source_index:]).lstrip("\n")


def _strip_prompt_prefix_preserving_source(prompt: str, source: str) -> str:
    text = str(source or "")
    stripped = _strip_prompt_prefix_once(prompt, text)
    if stripped != text:
        return stripped
    dedented = _dedent_source(text)
    if dedented != text:
        dedented_stripped = _strip_prompt_prefix_once(prompt, dedented)
        if dedented_stripped != dedented:
            return dedented_stripped
    return text


def _looks_like_standalone_source(source: str) -> bool:
    for candidate in (str(source or ""), _dedent_source(source)):
        for line in _split_source_lines(candidate):
            if not line.strip():
                continue
            stripped = line.lstrip()
            return not line.startswith((" ", "\t")) and stripped.startswith(STANDALONE_CODE_PREFIXES)
    return False


def _extract_fenced_code_block(source: str, *, language: str) -> str | None:
    matches = list(FENCED_CODE_BLOCK_RE.finditer(str(source or "")))
    if not matches:
        return None
    preferred_labels = LANGUAGE_FENCE_LABELS.get(str(language or "").strip().lower(), set())
    unlabeled: list[str] = []
    fallback: list[str] = []
    for match in matches:
        label = str(match.group("label") or "").strip().lower()
        body = str(match.group("body") or "").strip("\n")
        if not body:
            continue
        if label in preferred_labels:
            return body
        if not label:
            unlabeled.append(body)
        else:
            fallback.append(body)
    if unlabeled:
        return unlabeled[0]
    if fallback:
        return fallback[0]
    return None


def _trim_to_first_standalone_block(source: str) -> str:
    for candidate in (str(source or ""), _dedent_source(source)):
        lines = _split_source_lines(candidate)
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            stripped = line.lstrip()
            if not line.startswith((" ", "\t")) and stripped.startswith(STANDALONE_CODE_PREFIXES):
                return "\n".join(lines[index:]).lstrip("\n")
    return source


def _normalize_completion_surface_source(example: BenchmarkExample, source: str) -> str:
    text = str(source or "")
    if not text or not uses_completion_surface(example):
        return text
    stripped = _strip_prompt_prefix_preserving_source(example.prompt, text)
    fenced = _extract_fenced_code_block(stripped, language=example.language)
    if fenced is None and stripped != text:
        fenced = _extract_fenced_code_block(text, language=example.language)
    if fenced is not None:
        return fenced
    return _trim_to_first_standalone_block(stripped)


def visible_evaluation_source(example: BenchmarkExample, source: str) -> str:
    text = str(source or "")
    if not text or not uses_completion_surface(example):
        return text
    return _normalize_completion_surface_source(example, text)


def executable_validation_source(example: BenchmarkExample, source: str) -> str:
    text = str(source or "")
    if not text or not uses_completion_surface(example):
        return text
    normalized = _normalize_completion_surface_source(example, text)
    if _looks_like_standalone_source(normalized):
        return normalized
    prompt = str(example.prompt or "").rstrip("\n")
    completion = normalized.lstrip("\n")
    if not prompt:
        return completion
    if not completion:
        return prompt
    return f"{prompt}\n{completion}"


def validate_semantics(example: BenchmarkExample, source: str) -> SemanticValidationResult:
    tests = tuple(example.execution_tests)
    language = example.language.lower()
    metadata = dict(example.metadata)
    validation_mode = str(metadata.get("validation_mode", "unavailable"))
    language_family = str(metadata.get("language_family", language))
    validation_scope = str(metadata.get("validation_scope", "python_first"))
    evaluation_backend = str(metadata.get("evaluation_backend", validation_mode)).lower()
    runner = str(metadata.get("runner_image", ""))
    toolchain_metadata = _toolchain_metadata(language)
    executable_source = executable_validation_source(example, source)

    if evaluation_backend == "mock_multilingual":
        passed = bool(executable_source.strip()) and bool(tests)
        return SemanticValidationResult(
            example_id=example.example_id,
            language=example.language,
            available=True,
            passed=passed,
            failures=() if passed else ("mock_multilingual_failed",),
            metadata={
                "test_count": len(tests),
                "validation_mode": validation_mode,
                "language_family": language_family,
                "validation_scope": validation_scope,
                "validation_supported": True,
                "evaluation_backend": evaluation_backend,
                "runner_image": runner,
                "compile_success": bool(executable_source.strip()),
                "test_pass": passed,
                "error_kind": "" if passed else "runtime",
                "claimed_languages": list(metadata.get("claimed_languages", [])),
                **toolchain_metadata,
            },
        )

    if language == "python" and tests:
        compile_success, passed, failures, error_kind = _python_exec(executable_source, tests)
        return SemanticValidationResult(
            example_id=example.example_id,
            language=example.language,
            available=True,
            passed=passed,
            failures=failures,
            metadata={
                "test_count": len(tests),
                "validation_mode": validation_mode,
                "language_family": language_family,
                "validation_scope": validation_scope,
                "validation_supported": True,
                "evaluation_backend": evaluation_backend,
                "runner_image": runner,
                "compile_success": compile_success,
                "test_pass": passed,
                "error_kind": error_kind,
                "claimed_languages": list(metadata.get("claimed_languages", [])),
                **toolchain_metadata,
            },
        )

    if evaluation_backend == "docker_remote" and tests and not bool(toolchain_metadata.get("toolchain_verified", False)):
        issues = tuple(
            f"execution_validation_unavailable:{item}"
            for item in toolchain_metadata.get("toolchain_issues", [])
        ) or ("execution_validation_unavailable:toolchain_version_mismatch",)
        return SemanticValidationResult(
            example_id=example.example_id,
            language=example.language,
            available=False,
            passed=None,
            failures=issues,
            metadata={
                "reason": "toolchain_version_mismatch",
                "test_count": len(tests),
                "validation_mode": validation_mode,
                "language_family": language_family,
                "validation_scope": validation_scope,
                "validation_supported": bool(tests),
                "evaluation_backend": evaluation_backend,
                "runner_image": runner,
                "compile_success": None,
                "test_pass": None,
                "error_kind": "environment",
                "claimed_languages": list(metadata.get("claimed_languages", [])),
                **toolchain_metadata,
            },
        )

    compiled = _compiled_exec(language, executable_source, tests)
    if compiled is not None:
        compile_success, passed, failures, error_kind = compiled
        return SemanticValidationResult(
            example_id=example.example_id,
            language=example.language,
            available=True,
            passed=passed,
            failures=failures,
            metadata={
                "test_count": len(tests),
                "validation_mode": validation_mode,
                "language_family": language_family,
                "validation_scope": validation_scope,
                "validation_supported": True,
                "evaluation_backend": evaluation_backend,
                "runner_image": runner,
                "compile_success": compile_success,
                "test_pass": passed,
                "error_kind": error_kind,
                "claimed_languages": list(metadata.get("claimed_languages", [])),
                **toolchain_metadata,
            },
        )

    if not tests:
        reason = "missing_tests"
    else:
        reason = "missing_toolchain"
    return SemanticValidationResult(
        example_id=example.example_id,
        language=example.language,
        available=False,
        passed=None,
        failures=(f"execution_validation_unavailable:{reason}",),
        metadata={
            "reason": reason,
            "test_count": len(tests),
            "validation_mode": validation_mode,
            "language_family": language_family,
            "validation_scope": validation_scope,
            "validation_supported": False,
            "evaluation_backend": evaluation_backend,
            "runner_image": runner,
            "compile_success": None,
            "test_pass": None,
            "error_kind": "unavailable",
            "claimed_languages": list(metadata.get("claimed_languages", [])),
            **toolchain_metadata,
        },
    )
