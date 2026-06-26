from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
import textwrap
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
import io
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .language_support import (
    default_evaluation_backend,
    language_family,
    language_version,
    normalize_language_name,
    runner_image,
    source_relative_to,
    supports_execution,
    validation_mode,
)
from .models import BenchmarkExample
from .utils import ensure_parent


@dataclass(frozen=True, slots=True)
class PublicBenchmarkSpec:
    name: str
    dataset_label: str
    source_url: str
    source_revision: str
    license_note: str
    split: str
    task_count: int
    adapter_name: str
    validation_scope: str = "python_first"
    source_format: str = "jsonl.gz"
    notes: str = ""
    source_kind: str = "remote_archive"
    included_in_core: bool = True
    artifact_policy: str = "allowed"
    validate_python_references: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset_label": self.dataset_label,
            "source_url": self.source_url,
            "source_revision": self.source_revision,
            "license_note": self.license_note,
            "split": self.split,
            "task_count": self.task_count,
            "adapter_name": self.adapter_name,
            "validation_scope": self.validation_scope,
            "source_format": self.source_format,
            "notes": self.notes,
            "source_kind": self.source_kind,
            "included_in_core": self.included_in_core,
            "artifact_policy": self.artifact_policy,
            "validate_python_references": self.validate_python_references,
        }


_PUBLIC_SPECS: dict[str, PublicBenchmarkSpec] = {
    "human_eval": PublicBenchmarkSpec(
        name="human_eval",
        dataset_label="HumanEval",
        source_url="https://raw.githubusercontent.com/openai/human-eval/6d43fb980f9fee3c892a914eda09951f772ad10d/data/HumanEval.jsonl.gz",
        source_revision="6d43fb980f9fee3c892a914eda09951f772ad10d",
        license_note="MIT; original HumanEval repository",
        split="test",
        task_count=164,
        adapter_name="human-eval",
        validation_scope="python_first",
    ),
    "humaneval_plus": PublicBenchmarkSpec(
        name="humaneval_plus",
        dataset_label="HumanEval+",
        source_url="https://raw.githubusercontent.com/evalplus/humanevalplus_release/200defce9e3429d28ca215b6dd061c0f7f31c18b/HumanEvalPlus.jsonl.gz",
        source_revision="200defce9e3429d28ca215b6dd061c0f7f31c18b",
        license_note="Apache-2.0; EvalPlus HumanEval+ release",
        split="test",
        task_count=164,
        adapter_name="human-eval-plus",
        validation_scope="python_first",
    ),
    "mbpp_plus": PublicBenchmarkSpec(
        name="mbpp_plus",
        dataset_label="MBPP+",
        source_url="https://raw.githubusercontent.com/evalplus/mbppplus_release/dadf43da556a00f7bacd71cb154f2932757d9144/MbppPlus.jsonl.gz",
        source_revision="dadf43da556a00f7bacd71cb154f2932757d9144",
        license_note="Apache-2.0; EvalPlus MBPP+ release",
        split="test",
        task_count=378,
        adapter_name="mbpp-plus",
        validation_scope="python_first",
    ),
    "humaneval_x": PublicBenchmarkSpec(
        name="humaneval_x",
        dataset_label="HumanEval-X (5-language balanced slice)",
        source_url="https://github.com/zai-org/CodeGeeX",
        source_revision="2838420b7b4492cf3d16bce5320e26e65960c9e2",
        license_note="Apache-2.0; CodeGeeX HumanEval-X benchmark",
        split="test",
        task_count=820,
        adapter_name="humaneval-x",
        validation_scope="multilingual_exec",
        source_format="git_checkout",
        source_kind="git_checkout",
        notes="Extracted from the official CodeGeeX repository at a pinned commit using the five benchmark language files.",
    ),
    "mbxp_5lang": PublicBenchmarkSpec(
        name="mbxp_5lang",
        dataset_label="MBXP-5lang (5-language balanced slice)",
        source_url="https://github.com/amazon-science/mxeval",
        source_revision="e09974f990eeaf0c0e8f2b5eaff4be66effb2c86",
        license_note="Apache-2.0; MXEval MBXP release",
        split="test",
        task_count=4693,
        adapter_name="mbxp-5lang",
        validation_scope="multilingual_exec",
        source_format="git_checkout",
        source_kind="git_checkout",
        notes="Extracted from the official MXEval repository at a pinned commit using the five MBXP language files plus example overlays.",
    ),
    "class_eval": PublicBenchmarkSpec(
        name="class_eval",
        dataset_label="ClassEval",
        source_url="https://github.com/FudanSELab/ClassEval",
        source_revision="eaeac44d0d5dcd8a95feec50726d66fedc73a98f",
        license_note="Repository MIT; dataset CC BY-NC 4.0; excluded from the executable suite aggregate because of artifact policy constraints.",
        split="test",
        task_count=100,
        adapter_name="class-eval",
        validation_scope="python_first",
        source_format="git_checkout",
        source_kind="git_checkout",
        notes="Supported as an optional Python control-set fetch/normalize path only; not included in the default suite aggregate sources.",
        included_in_core=False,
        artifact_policy="restricted_noncommercial",
        validate_python_references=False,
    ),
}


_ALIASES = {
    "humaneval": "human_eval",
    "human-eval": "human_eval",
    "human_eval": "human_eval",
    "evalplus_humaneval": "humaneval_plus",
    "evalplus-humaneval": "humaneval_plus",
    "human-eval-plus": "humaneval_plus",
    "humaneval-plus": "humaneval_plus",
    "mbpp": "mbpp_plus",
    "mbpp-plus": "mbpp_plus",
    "evalplus-mbpp": "mbpp_plus",
    "humanevalx": "humaneval_x",
    "humaneval-x": "humaneval_x",
    "human_eval_x": "humaneval_x",
    "mbxp": "mbxp_5lang",
    "mxeval": "mbxp_5lang",
    "mbxp-5lang": "mbxp_5lang",
    "classeval": "class_eval",
    "class_eval": "class_eval",
    "class-eval": "class_eval",
}


_MBXP_LANGUAGE_FILES = {
    "python": "mbpp_release_v1.jsonl",
    "cpp": "mbcpp_release_v1.2.jsonl",
    "java": "mbjp_release_v1.2.jsonl",
    "javascript": "mbjsp_release_v1.2.jsonl",
    "go": "mbgp_release_v1.1.jsonl",
}


_HUMANEVAL_X_LANGUAGE_FILES = {
    "python": ("python", "humaneval_python.jsonl.gz"),
    "cpp": ("cpp", "humaneval_cpp.jsonl.gz"),
    "java": ("java", "humaneval_java.jsonl.gz"),
    "javascript": ("js", "humaneval_js.jsonl.gz"),
    "go": ("go", "humaneval_go.jsonl.gz"),
}


def available_public_sources() -> tuple[str, ...]:
    return tuple(sorted(_PUBLIC_SPECS))


def resolve_public_source_name(name: str) -> str:
    normalized = str(name).strip().lower().replace(" ", "_")
    return _ALIASES.get(normalized, normalized)


def get_public_source_spec(name: str) -> PublicBenchmarkSpec:
    resolved = resolve_public_source_name(name)
    if resolved not in _PUBLIC_SPECS:
        raise KeyError(f"unknown public benchmark source: {name}")
    return _PUBLIC_SPECS[resolved]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _release_path(path: Path) -> str:
    return source_relative_to(_project_root(), path).replace("\\", "/")


def _run_git(args: Sequence[str], *, cwd: Path | None = None, timeout: float = 120.0) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr or completed.returncode}")
    return (completed.stdout or "").strip()


def _repo_checkout_dir(cache_dir: Path, spec: PublicBenchmarkSpec) -> Path:
    owner_repo = spec.source_url.rstrip("/").rsplit("/", 2)[-2:]
    repo_label = "-".join(owner_repo)
    return cache_dir / "repos" / f"{repo_label}-{spec.source_revision[:12]}"


def _ensure_git_checkout(spec: PublicBenchmarkSpec, *, cache_dir: Path, fetch: bool = True) -> Path:
    checkout_root = _repo_checkout_dir(cache_dir, spec)
    git_dir = checkout_root / ".git"
    if not git_dir.exists():
        if not fetch:
            raise FileNotFoundError(f"missing cached checkout for {spec.name}: {checkout_root}")
        ensure_parent(git_dir / "placeholder")
        _run_git(["init", str(checkout_root)], timeout=60.0)
        _run_git(["-C", str(checkout_root), "remote", "add", "origin", spec.source_url], timeout=30.0)
    remote_url = _run_git(["-C", str(checkout_root), "remote", "get-url", "origin"], timeout=30.0)
    if remote_url.rstrip("/") != spec.source_url.rstrip("/"):
        raise RuntimeError(
            f"public source checkout {checkout_root} points at {remote_url}, expected {spec.source_url}"
        )
    current_head = ""
    if git_dir.exists():
        try:
            current_head = _run_git(["-C", str(checkout_root), "rev-parse", "HEAD"], timeout=30.0)
        except RuntimeError:
            current_head = ""
    if current_head != spec.source_revision:
        if not fetch:
            raise FileNotFoundError(
                f"cached checkout {checkout_root} is at {current_head or 'unknown'}, expected {spec.source_revision}; re-run with fetch=True"
            )
        _run_git(["-C", str(checkout_root), "fetch", "--depth", "1", "origin", spec.source_revision], timeout=300.0)
        _run_git(["-C", str(checkout_root), "checkout", "--detach", spec.source_revision], timeout=120.0)
    head = _run_git(["-C", str(checkout_root), "rev-parse", "HEAD"], timeout=30.0)
    if head != spec.source_revision:
        raise RuntimeError(f"expected {spec.source_revision} in {checkout_root}, found {head}")
    dirty = _run_git(["-C", str(checkout_root), "status", "--porcelain"], timeout=30.0)
    if dirty:
        raise RuntimeError(f"public source checkout {checkout_root} must be clean before normalization")
    return checkout_root


def _download_bytes(url: str, *, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "posteditbench-artifact"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.URLError as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"failed to download {url}: {exc}") from exc


def _read_records_from_archive(archive_bytes: bytes) -> list[dict[str, Any]]:
    try:
        payload = gzip.decompress(archive_bytes).decode("utf-8")
    except OSError:
        payload = archive_bytes.decode("utf-8")
    rows: list[dict[str, Any]] = []
    for line in payload.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError("public benchmark archive must contain JSON objects")
        rows.append(row)
    return rows


def _archive_sha256(archive_bytes: bytes) -> str:
    return hashlib.sha256(archive_bytes).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_python_body(body: str) -> str:
    expanded = textwrap.dedent(body).expandtabs(4)
    lines = [line.rstrip() for line in expanded.splitlines()]
    non_empty = [line for line in lines if line.strip()]
    block_openers = ("def ", "async def ", "class ", "if ", "elif ", "else:", "for ", "while ", "try:", "except", "finally:", "with ")
    has_block_structure = any(
        line.lstrip().startswith(block_openers) or line.rstrip().endswith(":")
        for line in non_empty
    )
    if not has_block_structure:
        return "\n".join(line.lstrip(" ") if line.strip() else "" for line in lines).strip("\n")

    indent_levels = sorted({len(line) - len(line.lstrip(" ")) for line in non_empty})
    if not indent_levels:
        return ""
    indent_map = {indent: index * 4 for index, indent in enumerate(indent_levels)}
    normalized: list[str] = []
    for line in lines:
        if not line.strip():
            normalized.append("")
            continue
        indent = len(line) - len(line.lstrip(" "))
        mapped = indent_map.get(indent, indent_map[indent_levels[-1]])
        normalized.append(" " * mapped + line.lstrip(" "))
    return "\n".join(normalized).strip("\n")


def _split_python_solution(solution: str) -> tuple[str, str]:
    lines = textwrap.dedent(solution).expandtabs(4).splitlines()
    seen_body_text: list[str] = []
    split_index: int | None = None
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped:
            continue
        if re.match(r"^(?:async\s+def|def)\s+\w+", stripped):
            name_match = re.match(r"^(?:async\s+def|def)\s+([A-Za-z_]\w*)", stripped)
            if name_match:
                name = name_match.group(1)
                earlier_text = "\n".join(seen_body_text)
                if re.search(rf"\b{name}\s*\(", earlier_text):
                    split_index = index
                    break
        seen_body_text.append(line.rstrip())
    if split_index is None:
        return "\n".join(lines).strip("\n"), ""
    return "\n".join(lines[:split_index]).strip("\n"), "\n".join(lines[split_index:]).strip("\n")


def _stitch_python_reference(prompt: str, solution: str) -> str:
    prompt_lines = prompt.rstrip().splitlines()
    if not prompt_lines:
        body = _normalize_python_body(solution)
        return f"{textwrap.indent(body, '    ')}\n" if body else ""

    def_matches = [index for index, line in enumerate(prompt_lines) if re.match(r"^\s*(?:async\s+def|def)\s+\w+", line)]
    def_index = def_matches[-1] if def_matches else None
    if def_index is None:
        body_source, tail_source = _split_python_solution(solution)
        combined_body = _normalize_python_body("\n".join([prompt, body_source, tail_source]))
        return f"{textwrap.indent(combined_body, '    ')}\n" if combined_body else prompt.rstrip()

    prefix = "\n".join(line.rstrip() for line in prompt_lines[:def_index]).rstrip()
    header = prompt_lines[def_index].rstrip()
    prompt_body = "\n".join(prompt_lines[def_index + 1 :])
    solution_body, solution_tail = _split_python_solution(solution)
    normalized_prompt_body = _normalize_python_body(prompt_body) if prompt_body.strip() else ""
    normalized_solution_body = _normalize_python_body(solution_body) if solution_body.strip() else ""
    combined_body = "\n".join(part for part in [normalized_prompt_body, normalized_solution_body] if part.strip())
    indented_body = textwrap.indent(combined_body, "    ") if combined_body else ""
    normalized_tail = _normalize_python_body(solution_tail) if solution_tail else ""
    parts = [part for part in [prefix, header, indented_body, normalized_tail] if part]
    return "\n".join(parts) + "\n"


def _wrap_humaneval_test(entry_point: str, test: str) -> tuple[str, ...]:
    cleaned = test.strip()
    if not cleaned:
        return ()
    if "check(" in cleaned and entry_point and "check(candidate)" in cleaned:
        return (f"{cleaned}\ncheck({entry_point})",)
    if "check(" in cleaned and entry_point:
        return (f"{cleaned}\ncheck({entry_point})",)
    return (cleaned,)


def _difficulty_from_payload(prompt: str, test: str, *, language: str, bias: int = 0) -> str:
    complexity = len(prompt.split()) + len(test.split()) * 2 + bias
    if language in {"cpp", "java", "go"}:
        complexity += 12
    if complexity < 140:
        return "easy"
    if complexity < 320:
        return "medium"
    return "hard"


def _source_group_for(spec: PublicBenchmarkSpec) -> str:
    return f"public_{spec.name}"


def _common_public_fields(
    *,
    spec: PublicBenchmarkSpec,
    language: str,
    task_id: str,
    prompt: str,
    reference_solution: str,
    test: str,
    source_path: Path,
    source_index: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_language = normalize_language_name(language)
    family_id = task_id.split("/")[-1].replace("-", "_")
    evaluation_backend = default_evaluation_backend(normalized_language)
    reference_tests = extra.get("reference_tests", ()) if extra else ()
    execution_tests = extra.get("execution_tests", ()) if extra else ()
    source_sha = _text_sha256(reference_solution + test)
    reference_kind = str(extra.get("reference_kind", "canonical")).strip().lower() if extra else "canonical"
    payload = {
        "task_id": task_id,
        "dataset": spec.dataset_label,
        "language": normalized_language,
        "prompt": prompt.rstrip(),
        "reference_solution": reference_solution.rstrip(),
        "reference_tests": list(reference_tests),
        "execution_tests": list(execution_tests),
        "claimed_languages": [normalized_language],
        "language_family": language_family(normalized_language),
        "validation_mode": validation_mode(normalized_language),
        "validation_supported": supports_execution(normalized_language, list(execution_tests), backend=evaluation_backend),
        "source": str(source_path.name),
        "source_path": _release_path(source_path),
        "source_index": source_index,
        "source_url": spec.source_url,
        "source_revision": spec.source_revision,
        "source_sha256": source_sha,
        "source_digest": _text_sha256(f"{task_id}|{prompt}|{reference_solution}"),
        "prompt_digest": _text_sha256(prompt),
        "solution_digest": _text_sha256(reference_solution),
        "split": spec.split,
        "license_note": spec.license_note,
        "adapter_name": spec.adapter_name,
        "validation_scope": spec.validation_scope,
        "public_source": spec.name,
        "record_kind": "public_benchmark",
        "reference_kind": reference_kind,
        "source_group": _source_group_for(spec),
        "origin_type": "public",
        "family_id": f"{spec.name}_{family_id}",
        "difficulty": _difficulty_from_payload(prompt, test, language=normalized_language),
        "evaluation_backend": evaluation_backend,
        "runner_image": runner_image(normalized_language),
        "source_path": _release_path(source_path),
        "official_problem_file": _release_path(source_path),
        "language_version": language_version(normalized_language),
    }
    if extra:
        payload.update({str(key): value for key, value in extra.items() if value is not None})
    return payload


def _normalize_remote_example(record: Mapping[str, Any], index: int, spec: PublicBenchmarkSpec, source_path: Path) -> dict[str, Any]:
    task_id = str(record.get("task_id") or record.get("id") or f"{spec.name}-{index:04d}")
    prompt = str(record.get("prompt", "")).rstrip()
    entry_point = str(record.get("entry_point", "")).strip()
    if spec.name in {"human_eval", "humaneval_plus"}:
        canonical_solution = str(record.get("canonical_solution", "")).rstrip()
        test = str(record.get("test", "")).strip()
        reference_solution = _stitch_python_reference(prompt, canonical_solution)
        execution_tests = _wrap_humaneval_test(entry_point, test)
        reference_tests = execution_tests
        extra = {
            "entry_point": entry_point,
            "reference_tests": reference_tests,
            "execution_tests": execution_tests,
            "stress_suite": False,
            "reference_kind": "canonical",
        }
        if record.get("contract") is not None:
            extra["contract"] = record.get("contract")
        return _common_public_fields(
            spec=spec,
            language="python",
            task_id=task_id,
            prompt=prompt,
            reference_solution=reference_solution,
            test=test,
            source_path=source_path,
            source_index=index,
            extra=extra,
        )

    canonical_solution = str(record.get("canonical_solution", "")).rstrip()
    assertion = str(record.get("assertion", "")).strip()
    reference_solution = canonical_solution if canonical_solution.endswith("\n") else canonical_solution + "\n"
    base_inputs = record.get("base_input", [])
    plus_inputs = record.get("plus_input", [])
    extra = {
        "entry_point": entry_point,
        "reference_tests": (assertion,) if assertion else (),
        "execution_tests": (assertion,) if assertion else (),
        "stress_suite": bool(plus_inputs),
        "stress_base_input_count": len(base_inputs) if isinstance(base_inputs, list) else 0,
        "stress_plus_input_count": len(plus_inputs) if isinstance(plus_inputs, list) else 0,
        "reference_kind": "canonical",
    }
    if record.get("contract") is not None:
        extra["contract"] = record.get("contract")
    return _common_public_fields(
        spec=spec,
        language="python",
        task_id=task_id,
        prompt=prompt,
        reference_solution=reference_solution,
        test=assertion,
        source_path=source_path,
        source_index=index,
        extra=extra,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_tree_sha256(root: Path, relative_paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(dict.fromkeys(str(path).replace("\\", "/") for path in relative_paths)):
        source_path = root / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(source_path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"{path} must contain JSON objects on each line")
            rows.append(payload)
    return rows


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"{path} must contain JSON objects on each line")
            rows.append(payload)
    return rows


def _normalize_humaneval_x_records(
    spec: PublicBenchmarkSpec,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    benchmark_root = repo_root / "codegeex" / "benchmark" / "humaneval-x"
    all_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    extraction_rules: list[str] = []
    for language, (folder_name, filename) in _HUMANEVAL_X_LANGUAGE_FILES.items():
        relative_path = f"codegeex/benchmark/humaneval-x/{folder_name}/data/{filename}"
        source_path = benchmark_root / folder_name / "data" / filename
        rows = _read_jsonl_gz(source_path)
        task_ids = [str(row.get("task_id") or f"{language}/{index}") for index, row in enumerate(rows)]
        extraction_rules.append(relative_path)
        sources.append(
            {
                "source_group": _source_group_for(spec),
                "benchmark": spec.name,
                "dataset_label": spec.dataset_label,
                "language": language,
                "task_count": len(rows),
                "source_path": _release_path(source_path),
                "source_url": spec.source_url,
                "source_revision": spec.source_revision,
                "source_sha256": _file_sha256(source_path),
                "license_note": spec.license_note,
                "split": spec.split,
                "task_id_digest": _text_sha256("\n".join(task_ids)),
                "sample_task_ids": task_ids[:3] + task_ids[-1:],
            }
        )
        for index, row in enumerate(rows):
            task_id = str(row.get("task_id") or f"{language}/{index}")
            prompt = str(row.get("prompt", "")).rstrip()
            canonical_solution = str(row.get("canonical_solution", "")).rstrip()
            hidden_test = str(row.get("test", "")).strip()
            example_test = str(row.get("example_test", "")).strip()
            reference_solution = _stitch_python_reference(prompt, canonical_solution)
            extra = {
                "entry_point": str(row.get("entry_point") or ""),
                "reference_tests": (example_test,) if example_test else ((hidden_test,) if hidden_test else ()),
                "execution_tests": (hidden_test,) if hidden_test else (),
                "stress_suite": False,
                "example_test": example_test or None,
                "notes": "Canonical solution from HumanEval-X.",
                "difficulty": _difficulty_from_payload(prompt, hidden_test, language=language, bias=18),
                "family_id": f"humaneval_x_{task_id.split('/')[-1]}",
                "reference_kind": "canonical",
            }
            payload = _common_public_fields(
                spec=spec,
                language=language,
                task_id=task_id,
                prompt=prompt,
                reference_solution=reference_solution,
                test=hidden_test,
                source_path=source_path,
                source_index=index,
                extra=extra,
            )
            all_rows.append(payload)
    return all_rows, sources, extraction_rules


def _load_mbxp_sample_overlays(root: Path) -> tuple[dict[str, str], list[str]]:
    examples_root = root / "examples"
    overlays: dict[str, str] = {}
    overlay_files: list[str] = []
    if not examples_root.exists():
        return overlays, overlay_files
    for sample_path in sorted(examples_root.glob("*_samples.jsonl")):
        overlay_files.append(source_relative_to(root.parent.parent, sample_path))
        for row in _read_jsonl(sample_path):
            task_id = str(row.get("task_id", "")).replace("\\/", "/")
            completion = str(row.get("completion", "")).rstrip()
            if task_id and completion and task_id not in overlays:
                overlays[task_id] = completion
    return overlays, overlay_files


def _normalize_mbxp_records(
    spec: PublicBenchmarkSpec,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    benchmark_root = repo_root / "data" / "mbxp"
    sample_overlays, overlay_files = _load_mbxp_sample_overlays(benchmark_root)
    all_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    extraction_rules = [f"data/mbxp/{filename}" for filename in _MBXP_LANGUAGE_FILES.values()]
    extraction_rules.extend(overlay_files)
    for language, filename in _MBXP_LANGUAGE_FILES.items():
        source_path = benchmark_root / filename
        rows = _read_jsonl(source_path)
        task_ids = [str(row.get("task_id") or f"{language}/{index}") for index, row in enumerate(rows)]
        sources.append(
            {
                "source_group": _source_group_for(spec),
                "benchmark": spec.name,
                "dataset_label": spec.dataset_label,
                "language": language,
                "task_count": len(rows),
                "source_path": _release_path(source_path),
                "source_url": spec.source_url,
                "source_revision": spec.source_revision,
                "source_sha256": _file_sha256(source_path),
                "license_note": spec.license_note,
                "split": spec.split,
                "task_id_digest": _text_sha256("\n".join(task_ids)),
                "sample_task_ids": task_ids[:3] + task_ids[-1:],
            }
        )
        for index, row in enumerate(rows):
            task_id = str(row.get("task_id") or f"{language}/{index}")
            prompt = str(row.get("prompt", "")).rstrip()
            hidden_test = str(row.get("test", "")).strip()
            canonical_solution = str(row.get("canonical_solution") or "").rstrip()
            sample_completion = sample_overlays.get(task_id, "").rstrip()
            completion = canonical_solution or sample_completion
            reference_solution = _stitch_python_reference(prompt, completion) if completion else prompt
            notes = "Canonical solution from MBXP." if canonical_solution else "Smoke overlay completion from MXEval examples."
            extra = {
                "entry_point": str(row.get("entry_point") or ""),
                "reference_tests": (hidden_test,) if hidden_test else (),
                "execution_tests": (hidden_test,) if hidden_test else (),
                "description": str(row.get("description") or ""),
                "reference_kind": "canonical" if canonical_solution else ("smoke_overlay" if sample_completion else "prompt_only"),
                "smoke_completion_available": bool(sample_completion),
                "canonical_available": bool(canonical_solution),
                "notes": notes,
                "difficulty": _difficulty_from_payload(prompt, hidden_test, language=language, bias=10),
                "family_id": f"mbxp_{task_id.split('/')[-1]}",
            }
            payload = _common_public_fields(
                spec=spec,
                language=language,
                task_id=task_id,
                prompt=prompt,
                reference_solution=reference_solution,
                test=hidden_test,
                source_path=source_path,
                source_index=index,
                extra=extra,
            )
            all_rows.append(payload)
    return all_rows, sources, extraction_rules


def _normalize_classeval_records(
    spec: PublicBenchmarkSpec,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    relative_path = "data/ClassEval_data.json"
    source_path = repo_root / relative_path
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"{source_path} must contain a JSON list")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise TypeError(f"{source_path} must contain JSON objects")
        task_id = str(item.get("task_id") or f"ClassEval/{index:03d}")
        prompt = str(item.get("skeleton") or "").rstrip()
        test = str(item.get("test") or "").strip()
        solution = str(item.get("solution_code") or "").rstrip()
        method_infos = item.get("methods_info") if isinstance(item.get("methods_info"), list) else []
        metadata = {
            "entry_point": str(item.get("class_name") or ""),
            "reference_tests": (test,) if test else (),
            "execution_tests": (test,) if test else (),
            "reference_kind": "canonical",
            "notes": "ClassEval class-level benchmark task.",
            "difficulty": _difficulty_from_payload(prompt, test, language="python", bias=30),
            "family_id": f"class_eval_{task_id.split('/')[-1]}",
            "class_name": str(item.get("class_name") or ""),
            "class_level_task": True,
            "method_count": len(method_infos),
        }
        row = _common_public_fields(
            spec=spec,
            language="python",
            task_id=task_id,
            prompt=prompt,
            reference_solution=solution if solution.endswith("\n") else solution + "\n",
            test=test,
            source_path=source_path,
            source_index=index,
            extra=metadata,
        )
        rows.append(row)
    source_manifest = {
        "source_group": _source_group_for(spec),
        "benchmark": spec.name,
        "dataset_label": spec.dataset_label,
        "language": "python",
        "task_count": len(rows),
        "source_path": _release_path(source_path),
        "source_url": spec.source_url,
        "source_revision": spec.source_revision,
        "source_sha256": _file_sha256(source_path),
        "license_note": spec.license_note,
        "split": spec.split,
        "task_id_digest": _text_sha256("\n".join(str(row.get("task_id")) for row in rows)),
        "sample_task_ids": [str(row.get("task_id")) for row in rows[:3]] + [str(rows[-1].get("task_id"))] if rows else [],
    }
    return rows, [source_manifest], [relative_path]


def normalize_public_records(spec: PublicBenchmarkSpec, raw_rows: Sequence[Mapping[str, Any]], source_path: Path) -> list[dict[str, Any]]:
    return [_normalize_remote_example(record, index, spec, source_path) for index, record in enumerate(raw_rows)]


def _build_validation_example(row: Mapping[str, Any]) -> BenchmarkExample:
    reference_tests = row.get("reference_tests", [])
    execution_tests = row.get("execution_tests", [])
    if isinstance(reference_tests, (str, bytes)):
        reference_tests = [reference_tests]
    if isinstance(execution_tests, (str, bytes)):
        execution_tests = [execution_tests]
    if not isinstance(reference_tests, Sequence):
        reference_tests = []
    if not isinstance(execution_tests, Sequence):
        execution_tests = []
    return BenchmarkExample(
        example_id=str(row.get("example_id") or row.get("task_id") or row.get("id") or "unknown"),
        language=str(row.get("language", "")).strip() or "python",
        prompt=str(row.get("prompt", "")),
        reference_solution=str(row.get("reference_solution", "")),
        reference_tests=tuple(str(item) for item in reference_tests if str(item).strip()),
        execution_tests=tuple(str(item) for item in execution_tests if str(item).strip()),
        metadata=dict(row),
    )


def _python_public_exec(source: str, tests: tuple[str, ...]) -> tuple[bool, bool, tuple[str, ...], str]:
    namespace = {
        "__builtins__": __builtins__,
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
    try:
        exec(compile(source, "<posteditbench-public>", "exec"), namespace, namespace)
    except Exception as exc:
        return False, False, (f"source:{exc.__class__.__name__}:{exc}",), "compile"
    failures: list[str] = []
    for index, test in enumerate(tests):
        try:
            exec(compile(test, f"<posteditbench-public-test-{index}>", "exec"), namespace, namespace)
        except AssertionError as exc:
            failures.append(f"test_{index}:AssertionError:{exc}")
        except Exception as exc:
            failures.append(f"test_{index}:{exc.__class__.__name__}:{exc}")
    if failures:
        kind = "assertion" if any("AssertionError" in item for item in failures) else "runtime"
        return True, False, tuple(failures), kind
    return True, True, (), ""


def _validate_python_public_rows(rows: Sequence[Mapping[str, Any]], *, source_path: Path) -> None:
    failures: list[str] = []
    for index, row in enumerate(rows):
        if str(row.get("language", "")).lower() != "python":
            continue
        if not bool(row.get("validation_supported")):
            continue
        example = _build_validation_example(row)
        tests = tuple(example.execution_tests)
        if not tests:
            failures.append(f"{example.example_id}: available=False passed=None failures=['execution_validation_unavailable:missing_tests']")
            continue
        compile_success, passed, validation_failures, error_kind = _python_public_exec(example.reference_solution, tests)
        if not compile_success or not passed:
            failures.append(
                f"{row.get('task_id', f'row-{index}')}: available=True passed={passed} failures={list(validation_failures)}"
            )
    if failures:
        preview = "; ".join(failures[:5])
        raise ValueError(f"python public clean reference validation failed for {source_path}: {preview}")


def build_public_manifest(
    spec: PublicBenchmarkSpec,
    *,
    source_path: Path,
    normalized_path: Path,
    archive_sha256: str,
    normalized_rows: Sequence[Mapping[str, Any]],
    source_manifests: Sequence[Mapping[str, Any]] | None = None,
    extraction_rules: Sequence[str] | None = None,
    sample_ids_path: Path | None = None,
    validation_policy: str = "python_first",
) -> dict[str, Any]:
    observed_languages = sorted({str(row.get("language", "")).lower() for row in normalized_rows if str(row.get("language", "")).strip()})
    dataset_names = sorted({str(row.get("dataset", spec.dataset_label)).strip() or spec.dataset_label for row in normalized_rows})
    validation_supported = sorted(
        {str(row.get("language", "")).lower() for row in normalized_rows if bool(row.get("validation_supported"))}
    )
    source_group_counts: dict[str, int] = {}
    for row in normalized_rows:
        source_group = str(row.get("source_group", "")).lower()
        if source_group:
            source_group_counts[source_group] = source_group_counts.get(source_group, 0) + 1
    origin_type_counts: dict[str, int] = {}
    for row in normalized_rows:
        origin_type = str(row.get("origin_type", "")).lower()
        if origin_type:
            origin_type_counts[origin_type] = origin_type_counts.get(origin_type, 0) + 1
    reference_kind_counts: dict[str, int] = {}
    for row in normalized_rows:
        reference_kind = str(row.get("reference_kind", "")).lower()
        if reference_kind:
            reference_kind_counts[reference_kind] = reference_kind_counts.get(reference_kind, 0) + 1
    difficulty_counts: dict[str, int] = {}
    for row in normalized_rows:
        difficulty = str(row.get("difficulty", "")).lower()
        if difficulty:
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    materialized_source_manifests = [dict(item) for item in source_manifests or []]
    if not materialized_source_manifests:
        materialized_source_manifests = [
            {
                "source_url": spec.source_url,
                "source_revision": spec.source_revision,
                "source_sha256": archive_sha256,
                "source_archive_sha256": archive_sha256,
                "source_path": _release_path(source_path),
                "record_count": len(normalized_rows),
            }
        ]
    return {
        "schema_version": 2,
        "benchmark": spec.name,
        "dataset_label": spec.dataset_label,
        "source_group": _source_group_for(spec),
        "source_url": spec.source_url,
        "source_revision": spec.source_revision,
        "source_archive_sha256": archive_sha256,
        "source_path": _release_path(source_path),
        "normalized_path": _release_path(normalized_path),
        "sample_ids_path": _release_path(sample_ids_path) if sample_ids_path is not None else "",
        "split": spec.split,
        "license_note": spec.license_note,
        "adapter_name": spec.adapter_name,
        "validation_scope": spec.validation_scope,
        "validation_policy": validation_policy,
        "included_in_core": spec.included_in_core,
        "artifact_policy": spec.artifact_policy,
        "task_count": len(normalized_rows),
        "expected_task_count": spec.task_count,
        "observed_languages": observed_languages,
        "datasets": dataset_names,
        "validation_supported_languages": validation_supported,
        "source_group_counts": source_group_counts,
        "origin_type_counts": origin_type_counts,
        "reference_kind_counts": reference_kind_counts,
        "canonical_reference_rate": round(reference_kind_counts.get("canonical", 0) / max(1, len(normalized_rows)), 4),
        "difficulty_counts": difficulty_counts,
        "counts": {
            "observed": len(normalized_rows),
            "expected": spec.task_count,
        },
        "extraction_rules": [str(item).replace("\\", "/") for item in extraction_rules or []],
        "source_manifests": materialized_source_manifests,
        "notes": spec.notes,
    }


def _write_public_sample_ids(output_path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    sample_ids_path = output_path.with_suffix(".sample_ids.json")
    task_ids = sorted({str(row.get("task_id", "")).strip() for row in rows if str(row.get("task_id", "")).strip()})
    task_ids_by_language: dict[str, list[str]] = {}
    for row in rows:
        language = str(row.get("language", "")).strip().lower()
        task_id = str(row.get("task_id", "")).strip()
        if not language or not task_id:
            continue
        task_ids_by_language.setdefault(language, []).append(task_id)
    for language, values in task_ids_by_language.items():
        task_ids_by_language[language] = sorted(dict.fromkeys(values))
    sample_ids_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_count": len(task_ids),
                "task_ids": task_ids,
                "task_ids_by_language": task_ids_by_language,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return sample_ids_path


def _prepare_remote_archive_benchmark(
    spec: PublicBenchmarkSpec,
    *,
    output_path: Path,
    fetch: bool,
    cache_dir: Path,
) -> dict[str, Any]:
    raw_bytes: bytes
    source_path = cache_dir / f"{spec.name}.source.jsonl.gz"
    if source_path.exists():
        raw_bytes = source_path.read_bytes()
    else:
        if not fetch:
            raise FileNotFoundError(
                f"missing cached source for {spec.name}: {source_path}. Pass fetch=True to download it."
            )
        raw_bytes = _download_bytes(spec.source_url)
        ensure_parent(source_path)
        source_path.write_bytes(raw_bytes)

    raw_rows = _read_records_from_archive(raw_bytes)
    normalized_rows = normalize_public_records(spec, raw_rows, source_path)
    if spec.validate_python_references:
        _validate_python_public_rows(normalized_rows, source_path=source_path)
    archive_sha256 = _archive_sha256(raw_bytes)
    normalized_text = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in normalized_rows) + "\n"
    output_path.write_text(normalized_text, encoding="utf-8", newline="\n")
    sample_ids_path = _write_public_sample_ids(output_path, normalized_rows)
    manifest = build_public_manifest(
        spec,
        source_path=source_path,
        normalized_path=output_path,
        archive_sha256=archive_sha256,
        normalized_rows=normalized_rows,
        sample_ids_path=sample_ids_path,
        validation_policy="validated_python_rows" if spec.validate_python_references else "metadata_only",
    )
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _prepare_git_checkout_benchmark(
    spec: PublicBenchmarkSpec,
    *,
    output_path: Path,
    fetch: bool,
    cache_dir: Path,
) -> dict[str, Any]:
    checkout_dir = _repo_checkout_dir(cache_dir, spec)
    if checkout_dir.exists():
        source_path = _ensure_git_checkout(spec, cache_dir=cache_dir, fetch=fetch)
    else:
        if not fetch:
            raise FileNotFoundError(
                f"{spec.name} requires an upstream checkout in {cache_dir}. Re-run with fetch=True to materialize the pinned source."
            )
        source_path = _ensure_git_checkout(spec, cache_dir=cache_dir, fetch=True)
    if spec.name == "humaneval_x":
        normalized_rows, source_manifests, extraction_rules = _normalize_humaneval_x_records(spec, source_path)
    elif spec.name == "mbxp_5lang":
        normalized_rows, source_manifests, extraction_rules = _normalize_mbxp_records(spec, source_path)
    elif spec.name == "class_eval":
        normalized_rows, source_manifests, extraction_rules = _normalize_classeval_records(spec, source_path)
    else:  # pragma: no cover - defensive
        raise KeyError(f"unsupported git-backed benchmark: {spec.name}")

    if spec.validate_python_references:
        _validate_python_public_rows(normalized_rows, source_path=source_path)
    normalized_text = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in normalized_rows) + "\n"
    output_path.write_text(normalized_text, encoding="utf-8", newline="\n")
    sample_ids_path = _write_public_sample_ids(output_path, normalized_rows)
    manifest = build_public_manifest(
        spec,
        source_path=source_path,
        normalized_path=output_path,
        archive_sha256=_selected_tree_sha256(source_path, extraction_rules),
        normalized_rows=normalized_rows,
        source_manifests=source_manifests,
        extraction_rules=extraction_rules,
        sample_ids_path=sample_ids_path,
        validation_policy="validated_python_rows" if spec.validate_python_references else "metadata_only",
    )
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def prepare_public_benchmark(
    benchmark_name: str,
    *,
    output_path: str | Path,
    fetch: bool = False,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    spec = get_public_source_spec(benchmark_name)
    output_path = Path(output_path)
    cache_dir = Path(cache_dir) if cache_dir is not None else output_path.parent
    ensure_parent(output_path)
    if spec.source_kind == "remote_archive":
        return _prepare_remote_archive_benchmark(spec, output_path=output_path, fetch=fetch, cache_dir=cache_dir)
    return _prepare_git_checkout_benchmark(spec, output_path=output_path, fetch=fetch, cache_dir=cache_dir)
