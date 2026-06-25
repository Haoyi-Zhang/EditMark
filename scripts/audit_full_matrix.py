from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _shared import dump_json, load_json
from evaluate_baseline_family import derived_baseline_eval_requirement
from _hf_readiness import (
    HFModelRequirement,
    cache_entry_paths,
    smoke_load_local_hf_evaluator,
    smoke_load_local_hf_model,
    validate_local_hf_cache,
)
from posteditbench.config import build_experiment_config, load_config, merge_config_source, validate_experiment_config
from posteditbench.hf_auth import resolve_token_env_value
from posteditbench.models import BenchmarkExample, WatermarkSpec
from posteditbench.providers import summarize_provider_configuration
from posteditbench.suite import resolve_model_revision
from posteditbench.watermarks.registry import all_watermarks, build_watermark_bundle, internal_watermarks

RUNTIME_METHODS = ("stone_runtime", "sweet_runtime", "ewd_runtime", "kgw_runtime")
RUNTIME_SOURCE_PREFERENCE = (
    "crafted_original",
    "humaneval_x",
    "mbxp_5lang",
    "crafted_translation",
    "crafted_stress",
    "humaneval_plus",
    "mbpp_plus",
)
RUNTIME_SOURCE_RANK = {source: index for index, source in enumerate(RUNTIME_SOURCE_PREFERENCE)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-run audit for the PostEditBench suite matrix.")
    parser.add_argument("--manifest", type=Path, default=Path("configs/matrices/suite_all_models_methods.json"))
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--runtime-smoke", action="store_true", help="Execute a one-example smoke run for runtime methods.")
    parser.add_argument(
        "--runtime-smoke-timeout-seconds",
        type=int,
        default=259200,
        help="Timeout applied to each runtime smoke subprocess when --runtime-smoke is enabled.",
    )
    parser.add_argument("--strict-hf-cache", action="store_true", help="Require official root caches and validate shard integrity.")
    parser.add_argument("--model-load-smoke", action="store_true", help="Offline-load all required local_hf models and run a minimal generate.")
    parser.add_argument("--skip-provider-credentials", action="store_true")
    parser.add_argument("--skip-hf-access", action="store_true")
    return parser.parse_args()


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    return dict(load_json(manifest_path))


def _load_profile_runs(manifest: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    return [dict(item) for item in manifest.get("runs", []) if str(item.get("profile", profile)) == profile]


def _resolved_profile(manifest: dict[str, Any], requested_profile: str | None) -> str:
    profile = str(requested_profile or manifest.get("profile", "suite_all_models_methods")).strip()
    if not profile:
        return "suite_all_models_methods"
    return profile


def _resolved_run_raw(run_item: dict[str, Any]) -> dict[str, Any]:
    source = load_config(Path(str(run_item["config"])))
    overrides = dict(run_item.get("config_overrides", {})) if isinstance(run_item.get("config_overrides"), dict) else {}
    return merge_config_source(source, **overrides)


def _filter_issues(issues: list[str], *, skip_provider_credentials: bool) -> list[str]:
    if not skip_provider_credentials:
        return issues
    return [issue for issue in issues if "requires credential" not in issue]


def _example_for(language: str) -> BenchmarkExample:
    solution = {
        "python": "def add(a, b):\n    return a + b\n",
        "cpp": "#include <cstdint>\n\nint add(int a, int b) {\n    return a + b;\n}\n",
        "javascript": "function add(a, b) {\n  return a + b;\n}\n",
        "java": "class Add { int add(int a, int b) { return a + b; } }\n",
        "go": "package main\n\nfunc add(a int, b int) int {\n\treturn a + b\n}\n",
    }.get(language, "def add(a, b):\n    return a + b\n")
    return BenchmarkExample(
        example_id=f"audit_{language}",
        language=language,
        prompt="Write a function that adds two integers.",
        reference_solution=solution,
        execution_tests=("assert add(1, 2) == 3",) if language == "python" else (),
        metadata={"task_id": f"audit_{language}", "dataset": "audit"},
    )


def _smoke_project_native(method: str) -> dict[str, Any]:
    example = _example_for("python")
    spec = WatermarkSpec(name=method, secret="audit", payload="wm", strength=0.55, parameters={"threshold": 0.5, "seed": 7})
    bundle = build_watermark_bundle(method, allow_internal=method in set(internal_watermarks()))
    prepared = bundle.prepare_example(example, spec)
    watermarked = bundle.embed(prepared, spec)
    detection = bundle.detect(watermarked, spec, example_id=example.example_id)
    return {
        "method": method,
        "status": "ok" if detection.detected else "failed",
        "prepared_language": prepared.language,
        "watermarked_length": len(watermarked.source),
        "detected": bool(detection.detected),
        "score": float(detection.score),
    }


def _resolved_revision(model_id: str, explicit_revision: str = "") -> str:
    return resolve_model_revision(model_id, explicit_revision, require_canonical=False)


def _probe_hf_model(model_id: str, token: str, *, revision: str = "", timeout: float = 20.0) -> dict[str, Any]:
    resolved_revision = _resolved_revision(model_id, revision)
    request = urllib.request.Request(f"https://huggingface.co/{model_id}/resolve/{resolved_revision}/config.json", method="HEAD")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {
                "model": model_id,
                "requested_revision": resolved_revision,
                "accessible": True,
                "status": getattr(response, "status", 200),
                "reason": "ok",
            }
    except urllib.error.HTTPError as exc:
        return {
            "model": model_id,
            "requested_revision": resolved_revision,
            "accessible": False,
            "status": exc.code,
            "reason": str(exc.reason),
        }
    except urllib.error.URLError as exc:
        return {
            "model": model_id,
            "requested_revision": resolved_revision,
            "accessible": False,
            "status": None,
            "reason": str(exc.reason),
        }


def _hf_cache_entry(model_id: str, cache_dir: str) -> Path:
    root_entry, hub_entry = cache_entry_paths(model_id, cache_dir)
    if root_entry.exists():
        return root_entry
    if hub_entry.exists():
        return hub_entry
    return root_entry


def _merge_hf_model(
    bucket: dict[str, dict[str, Any]],
    *,
    model_name: str,
    revision: str,
    cache_dir: str,
    local_files_only: bool,
    token_env: str,
    trust_remote_code: bool,
    device: str,
    dtype: str,
    usage: str,
    config_path: Path,
) -> None:
    entry = bucket.get(model_name)
    if entry is None:
        entry = {
            "model": model_name,
            "revision": revision,
            "token_env": token_env,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
            "device": device,
            "dtype": dtype,
            "usage": set(),
            "config_paths": set(),
        }
        bucket[model_name] = entry
    else:
        conflicts: list[str] = []
        comparable_fields = {
            "revision": revision,
            "cache_dir": cache_dir,
            "token_env": token_env,
            "device": device,
            "dtype": dtype,
        }
        for field, incoming in comparable_fields.items():
            existing = str(entry.get(field, "") or "").strip()
            current = str(incoming or "").strip()
            if existing and current and existing != current:
                conflicts.append(f"{field} {existing!r} != {current!r}")
        if bool(entry.get("local_files_only", False)) != bool(local_files_only):
            conflicts.append(
                f"local_files_only {bool(entry.get('local_files_only', False))!r} != {bool(local_files_only)!r}"
            )
        if bool(entry.get("trust_remote_code", False)) != bool(trust_remote_code):
            conflicts.append(
                f"trust_remote_code {bool(entry.get('trust_remote_code', False))!r} != {bool(trust_remote_code)!r}"
            )
        if conflicts:
            raise ValueError(
                f"HF model requirement conflict for {model_name}: " + "; ".join(conflicts)
            )
    entry["usage"].add(usage)
    entry["config_paths"].add(str(config_path))
    if revision and not entry.get("revision"):
        entry["revision"] = revision
    if cache_dir and not entry.get("cache_dir"):
        entry["cache_dir"] = cache_dir
    entry["local_files_only"] = bool(local_files_only)
    entry["trust_remote_code"] = bool(trust_remote_code)
    if token_env and not entry.get("token_env"):
        entry["token_env"] = token_env
    if device and not entry.get("device"):
        entry["device"] = device
    if dtype and not entry.get("dtype"):
        entry["dtype"] = dtype


def _requirement_from_model_config(item: dict[str, Any]) -> HFModelRequirement:
    return HFModelRequirement(
        model=str(item["model"]),
        cache_dir=str(item.get("cache_dir", "")),
        local_files_only=bool(item.get("local_files_only", False)),
        revision=str(item.get("revision", "")),
        trust_remote_code=bool(item.get("trust_remote_code", False)),
        device=str(item.get("device", "cuda")),
        dtype=str(item.get("dtype", "float16")),
        token_env=str(item.get("token_env", "HF_ACCESS_TOKEN")),
        usage=tuple(sorted(str(entry) for entry in item.get("usage", set()))),
        config_paths=tuple(sorted(str(entry) for entry in item.get("config_paths", set()))),
    )


def _runtime_smoke(run_item: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
    resolved = _resolved_run_raw(run_item)
    raw = json.loads(json.dumps(resolved))
    raw.setdefault("project", {})
    raw["project"]["name"] = f"{raw['project'].get('name', 'audit')}-audit"
    benchmark = dict(raw.get("benchmark", {}))
    benchmark["limit"] = 1
    raw["benchmark"] = benchmark
    raw["attacks"] = {"include": ["comment_strip"]}
    audit_root = ROOT / "results" / "audits"
    audit_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="posteditbench-audit-", dir=str(audit_root)))
    try:
        report_path = temp_dir / "report.json"
        payload = base64.b64encode(json.dumps(raw).encode("utf-8")).decode("ascii")
        child = (
            "import base64, json, os; "
            "from posteditbench.config import build_experiment_config; "
            "from posteditbench.pipeline import run_experiment; "
            "raw = json.loads(base64.b64decode(os.environ['POSTEDITBENCH_AUDIT_RAW_CONFIG']).decode('utf-8')); "
            "config = build_experiment_config(raw, output_path=os.environ['POSTEDITBENCH_AUDIT_REPORT_PATH']); "
            "run_experiment(config)"
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                child,
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            env={
                **os.environ,
                "POSTEDITBENCH_AUDIT_RAW_CONFIG": payload,
                "POSTEDITBENCH_AUDIT_REPORT_PATH": str(report_path),
                "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
            },
            timeout=max(1, int(timeout_seconds)),
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(details or f"run_experiment.py exited with status {completed.returncode}")
        payload = load_json(report_path)
        row_count = len(payload.get("rows", []))
        if row_count <= 0:
            raise RuntimeError("runtime smoke produced an empty report")
        return {
            "method": str(raw.get("watermark", {}).get("scheme", "runtime")),
            "run_id": str(run_item.get("run_id", "")),
            "effective_model": str(
                run_item.get("effective_model")
                or raw.get("provider", {}).get("parameters", {}).get("model", "")
            ),
            "benchmark_label": str(benchmark.get("dataset_label", "")),
            "status": "ok",
            "row_count": row_count,
        }
    except Exception as exc:
        return {
            "method": str(raw.get("watermark", {}).get("scheme", "runtime")),
            "run_id": str(run_item.get("run_id", "")),
            "effective_model": str(
                run_item.get("effective_model")
                or raw.get("provider", {}).get("parameters", {}).get("model", "")
            ),
            "benchmark_label": str(benchmark.get("dataset_label", "")),
            "status": "failed",
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _smoke_methods_for_matrix(matrix_methods: set[str]) -> list[str]:
    preferred_order = tuple(all_watermarks())
    seen: set[str] = set()
    ordered: list[str] = []
    for method in preferred_order:
        if method in matrix_methods and method not in seen:
            ordered.append(method)
            seen.add(method)
    extras = sorted(method for method in matrix_methods if method not in ordered and method in preferred_order)
    return ordered + extras


def _required_roster(values: list[Any] | None) -> list[str]:
    if not values:
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _effective_model(config: Any) -> str:
    provider_model = str(config.provider_parameters.get("model", "")).strip()
    if provider_model:
        return provider_model
    return str(config.metadata.get("watermark", {}).get("model_name", "")).strip()


def _runtime_candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, str, str, str]:
    source_slug = str(candidate.get("source_slug", "")).strip()
    config_issue_count = len(candidate.get("config_issues", []))
    return (
        1 if config_issue_count else 0,
        RUNTIME_SOURCE_RANK.get(source_slug, len(RUNTIME_SOURCE_RANK)),
        str(candidate.get("benchmark_label", "")).strip(),
        str(candidate.get("run_id", "")).strip(),
        str(candidate.get("config", "")).strip(),
    )


def _runtime_smoke_representatives(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen_by_model: dict[str, dict[str, Any]] = {}
    for candidate in sorted(candidates, key=_runtime_candidate_sort_key):
        effective_model = str(candidate.get("effective_model", "")).strip()
        if not effective_model or effective_model in chosen_by_model:
            continue
        chosen_by_model[effective_model] = dict(candidate)
    return [chosen_by_model[model_name] for model_name in sorted(chosen_by_model)]


def main() -> int:
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    profile = _resolved_profile(manifest, args.profile)
    output_path = args.output or Path(f"results/audits/{profile}_audit.json")
    run_items = _load_profile_runs(manifest, profile)
    configs: list[dict[str, Any]] = []
    matrix_methods: set[str] = set()
    provider_modes: set[str] = set()
    gpu_pools: set[str] = set()
    matrix_models: set[str] = set()
    benchmark_labels: set[str] = set()
    hf_models: dict[str, dict[str, Any]] = {}
    runtime_run_lookup: dict[str, list[dict[str, Any]]] = {}
    runtime_checkout_issues: dict[str, set[str]] = defaultdict(set)
    slice_coverage: dict[tuple[str, str], set[str]] = {}
    issues: list[str] = []

    for item in run_items:
        config_path = Path(str(item["config"]))
        config = build_experiment_config(_resolved_run_raw(item))
        config_issues = _filter_issues(
            validate_experiment_config(config),
            skip_provider_credentials=args.skip_provider_credentials,
        )
        matrix_methods.add(config.watermark_name)
        provider_modes.add(config.provider_mode)
        effective_model = _effective_model(config)
        benchmark_label = str(config.corpus_parameters.get("dataset_label", "")).strip()
        source_slug = str(item.get("source_slug", "")).strip()
        gpu_pool = str(item.get("gpu_pool", "")).strip()
        if gpu_pool:
            gpu_pools.add(gpu_pool)
        if effective_model:
            matrix_models.add(effective_model)
        if benchmark_label:
            benchmark_labels.add(benchmark_label)
        if effective_model and benchmark_label:
            slice_coverage.setdefault((effective_model, benchmark_label), set()).add(config.watermark_name)
        if config.watermark_name in RUNTIME_METHODS:
            method_checkout_issues = runtime_checkout_issues[config.watermark_name]
            if not method_checkout_issues:
                from posteditbench.baselines.stone_family.common import validate_checkout as validate_runtime_checkout

                method_checkout_issues.update(validate_runtime_checkout(config.watermark_name))
            if method_checkout_issues:
                config_issues = [issue for issue in config_issues if issue not in method_checkout_issues]
        if config_issues:
            issues.extend(f"{config_path}: {issue}" for issue in config_issues)
        provider_summary = summarize_provider_configuration(config.provider_mode, dict(config.provider_parameters))
        record = {
            "run_id": str(item["run_id"]),
            "config": str(config_path),
            "provider_mode": config.provider_mode,
            "watermark_name": config.watermark_name,
            "effective_model": effective_model,
            "benchmark_label": benchmark_label,
            "source_slug": source_slug,
            "corpus_size": config.corpus_size,
            "provider_summary": provider_summary,
            "config_issues": config_issues,
        }
        if config.provider_mode == "local_hf":
            model_name = str(config.provider_parameters.get("model", "")).strip()
            token_env = str(config.provider_parameters.get("token_env", "HF_ACCESS_TOKEN")).strip()
            cache_dir = str(config.provider_parameters.get("cache_dir", "")).strip()
            local_files_only = bool(config.provider_parameters.get("local_files_only", False))
            if model_name:
                try:
                    _merge_hf_model(
                        hf_models,
                        model_name=model_name,
                        revision=str(config.provider_parameters.get("revision", "")),
                        cache_dir=cache_dir,
                        local_files_only=local_files_only,
                        token_env=token_env,
                        trust_remote_code=bool(config.provider_parameters.get("trust_remote_code", False)),
                        device=str(config.provider_parameters.get("device", "cuda")),
                        dtype=str(config.provider_parameters.get("dtype", "float16")),
                        usage="local_hf",
                        config_path=config_path,
                    )
                except ValueError as exc:
                    issues.append(str(exc))
        watermark_meta = dict(config.metadata.get("watermark", {})) if isinstance(config.metadata, dict) else {}
        runtime_model = str(watermark_meta.get("model_name", "")).strip()
        runtime_token_env = str(watermark_meta.get("token_env", "HF_ACCESS_TOKEN")).strip()
        runtime_cache_dir = str(watermark_meta.get("cache_dir", "")).strip()
        runtime_local_files_only = bool(watermark_meta.get("local_files_only", False))
        if runtime_model:
            try:
                _merge_hf_model(
                    hf_models,
                    model_name=runtime_model,
                    revision=str(watermark_meta.get("revision", "")),
                    cache_dir=runtime_cache_dir,
                    local_files_only=runtime_local_files_only,
                    token_env=runtime_token_env,
                    trust_remote_code=bool(watermark_meta.get("trust_remote_code", False)),
                    device=str(watermark_meta.get("device", "cuda")),
                    dtype="float16",
                    usage="runtime",
                    config_path=config_path,
                )
            except ValueError as exc:
                issues.append(str(exc))
        if config.watermark_name in RUNTIME_METHODS:
            runtime_run_lookup.setdefault(config.watermark_name, []).append(
                {
                    **dict(item),
                    "config": str(config_path),
                    "effective_model": effective_model,
                    "benchmark_label": benchmark_label,
                    "source_slug": source_slug,
                    "config_issues": list(config_issues),
                }
            )
        configs.append(record)

    for item in run_items:
        if not bool(item.get("baseline_eval", False)):
            continue
        baseline_config = build_experiment_config(_resolved_run_raw(dict(item)))
        baseline_eval_requirement = derived_baseline_eval_requirement(
            metadata=dict(baseline_config.metadata or {}),
            device="cuda",
            usage=("baseline_eval", "evaluator"),
        )
        try:
            _merge_hf_model(
                hf_models,
                model_name=baseline_eval_requirement.model,
                revision=baseline_eval_requirement.revision,
                cache_dir=baseline_eval_requirement.cache_dir,
                local_files_only=baseline_eval_requirement.local_files_only,
                token_env=baseline_eval_requirement.token_env,
                trust_remote_code=baseline_eval_requirement.trust_remote_code,
                device=baseline_eval_requirement.device,
                dtype=baseline_eval_requirement.dtype,
                usage="baseline_eval",
                config_path=Path(str(item.get("config", "scripts/evaluate_baseline_family.py"))),
            )
        except ValueError as exc:
            issues.append(str(exc))

    required_methods_source = manifest.get("required_watermark_methods")
    if required_methods_source is None:
        required_methods_source = manifest.get("method_roster")
    if required_methods_source is None:
        required_methods_source = sorted(matrix_methods)
    required_methods = [
        str(method).strip()
        for method in required_methods_source
        if str(method).strip()
    ]
    required_provider_modes = [
        str(mode).strip()
        for mode in manifest.get("required_provider_modes", [])
        if str(mode).strip()
    ]
    required_gpu_pools = [
        str(pool).strip()
        for pool in manifest.get("required_gpu_pools", [])
        if str(pool).strip()
    ]
    required_model_roster = _required_roster(manifest.get("model_roster"))
    required_benchmark_roster = _required_roster(manifest.get("benchmark_roster"))

    missing_methods = sorted(set(required_methods) - matrix_methods)
    if missing_methods:
        issues.append(f"matrix is missing watermark methods: {missing_methods}")
    missing_provider_modes = sorted(set(required_provider_modes) - provider_modes)
    if missing_provider_modes:
        issues.append(f"matrix is missing provider modes: {missing_provider_modes}")
    missing_gpu_pools = sorted(set(required_gpu_pools) - gpu_pools)
    if missing_gpu_pools:
        issues.append(f"matrix is missing gpu pools: {missing_gpu_pools}")
    missing_model_roster = sorted(set(required_model_roster) - matrix_models)
    if missing_model_roster:
        issues.append(f"matrix is missing model_roster entries: {missing_model_roster}")
    missing_benchmark_roster = sorted(set(required_benchmark_roster) - benchmark_labels)
    if missing_benchmark_roster:
        issues.append(f"matrix is missing benchmark_roster entries: {missing_benchmark_roster}")

    missing_slice_methods: list[dict[str, Any]] = []
    if required_model_roster and required_benchmark_roster and required_methods:
        required_method_set = set(required_methods)
        for model_name in required_model_roster:
            for benchmark_label in required_benchmark_roster:
                present = slice_coverage.get((model_name, benchmark_label), set())
                missing = sorted(required_method_set - present)
                if missing:
                    missing_slice_methods.append(
                        {
                            "model": model_name,
                            "benchmark": benchmark_label,
                            "missing_methods": missing,
                        }
                    )
    if missing_slice_methods:
        issues.append(
            "matrix fairness coverage is incomplete for one or more (model, benchmark) slices"
        )

    method_smoke: list[dict[str, Any]] = []
    for method in _smoke_methods_for_matrix(matrix_methods):
        if method in RUNTIME_METHODS:
            if args.runtime_smoke:
                matching_runs = runtime_run_lookup.get(method, [])
                if not matching_runs:
                    method_smoke.append({"method": method, "status": "failed", "error": "no_matrix_run"})
                else:
                    representative_runs = _runtime_smoke_representatives(matching_runs)
                    checkout_issues = sorted(runtime_checkout_issues.get(method, set()))
                    for issue in checkout_issues:
                        if issue not in issues:
                            issues.append(issue)
                    if checkout_issues:
                        for matching in representative_runs:
                            method_smoke.append(
                                {
                                    "method": method,
                                    "run_id": str(matching.get("run_id", "")),
                                    "effective_model": str(matching.get("effective_model", "")),
                                    "benchmark_label": str(matching.get("benchmark_label", "")),
                                    "source_slug": str(matching.get("source_slug", "")),
                                    "status": "skipped",
                                    "reason": "checkout_validation_failed",
                                    "issues": checkout_issues,
                                }
                            )
                    elif not representative_runs:
                        method_smoke.append(
                            {
                                "method": method,
                                "status": "failed",
                                "error": "no_representative_runtime_smoke_candidate",
                                "reason_detail": "missing_effective_model",
                            }
                        )
                        issues.append(
                            f"runtime smoke failed for {method}: no representative Method x Model candidate resolved from the manifest"
                        )
                    else:
                        for matching in representative_runs:
                            smoke = _runtime_smoke(matching, timeout_seconds=args.runtime_smoke_timeout_seconds)
                            smoke["source_slug"] = str(matching.get("source_slug", ""))
                            method_smoke.append(smoke)
                            if smoke["status"] != "ok":
                                issue_scope = ", ".join(
                                    value
                                    for value in (
                                        smoke.get("effective_model", ""),
                                        smoke.get("benchmark_label", ""),
                                        smoke.get("run_id", ""),
                                    )
                                    if value
                                )
                                if issue_scope:
                                    issues.append(
                                        f"runtime smoke failed for {method} ({issue_scope}): {smoke.get('error', 'unknown error')}"
                                    )
                                else:
                                    issues.append(f"runtime smoke failed for {method}: {smoke.get('error', 'unknown error')}")
            else:
                method_smoke.append({"method": method, "status": "skipped", "reason": "runtime_smoke_disabled"})
            continue
        smoke = _smoke_project_native(method)
        method_smoke.append(smoke)
        if smoke["status"] != "ok":
            issues.append(f"embed/detect smoke failed for {method}")

    required_hf_models = sorted(hf_models)
    hf_cache_validation: list[dict[str, Any]] = []
    if args.strict_hf_cache:
        for model_name in required_hf_models:
            requirement = _requirement_from_model_config(hf_models[model_name])
            result = validate_local_hf_cache(requirement, require_root_entry=True)
            result["usage"] = list(requirement.usage)
            result["config_paths"] = list(requirement.config_paths)
            hf_cache_validation.append(result)
            if result["status"] != "ok":
                issues.append(f"strict HF cache validation failed for {model_name}")

    hf_model_smoke: list[dict[str, Any]] = []
    hf_evaluator_smoke: list[dict[str, Any]] = []
    if args.model_load_smoke:
        cache_status = {item["model"]: item for item in hf_cache_validation}
        for model_name in required_hf_models:
            requirement = _requirement_from_model_config(hf_models[model_name])
            if cache_status.get(model_name, {}).get("status") == "failed":
                smoke = {
                    "model": model_name,
                    "status": "skipped",
                    "issues": ["cache_validation_failed"],
                    "usage": list(requirement.usage),
                    "config_paths": list(requirement.config_paths),
                }
                evaluator_smoke = {
                    "model": model_name,
                    "status": "skipped",
                    "issues": ["cache_validation_failed"],
                    "usage": list(requirement.usage),
                    "config_paths": list(requirement.config_paths),
                }
            else:
                smoke = smoke_load_local_hf_model(requirement)
                smoke["usage"] = list(requirement.usage)
                smoke["config_paths"] = list(requirement.config_paths)
                if smoke["status"] != "ok":
                    issues.append(f"offline model-load smoke failed for {model_name}: {smoke['issues'][0]}")
                evaluator_smoke = smoke_load_local_hf_evaluator(requirement)
                evaluator_smoke["usage"] = list(requirement.usage)
                evaluator_smoke["config_paths"] = list(requirement.config_paths)
                if evaluator_smoke["status"] != "ok":
                    issues.append(f"offline evaluator-load smoke failed for {model_name}: {evaluator_smoke['issues'][0]}")
            hf_model_smoke.append(smoke)
            hf_evaluator_smoke.append(evaluator_smoke)

    hf_access: list[dict[str, Any]] = []
    if not args.skip_hf_access:
        for model_name, access_config in sorted(hf_models.items()):
            token_env = str(access_config.get("token_env", "HF_ACCESS_TOKEN"))
            cache_dir = str(access_config.get("cache_dir", "")).strip()
            local_files_only = bool(access_config.get("local_files_only", False))
            try:
                requested_revision = _resolved_revision(model_name, str(access_config.get("revision", "")))
            except ValueError as exc:
                hf_access.append(
                    {
                        "model": model_name,
                        "requested_revision": "",
                        "accessible": False,
                        "status": "failed",
                        "reason": str(exc),
                        "token_env": token_env,
                        "cache_dir": cache_dir,
                        "local_files_only": local_files_only,
                    }
                )
                issues.append(f"Hugging Face access check failed for {model_name}: failed {exc}")
                continue
            if local_files_only and cache_dir:
                cache_entry = _hf_cache_entry(model_name, cache_dir)
                snapshots_dir = cache_entry / "snapshots"
                refs_dir = cache_entry / "refs"
                cached = cache_entry.exists() and (snapshots_dir.exists() or refs_dir.exists())
                result = {
                    "model": model_name,
                    "requested_revision": requested_revision,
                    "accessible": cached,
                    "status": "cache",
                    "reason": "local_cache" if cached else "missing_local_cache",
                }
            else:
                token = resolve_token_env_value(token_env)
                result = _probe_hf_model(model_name, token, revision=requested_revision)
            result["token_env"] = token_env
            result["cache_dir"] = cache_dir
            result["local_files_only"] = local_files_only
            hf_access.append(result)
            if not result["accessible"]:
                issues.append(f"Hugging Face access check failed for {model_name}: {result['status']} {result['reason']}")

    payload = {
        "profile": profile,
        "manifest": str(args.manifest),
        "config_count": len(configs),
        "method_count": len(required_methods),
        "required_methods": required_methods,
        "matrix_methods": sorted(matrix_methods),
        "missing_methods": missing_methods,
        "required_provider_modes": required_provider_modes,
        "provider_modes": sorted(provider_modes),
        "missing_provider_modes": missing_provider_modes,
        "required_gpu_pools": required_gpu_pools,
        "gpu_pools": sorted(gpu_pools),
        "missing_gpu_pools": missing_gpu_pools,
        "required_model_roster": required_model_roster,
        "matrix_models": sorted(matrix_models),
        "missing_model_roster": missing_model_roster,
        "required_benchmark_roster": required_benchmark_roster,
        "benchmark_labels": sorted(benchmark_labels),
        "missing_benchmark_roster": missing_benchmark_roster,
        "missing_slice_methods": missing_slice_methods,
        "configs": configs,
        "runtime_smoke_scope": "method_x_model" if args.runtime_smoke else "disabled",
        "method_smoke": method_smoke,
        "required_hf_models": required_hf_models,
        "hf_cache_validation": hf_cache_validation,
        "hf_model_smoke": hf_model_smoke,
        "hf_evaluator_smoke": hf_evaluator_smoke,
        "hf_access": hf_access,
        "issues": issues,
        "status": "clean" if not issues else "has_issues",
    }
    dump_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
