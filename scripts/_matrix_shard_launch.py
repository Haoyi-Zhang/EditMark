from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts import _repo_snapshot, audit_full_matrix, capture_environment
from scripts._hf_readiness import (
    HFModelRequirement,
    smoke_load_local_hf_evaluator,
    smoke_load_local_hf_model,
    validate_local_hf_cache,
)
from scripts.evaluate_baseline_family import derived_baseline_eval_requirement

EXPECTED_EXECUTION_MODE = "sharded_identical_execution_class"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch-time guards for matrix shard execution.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-existing-receipt",
        help="Validate that a passed readiness receipt is still safe to reuse at launch time.",
    )
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--profile", required=True)
    validate.add_argument("--manifest-rel", required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--canonical-manifest", type=Path, required=True)
    validate.add_argument("--shard-index", type=int, required=True)
    validate.add_argument("--shard-count", type=int, required=True)
    validate.add_argument("--gpu-slots", type=int, required=True)
    validate.add_argument("--gpu-pool-mode", required=True)
    validate.add_argument("--cpu-workers", type=int, required=True)
    validate.add_argument("--retry-count", type=int, required=True)

    clean = subparsers.add_parser(
        "prepare-clean-launch-tree",
        help="Delete shard-local transient outputs and ensure the shard output tree is empty before launch.",
    )
    clean.add_argument("--root", type=Path, default=ROOT)
    clean.add_argument("--output-dir", type=Path, required=True)
    clean.add_argument("--extra-dir", action="append", type=Path, default=[])
    return parser.parse_args()


def _manifest_digests(manifest_path: Path, canonical_manifest_path: Path) -> tuple[str, str]:
    return (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        hashlib.sha256(canonical_manifest_path.read_bytes()).hexdigest(),
    )


def _current_suite_model_revisions(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return dict(payload.get("canonical_model_revisions") or payload.get("model_revisions") or {})


def collect_current_hf_requirements(manifest_path: Path, profile: str) -> dict[str, HFModelRequirement]:
    manifest = audit_full_matrix._load_manifest(manifest_path)
    resolved_profile = audit_full_matrix._resolved_profile(manifest, profile)
    run_items = audit_full_matrix._load_profile_runs(manifest, resolved_profile)
    hf_models: dict[str, dict[str, Any]] = {}
    for item in run_items:
        config_path = Path(str(item["config"]))
        config = audit_full_matrix.build_experiment_config(audit_full_matrix._resolved_run_raw(dict(item)))
        if config.provider_mode == "local_hf":
            model_name = str(config.provider_parameters.get("model", "")).strip()
            if model_name:
                audit_full_matrix._merge_hf_model(
                    hf_models,
                    model_name=model_name,
                    revision=str(config.provider_parameters.get("revision", "")),
                    cache_dir=str(config.provider_parameters.get("cache_dir", "")).strip(),
                    local_files_only=bool(config.provider_parameters.get("local_files_only", False)),
                    token_env=str(config.provider_parameters.get("token_env", "HF_ACCESS_TOKEN")).strip(),
                    trust_remote_code=bool(config.provider_parameters.get("trust_remote_code", False)),
                    device=str(config.provider_parameters.get("device", "cuda")),
                    dtype=str(config.provider_parameters.get("dtype", "float16")),
                    usage="local_hf",
                    config_path=config_path,
                )
        watermark_meta = dict(config.metadata.get("watermark", {})) if isinstance(config.metadata, dict) else {}
        runtime_model = str(watermark_meta.get("model_name", "")).strip()
        if runtime_model:
            audit_full_matrix._merge_hf_model(
                hf_models,
                model_name=runtime_model,
                revision=str(watermark_meta.get("revision", "")),
                cache_dir=str(watermark_meta.get("cache_dir", "")).strip(),
                local_files_only=bool(watermark_meta.get("local_files_only", False)),
                token_env=str(watermark_meta.get("token_env", "HF_ACCESS_TOKEN")).strip(),
                trust_remote_code=bool(watermark_meta.get("trust_remote_code", False)),
                device=str(watermark_meta.get("device", "cuda")),
                dtype=str(watermark_meta.get("dtype", "float16")),
                usage="runtime",
                config_path=config_path,
            )
        if bool(item.get("baseline_eval", False)):
            baseline_requirement = derived_baseline_eval_requirement(
                metadata=dict(config.metadata or {}),
                device="cuda",
                usage=("baseline_eval", "evaluator"),
            )
            audit_full_matrix._merge_hf_model(
                hf_models,
                model_name=baseline_requirement.model,
                revision=baseline_requirement.revision,
                cache_dir=baseline_requirement.cache_dir,
                local_files_only=baseline_requirement.local_files_only,
                token_env=baseline_requirement.token_env,
                trust_remote_code=baseline_requirement.trust_remote_code,
                device=baseline_requirement.device,
                dtype=baseline_requirement.dtype,
                usage="baseline_eval",
                config_path=config_path,
            )
    return {
        model_name: audit_full_matrix._requirement_from_model_config(config)
        for model_name, config in hf_models.items()
    }


def validate_current_hf_requirements(
    requirements: dict[str, HFModelRequirement],
) -> tuple[list[dict[str, Any]], list[str]]:
    payloads: list[dict[str, Any]] = []
    issues: list[str] = []
    for model_name in sorted(requirements):
        requirement = requirements[model_name]
        payload = validate_local_hf_cache(requirement, require_root_entry=True)
        payload["usage"] = list(requirement.usage)
        payload["config_paths"] = list(requirement.config_paths)
        if payload.get("status") == "ok":
            model_smoke = smoke_load_local_hf_model(requirement)
            evaluator_smoke = smoke_load_local_hf_evaluator(requirement)
        else:
            model_smoke = {"model": requirement.model, "status": "skipped", "issues": ["cache_validation_failed"]}
            evaluator_smoke = {"model": requirement.model, "status": "skipped", "issues": ["cache_validation_failed"]}
        payload["model_smoke"] = model_smoke
        payload["evaluator_smoke"] = evaluator_smoke
        payloads.append(payload)
        if payload.get("status") != "ok":
            issues.append(f"current HF cache validation failed for {model_name}: {payload.get('issues', ['unknown'])}")
            continue
        if model_smoke.get("status") != "ok":
            issues.append(f"current HF model smoke failed for {model_name}: {model_smoke.get('issues', ['unknown'])}")
        if evaluator_smoke.get("status") != "ok":
            issues.append(f"current HF evaluator smoke failed for {model_name}: {evaluator_smoke.get('issues', ['unknown'])}")
    return payloads, issues


def _status_by_model(items: list[dict[str, Any]] | None) -> dict[str, str]:
    payload: dict[str, str] = {}
    for item in items or []:
        model_name = str(item.get("model", "")).strip()
        if model_name:
            payload[model_name] = str(item.get("status", "")).strip()
    return payload


def _require_receipt_hf_audits(receipt: dict[str, Any], requirements: dict[str, HFModelRequirement]) -> None:
    audits = dict(receipt.get("audits", {}) or {})
    full_matrix_audit = dict(audits.get("full_matrix_audit", {}) or {})
    if not full_matrix_audit:
        raise SystemExit("readiness receipt is missing full_matrix_audit payload")
    expected_models = sorted(requirements)
    observed_models = sorted(str(item).strip() for item in full_matrix_audit.get("required_hf_models", []) if str(item).strip())
    if observed_models != expected_models:
        raise SystemExit("readiness receipt full_matrix_audit required_hf_models mismatch at launch time")
    cache_status = _status_by_model(list(full_matrix_audit.get("hf_cache_validation", []) or []))
    model_smoke_status = _status_by_model(list(full_matrix_audit.get("hf_model_smoke", []) or []))
    evaluator_smoke_status = _status_by_model(list(full_matrix_audit.get("hf_evaluator_smoke", []) or []))
    for model_name in expected_models:
        if cache_status.get(model_name) != "ok":
            raise SystemExit(f"readiness receipt cache audit is not ok for {model_name}")
        if model_smoke_status.get(model_name) != "ok":
            raise SystemExit(f"readiness receipt model smoke is not ok for {model_name}")
        if evaluator_smoke_status.get(model_name) != "ok":
            raise SystemExit(f"readiness receipt evaluator smoke is not ok for {model_name}")


def validate_existing_readiness_receipt(
    *,
    root: Path,
    receipt_path: Path,
    profile: str,
    manifest_rel: str,
    manifest_path: Path,
    canonical_manifest_path: Path,
    shard_index: int,
    shard_count: int,
    gpu_slots: int,
    gpu_pool_mode: str,
    cpu_workers: int,
    retry_count: int,
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    canonical_manifest_path = canonical_manifest_path.resolve()
    if not receipt_path.exists():
        raise SystemExit(f"missing readiness receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if str(receipt.get("receipt_type", "")).strip() != "matrix_shard_readiness":
        raise SystemExit(f"{receipt_path} must be a matrix_shard_readiness receipt")
    if str(receipt.get("status", "")).strip() != "passed":
        raise SystemExit(f"{receipt_path} must be passed before launch-only execution")
    if str(receipt.get("execution_mode", "")).strip() != EXPECTED_EXECUTION_MODE:
        raise SystemExit(f"{receipt_path} execution_mode must be {EXPECTED_EXECUTION_MODE}")
    if str(receipt.get("profile", "")).strip() != profile:
        raise SystemExit(f"{receipt_path} profile mismatch")
    if str(receipt.get("manifest", "")).strip() != manifest_rel:
        raise SystemExit(f"{receipt_path} manifest mismatch")
    expected_canonical_manifest_rel = str(canonical_manifest_path.relative_to(root)).replace("\\", "/")
    if str(receipt.get("canonical_manifest", "")).strip() != expected_canonical_manifest_rel:
        raise SystemExit(f"{receipt_path} canonical_manifest mismatch")
    if int(receipt.get("shard_index", 0) or 0) != shard_index:
        raise SystemExit(f"{receipt_path} shard_index mismatch")
    if int(receipt.get("shard_count", 0) or 0) != shard_count:
        raise SystemExit(f"{receipt_path} shard_count mismatch")
    if int(receipt.get("gpu_slots", 0) or 0) != gpu_slots:
        raise SystemExit(f"{receipt_path} gpu_slots mismatch")
    if str(receipt.get("gpu_pool_mode", "")).strip() != gpu_pool_mode:
        raise SystemExit(f"{receipt_path} gpu_pool_mode mismatch")
    if int(receipt.get("cpu_workers", 0) or 0) != cpu_workers:
        raise SystemExit(f"{receipt_path} cpu_workers mismatch")
    if int(receipt.get("retry_count", 0) or 0) != retry_count:
        raise SystemExit(f"{receipt_path} retry_count mismatch")

    manifest_digest, canonical_manifest_digest = _manifest_digests(manifest_path, canonical_manifest_path)
    manifest_digests = dict(receipt.get("manifest_digests", {}) or {})
    if str(manifest_digests.get("manifest", "")).strip() != manifest_digest:
        raise SystemExit(f"{receipt_path} manifest digest mismatch at launch time")
    if str(manifest_digests.get("canonical_manifest", "")).strip() != canonical_manifest_digest:
        raise SystemExit(f"{receipt_path} canonical manifest digest mismatch at launch time")

    expected_suite_model_revisions = dict(receipt.get("suite_model_revisions", {}) or {})
    current_suite_model_revisions = _current_suite_model_revisions(manifest_path)
    if current_suite_model_revisions != expected_suite_model_revisions:
        raise SystemExit(f"{receipt_path} suite_model_revisions mismatch at launch time")

    expected_code_snapshot_digest = str(receipt.get("code_snapshot_digest", "")).strip()
    if not expected_code_snapshot_digest:
        raise SystemExit(f"{receipt_path} is missing code_snapshot_digest")
    current_code_snapshot_digest = _repo_snapshot.repo_snapshot_sha256(root)
    if current_code_snapshot_digest != expected_code_snapshot_digest:
        raise SystemExit(f"{receipt_path} code_snapshot_digest mismatch at launch time")

    env_receipt = dict(receipt.get("environment_receipt", {}) or {})
    expected_execution_environment_fingerprint = str(env_receipt.get("execution_environment_fingerprint", "")).strip()
    expected_cuda_visible_devices = str(env_receipt.get("cuda_visible_devices", "")).strip()
    expected_visible_gpu_count = int(env_receipt.get("visible_gpu_count", 0) or 0)
    if not expected_execution_environment_fingerprint:
        raise SystemExit(f"{receipt_path} is missing environment_receipt.execution_environment_fingerprint")
    current_environment = capture_environment._collect()
    current_cuda_visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    current_execution_environment_fingerprint = capture_environment.execution_environment_fingerprint_sha256(
        current_environment,
        cuda_visible_devices=current_cuda_visible_devices,
    )
    current_visible_gpu_count = len(
        capture_environment.execution_class_gpu_devices(
            current_environment,
            cuda_visible_devices=current_cuda_visible_devices,
        )
    )
    if current_execution_environment_fingerprint != expected_execution_environment_fingerprint:
        raise SystemExit(f"{receipt_path} execution_environment_fingerprint mismatch at launch time")
    if current_cuda_visible_devices != expected_cuda_visible_devices:
        raise SystemExit(f"{receipt_path} cuda_visible_devices mismatch at launch time")
    if current_visible_gpu_count != expected_visible_gpu_count:
        raise SystemExit(f"{receipt_path} visible_gpu_count mismatch at launch time")

    current_hf_requirements = collect_current_hf_requirements(manifest_path, profile)
    _require_receipt_hf_audits(receipt, current_hf_requirements)
    current_hf_cache_validation, hf_issues = validate_current_hf_requirements(current_hf_requirements)
    if hf_issues:
        raise SystemExit("; ".join(hf_issues))

    return {
        "receipt": str(receipt_path),
        "status": "launch_ready",
        "code_snapshot_digest": current_code_snapshot_digest,
        "execution_environment_fingerprint": current_execution_environment_fingerprint,
        "cuda_visible_devices": current_cuda_visible_devices,
        "visible_gpu_count": current_visible_gpu_count,
        "required_hf_models": sorted(current_hf_requirements),
        "current_hf_cache_validation": current_hf_cache_validation,
    }


def _allowed_clean_launch_roots(root: Path) -> tuple[Path, ...]:
    results_root = root.resolve() / "results"
    return tuple(
        results_root / name
        for name in ("matrix", "matrix_shards", "certifications", "environment", "audits", "figures", "tables")
    )


def _resolve_clean_launch_target(root: Path, target: Path) -> Path:
    resolved = target.resolve(strict=False)
    if not any(resolved == allowed or allowed in resolved.parents for allowed in _allowed_clean_launch_roots(root)):
        allowed = ", ".join(str(path) for path in _allowed_clean_launch_roots(root))
        raise SystemExit(f"clean launch targets must stay under one of: {allowed}")
    return resolved


def prepare_clean_launch_tree(root: Path, output_dir: Path, extra_dirs: list[Path]) -> dict[str, Any]:
    cleanup_targets = [_resolve_clean_launch_target(root, Path(item)) for item in extra_dirs]
    output_dir = _resolve_clean_launch_target(root, Path(output_dir))
    cleanup_targets.append(output_dir)
    deleted: list[str] = []
    for path in cleanup_targets:
        if path.exists():
            shutil.rmtree(path)
            deleted.append(str(path))
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise SystemExit(f"Shard output tree is not clean before launch: {output_dir}")
    return {
        "status": "clean",
        "output_dir": str(output_dir),
        "deleted": deleted,
    }


def main() -> int:
    args = parse_args()
    if args.command == "validate-existing-receipt":
        payload = validate_existing_readiness_receipt(
            root=args.root,
            receipt_path=args.receipt,
            profile=args.profile,
            manifest_rel=args.manifest_rel,
            manifest_path=args.manifest,
            canonical_manifest_path=args.canonical_manifest,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            gpu_slots=args.gpu_slots,
            gpu_pool_mode=args.gpu_pool_mode,
            cpu_workers=args.cpu_workers,
            retry_count=args.retry_count,
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    if args.command == "prepare-clean-launch-tree":
        payload = prepare_clean_launch_tree(args.root, args.output_dir, list(args.extra_dir))
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
