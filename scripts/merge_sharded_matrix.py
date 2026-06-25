from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from _shared import dump_json, load_json
except ModuleNotFoundError:  # pragma: no cover
    from scripts._shared import dump_json, load_json


CANONICAL_PROFILE = "suite_all_models_methods"
CANONICAL_MANIFEST = Path("configs/matrices/suite_all_models_methods.json")
EXPECTED_EXECUTION_MODE = "sharded_identical_execution_class"
EXPECTED_GPU_SLOTS = 8
EXPECTED_GPU_POOL_MODE = "shared"
EXPECTED_CPU_WORKERS = 9
EXPECTED_RETRY_COUNT = 1
_ALLOWED_SHARD_AUDIT_ISSUES = {
    "matrix fairness coverage is incomplete for one or more (model, benchmark) slices",
}
_DEFAULT_INSPECTION_OUTPUT_INDEX = Path(
    "results/matrix/reviewer_sharded_inspection/suite_all_models_methods_sharded_merged/matrix_index.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge shard-local matrix indexes from one identical execution class into one reviewer-local "
            "inspection-only suite_all_models_methods index."
        )
    )
    parser.add_argument("--manifest", type=Path, default=CANONICAL_MANIFEST)
    parser.add_argument("--profile", type=str, default=CANONICAL_PROFILE)
    parser.add_argument("--shard-index", action="append", type=Path, required=True)
    parser.add_argument("--host-receipt", action="append", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, default=_DEFAULT_INSPECTION_OUTPUT_INDEX)
    return parser.parse_args()


def _load_payload(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a top-level JSON object")
    return payload


def _repo_relpath(path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def _resolve_payload_path(value: str | Path, *, anchor: Path = ROOT) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    return (anchor / candidate).resolve(strict=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_run_ids(manifest_payload: Mapping[str, Any], canonical_profile: str) -> list[str]:
    run_ids: list[str] = []
    for run in manifest_payload.get("runs", []):
        if not isinstance(run, Mapping):
            continue
        if str(run.get("profile", canonical_profile)).strip() != canonical_profile:
            continue
        run_id = str(run.get("run_id", "")).strip()
        if run_id:
            run_ids.append(run_id)
    return run_ids


def _resolve_manifest_reference(value: str, *, label: str) -> Path:
    resolved = _resolve_payload_path(value, anchor=ROOT)
    if not resolved.exists():
        raise SystemExit(f"{label} does not resolve to an existing file: {value}")
    return resolved


def _shard_audit_is_merge_safe(full_matrix_audit: Mapping[str, Any]) -> bool:
    status = str(full_matrix_audit.get("status", "")).strip()
    if status == "clean":
        return True
    if status != "has_issues":
        return False
    issues = {
        str(issue).strip()
        for issue in full_matrix_audit.get("issues", [])
        if str(issue).strip()
    }
    if not issues:
        return False
    if not issues.issubset(_ALLOWED_SHARD_AUDIT_ISSUES):
        return False
    for field in (
        "missing_methods",
        "missing_model_roster",
        "missing_benchmark_roster",
        "missing_provider_modes",
        "missing_gpu_pools",
    ):
        values = full_matrix_audit.get(field, [])
        if isinstance(values, list) and values:
            return False
    hf_model_smoke = full_matrix_audit.get("hf_model_smoke", [])
    if isinstance(hf_model_smoke, list):
        for item in hf_model_smoke:
            if not isinstance(item, Mapping):
                continue
            item_status = str(item.get("status", "")).strip()
            if item_status and item_status != "ok":
                return False
    method_smoke = full_matrix_audit.get("method_smoke", [])
    if isinstance(method_smoke, list):
        for item in method_smoke:
            if not isinstance(item, Mapping):
                continue
            item_status = str(item.get("status", "")).strip()
            if item_status and item_status != "ok":
                return False
    return True


def _validate_receipt(
    receipt_path: Path,
    receipt: Mapping[str, Any],
    *,
    canonical_manifest_path: Path,
    canonical_manifest_digest: str,
    canonical_profile: str,
    canonical_model_revisions: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    if str(receipt.get("receipt_type", "")).strip() != "matrix_shard_readiness":
        raise SystemExit(f"{receipt_path} must be a matrix_shard_readiness receipt")
    if str(receipt.get("status", "")).strip() != "passed":
        raise SystemExit(f"{receipt_path} is not passed")
    execution_mode = str(receipt.get("execution_mode", "")).strip()
    if execution_mode != EXPECTED_EXECUTION_MODE:
        raise SystemExit(f"{receipt_path} execution_mode must be {EXPECTED_EXECUTION_MODE}")
    if int(receipt.get("gpu_slots", 0) or 0) != EXPECTED_GPU_SLOTS:
        raise SystemExit(f"{receipt_path} gpu_slots must be {EXPECTED_GPU_SLOTS}")
    if str(receipt.get("gpu_pool_mode", "")).strip() != EXPECTED_GPU_POOL_MODE:
        raise SystemExit(f"{receipt_path} gpu_pool_mode must be {EXPECTED_GPU_POOL_MODE}")
    if int(receipt.get("cpu_workers", 0) or 0) != EXPECTED_CPU_WORKERS:
        raise SystemExit(f"{receipt_path} cpu_workers must be {EXPECTED_CPU_WORKERS}")
    if int(receipt.get("retry_count", 0) or 0) != EXPECTED_RETRY_COUNT:
        raise SystemExit(f"{receipt_path} retry_count must be {EXPECTED_RETRY_COUNT}")

    profile = str(receipt.get("profile", "")).strip()
    if not profile:
        raise SystemExit(f"{receipt_path} is missing profile")
    if str(receipt.get("canonical_profile", "")).strip() != canonical_profile:
        raise SystemExit(f"{receipt_path} canonical_profile mismatch")
    receipt_canonical_manifest = _resolve_manifest_reference(
        str(receipt.get("canonical_manifest", "")).strip(),
        label=f"{receipt_path} canonical_manifest",
    )
    if receipt_canonical_manifest != canonical_manifest_path:
        raise SystemExit(f"{receipt_path} canonical_manifest path mismatch")

    manifest_digests = dict(receipt.get("manifest_digests", {}))
    if str(manifest_digests.get("canonical_manifest", "")).strip() != canonical_manifest_digest:
        raise SystemExit(f"{receipt_path} canonical manifest digest mismatch")
    if not str(manifest_digests.get("manifest", "")).strip():
        raise SystemExit(f"{receipt_path} shard manifest digest metadata is missing")
    if dict(receipt.get("suite_model_revisions", {})) != dict(canonical_model_revisions):
        raise SystemExit(f"{receipt_path} suite_model_revisions mismatch")
    code_snapshot_digest = str(receipt.get("code_snapshot_digest", "")).strip()
    if not code_snapshot_digest:
        raise SystemExit(f"{receipt_path} is missing code_snapshot_digest")

    env = dict(receipt.get("environment_receipt", {}))
    execution_environment_fingerprint = str(env.get("execution_environment_fingerprint", "")).strip()
    if not execution_environment_fingerprint:
        raise SystemExit(f"{receipt_path} is missing environment_receipt.execution_environment_fingerprint")
    host_environment_fingerprint = str(env.get("host_environment_fingerprint", "")).strip()

    benchmark_audit = dict(receipt.get("audits", {}).get("benchmark_audit", {}) or {})
    full_matrix_audit = dict(receipt.get("audits", {}).get("full_matrix_audit", {}) or {})
    if str(benchmark_audit.get("status", "")).strip() != "ok":
        raise SystemExit(f"{receipt_path} benchmark audit is not clean")
    if not _shard_audit_is_merge_safe(full_matrix_audit):
        raise SystemExit(f"{receipt_path} full matrix audit is not merge-safe")

    receipt_manifest_ref = str(receipt.get("manifest", "")).strip()
    if not receipt_manifest_ref:
        raise SystemExit(f"{receipt_path} is missing manifest")
    _resolve_manifest_reference(receipt_manifest_ref, label=f"{receipt_path} manifest")
    return profile, execution_environment_fingerprint, host_environment_fingerprint, code_snapshot_digest


def _fallback_artifact_path(*, field: str, shard_output_dir: Path, run_id: str) -> Path:
    run_dir = shard_output_dir / run_id
    mapping = {
        "output_dir": run_dir,
        "report_path": run_dir / "report.json",
        "baseline_eval_path": run_dir / "baseline_eval.json",
        "log_path": run_dir / "run.log",
        "resolved_config_path": run_dir / "_resolved_config.yaml",
    }
    return mapping[field]


def _normalize_run_record(run: Mapping[str, Any], *, shard_output_dir: Path) -> dict[str, Any]:
    payload = dict(run)
    run_id = str(payload.get("run_id", "")).strip()
    for field in ("output_dir", "report_path", "baseline_eval_path", "log_path", "resolved_config_path"):
        raw_value = str(payload.get(field, "")).strip()
        resolved: Path | None = None
        if raw_value:
            candidate = _resolve_payload_path(raw_value, anchor=ROOT)
            if candidate.exists():
                resolved = candidate
        if resolved is None:
            fallback = _fallback_artifact_path(field=field, shard_output_dir=shard_output_dir, run_id=run_id)
            if fallback.exists():
                resolved = fallback
        payload[field] = str(resolved) if resolved is not None else raw_value
    return payload


def main() -> int:
    args = _parse_args()
    canonical_manifest_path = _resolve_payload_path(args.manifest, anchor=ROOT)
    if not canonical_manifest_path.exists():
        raise SystemExit(f"canonical manifest not found: {canonical_manifest_path}")
    canonical_manifest_payload = _load_payload(canonical_manifest_path)
    canonical_profile = str(args.profile).strip() or CANONICAL_PROFILE
    canonical_manifest_digest = _sha256(canonical_manifest_path)
    canonical_model_revisions = dict(canonical_manifest_payload.get("model_revisions", {}))
    canonical_run_ids = _canonical_run_ids(canonical_manifest_payload, canonical_profile)
    canonical_run_index_by_id = {run_id: index for index, run_id in enumerate(canonical_run_ids)}
    if not canonical_run_ids:
        raise SystemExit(f"{canonical_manifest_path} does not contain any runs for profile {canonical_profile}")

    receipt_payloads: dict[str, tuple[Path, dict[str, Any]]] = {}
    execution_environment_fingerprints: set[str] = set()
    host_environment_fingerprints: dict[str, str] = {}
    code_snapshot_digests: set[str] = set()
    for receipt_arg in args.host_receipt:
        receipt_path = _resolve_payload_path(receipt_arg, anchor=ROOT)
        receipt = _load_payload(receipt_path)
        profile, execution_environment_fingerprint, host_environment_fingerprint, code_snapshot_digest = _validate_receipt(
            receipt_path,
            receipt,
            canonical_manifest_path=canonical_manifest_path,
            canonical_manifest_digest=canonical_manifest_digest,
            canonical_profile=canonical_profile,
            canonical_model_revisions=canonical_model_revisions,
        )
        if profile in receipt_payloads:
            raise SystemExit(f"duplicate host receipt for shard profile {profile}")
        receipt_payloads[profile] = (receipt_path, receipt)
        execution_environment_fingerprints.add(execution_environment_fingerprint)
        code_snapshot_digests.add(code_snapshot_digest)
        if host_environment_fingerprint:
            host_environment_fingerprints[profile] = host_environment_fingerprint
    if len(execution_environment_fingerprints) != 1:
        raise SystemExit("host readiness receipts do not share one execution environment fingerprint")
    if len(code_snapshot_digests) != 1:
        raise SystemExit("host readiness receipts do not share one code_snapshot_digest")

    merged_runs: dict[str, dict[str, Any]] = {}
    shard_profiles: list[str] = []
    shard_count_values: set[int] = set()
    gpu_pool_modes: set[str] = set()
    for shard_index_arg in args.shard_index:
        shard_index_path = _resolve_payload_path(shard_index_arg, anchor=ROOT)
        shard_index = _load_payload(shard_index_path)
        shard_profile = str(shard_index.get("profile", "")).strip()
        if not shard_profile:
            raise SystemExit(f"{shard_index_path} is missing profile")
        if shard_profile in shard_profiles:
            raise SystemExit(f"duplicate shard profile in shard indexes: {shard_profile}")
        shard_profiles.append(shard_profile)

        receipt_tuple = receipt_payloads.get(shard_profile)
        if receipt_tuple is None:
            raise SystemExit(f"missing host receipt for shard profile {shard_profile}")
        receipt_path, receipt = receipt_tuple

        shard_manifest_ref = str(shard_index.get("manifest", "")).strip()
        if not shard_manifest_ref:
            raise SystemExit(f"{shard_index_path} is missing manifest")
        shard_manifest_path = _resolve_manifest_reference(shard_manifest_ref, label=f"{shard_index_path} manifest")
        if str(receipt.get("manifest", "")).strip():
            receipt_manifest_path = _resolve_manifest_reference(
                str(receipt.get("manifest", "")).strip(),
                label=f"{receipt_path} manifest",
            )
            if receipt_manifest_path != shard_manifest_path:
                raise SystemExit(f"{receipt_path} manifest does not match shard index manifest for {shard_profile}")

        shard_manifest = _load_payload(shard_manifest_path)
        if str(shard_manifest.get("profile", "")).strip() != shard_profile:
            raise SystemExit(f"{shard_manifest_path} profile mismatch")
        shard_execution_mode = str(shard_manifest.get("execution_mode", "")).strip()
        if shard_execution_mode != EXPECTED_EXECUTION_MODE:
            raise SystemExit(
                f"{shard_manifest_path} execution_mode must be {EXPECTED_EXECUTION_MODE}"
            )
        if str(shard_manifest.get("canonical_profile", "")).strip() != canonical_profile:
            raise SystemExit(f"{shard_manifest_path} canonical_profile mismatch")
        shard_manifest_canonical = _resolve_manifest_reference(
            str(shard_manifest.get("canonical_manifest", "")).strip(),
            label=f"{shard_manifest_path} canonical_manifest",
        )
        if shard_manifest_canonical != canonical_manifest_path:
            raise SystemExit(f"{shard_manifest_path} canonical_manifest path mismatch")
        if str(shard_manifest.get("canonical_manifest_digest", "")).strip() != canonical_manifest_digest:
            raise SystemExit(f"{shard_manifest_path} canonical_manifest_digest mismatch")
        receipt_manifest_digest = str(dict(receipt.get("manifest_digests", {}) or {}).get("manifest", "")).strip()
        live_shard_manifest_digest = _sha256(shard_manifest_path)
        if receipt_manifest_digest != live_shard_manifest_digest:
            raise SystemExit(f"{receipt_path} shard manifest digest mismatch for {shard_profile}")
        shard_manifest_model_revisions = dict(
            shard_manifest.get("canonical_model_revisions") or shard_manifest.get("model_revisions") or {}
        )
        if shard_manifest_model_revisions != canonical_model_revisions:
            raise SystemExit(f"{shard_manifest_path} canonical model revisions mismatch")
        expected_shard_run_ids = [str(run_id).strip() for run_id in shard_manifest.get("shard_run_ids", []) if str(run_id).strip()]
        if not expected_shard_run_ids:
            expected_shard_run_ids = [
                str(run.get("run_id", "")).strip()
                for run in shard_manifest.get("runs", [])
                if isinstance(run, Mapping) and str(run.get("run_id", "")).strip()
            ]
        expected_shard_canonical_run_indices = [
            int(index)
            for index in shard_manifest.get("shard_canonical_run_indices", [])
        ]
        shard_count_values.add(int(shard_manifest.get("shard_count", 0) or 0))

        shard_runs = [run for run in shard_index.get("runs", []) if isinstance(run, Mapping)]
        declared_run_count = int(shard_index.get("run_count", len(shard_runs)) or len(shard_runs))
        if declared_run_count != len(shard_runs):
            raise SystemExit(f"{shard_index_path} run_count does not match the number of run records")
        if str(shard_index.get("manifest", "")).strip() != str(receipt.get("manifest", "")).strip():
            raise SystemExit(f"{shard_index_path} manifest/ref mismatch against {receipt_path}")
        gpu_pool_mode = str(shard_index.get("gpu_pool_mode", "")).strip() or EXPECTED_GPU_POOL_MODE
        gpu_pool_modes.add(gpu_pool_mode)
        actual_shard_run_ids = [str(run.get("run_id", "")).strip() for run in shard_runs]
        if actual_shard_run_ids != expected_shard_run_ids:
            raise SystemExit(f"{shard_index_path} run set does not match shard manifest membership for {shard_profile}")
        actual_shard_canonical_run_indices: list[int] = []
        for run_id in actual_shard_run_ids:
            if run_id not in canonical_run_index_by_id:
                raise SystemExit(f"{shard_index_path} contains unknown canonical run_id {run_id}")
            actual_shard_canonical_run_indices.append(int(canonical_run_index_by_id[run_id]))
        if expected_shard_canonical_run_indices and actual_shard_canonical_run_indices != expected_shard_canonical_run_indices:
            raise SystemExit(
                f"{shard_index_path} canonical run indices do not match shard manifest membership for {shard_profile}"
            )

        shard_output_dir = shard_index_path.parent
        for run in shard_runs:
            run_id = str(run.get("run_id", "")).strip()
            if not run_id:
                raise SystemExit(f"{shard_index_path} contains a run without run_id")
            status = str(run.get("status", "")).strip()
            reason = str(run.get("reason", "")).strip()
            if status not in {"success", "skipped"}:
                raise SystemExit(f"{shard_index_path} contains non-exportable status for {run_id}: {status}")
            if status == "skipped" and reason != "resume_existing_report":
                raise SystemExit(f"{shard_index_path} contains unsupported skipped run for {run_id}: {reason}")
            if run_id in merged_runs:
                raise SystemExit(f"duplicate run_id across shard indexes: {run_id}")
            normalized = _normalize_run_record(run, shard_output_dir=shard_output_dir)
            report_path = Path(str(normalized.get("report_path", "")).strip())
            if not report_path.exists():
                raise SystemExit(f"merged run {run_id} is missing a resolvable report_path")
            baseline_eval_value = str(normalized.get("baseline_eval_path", "")).strip()
            if baseline_eval_value and not Path(baseline_eval_value).exists():
                raise SystemExit(f"merged run {run_id} points at a missing baseline_eval_path")
            merged_runs[run_id] = normalized

    if len(shard_count_values) != 1:
        raise SystemExit("shard manifests do not agree on shard_count")
    expected_shard_count = shard_count_values.pop()
    if expected_shard_count <= 0:
        raise SystemExit("invalid shard_count metadata in shard manifests")
    if len(shard_profiles) != expected_shard_count:
        raise SystemExit(f"expected {expected_shard_count} shard indexes, found {len(shard_profiles)}")
    if len(receipt_payloads) != expected_shard_count:
        raise SystemExit(f"expected {expected_shard_count} host receipts, found {len(receipt_payloads)}")
    if gpu_pool_modes != {EXPECTED_GPU_POOL_MODE}:
        raise SystemExit(f"merged shard indexes must all use gpu_pool_mode={EXPECTED_GPU_POOL_MODE}")

    canonical_run_id_set = set(canonical_run_ids)
    merged_run_id_set = set(merged_runs)
    missing = sorted(canonical_run_id_set - merged_run_id_set)
    extra = sorted(merged_run_id_set - canonical_run_id_set)
    if missing:
        raise SystemExit(f"merged shard indexes are missing canonical runs: {missing[:8]}")
    if extra:
        raise SystemExit(f"merged shard indexes contain unexpected runs: {extra[:8]}")

    ordered_runs = [merged_runs[run_id] for run_id in canonical_run_ids]
    success_count = sum(1 for run in ordered_runs if str(run.get("status", "")).strip() == "success")
    skipped_count = sum(1 for run in ordered_runs if str(run.get("status", "")).strip() == "skipped")

    output_index = _resolve_payload_path(args.output_index, anchor=ROOT)
    canonical_release_index = (ROOT / "results" / "matrix" / "suite_all_models_methods" / "matrix_index.json").resolve(strict=False)
    if output_index == canonical_release_index:
        raise SystemExit(
            "merge_sharded_matrix.py is inspection-only and must not overwrite "
            "results/matrix/suite_all_models_methods/matrix_index.json. "
            "Use a reviewer-local inspection path for --output-index."
        )
    output_payload = {
        "schema_version": 1,
        "profile": canonical_profile,
        "manifest": _repo_relpath(canonical_manifest_path),
        "canonical_manifest_digest": canonical_manifest_digest,
        "canonical_model_revisions": canonical_model_revisions,
        "run_count": len(canonical_run_ids),
        "planned_run_count": len(canonical_run_ids),
        "completed_count": success_count + skipped_count,
        "success_count": success_count,
        "skipped_count": skipped_count,
        "running_count": 0,
        "failed_count": 0,
        "pending_count": 0,
        "stop_requested": False,
        "updated_at": time.time(),
        "gpu_pool_mode": EXPECTED_GPU_POOL_MODE,
        "execution_mode": EXPECTED_EXECUTION_MODE,
        "environment_fingerprint": next(iter(execution_environment_fingerprints)),
        "execution_environment_fingerprint": next(iter(execution_environment_fingerprints)),
        "code_snapshot_digest": next(iter(code_snapshot_digests)),
        "host_environment_fingerprints": host_environment_fingerprints,
        "shard_profiles": shard_profiles,
        "host_receipts": [str(path.resolve(strict=False)) for path, _ in receipt_payloads.values()],
        "runs": ordered_runs,
    }
    dump_json(output_index, output_payload)
    print(__import__("json").dumps(output_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
