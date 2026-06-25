from __future__ import annotations

import argparse
import hashlib
import shutil
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
PUBLICATION_EXECUTION_MODE = "single_host_canonical"
KNOWN_SOURCE_EXECUTION_MODES = {"single_host_canonical"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble one same-host convergence index from multiple successful result segments "
            "that share the same canonical manifest contract. This helper is for recovery or "
            "inspection workflows, not for relabeling segmented history as the formal one-shot "
            "release result."
        )
    )
    parser.add_argument("--manifest", type=Path, default=CANONICAL_MANIFEST)
    parser.add_argument("--profile", type=str, default=CANONICAL_PROFILE)
    parser.add_argument("--input-index", action="append", type=Path, required=True)
    parser.add_argument(
        "--output-index",
        type=Path,
        default=Path("results/matrix/suite_all_models_methods/matrix_index.json"),
    )
    parser.add_argument(
        "--publication-execution-mode",
        type=str,
        default=PUBLICATION_EXECUTION_MODE,
        choices=("single_host_canonical",),
        help="Publication-facing execution mode written into the assembled canonical index.",
    )
    parser.add_argument(
        "--stage-external-runs-under",
        type=Path,
        default=None,
        help=(
            "Optional repo-relative directory used to copy in any run directories that currently live "
            "outside the repository root before writing the assembled canonical index."
        ),
    )
    parser.add_argument(
        "--code-snapshot-digest",
        type=str,
        default=None,
        help=(
            "Explicit publication-facing code snapshot digest to write into the assembled canonical index "
            "when the source indexes do not already carry one shared digest."
        ),
    )
    parser.add_argument(
        "--execution-environment-fingerprint",
        type=str,
        default=None,
        help=(
            "Explicit publication-facing execution-environment fingerprint to write into the assembled canonical "
            "index when the source indexes do not already carry one shared fingerprint."
        ),
    )
    return parser.parse_args()


def _load_payload(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a top-level JSON object")
    return payload


def _resolve_payload_path(value: str | Path, *, anchor: Path = ROOT) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    return (anchor / candidate).resolve(strict=False)


def _repo_relpath(path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


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


def _fallback_artifact_path(*, field: str, source_output_dir: Path, run_id: str) -> Path:
    run_dir = source_output_dir / run_id
    mapping = {
        "output_dir": run_dir,
        "report_path": run_dir / "report.json",
        "baseline_eval_path": run_dir / "baseline_eval.json",
        "log_path": run_dir / "run.log",
        "resolved_config_path": run_dir / "_resolved_config.yaml",
    }
    return mapping[field]


def _stage_external_run_dir(*, source_run_dir: Path, stage_root: Path, run_id: str) -> Path:
    if not source_run_dir.exists():
        raise SystemExit(f"cannot stage external run {run_id}: missing source directory {source_run_dir}")
    staged_dir = stage_root / run_id
    if not staged_dir.exists():
        staged_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_run_dir, staged_dir)
    return staged_dir


def _normalize_run_record(
    run: Mapping[str, Any],
    *,
    source_output_dir: Path,
    stage_external_runs_under: Path | None,
) -> dict[str, Any]:
    payload = dict(run)
    run_id = str(payload.get("run_id", "")).strip()
    resolved_paths: dict[str, Path | None] = {}
    for field in ("output_dir", "report_path", "baseline_eval_path", "log_path", "resolved_config_path"):
        raw_value = str(payload.get(field, "")).strip()
        resolved: Path | None = None
        if raw_value:
            candidate = _resolve_payload_path(raw_value, anchor=ROOT)
            if candidate.exists():
                resolved = candidate
        if resolved is None:
            fallback = _fallback_artifact_path(field=field, source_output_dir=source_output_dir, run_id=run_id)
            if fallback.exists():
                resolved = fallback
        resolved_paths[field] = resolved

    output_dir = resolved_paths.get("output_dir")
    if output_dir is not None:
        try:
            output_dir.relative_to(ROOT)
        except ValueError:
            if stage_external_runs_under is None:
                raise SystemExit(
                    f"{run_id} resolves outside the repository root at {output_dir}. "
                    "Re-run with --stage-external-runs-under to stage external run directories into the repo tree."
                )
            staged_output_dir = _stage_external_run_dir(
                source_run_dir=output_dir,
                stage_root=stage_external_runs_under,
                run_id=run_id,
            )
            resolved_paths["output_dir"] = staged_output_dir
            resolved_paths["report_path"] = staged_output_dir / "report.json"
            if resolved_paths.get("baseline_eval_path") is not None:
                resolved_paths["baseline_eval_path"] = staged_output_dir / "baseline_eval.json"
            resolved_paths["log_path"] = staged_output_dir / "run.log"
            resolved_paths["resolved_config_path"] = staged_output_dir / "_resolved_config.yaml"

    for field in ("output_dir", "report_path", "baseline_eval_path", "log_path", "resolved_config_path"):
        resolved = resolved_paths.get(field)
        payload[field] = _repo_relpath(resolved) if resolved is not None else str(payload.get(field, "")).strip()
    return payload


def _validate_source_index(
    source_index_path: Path,
    payload: Mapping[str, Any],
    *,
    canonical_manifest_path: Path,
    canonical_manifest_digest: str,
    canonical_profile: str,
    canonical_model_revisions: Mapping[str, Any],
    canonical_run_index_by_id: Mapping[str, int],
    stage_external_runs_under: Path | None,
 ) -> tuple[str, str, str, str, str, list[dict[str, Any]]]:
    profile = str(payload.get("profile", "")).strip()
    if not profile:
        raise SystemExit(f"{source_index_path} is missing profile")
    source_execution_mode = str(payload.get("execution_mode", "")).strip()
    if not source_execution_mode:
        raise SystemExit(f"{source_index_path} is missing execution_mode")
    if source_execution_mode not in KNOWN_SOURCE_EXECUTION_MODES:
        raise SystemExit(
            f"{source_index_path} execution_mode must be one of {sorted(KNOWN_SOURCE_EXECUTION_MODES)}; "
            f"observed {source_execution_mode or '<missing>'}."
        )
    run_records = [run for run in payload.get("runs", []) if isinstance(run, Mapping)]
    declared_run_count = int(payload.get("run_count", len(run_records)) or len(run_records))
    if declared_run_count != len(run_records):
        raise SystemExit(f"{source_index_path} run_count does not match the number of run records")
    if int(payload.get("failed_count", 0) or 0) != 0:
        raise SystemExit(f"{source_index_path} contains failed runs and is not publication-assembly-safe")
    if int(payload.get("running_count", 0) or 0) != 0:
        raise SystemExit(f"{source_index_path} still reports running_count > 0")
    if int(payload.get("pending_count", 0) or 0) != 0:
        raise SystemExit(f"{source_index_path} still reports pending_count > 0")
    observed_success = int(payload.get("success_count", 0) or 0)
    observed_skipped = int(payload.get("skipped_count", 0) or 0)
    if observed_success + observed_skipped != len(run_records):
        raise SystemExit(
            f"{source_index_path} success/skipped counts do not cover all run records "
            f"({observed_success}+{observed_skipped}!={len(run_records)})."
        )
    source_canonical_manifest = str(payload.get("canonical_manifest", "")).strip()
    if source_canonical_manifest:
        resolved_canonical_manifest = _resolve_payload_path(source_canonical_manifest, anchor=ROOT)
        if resolved_canonical_manifest != canonical_manifest_path:
            raise SystemExit(f"{source_index_path} canonical_manifest mismatch")
    source_canonical_manifest_digest = str(payload.get("canonical_manifest_digest", "")).strip()
    if source_canonical_manifest_digest and source_canonical_manifest_digest != canonical_manifest_digest:
        raise SystemExit(f"{source_index_path} canonical_manifest_digest mismatch")
    source_model_revisions = dict(payload.get("canonical_model_revisions") or payload.get("model_revisions") or {})
    if source_model_revisions and source_model_revisions != canonical_model_revisions:
        raise SystemExit(f"{source_index_path} canonical_model_revisions mismatch")
    source_code_snapshot_digest = str(payload.get("code_snapshot_digest", "")).strip()
    source_execution_environment_fingerprint = str(payload.get("execution_environment_fingerprint", "")).strip()

    source_output_dir = source_index_path.parent
    normalized_runs: list[dict[str, Any]] = []
    actual_run_ids: list[str] = []
    for run in run_records:
        run_id = str(run.get("run_id", "")).strip()
        if not run_id:
            raise SystemExit(f"{source_index_path} contains a run without run_id")
        status = str(run.get("status", "")).strip()
        reason = str(run.get("reason", "")).strip()
        if status not in {"success", "skipped"}:
            raise SystemExit(f"{source_index_path} contains non-exportable status for {run_id}: {status}")
        if status == "skipped" and reason != "resume_existing_report":
            raise SystemExit(f"{source_index_path} contains unsupported skipped run for {run_id}: {reason}")
        if run_id not in canonical_run_index_by_id:
            raise SystemExit(f"{source_index_path} contains unknown canonical run_id {run_id}")
        normalized = _normalize_run_record(
            run,
            source_output_dir=source_output_dir,
            stage_external_runs_under=stage_external_runs_under,
        )
        report_path = Path(str(normalized.get("report_path", "")).strip())
        if not report_path.is_absolute():
            report_path = _resolve_payload_path(report_path, anchor=ROOT)
        if not report_path.exists():
            raise SystemExit(f"{source_index_path} merged run {run_id} is missing a resolvable report_path")
        baseline_eval_value = str(normalized.get("baseline_eval_path", "")).strip()
        if baseline_eval_value:
            baseline_eval_path = Path(baseline_eval_value)
            if not baseline_eval_path.is_absolute():
                baseline_eval_path = _resolve_payload_path(baseline_eval_path, anchor=ROOT)
            if not baseline_eval_path.exists():
                raise SystemExit(f"{source_index_path} merged run {run_id} points at a missing baseline_eval_path")
        actual_run_ids.append(run_id)
        normalized_runs.append(normalized)
    return (
        profile,
        source_execution_mode,
        str(payload.get("manifest", "")).strip(),
        source_code_snapshot_digest,
        source_execution_environment_fingerprint,
        normalized_runs,
    )


def _resolve_publication_identity(
    *,
    field_label: str,
    source_values: set[str],
    explicit_value: str,
) -> str:
    normalized_explicit_value = explicit_value.strip()
    normalized_source_values = {value.strip() for value in source_values if value.strip()}
    if normalized_explicit_value:
        if normalized_source_values and normalized_source_values != {normalized_explicit_value}:
            raise SystemExit(
                f"explicit {field_label}={normalized_explicit_value} disagrees with source-index values "
                f"{sorted(normalized_source_values)}"
            )
        return normalized_explicit_value
    if len(normalized_source_values) == 1:
        return next(iter(normalized_source_values))
    if not normalized_source_values:
        raise SystemExit(
            f"assembled publication index is missing {field_label}; pass --{field_label.replace('_', '-')} "
            "or use source indexes that already carry one shared value."
        )
    raise SystemExit(
        f"assembled publication index does not have one shared {field_label}: {sorted(normalized_source_values)}"
    )


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
    stage_external_runs_under = (
        _resolve_payload_path(args.stage_external_runs_under, anchor=ROOT)
        if args.stage_external_runs_under is not None
        else None
    )
    if not canonical_run_ids:
        raise SystemExit(f"{canonical_manifest_path} does not contain any runs for profile {canonical_profile}")
    canonical_run_index_by_id = {run_id: index for index, run_id in enumerate(canonical_run_ids)}
    output_index = _resolve_payload_path(args.output_index, anchor=ROOT)
    canonical_release_output = _resolve_payload_path(
        Path("results/matrix/suite_all_models_methods/matrix_index.json"),
        anchor=ROOT,
    )
    if len(args.input_index) != 1 and output_index == canonical_release_output:
        raise SystemExit(
            "formal canonical release path is reserved for the one-shot run_full_matrix result; "
            "assembled same-host convergence indexes must use a custom output path"
        )

    merged_runs: dict[str, dict[str, Any]] = {}
    source_execution_modes: list[str] = []
    source_indexes: list[dict[str, Any]] = []
    gpu_pool_modes: set[str] = set()
    source_code_snapshot_digests: set[str] = set()
    source_execution_environment_fingerprints: set[str] = set()

    for input_index_arg in args.input_index:
        source_index_path = _resolve_payload_path(input_index_arg, anchor=ROOT)
        payload = _load_payload(source_index_path)
        (
            profile,
            source_execution_mode,
            manifest_ref,
            source_code_snapshot_digest,
            source_execution_environment_fingerprint,
            normalized_runs,
        ) = _validate_source_index(
            source_index_path,
            payload,
            canonical_manifest_path=canonical_manifest_path,
            canonical_manifest_digest=canonical_manifest_digest,
            canonical_profile=canonical_profile,
            canonical_model_revisions=canonical_model_revisions,
            canonical_run_index_by_id=canonical_run_index_by_id,
            stage_external_runs_under=stage_external_runs_under,
        )
        source_execution_modes.append(source_execution_mode)
        gpu_pool_mode = str(payload.get("gpu_pool_mode", "")).strip()
        if gpu_pool_mode:
            gpu_pool_modes.add(gpu_pool_mode)
        if source_code_snapshot_digest:
            source_code_snapshot_digests.add(source_code_snapshot_digest)
        if source_execution_environment_fingerprint:
            source_execution_environment_fingerprints.add(source_execution_environment_fingerprint)
        for normalized in normalized_runs:
            run_id = str(normalized.get("run_id", "")).strip()
            if run_id in merged_runs:
                raise SystemExit(f"duplicate run_id across source indexes: {run_id}")
            merged_runs[run_id] = normalized
        source_indexes.append(
            {
                "path": str(source_index_path.resolve(strict=False)),
                "profile": profile,
                "manifest": manifest_ref,
                "execution_mode": source_execution_mode,
                "run_count": len(normalized_runs),
            }
        )

    canonical_run_id_set = set(canonical_run_ids)
    merged_run_id_set = set(merged_runs)
    missing = sorted(canonical_run_id_set - merged_run_id_set)
    extra = sorted(merged_run_id_set - canonical_run_id_set)
    if missing:
        raise SystemExit(f"assembled publication index is missing canonical runs: {missing[:8]}")
    if extra:
        raise SystemExit(f"assembled publication index contains unexpected runs: {extra[:8]}")
    if len(gpu_pool_modes) > 1:
        raise SystemExit(f"source indexes disagree on gpu_pool_mode: {sorted(gpu_pool_modes)}")

    ordered_runs = [merged_runs[run_id] for run_id in canonical_run_ids]
    success_count = sum(1 for run in ordered_runs if str(run.get("status", "")).strip() == "success")
    skipped_count = sum(1 for run in ordered_runs if str(run.get("status", "")).strip() == "skipped")
    publication_gpu_pool_mode = "shared" if not gpu_pool_modes else sorted(gpu_pool_modes)[0]
    publication_code_snapshot_digest = _resolve_publication_identity(
        field_label="code_snapshot_digest",
        source_values=source_code_snapshot_digests,
        explicit_value=str(args.code_snapshot_digest or ""),
    )
    publication_execution_environment_fingerprint = _resolve_publication_identity(
        field_label="execution_environment_fingerprint",
        source_values=source_execution_environment_fingerprints,
        explicit_value=str(args.execution_environment_fingerprint or ""),
    )

    output_payload = {
        "schema_version": 1,
        "profile": canonical_profile,
        "manifest": _repo_relpath(canonical_manifest_path),
        "canonical_manifest": _repo_relpath(canonical_manifest_path),
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
        "gpu_pool_mode": publication_gpu_pool_mode,
        "execution_mode": args.publication_execution_mode,
        "code_snapshot_digest": publication_code_snapshot_digest,
        "execution_environment_fingerprint": publication_execution_environment_fingerprint,
        "environment_fingerprint": publication_execution_environment_fingerprint,
        "assembly_source_indexes": source_indexes,
        "assembly_source_execution_modes": sorted(set(source_execution_modes)),
        "runs": ordered_runs,
    }
    if stage_external_runs_under is not None:
        output_payload["staged_external_runs_under"] = _repo_relpath(stage_external_runs_under)
    dump_json(output_index, output_payload)
    print(__import__("json").dumps(output_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
