from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _python_bin(explicit: str | None) -> str:
    if explicit:
        return str(explicit)
    env_python = str(os.environ.get("PYTHON_BIN", "")).strip()
    if env_python:
        return env_python
    repo_python_candidates = [
        ROOT / ".venv" / "bin" / "python",
        ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    for repo_python in repo_python_candidates:
        if repo_python.exists():
            return str(repo_python)
    current_python = Path(sys.executable)
    active_venv = str(os.environ.get("VIRTUAL_ENV", "")).strip()
    if active_venv:
        try:
            current_python.relative_to(Path(active_venv).resolve())
            return str(current_python)
        except Exception:
            pass
    if any(token in current_python.parts for token in (".venv", "posteditbench_release", "posteditbench_release_clean")):
        return str(current_python)
    raise SystemExit(
        "Missing pinned Python interpreter. Set --python or PYTHON_BIN, activate a dedicated virtualenv, "
        f"or create {repo_python_candidates[0]}."
    )


def _workflow_log_path(command: str, *, profile: str | None, output_root: Path | None) -> Path | None:
    if command == "browse":
        return None
    if command == "regenerate":
        return ROOT / "results" / "audits" / "reviewer_regenerate.log"
    if command in {"subset", "full"} and output_root is not None and profile:
        return _matrix_root_for_profile(output_root, profile) / "workflow.log"
    return ROOT / "results" / "audits" / f"reviewer_{command}.log"


def _run(command: list[str], *, log_path: Path | None = None) -> None:
    rendered = " ".join(command)
    print("+ " + rendered, flush=True)
    if log_path is None:
        subprocess.run(command, cwd=ROOT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"+ {rendered}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def _print_command(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)


def _normalize_manifest_path(manifest: Path) -> Path:
    candidate = manifest if manifest.is_absolute() else (ROOT / manifest)
    return candidate.resolve(strict=False)


def _manifest_profile(manifest: Path) -> str:
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise SystemExit(f"Unable to parse manifest at {manifest}: {exc}") from exc
    profile = str(payload.get("profile", "")).strip()
    if not profile:
        raise SystemExit(f"Manifest at {manifest} is missing a top-level 'profile' field.")
    return profile


def _manifest_payload(manifest: Path) -> dict[str, object]:
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise SystemExit(f"Unable to parse manifest at {manifest}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Manifest at {manifest} must be a JSON object.")
    return payload


def _ensure_manifest_model_revisions(manifest: Path) -> dict[str, str]:
    payload = _manifest_payload(manifest)
    roster = [str(model).strip() for model in payload.get("model_roster", []) if str(model).strip()]
    revisions_raw = payload.get("model_revisions") or {}
    if not isinstance(revisions_raw, dict):
        raise SystemExit(f"Manifest at {manifest} is missing a valid top-level 'model_revisions' mapping.")
    revisions = {
        str(model).strip(): str(revision).strip()
        for model, revision in revisions_raw.items()
        if str(model).strip()
    }
    missing = [model for model in roster if not revisions.get(model)]
    if missing:
        raise SystemExit(
            f"Manifest at {manifest} is missing pinned model revisions for: {', '.join(sorted(missing))}."
        )
    return {model: revisions[model] for model in roster}


def _validate_manifest_profile_pair(manifest: Path, profile: str) -> Path:
    resolved = _normalize_manifest_path(manifest)
    if not resolved.exists():
        raise SystemExit(
            "Custom full-manifest runs require an existing manifest file. "
            "Generate it explicitly first, or use the canonical defaults."
        )
    manifest_profile = _manifest_profile(resolved)
    if manifest_profile != profile:
        raise SystemExit(
            f"Manifest/profile mismatch for full reviewer workflow: manifest profile is "
            f"'{manifest_profile}', but --profile was '{profile}'."
        )
    return resolved


def _default_matrix_index() -> Path:
    return ROOT / "results" / "matrix" / "suite_all_models_methods" / "matrix_index.json"


def _default_figure_dir() -> Path:
    return ROOT / "results" / "figures" / "suite_all_models_methods"


def _default_table_dir() -> Path:
    return ROOT / "results" / "tables" / "suite_all_models_methods"


def _default_subset_manifest(profile: str) -> Path:
    safe_profile = _validated_profile_name(profile)
    return ROOT / "results" / "matrix" / "subsets" / f"{safe_profile}.json"


def _matrix_root_for_profile(output_root: Path, profile: str) -> Path:
    return output_root / _validated_profile_name(profile)


def _matrix_index_for_profile(output_root: Path, profile: str) -> Path:
    return _matrix_root_for_profile(output_root, profile) / "matrix_index.json"


def _default_environment_json() -> Path:
    return ROOT / "results" / "environment" / "runtime_environment.json"


def _default_environment_md() -> Path:
    return ROOT / "results" / "environment" / "runtime_environment.md"


def _subset_environment_paths(profile: str, json_path: Path, md_path: Path) -> tuple[Path, Path]:
    default_json = _default_environment_json()
    default_md = _default_environment_md()
    if json_path == default_json and md_path == default_md:
        safe_profile = _validated_profile_name(profile or "subset")
        return (
            default_json.parent / f"{safe_profile}_runtime_environment.json",
            default_md.parent / f"{safe_profile}_runtime_environment.md",
        )
    return json_path, md_path


def _validated_profile_name(profile: str) -> str:
    text = str(profile or "").strip()
    candidate = Path(text)
    if not text:
        raise SystemExit("Profile names must be non-empty.")
    if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.parts[0] in {".", ".."}:
        raise SystemExit(
            "Profile names must be a single safe path segment without path traversal."
        )
    return candidate.parts[0]


def _add_common_matrix_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", type=str, default=None, help="Python interpreter to use for subcommands.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "matrix", help="Matrix output root.")
    parser.add_argument("--matrix-index", type=Path, default=_default_matrix_index(), help="Matrix index path for report regeneration.")
    parser.add_argument("--figure-dir", type=Path, default=_default_figure_dir(), help="Figure output directory.")
    parser.add_argument("--table-dir", type=Path, default=_default_table_dir(), help="Table output directory.")
    parser.add_argument("--environment-json", type=Path, default=_default_environment_json(), help="Machine-readable environment capture path.")
    parser.add_argument("--environment-md", type=Path, default=_default_environment_md(), help="Markdown environment capture path.")


def _add_fail_fast_arguments(parser: argparse.ArgumentParser, *, default: bool, noun: str) -> None:
    parser.set_defaults(fail_fast=default)
    parser.add_argument(
        "--fail-fast",
        dest="fail_fast",
        action="store_true",
        help=f"Stop the {noun} after the first failed run.",
    )
    parser.add_argument(
        "--no-fail-fast",
        dest="fail_fast",
        action="store_false",
        help=f"Let the {noun} continue and record per-run failures without terminating the whole matrix.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical reviewer workflow for the anonymous post-edit provenance artifact.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    browse = subparsers.add_parser("browse", help="Show the canonical reviewer-facing summary files.")
    _add_common_matrix_arguments(browse)
    browse.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print the shipped summary paths without additional notes.",
    )

    regenerate = subparsers.add_parser("regenerate", help="Regenerate figures and tables from an existing matrix index.")
    _add_common_matrix_arguments(regenerate)
    regenerate.add_argument(
        "--require-times-new-roman",
        dest="require_times_new_roman",
        action="store_true",
        default=True,
        help="Fail closed if Times New Roman is unavailable.",
    )
    regenerate.add_argument(
        "--allow-font-fallback",
        dest="require_times_new_roman",
        action="store_false",
        help="Allow serif fallback fonts instead of failing on missing Times New Roman.",
    )

    subset = subparsers.add_parser("subset", help="Build and run a reviewer subset manifest.")
    _add_common_matrix_arguments(subset)
    subset.add_argument("--models", type=str, default="", help="Comma-separated model filter.")
    subset.add_argument("--methods", type=str, default="", help="Comma-separated method filter.")
    subset.add_argument(
        "--sources",
        "--benchmark-source",
        dest="sources",
        type=str,
        default="",
        help="Comma-separated source filter. `--benchmark-source` is accepted as a reviewer-friendly alias.",
    )
    subset.add_argument("--limit", type=int, default=None, help="Optional micro-smoke limit for the selected source slice.")
    subset.add_argument("--profile", type=str, default="suite_reviewer_subset", help="Subset manifest profile name.")
    subset.add_argument("--output-manifest", type=Path, default=None, help="Custom subset manifest path.")
    subset.add_argument("--no-run", dest="run", action="store_false", help="Only build the subset manifest.")
    subset.add_argument("--dry-run", action="store_true", help="Print the commands without executing them.")
    subset.add_argument("--resume", action="store_true", help="Resume a previously interrupted subset matrix when possible.")
    subset.add_argument("--gpu-slots", type=int, default=4, help="Total visible GPU slots to schedule across.")
    subset.add_argument("--gpu-pool-mode", choices=("split", "shared"), default="shared", help="GPU scheduling mode.")
    subset.add_argument("--cpu-workers", type=int, default=6, help="CPU worker count.")
    subset.add_argument("--retry-count", type=int, default=1, help="Run retry count.")
    _add_fail_fast_arguments(subset, default=True, noun="reviewer subset")
    subset.add_argument(
        "--probe-hf-access",
        action="store_true",
        help="Probe token-backed Hugging Face access in the readiness gate instead of cache-only validation.",
    )
    subset.set_defaults(run=True)

    full = subparsers.add_parser(
        "full",
        help="Run the formal single-host direct-full helper that reproduces the release-facing Linux 8-GPU contract.",
    )
    _add_common_matrix_arguments(full)
    full.add_argument("--manifest", type=Path, default=ROOT / "configs" / "matrices" / "suite_all_models_methods.json", help="Full-suite manifest path.")
    full.add_argument("--profile", type=str, default="suite_all_models_methods", help="Full-suite profile name.")
    full.add_argument("--dry-run", action="store_true", help="Print the commands without executing them.")
    full.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previously interrupted non-canonical full matrix. Not supported for the formal single-host release path.",
    )
    full.add_argument("--gpu-slots", type=int, default=8, help="Total visible GPU slots to schedule across.")
    full.add_argument("--gpu-pool-mode", choices=("split", "shared"), default="shared", help="GPU scheduling mode.")
    full.add_argument("--cpu-workers", type=int, default=9, help="CPU worker count.")
    full.add_argument("--retry-count", type=int, default=1, help="Run retry count.")
    full.add_argument(
        "--command-timeout-seconds",
        type=int,
        default=259200,
        help="Per-run command timeout for the formal full-suite helper.",
    )
    _add_fail_fast_arguments(full, default=False, noun="full suite")
    full.add_argument(
        "--probe-hf-access",
        action="store_true",
        help="Probe token-backed Hugging Face access in the readiness gate instead of cache-only validation.",
    )

    return parser


def _relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _status_label(path: Path, *, expect_rerun_backed: bool = False) -> str:
    if path.exists():
        if expect_rerun_backed and path.is_dir():
            materialized_entries = [
                child.name
                for child in path.iterdir()
                if child.name not in {".gitkeep", "__pycache__"}
            ]
            readme_only_entries = {"README.md", "README.txt", "README"}
            if not materialized_entries or set(materialized_entries).issubset(readme_only_entries):
                return "not materialized"
        return "present"
    if expect_rerun_backed:
        return "not materialized"
    return "missing"


def _manifest_path_from_matrix_index(matrix_index: Path) -> Path | None:
    if not matrix_index.exists():
        return None
    try:
        payload = _manifest_payload(matrix_index)
    except SystemExit:
        return None
    manifest = str(payload.get("manifest", "")).strip()
    if not manifest:
        return None
    return _normalize_manifest_path(Path(manifest))


def _browse_model_revision_surface(matrix_index: Path) -> tuple[str, dict[str, str] | None]:
    manifest_path = _manifest_path_from_matrix_index(matrix_index)
    if manifest_path is None or not manifest_path.exists():
        manifest_path = ROOT / "configs" / "matrices" / "suite_all_models_methods.json"
    if manifest_path.exists():
        return "Pinned model revisions:", _ensure_manifest_model_revisions(manifest_path)
    return "Pinned model revisions: canonical manifest unavailable", None


def _gap_check_line(table_dir: Path) -> str | None:
    table = table_dir / "model_method_summary.csv"
    if not table.exists():
        return None
    gaps: list[float] = []
    with table.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                detection = float(row["detection_separability"])
                robustness = float(row["robustness"])
            except (KeyError, TypeError, ValueError):
                continue
            gaps.append(detection - robustness)
    if not gaps:
        return None
    mean_gap = sum(gaps) / len(gaps)
    loo_means = [
        (sum(gaps) - gap) / (len(gaps) - 1)
        for gap in gaps
    ] if len(gaps) > 1 else [mean_gap]
    return (
        "Main gap check: rounded 20-slice mean detection-minus-robustness "
        f"= {mean_gap:.4f} across {len(gaps)} method-by-generator slices "
        "(paper reports 0.3220 from the unrounded frozen analysis; "
        "leave-one-out mean range "
        f"{min(loo_means):.4f} to {max(loo_means):.4f}; slice counts are descriptive support, not separate tests; "
        "see artifact/headline_gap_statistics.*)."
    )


def _print_browse(summary_only: bool, matrix_index: Path, figure_dir: Path, table_dir: Path) -> None:
    revision_label, model_revisions = _browse_model_revision_surface(matrix_index)
    print("Post-edit provenance reviewer browse path", flush=True)
    print(f"Matrix index: {matrix_index} [{_status_label(matrix_index, expect_rerun_backed=True)}]", flush=True)
    print(f"Figures: {figure_dir} [{_status_label(figure_dir, expect_rerun_backed=True)}]", flush=True)
    print(f"Tables: {table_dir} [{_status_label(table_dir, expect_rerun_backed=True)}]", flush=True)
    gap_line = _gap_check_line(table_dir)
    if gap_line is not None:
        print(gap_line, flush=True)
    print(revision_label, flush=True)
    if model_revisions is not None:
        for model_name, revision in model_revisions.items():
            print(f"  - {model_name} @ {revision}", flush=True)
    if not summary_only:
        print("Suggested review files:", flush=True)
        review_paths = (
            (ROOT / "results" / "tables" / "dataset_statistics" / "benchmark_definition_summary.csv", False),
            (ROOT / "results" / "tables" / "dataset_statistics" / "release_slice_language_breakdown.csv", False),
            (ROOT / "results" / "tables" / "dataset_statistics" / "dataset_task_category_breakdown.csv", False),
            (ROOT / "results" / "tables" / "dataset_statistics" / "dataset_family_breakdown.csv", False),
            (ROOT / "results" / "tables" / "dataset_statistics" / "release_source_manifest_index.csv", False),
            (ROOT / "results" / "figures" / "dataset_statistics" / "release_slice_composition.png", False),
            (ROOT / "results" / "figures" / "dataset_statistics" / "evaluation_dimensions_overview.png", False),
            (figure_dir / "suite_all_models_methods_score_decomposition.png", True),
            (figure_dir / "suite_all_models_methods_detection_vs_utility.png", True),
            (table_dir / "suite_all_models_methods_method_master_leaderboard.csv", True),
            (table_dir / "suite_all_models_methods_method_model_leaderboard.csv", True),
            (table_dir / "method_language_summary.csv", True),
            (table_dir / "suite_all_models_methods_model_method_functional_quality.csv", True),
            (table_dir / "suite_all_models_methods_utility_robustness_summary.csv", True),
            (table_dir / "suite_all_models_methods_model_method_timing.csv", True),
        )
        for relpath, expect_rerun_backed in review_paths:
            print(
                f"  - {_relpath(relpath)} [{_status_label(relpath, expect_rerun_backed=expect_rerun_backed)}]",
                flush=True,
            )
        print(
            "Note: the publication-facing review packet is rooted in the canonical suite_all_models_methods figure and table directories. "
            "The tracked figure roster is intentionally narrow: score decomposition, detection-vs-utility, release-slice composition, and one conceptual evaluation-overview panel, while exact-value leaderboard and breakdown evidence stays table-first. "
            "If those exports are missing in your checkout, materialize or restore the canonical matrix index first and then run the canonical regenerate path.",
            flush=True,
        )


def _ensure_matrix_index_exists(matrix_index: Path) -> None:
    if matrix_index.exists():
        return
    raise SystemExit(
        "Matrix index not found for regeneration. Restore a raw matrix tree under results/matrix/, or wait until result-backed outputs are materialized before using the regenerate path."
    )


def _load_matrix_index_payload(matrix_index: Path) -> dict[str, object]:
    try:
        payload = json.loads(matrix_index.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Unable to parse matrix index at {matrix_index}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Matrix index at {matrix_index} must be a JSON object.")
    return payload


def _require_release_facing_matrix_status(matrix_index: Path) -> None:
    payload = _load_matrix_index_payload(matrix_index)
    run_count = int(payload.get("run_count", 0) or 0)
    success_count = int(payload.get("success_count", 0) or 0)
    failed_count = int(payload.get("failed_count", 0) or 0)
    if run_count != 140 or success_count != 140 or failed_count != 0:
        raise SystemExit(
            "Canonical suite_all_models_methods publication-facing paths require a completed matrix with "
            f"run_count=140, success_count=140, failed_count=0; observed "
            f"run_count={run_count}, success_count={success_count}, failed_count={failed_count}."
        )


def _ensure_regenerate_output_contract(matrix_index: Path, figure_dir: Path, table_dir: Path) -> None:
    canonical_matrix_index = _default_matrix_index().resolve(strict=False)
    canonical_figure_dir = _default_figure_dir().resolve(strict=False)
    canonical_table_dir = _default_table_dir().resolve(strict=False)
    requested_matrix_index = matrix_index.resolve(strict=False)
    requested_figure_dir = figure_dir.resolve(strict=False)
    requested_table_dir = table_dir.resolve(strict=False)
    using_canonical_named_outputs = (
        requested_figure_dir == canonical_figure_dir
        or requested_table_dir == canonical_table_dir
    )
    if using_canonical_named_outputs and requested_matrix_index != canonical_matrix_index:
        raise SystemExit(
            "Canonical suite_all_models_methods output paths are reserved for the canonical full-suite matrix index. "
            "For a custom matrix, pass custom --figure-dir and --table-dir."
        )
    if using_canonical_named_outputs and requested_matrix_index == canonical_matrix_index:
        _require_release_facing_matrix_status(requested_matrix_index)
    if using_canonical_named_outputs:
        payload = _manifest_payload(requested_matrix_index)
        manifest = str(payload.get("manifest", "")).strip()
        profile = str(payload.get("profile", "")).strip()
        execution_mode = str(payload.get("execution_mode", "")).strip()
        if manifest != "configs/matrices/suite_all_models_methods.json" or profile != "suite_all_models_methods":
            raise SystemExit(
                "Canonical suite_all_models_methods output paths require a matrix index produced from "
                "configs/matrices/suite_all_models_methods.json with profile suite_all_models_methods."
            )
        if execution_mode != "single_host_canonical":
            raise SystemExit(
                "Canonical suite_all_models_methods output paths require execution_mode=single_host_canonical. "
                "Sharded/inspection-only matrices must use custom --figure-dir and --table-dir."
            )


def _capture_environment(
    python_bin: str,
    json_path: Path,
    md_path: Path,
    label: str,
    *,
    execution_mode: str = "single_host_canonical",
    log_path: Path | None = None,
) -> None:
    _run(
        [
            python_bin,
            str(ROOT / "scripts" / "capture_environment.py"),
            "--label",
            label,
            "--execution-mode",
            execution_mode,
            "--output-json",
            str(json_path),
            "--output-md",
            str(md_path),
            "--public-safe-paths",
        ],
        log_path=log_path,
    )


def _audit_benchmarks_command(
    python_bin: str,
    *,
    manifest: Path | None = None,
    profile: str = "suite",
    matrix_profile: str | None = None,
) -> list[str]:
    command = [python_bin, str(ROOT / "scripts" / "audit_benchmarks.py")]
    if manifest is None:
        command.extend(["--profile", profile])
        return command
    command.extend(["--manifest", str(manifest)])
    effective_matrix_profile = str(matrix_profile or profile).strip()
    if effective_matrix_profile:
        command.extend(["--matrix-profile", effective_matrix_profile])
    command.extend(["--profile", profile])
    return command


def _audit_full_matrix_command(python_bin: str, manifest: Path, profile: str, *, probe_hf_access: bool) -> list[str]:
    command = [
        python_bin,
        str(ROOT / "scripts" / "audit_full_matrix.py"),
        "--manifest",
        str(manifest),
        "--profile",
        profile,
        "--strict-hf-cache",
        "--model-load-smoke",
        "--runtime-smoke",
        "--skip-provider-credentials",
    ]
    if not probe_hf_access:
        command.append("--skip-hf-access")
    return command


def _run_full_matrix_command(
    python_bin: str,
    *,
    manifest: Path,
    profile: str,
    output_root: Path,
    gpu_slots: int,
    gpu_pool_mode: str,
    cpu_workers: int,
    retry_count: int,
    resume: bool,
    fail_fast: bool,
    command_timeout_seconds: int | None = None,
    allow_resume: bool = False,
) -> list[str]:
    if resume and not allow_resume:
        raise SystemExit(
            "The formal single-host full helper is a one-shot rerun contract and does not support --resume. "
            "Use a custom profile/output root for engineering continuations instead."
        )
    command = [
        python_bin,
        str(ROOT / "scripts" / "run_full_matrix.py"),
        "--manifest",
        str(manifest),
        "--profile",
        profile,
        "--output-root",
        str(output_root),
        "--gpu-slots",
        str(int(gpu_slots)),
        "--gpu-pool-mode",
        gpu_pool_mode,
        "--cpu-workers",
        str(int(cpu_workers)),
        "--retry-count",
        str(int(retry_count)),
    ]
    if command_timeout_seconds is not None:
        command.extend(["--command-timeout-seconds", str(int(command_timeout_seconds))])
    if fail_fast:
        command.append("--fail-fast")
    if resume:
        command.append("--resume")
    return command


def _print_execution_artifacts(
    *,
    python_bin: str,
    manifest: Path,
    profile: str,
    output_root: Path,
    workflow_log: Path | None = None,
) -> None:
    matrix_root = _matrix_root_for_profile(output_root, profile)
    matrix_index = _matrix_index_for_profile(output_root, profile)
    model_revisions = _ensure_manifest_model_revisions(manifest) if manifest.exists() else None
    monitor_command = [
        python_bin,
        str(ROOT / "scripts" / "monitor_matrix.py"),
        "--matrix-index",
        str(matrix_index),
        "--watch-seconds",
        "5",
    ]
    print("Execution artifacts:", flush=True)
    print(f"Manifest: {_relpath(manifest)}", flush=True)
    if model_revisions is None:
        print("Pinned model revisions: pending manifest materialization", flush=True)
    else:
        print("Pinned model revisions:", flush=True)
        for model_name, revision in model_revisions.items():
            print(f"  - {model_name} @ {revision}", flush=True)
    print(f"Matrix index: {_relpath(matrix_index)}", flush=True)
    print(f"Run outputs: {_relpath(matrix_root)}", flush=True)
    if workflow_log is not None:
        print(f"Workflow log: {_relpath(workflow_log)}", flush=True)
    print(f"Per-run reports: {_relpath(matrix_root / '*' / 'report.json')}", flush=True)
    print(f"Per-run logs: {_relpath(matrix_root / '*' / 'run.log')}", flush=True)
    print("Monitor command:", flush=True)
    _print_command(monitor_command)


def _regenerate_outputs(
    python_bin: str,
    matrix_index: Path,
    figure_dir: Path,
    table_dir: Path,
    require_times_new_roman: bool,
    *,
    log_path: Path | None = None,
) -> None:
    _ensure_matrix_index_exists(matrix_index)
    _ensure_regenerate_output_contract(matrix_index, figure_dir, table_dir)
    _run([python_bin, str(ROOT / "scripts" / "refresh_report_metadata.py"), "--matrix-index", str(matrix_index)], log_path=log_path)
    _run([python_bin, str(ROOT / "scripts" / "export_full_run_tables.py"), "--matrix-index", str(matrix_index), "--output-dir", str(table_dir)], log_path=log_path)
    redraw_command = [
        python_bin,
        str(ROOT / "scripts" / "render_materialized_summary_figures.py"),
        "--table-dir",
        str(table_dir),
        "--output-dir",
        str(figure_dir),
        "--export-identity",
        str(table_dir / "suite_all_models_methods_export_identity.json"),
        "--prefix",
        "suite_all_models_methods",
    ]
    if require_times_new_roman:
        redraw_command.append("--require-times-new-roman")
    else:
        redraw_command.append("--allow-font-fallback")
    _run(redraw_command, log_path=log_path)
    _run([python_bin, str(ROOT / "scripts" / "export_dataset_statistics.py")], log_path=log_path)


def _canonical_full_selection(manifest: Path, profile: str) -> bool:
    default_manifest = (ROOT / "configs" / "matrices" / "suite_all_models_methods.json").resolve(strict=False)
    default_profile = "suite_all_models_methods"
    return _normalize_manifest_path(manifest) == default_manifest and profile == default_profile


def _use_remote_full_wrapper(manifest: Path, profile: str) -> bool:
    return _canonical_full_selection(manifest, profile)


def _canonical_remote_full_command(
    python_bin: str,
    *,
    output_root: Path,
    gpu_slots: int,
    gpu_pool_mode: str,
    cpu_workers: int,
    retry_count: int,
    command_timeout_seconds: int,
    resume: bool,
    fail_fast: bool,
    probe_hf_access: bool,
) -> list[str]:
    if resume:
        raise SystemExit(
            "The formal single-host full helper is a one-shot rerun contract and does not support --resume. "
            "Use a custom profile/output root for engineering continuations instead."
        )
    command = [
        "bash",
        str(ROOT / "scripts" / "remote" / "run_formal_single_host_full.sh"),
        "--python",
        python_bin,
        "--output-root",
        str(output_root),
        "--gpu-slots",
        str(int(gpu_slots)),
        "--gpu-pool-mode",
        gpu_pool_mode,
        "--cpu-workers",
        str(int(cpu_workers)),
        "--retry-count",
        str(int(retry_count)),
        "--command-timeout-seconds",
        str(int(command_timeout_seconds)),
    ]
    if fail_fast:
        command.append("--fail-fast")
    if not probe_hf_access:
        command.append("--skip-hf-access")
    return command


def _require_canonical_remote_scheduler_contract(
    *,
    gpu_slots: int,
    gpu_pool_mode: str,
    cpu_workers: int,
    retry_count: int,
    command_timeout_seconds: int,
) -> None:
    expected = {
        "gpu_slots": 8,
        "gpu_pool_mode": "shared",
        "cpu_workers": 9,
        "retry_count": 1,
        "command_timeout_seconds": 259200,
    }
    observed = {
        "gpu_slots": int(gpu_slots),
        "gpu_pool_mode": str(gpu_pool_mode),
        "cpu_workers": int(cpu_workers),
        "retry_count": int(retry_count),
        "command_timeout_seconds": int(command_timeout_seconds),
    }
    if observed != expected:
        raise SystemExit(
            "This helper only covers the formal single-host remote full-run workflow, which is fixed to "
            "--gpu-slots 8 --gpu-pool-mode shared --cpu-workers 9 --retry-count 1 --command-timeout-seconds 259200. "
            "Use the sharded identical-execution-class path only when you explicitly want the optional two-host reproduction or throughput mode, "
            "or use a non-canonical custom path if you need different scheduler settings."
        )


def _require_canonical_full_linux_host(*, allow_dry_run: bool = False) -> None:
    if allow_dry_run:
        return
    if platform.system() != "Linux":
        raise SystemExit(
            "The formal single-host full-suite helper reproduces the Linux 8-GPU contract used by the canonical result of record. "
            "Run it on an equivalent Linux cloud server, or use a non-canonical custom full-run path with a custom profile/output root."
        )


def _ensure_full_output_contract(manifest: Path, profile: str, output_root: Path) -> None:
    canonical_output_root = (ROOT / "results" / "matrix").resolve(strict=False)
    requested_output_root = output_root.resolve(strict=False)
    if _canonical_full_selection(manifest, profile):
        return
    if profile == "suite_all_models_methods" and requested_output_root == canonical_output_root:
        raise SystemExit(
            "The canonical suite_all_models_methods profile under results/matrix/ is reserved for the formal release-facing full run. "
            "Use a custom --profile and/or --output-root for non-canonical full runs."
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workflow_log = _workflow_log_path(
        args.command,
        profile=getattr(args, "profile", None),
        output_root=getattr(args, "output_root", None),
    )

    if args.command == "browse":
        _print_browse(args.summary_only, args.matrix_index, args.figure_dir, args.table_dir)
        return 0

    if args.command == "regenerate":
        _ensure_matrix_index_exists(args.matrix_index)
        python_bin = _python_bin(getattr(args, "python", None))
        print(f"[reviewer_workflow.py] Using Python interpreter: {python_bin}", flush=True)
        _regenerate_outputs(
            python_bin,
            args.matrix_index,
            args.figure_dir,
            args.table_dir,
            bool(args.require_times_new_roman),
            log_path=workflow_log,
        )
        _print_browse(False, args.matrix_index, args.figure_dir, args.table_dir)
        return 0

    python_bin = _python_bin(getattr(args, "python", None))
    print(f"[reviewer_workflow.py] Using Python interpreter: {python_bin}", flush=True)

    if args.command == "subset":
        output_manifest = args.output_manifest or _default_subset_manifest(args.profile)
        build_command = [
            python_bin,
            str(ROOT / "scripts" / "build_suite_manifests.py"),
            "--output-manifest",
            str(output_manifest),
            "--profile",
            args.profile,
            "--skip-refresh-prepared-inputs",
        ]
        if args.models:
            build_command.extend(["--models", args.models])
        if args.methods:
            build_command.extend(["--methods", args.methods])
        if args.sources:
            build_command.extend(["--sources", args.sources])
        if args.limit is not None:
            build_command.extend(["--limit", str(int(args.limit))])
        audit_benchmarks_command = _audit_benchmarks_command(
            python_bin,
            manifest=output_manifest,
            profile=args.profile,
            matrix_profile=args.profile,
        )
        audit_matrix_command = _audit_full_matrix_command(
            python_bin,
            output_manifest,
            args.profile,
            probe_hf_access=bool(args.probe_hf_access),
        )
        run_matrix_command = _run_full_matrix_command(
            python_bin,
            manifest=output_manifest,
            profile=args.profile,
            output_root=args.output_root,
            gpu_slots=args.gpu_slots,
            gpu_pool_mode=args.gpu_pool_mode,
            cpu_workers=args.cpu_workers,
            retry_count=args.retry_count,
            resume=bool(args.resume),
            fail_fast=bool(args.fail_fast),
            allow_resume=True,
        )
        if args.dry_run:
            _print_command(build_command)
            if args.run:
                _print_command(audit_benchmarks_command)
                _print_command(audit_matrix_command)
            if args.run:
                _print_command(
                    [
                        python_bin,
                        str(ROOT / "scripts" / "capture_environment.py"),
                        "--label",
                        f"{args.profile}-subset",
                        "--execution-mode",
                        "single_host_canonical",
                        "--output-json",
                        str(_subset_environment_paths(args.profile, args.environment_json, args.environment_md)[0]),
                        "--output-md",
                        str(_subset_environment_paths(args.profile, args.environment_json, args.environment_md)[1]),
                        "--public-safe-paths",
                    ]
                )
                _print_command(run_matrix_command)
                _print_execution_artifacts(
                    python_bin=python_bin,
                    manifest=output_manifest,
                    profile=args.profile,
                    output_root=args.output_root,
                    workflow_log=workflow_log,
                )
            else:
                print("Build-only subset dry-run: execution and environment-capture commands are intentionally omitted.", flush=True)
            return 0
        _run(build_command, log_path=workflow_log)
        if args.run:
            environment_json, environment_md = _subset_environment_paths(args.profile, args.environment_json, args.environment_md)
            if output_manifest.exists():
                _ensure_manifest_model_revisions(output_manifest)
            _run(audit_benchmarks_command, log_path=workflow_log)
            _run(audit_matrix_command, log_path=workflow_log)
            _print_execution_artifacts(
                python_bin=python_bin,
                manifest=output_manifest,
                profile=args.profile,
                output_root=args.output_root,
                workflow_log=workflow_log,
            )
            _capture_environment(
                python_bin,
                environment_json,
                environment_md,
                f"{args.profile}-subset",
                execution_mode="single_host_canonical",
                log_path=workflow_log,
            )
            _run(run_matrix_command, log_path=workflow_log)
        return 0

    if args.command == "full":
        resolved_manifest = _normalize_manifest_path(args.manifest)
        _ensure_full_output_contract(resolved_manifest, args.profile, args.output_root)
        use_remote_full_wrapper = _use_remote_full_wrapper(resolved_manifest, args.profile)
        if _canonical_full_selection(resolved_manifest, args.profile):
            _require_canonical_full_linux_host(allow_dry_run=bool(args.dry_run))
            build_command = None
        else:
            resolved_manifest = _validate_manifest_profile_pair(resolved_manifest, args.profile)
            build_command = None
        if use_remote_full_wrapper:
            _require_canonical_remote_scheduler_contract(
                gpu_slots=args.gpu_slots,
                gpu_pool_mode=args.gpu_pool_mode,
                cpu_workers=args.cpu_workers,
                retry_count=args.retry_count,
                command_timeout_seconds=args.command_timeout_seconds,
            )
            audit_benchmarks_command = None
            audit_matrix_command = None
            run_matrix_command = _canonical_remote_full_command(
                python_bin,
                output_root=args.output_root,
                gpu_slots=args.gpu_slots,
                gpu_pool_mode=args.gpu_pool_mode,
                cpu_workers=args.cpu_workers,
                retry_count=args.retry_count,
                command_timeout_seconds=args.command_timeout_seconds,
                resume=bool(args.resume),
                fail_fast=bool(args.fail_fast),
                probe_hf_access=bool(args.probe_hf_access),
            )
        else:
            audit_benchmarks_command = _audit_benchmarks_command(
                python_bin,
                manifest=resolved_manifest,
                profile=args.profile,
                matrix_profile=args.profile,
            )
            audit_matrix_command = _audit_full_matrix_command(
                python_bin,
                resolved_manifest,
                args.profile,
                probe_hf_access=bool(args.probe_hf_access),
            )
            run_matrix_command = _run_full_matrix_command(
                python_bin,
                manifest=resolved_manifest,
                profile=args.profile,
                output_root=args.output_root,
                gpu_slots=args.gpu_slots,
                gpu_pool_mode=args.gpu_pool_mode,
                cpu_workers=args.cpu_workers,
                retry_count=args.retry_count,
                command_timeout_seconds=args.command_timeout_seconds,
                resume=bool(args.resume),
                fail_fast=bool(args.fail_fast),
            )
        if args.dry_run:
            if build_command is not None:
                _print_command(build_command)
            if audit_benchmarks_command is not None:
                _print_command(audit_benchmarks_command)
            if audit_matrix_command is not None:
                _print_command(audit_matrix_command)
                _print_command(
                    [
                        python_bin,
                        str(ROOT / "scripts" / "capture_environment.py"),
                        "--label",
                        f"{args.profile}-full",
                        "--execution-mode",
                        "single_host_canonical",
                        "--output-json",
                        str(args.environment_json),
                        "--output-md",
                        str(args.environment_md),
                        "--public-safe-paths",
                    ]
                )
            _print_command(run_matrix_command)
            _print_execution_artifacts(
                python_bin=python_bin,
                manifest=resolved_manifest,
                profile=args.profile,
                output_root=args.output_root,
                workflow_log=workflow_log,
            )
            return 0
        if build_command is not None:
            _run(build_command, log_path=workflow_log)
        _ensure_manifest_model_revisions(resolved_manifest)
        if audit_benchmarks_command is not None and audit_matrix_command is not None:
            _run(audit_benchmarks_command, log_path=workflow_log)
            _run(audit_matrix_command, log_path=workflow_log)
            _print_execution_artifacts(
                python_bin=python_bin,
                manifest=resolved_manifest,
                profile=args.profile,
                output_root=args.output_root,
                workflow_log=workflow_log,
            )
            _capture_environment(
                python_bin,
                args.environment_json,
                args.environment_md,
                f"{args.profile}-full",
                execution_mode="single_host_canonical",
                log_path=workflow_log,
            )
        else:
            _print_execution_artifacts(
                python_bin=python_bin,
                manifest=resolved_manifest,
                profile=args.profile,
                output_root=args.output_root,
                workflow_log=workflow_log,
            )
        _run(run_matrix_command, log_path=workflow_log)
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
