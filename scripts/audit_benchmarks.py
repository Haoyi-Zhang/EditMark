from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from _shared import DEFAULT_INTERIM_DIR, PROJECT_ROOT, load_json, read_jsonl, resolve_prepared_benchmark_config
except ModuleNotFoundError:  # pragma: no cover - import style depends on entrypoint
    from scripts._shared import DEFAULT_INTERIM_DIR, PROJECT_ROOT, load_json, read_jsonl, resolve_prepared_benchmark_config

from posteditbench.suite import SUITE_AGGREGATE_SOURCES, SUITE_INVENTORY_SOURCES


MISSING_FIELD_SENTINELS = {"", "local_checkout", "unspecified", "unknown"}
MULTILINGUAL_LANGUAGES = {"python", "cpp", "java", "javascript", "go"}


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def _profile_inputs(profile: str) -> list[Path]:
    normalized = str(profile or "suite").strip().lower() or "suite"
    if normalized in {"suite", "suite_all_models_methods", "suite_canary_heavy", "model_invocation_smoke"}:
        candidates = [ROOT / source.prepared_output for source in SUITE_AGGREGATE_SOURCES]
        return _dedupe_paths([path.resolve() for path in candidates if path.exists()])
    if normalized == "suite_inventory":
        candidates = [ROOT / source.prepared_output for source in SUITE_INVENTORY_SOURCES]
        return _dedupe_paths([path.resolve() for path in candidates if path.exists()])
    raise ValueError(f"unsupported benchmark audit profile: {profile}")


def _matrix_inputs(manifest_path: Path, *, profile: str) -> list[Path]:
    payload = load_json(manifest_path)
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        raise TypeError(f"{manifest_path} must contain a list-valued runs field")
    resolved: list[Path] = []
    seen: set[Path] = set()
    for item in runs:
        if not isinstance(item, dict):
            continue
        item_profile = str(item.get("profile", payload.get("profile", ""))).strip()
        if item_profile and item_profile != profile:
            continue
        overrides = item.get("config_overrides", {})
        prepared_override = ""
        if isinstance(overrides, dict):
            paths_override = dict(overrides.get("paths", {}))
            benchmark_override = dict(overrides.get("benchmark", {}))
            prepared_override = str(
                paths_override.get("prepared_benchmark")
                or benchmark_override.get("prepared_output")
                or benchmark_override.get("source")
                or ""
            ).strip()
        if prepared_override:
            prepared_path = Path(prepared_override)
            if not prepared_path.is_absolute():
                prepared_path = ROOT / prepared_path
            prepared_path = prepared_path.resolve()
        else:
            config_value = str(item.get("config", "")).strip()
            if not config_value:
                continue
            prepared_path = Path(resolve_prepared_benchmark_config(Path(config_value), root=ROOT)["prepared_path"]).resolve()
        if prepared_path in seen:
            continue
        seen.add(prepared_path)
        resolved.append(prepared_path)
    return resolved


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest for {path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"manifest for {path} must be a JSON object")
    return payload


def _resolve_release_path(value: str) -> Path:
    candidate = Path(str(value).replace("\\", "/"))
    if candidate.is_absolute():
        return candidate
    direct = PROJECT_ROOT / candidate
    if direct.exists():
        return direct
    workspace_relative = PROJECT_ROOT.parent / candidate
    if workspace_relative.exists():
        return workspace_relative
    return direct


def _record_count_from_manifest(manifest: dict[str, Any]) -> int | None:
    if isinstance(manifest.get("record_count"), int):
        return int(manifest["record_count"])
    if isinstance(manifest.get("task_count"), int):
        return int(manifest["task_count"])
    counts = manifest.get("counts")
    if isinstance(counts, dict) and isinstance(counts.get("observed"), int):
        return int(counts["observed"])
    return None


def _normalize_language_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("language", "")).strip().lower() for row in rows if str(row.get("language", "")).strip())
    return dict(sorted(counts.items()))


def _normalize_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field, "")).strip().lower() for row in rows if str(row.get(field, "")).strip())
    return dict(sorted(counts.items()))


def _required_manifest_fields(path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    baseline_fields = ("schema_version", "reference_kind_counts")
    for field in baseline_fields:
        if manifest.get(field) in (None, {}, []):
            failures.append(f"manifest missing {field}")

    is_collection = "collection_name" in manifest
    benchmark_name = str(manifest.get("benchmark", "")).strip().lower()
    is_public = bool(benchmark_name and manifest.get("source_url")) and not is_collection
    is_crafted = benchmark_name.startswith("crafted_") and not is_collection

    if is_public:
        for field in ("source_url", "source_revision", "source_archive_sha256", "sample_ids_path"):
            value = str(manifest.get(field, "")).strip()
            if not value or value.lower() in MISSING_FIELD_SENTINELS:
                failures.append(f"public manifest missing hardened field {field}")
        if not isinstance(manifest.get("source_manifests"), list) or not manifest.get("source_manifests"):
            failures.append("public manifest missing source_manifests")
        else:
            for index, item in enumerate(manifest["source_manifests"]):
                source_revision = str(item.get("source_revision", "")).strip()
                source_sha256 = str(item.get("source_sha256", "")).strip()
                if not source_revision or source_revision.lower() in MISSING_FIELD_SENTINELS:
                    failures.append(f"source_manifests[{index}] missing source_revision")
                if not source_sha256:
                    failures.append(f"source_manifests[{index}] missing source_sha256")
    if is_collection:
        for field in ("language_counts", "source_group_counts", "origin_type_counts", "family_count"):
            if manifest.get(field) in (None, {}, []):
                failures.append(f"collection manifest missing {field}")
        if not isinstance(manifest.get("source_manifests"), list) or not manifest.get("source_manifests"):
            failures.append("collection manifest missing source_manifests")
        else:
            for index, item in enumerate(manifest["source_manifests"]):
                if item.get("source_url") and not str(item.get("source_revision", "")).strip():
                    failures.append(f"collection source_manifests[{index}] missing source_revision")
                if item.get("source_url") and not (str(item.get("source_sha256", "")).strip() or str(item.get("source_archive_sha256", "")).strip()):
                    failures.append(f"collection source_manifests[{index}] missing source checksum")
    if is_crafted:
        for field in ("category_counts", "template_family_counts", "task_count_per_family", "family_count"):
            if manifest.get(field) in (None, {}, []):
                failures.append(f"crafted manifest missing {field}")
    return failures


def _audit_duplicates(rows: list[dict[str, Any]], *, allow_cross_source_duplicates: bool) -> list[str]:
    failures: list[str] = []
    if allow_cross_source_duplicates:
        identifiers = [
            (
                str(row.get("source_group", "")).strip().lower(),
                str(row.get("task_id", "")).strip(),
            )
            for row in rows
            if str(row.get("task_id", "")).strip()
        ]
        duplicate_ids = [identifier for identifier, count in Counter(identifiers).items() if count > 1]
    else:
        task_ids = [str(row.get("task_id", "")).strip() for row in rows if str(row.get("task_id", "")).strip()]
        duplicate_ids = [task_id for task_id, count in Counter(task_ids).items() if count > 1]
    if duplicate_ids:
        failures.append(f"duplicate task_id values detected: {duplicate_ids[:5]}")

    if allow_cross_source_duplicates:
        return failures
    digest_sources: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        digest = str(row.get("source_digest", "")).strip()
        source_group = str(row.get("source_group", "")).strip().lower()
        if digest and source_group:
            digest_sources[digest].add(source_group)
    leaking = [digest for digest, source_groups in digest_sources.items() if len(source_groups) > 1]
    if leaking:
        failures.append(f"source leakage detected across source_group boundaries: {leaking[:5]}")
    return failures


def _audit_crafted_rows(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    family_languages: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family_id = str(row.get("family_id", "")).strip()
        language = str(row.get("language", "")).strip().lower()
        if family_id and language:
            family_languages[family_id].add(language)
        for field in ("semantic_contract", "difficulty", "category", "validation_backend", "reference_kind", "family_id"):
            if row.get(field) in (None, "", []):
                failures.append(f"crafted row missing {field}: {row.get('task_id', 'unknown')}")
                break
    uncovered = sorted(family_id for family_id, languages in family_languages.items() if languages != MULTILINGUAL_LANGUAGES)
    if uncovered:
        failures.append(f"crafted family coverage is incomplete: {uncovered[:5]}")
    if int(manifest.get("family_count", -1)) != len(family_languages):
        failures.append(
            f"crafted family_count mismatch: manifest={manifest.get('family_count')} observed={len(family_languages)}"
        )
    expected_task_count = int(manifest.get("task_count_per_family", 0))
    if expected_task_count and expected_task_count != len(MULTILINGUAL_LANGUAGES):
        failures.append(f"crafted task_count_per_family should be {len(MULTILINGUAL_LANGUAGES)}")
    difficulty_counts = {key for key, value in _normalize_counts(rows, "difficulty").items() if value > 0}
    if difficulty_counts != {"easy", "medium", "hard"}:
        failures.append(f"crafted difficulty coverage is incomplete: {sorted(difficulty_counts)}")
    return failures


def _audit_runtime_coverage(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    coverage = dict(manifest.get("coverage", {}))
    if not coverage:
        return failures
    basis = str(coverage.get("runtime_validation_basis", "")).strip().lower()
    if basis == "unavailable":
        for field in (
            "runtime_semantic_validation_rate",
            "runtime_semantic_validation_language_rate",
            "semantic_validation_rate",
            "semantic_validation_language_rate",
            "clean_reference_compile_rate",
            "clean_reference_pass_rate",
        ):
            if coverage.get(field) is not None:
                failures.append(f"runtime-unavailable collection must leave coverage.{field} = null")
        if coverage.get("runtime_validation_annotations_available") not in {False, None}:
            failures.append("runtime-unavailable collection must set runtime_validation_annotations_available=false")
    elif basis == "row_annotations":
        for field in ("runtime_semantic_validation_rate", "semantic_validation_rate"):
            if coverage.get(field) is None:
                failures.append(f"row-annotated collection is missing coverage.{field}")
        if coverage.get("runtime_validation_annotations_available") is False:
            failures.append("row-annotated collection cannot set runtime_validation_annotations_available=false")
    return failures


def _audit_collections(path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    language_counts = _normalize_language_counts(rows)
    manifest_language_counts = {str(key).lower(): int(value) for key, value in dict(manifest.get("language_counts", {})).items()}
    if manifest_language_counts and manifest_language_counts != language_counts:
        failures.append(f"collection language_counts mismatch for {path.name}")
    family_count = len({str(row.get("family_id", "")).strip() for row in rows if str(row.get("family_id", "")).strip()})
    if int(manifest.get("family_count", -1)) != family_count:
        failures.append(f"collection family_count mismatch: manifest={manifest.get('family_count')} observed={family_count}")

    if manifest.get("collection_name") in {
        "public_multilingual_core",
        "crafted_multilingual_core",
        "unified_multilingual_balanced_full",
        "unified_multilingual_full",
    }:
        if set(language_counts) != MULTILINGUAL_LANGUAGES:
            failures.append(f"{manifest.get('collection_name')} is missing one of the required 5 languages")

    balance_metadata = manifest.get("language_balance")
    if (
        manifest.get("collection_name") == "unified_multilingual_balanced_full"
        or (isinstance(balance_metadata, dict) and balance_metadata.get("applied") is True)
    ):
        if language_counts and len(set(language_counts.values())) != 1:
            failures.append(f"{manifest.get('collection_name')} is not language balanced: {language_counts}")

    failures.extend(_audit_runtime_coverage(manifest))
    return failures


def _audit_public_manifest(path: Path, manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    sample_ids_path = _resolve_release_path(str(manifest.get("sample_ids_path", "")).strip())
    if not sample_ids_path.exists():
        failures.append(f"sample_ids_path missing on disk: {sample_ids_path}")
    return failures


def _validation_rate_threshold(path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> float:
    if path.name == "benchmark.normalized.jsonl":
        return 0.1
    if manifest.get("collection_name") == "python_controls":
        return 1.0
    if any(str(row.get("record_kind", "")).strip() in {"public_benchmark", "crafted_benchmark"} for row in rows):
        return 0.8
    return 0.5


def audit_one(path: Path) -> dict[str, Any]:
    resolved_path = path.resolve()
    rows = read_jsonl(path)
    manifest = _load_manifest(path)
    failures: list[str] = []
    is_collection = "collection_name" in manifest
    benchmark_name = str(manifest.get("benchmark", "")).strip().lower()
    is_public_source = bool(benchmark_name and manifest.get("source_url")) and not is_collection
    is_crafted_source = benchmark_name.startswith("crafted_") and not is_collection

    manifest_count = _record_count_from_manifest(manifest)
    if manifest_count is None:
        failures.append("manifest missing record/task count")
    elif manifest_count != len(rows):
        failures.append(f"manifest count mismatch: manifest={manifest_count} observed={len(rows)}")

    observed_languages = sorted(_normalize_language_counts(rows))
    manifest_observed_languages = sorted(str(item).strip().lower() for item in manifest.get("observed_languages", []) if str(item).strip())
    if manifest_observed_languages and manifest_observed_languages != observed_languages:
        failures.append(
            f"manifest observed_languages mismatch: manifest={manifest_observed_languages} observed={observed_languages}"
        )

    validation_rate = sum(1 for row in rows if bool(row.get("validation_supported"))) / max(1, len(rows))
    if validation_rate < _validation_rate_threshold(path, rows, manifest):
        failures.append(f"validation_supported rate too low: {validation_rate:.3f}")

    failures.extend(_required_manifest_fields(path, rows, manifest))
    failures.extend(_audit_duplicates(rows, allow_cross_source_duplicates=is_collection))

    if is_crafted_source:
        failures.extend(_audit_crafted_rows(rows, manifest))
    if is_collection:
        failures.extend(_audit_collections(path, rows, manifest))
    if is_public_source:
        failures.extend(_audit_public_manifest(path, manifest))

    try:
        display_path = str(resolved_path.relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        display_path = str(path).replace("\\", "/")

    return {
        "path": display_path,
        "record_count": len(rows),
        "observed_languages": observed_languages,
        "validation_supported_rate": round(validation_rate, 4),
        "status": "ok" if not failures else "failed",
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit normalized benchmark snapshots and collections.")
    parser.add_argument("--input", action="append", default=None, help="Normalized JSONL paths to audit.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional matrix manifest to derive benchmark inputs from.")
    parser.add_argument("--matrix-profile", type=str, default=None, help="Profile within --manifest to select runs from.")
    parser.add_argument("--profile", default="suite", help="Audit profile to run when no --input or --manifest is supplied.")
    parser.add_argument("--output", type=Path, default=DEFAULT_INTERIM_DIR / "audit" / "benchmark_audit.json", help="Optional JSON report output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input:
        inputs = [Path(item).resolve() for item in args.input]
    elif args.manifest is not None:
        matrix_profile = str(args.matrix_profile or args.profile).strip()
        inputs = _matrix_inputs(Path(args.manifest).resolve(), profile=matrix_profile)
    else:
        inputs = _profile_inputs(args.profile)

    report = {"profile": args.profile, "audits": [], "status": "ok"}
    for path in inputs:
        if not path.exists():
            try:
                display_path = str(path.relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
            except ValueError:
                display_path = str(path).replace("\\", "/")
            report["audits"].append({"path": display_path, "status": "failed", "failures": ["missing file"]})
            report["status"] = "failed"
            continue
        result = audit_one(path)
        report["audits"].append(result)
        if result["status"] != "ok":
            report["status"] = "failed"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
