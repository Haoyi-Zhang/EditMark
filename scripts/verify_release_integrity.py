from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPORT_IDENTITY = ROOT / "results" / "tables" / "suite_all_models_methods" / "suite_all_models_methods_export_identity.json"
FIGURE_DIR = ROOT / "results" / "figures" / "suite_all_models_methods"
TABLE_DIR = ROOT / "results" / "tables" / "suite_all_models_methods"
RUN_INVENTORY = TABLE_DIR / "suite_all_models_methods_run_inventory.csv"
REQUIRED_ENVIRONMENT_FILES = (
    "results/environment/runtime_environment.json",
    "results/environment/runtime_environment.md",
    "results/environment/release_pip_freeze.txt",
)
REQUIRED_FREEZE_ANCHORS = (
    "torch==2.6.0+cu124",
    "transformers==4.57.6",
    "numpy==2.2.6",
)
EXPECTED_HASHED_TABLES = (
    "method_summary.json",
    "suite_all_models_methods_method_master_leaderboard.json",
    "suite_all_models_methods_method_model_leaderboard.json",
    "suite_all_models_methods_model_method_functional_quality.json",
    "per_attack_robustness_breakdown.csv",
    "per_attack_robustness_breakdown.json",
    "core_vs_stress_robustness_summary.csv",
    "core_vs_stress_robustness_summary.json",
    "robustness_factor_decomposition.csv",
    "robustness_factor_decomposition.json",
    "utility_factor_decomposition.csv",
    "utility_factor_decomposition.json",
    "generalization_axis_breakdown.csv",
    "generalization_axis_breakdown.json",
    "gate_decomposition.csv",
    "gate_decomposition.json",
)
SOURCE_COUNTS = {
    "data/release/sources/suite_humaneval_plus_release.normalized.jsonl": 164,
    "data/release/sources/suite_mbpp_plus_release.normalized.jsonl": 378,
    "data/release/sources/suite_humanevalx_release.normalized.jsonl": 200,
    "data/release/sources/suite_mbxp_release.normalized.jsonl": 200,
    "data/release/sources/crafted_original_release.normalized.jsonl": 240,
    "data/release/sources/crafted_translation_release.normalized.jsonl": 240,
    "data/release/sources/crafted_stress_release.normalized.jsonl": 240,
}
SKIP_SCAN_PREFIXES = (
    ".git/",
    ".pytest_cache/",
    "_PROJECT_CONTEXT/",
    "_review_outputs/",
    "_remote_preview_figures/",
    "data/interim/",
    "data/downloads/",
    "external_checkout/",
    "model_cache/",
    "results/audits/",
    "results/certifications/",
    "results/matrix/",
    "results/matrix_shards/",
    "results/release_bundle/",
    "results/runs/",
    "results/tmp/",
)
SECRET_TERMS = (
    "github" + "_pat_",
    "gh" + "p_",
    "BEGIN " + "OPENSSH PRIVATE KEY",
    "BEGIN " + "RSA PRIVATE KEY",
    "private-cloud" + ".example.invalid",
    "example-" + "ssh-port",
    "example-" + "credential-fragment",
    "/private" + "/execution-root",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _table_hash_matches(path: Path, expected: str, name: str) -> tuple[bool, str]:
    raw = path.read_bytes()
    actual = _sha256_bytes(raw)
    if actual == expected:
        return True, actual
    lf_actual = _sha256_bytes(raw.replace(b"\r\n", b"\n"))
    if lf_actual == expected:
        print(
            f"warning: {name} matches after CRLF-to-LF normalization; "
            "ensure the public release tree uses LF line endings.",
            file=sys.stderr,
        )
        return True, actual
    return False, actual


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _check_manifest_digest(identity: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    manifest_rel = str(identity.get("manifest", "")).strip()
    expected = str(identity.get("canonical_manifest_digest", "")).strip().lower()
    if not manifest_rel or not expected:
        return ["export identity is missing manifest/canonical_manifest_digest"]
    manifest = ROOT / manifest_rel
    if not manifest.exists():
        return [f"canonical manifest is missing: {manifest_rel}"]
    raw = manifest.read_bytes()
    actual = _sha256_bytes(raw)
    if actual == expected:
        return []
    lf_actual = _sha256_bytes(raw.replace(b"\r\n", b"\n"))
    if lf_actual == expected:
        print(
            f"warning: {manifest_rel} matches the canonical digest after CRLF-to-LF normalization; "
            "ensure the public release tree uses LF line endings.",
            file=sys.stderr,
        )
        return []
    errors.append(f"{manifest_rel} digest mismatch: expected {expected}, got {actual}")
    return errors


def _check_export_hashes(identity: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_tables = dict(identity.get("required_table_hashes", {}))
    for name in EXPECTED_HASHED_TABLES:
        if name not in required_tables:
            errors.append(f"export identity is missing required table hash: {name}")
    for name, expected in sorted(dict(identity.get("required_figure_hashes", {})).items()):
        path = FIGURE_DIR / name
        if not path.exists():
            errors.append(f"missing figure artifact: {path}")
            continue
        actual = _sha256(path)
        if actual != str(expected).strip().lower():
            errors.append(f"figure hash mismatch for {name}: expected {expected}, got {actual}")
    for name, expected in sorted(required_tables.items()):
        path = TABLE_DIR / name
        if not path.exists():
            errors.append(f"missing table artifact: {path}")
            continue
        expected_hash = str(expected).strip().lower()
        matched, actual = _table_hash_matches(path, expected_hash, name)
        if not matched:
            errors.append(f"table hash mismatch for {name}: expected {expected}, got {actual}")
    return errors


def _check_run_inventory() -> list[str]:
    if not RUN_INVENTORY.exists():
        return [f"run inventory missing: {RUN_INVENTORY}"]
    with RUN_INVENTORY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    success = sum(1 for row in rows if row.get("status") == "success")
    errors: list[str] = []
    if len(rows) != 140:
        errors.append(f"run inventory row count must be 140, got {len(rows)}")
    if success != 140:
        errors.append(f"run inventory success count must be 140, got {success}")
    return errors


def _check_source_counts() -> list[str]:
    errors: list[str] = []
    for relpath, expected in SOURCE_COUNTS.items():
        path = ROOT / relpath
        if not path.exists():
            errors.append(f"release source missing: {relpath}")
            continue
        count = sum(1 for _ in path.open(encoding="utf-8"))
        if count != expected:
            errors.append(f"{relpath} must contain {expected} JSONL rows, got {count}")
    return errors


def _check_environment_files() -> list[str]:
    errors: list[str] = []
    for relpath in REQUIRED_ENVIRONMENT_FILES:
        path = ROOT / relpath
        if not path.exists():
            errors.append(f"release environment artifact missing: {relpath}")
    freeze_path = ROOT / "results/environment/release_pip_freeze.txt"
    if freeze_path.exists():
        freeze = freeze_path.read_text(encoding="utf-8", errors="ignore")
        for anchor in REQUIRED_FREEZE_ANCHORS:
            if anchor not in freeze:
                errors.append(f"release pip freeze is missing anchor: {anchor}")
    return errors


def _scan_public_surface() -> list[str]:
    errors: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(rel.startswith(prefix) for prefix in SKIP_SCAN_PREFIXES):
            continue
        if path.stat().st_size > 2_000_000:
            continue
        if path.suffix.lower() in {".png", ".pdf", ".pyc", ".zst", ".tar", ".gz", ".zip"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for term in SECRET_TERMS:
            if term in text:
                errors.append(f"sensitive-looking token marker {term!r} appears in {rel}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the anonymous public release surface.")
    parser.add_argument("--skip-secret-scan", action="store_true", help="Skip lightweight token-marker scanning.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    identity = _load_json(EXPORT_IDENTITY)
    errors: list[str] = []
    errors.extend(_check_manifest_digest(identity))
    errors.extend(_check_export_hashes(identity))
    errors.extend(_check_run_inventory())
    errors.extend(_check_source_counts())
    errors.extend(_check_environment_files())
    if not args.skip_secret_scan:
        errors.extend(_scan_public_surface())
    if errors:
        print("release_integrity=failed")
        for error in errors:
            print(f"- {error}")
        return 1
    print("release_integrity=passed")
    print("run_count=140")
    print("success_count=140")
    print(f"canonical_manifest_digest={identity.get('canonical_manifest_digest')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
