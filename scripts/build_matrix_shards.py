from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from _shared import dump_json
except ModuleNotFoundError:  # pragma: no cover
    from scripts._shared import dump_json

from posteditbench.suite import (
    OFFICIAL_RUNTIME_BASELINES,
    SUITE_ATOMIC_SOURCE_ORDER,
    SUITE_MODEL_ROSTER,
    suite_model_revision,
)

CANONICAL_PROFILE = "suite_all_models_methods"
CANONICAL_MANIFEST = Path("configs/matrices/suite_all_models_methods.json")
DEFAULT_SHARD_COUNT = 5

_MODEL_PRIORITY_RANK = {name: len(SUITE_MODEL_ROSTER) - index for index, name in enumerate(SUITE_MODEL_ROSTER)}
_METHOD_PRIORITY_RANK = {
    name: len(OFFICIAL_RUNTIME_BASELINES) - index for index, name in enumerate(OFFICIAL_RUNTIME_BASELINES)
}
_SOURCE_PRIORITY_RANK = {name: len(SUITE_ATOMIC_SOURCE_ORDER) - index for index, name in enumerate(SUITE_ATOMIC_SOURCE_ORDER)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split the canonical suite manifest into weighted shard manifests.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CANONICAL_MANIFEST,
        help="Canonical suite manifest to shard.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=CANONICAL_PROFILE,
        help="Canonical profile name expected in the input manifest.",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=DEFAULT_SHARD_COUNT,
        help="Number of shard manifests to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "matrix_shards" / CANONICAL_PROFILE,
        help="Directory where shard manifests should be written.",
    )
    return parser.parse_args()


def _read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _canonical_manifest_digest(
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None = None,
) -> str:
    if manifest_path is not None:
        return hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return path.as_posix()


def _normalized_profile(profile: str) -> str:
    return str(profile).strip()


def _expected_priority(run: dict[str, Any]) -> int:
    priority = int(run.get("priority", 0) or 0)
    if priority:
        return priority
    model = str(run.get("model", "")).strip()
    method = str(run.get("method", "")).strip()
    source_slug = str(run.get("source_slug", "")).strip()
    benchmark = dict(run.get("config_overrides", {}).get("benchmark", {}) or {})
    benchmark_limit = int(benchmark.get("limit", 0) or 0)
    model_rank = int(_MODEL_PRIORITY_RANK.get(model, 0))
    method_rank = int(_METHOD_PRIORITY_RANK.get(method, 0))
    source_rank = int(_SOURCE_PRIORITY_RANK.get(source_slug, 0))
    capped_limit = max(0, min(benchmark_limit, 99_999))
    return model_rank * 1_000_000_000 + method_rank * 10_000_000 + source_rank * 100_000 + capped_limit


def _run_weight(run: dict[str, Any], *, manifest_index: int) -> tuple[int, int, str]:
    return (
        -int(_expected_priority(run)),
        int(manifest_index),
        str(run.get("run_id", "")).strip(),
    )


def _shard_profile(base_profile: str, shard_index: int, shard_count: int) -> str:
    width = max(2, len(str(int(shard_count))))
    return f"{base_profile}_shard_{int(shard_index):0{width}d}_of_{int(shard_count):0{width}d}"


def _validate_canonical_manifest(manifest: dict[str, Any], *, profile: str, manifest_path: Path) -> None:
    actual_profile = _normalized_profile(str(manifest.get("profile", "")))
    expected_profile = _normalized_profile(profile)
    if actual_profile != expected_profile:
        raise ValueError(
            f"{manifest_path} profile mismatch: expected '{expected_profile}', found '{actual_profile}'"
        )
    if expected_profile != CANONICAL_PROFILE:
        raise ValueError(
            f"only the canonical profile '{CANONICAL_PROFILE}' is supported for matrix sharding"
        )
    if _normalized_profile(str(manifest.get("profile", ""))) != CANONICAL_PROFILE:
        raise ValueError(f"{manifest_path} must be the canonical full-suite manifest for profile '{CANONICAL_PROFILE}'")
    if any(key in manifest for key in ("canonical_profile", "canonical_manifest_digest", "shard_profile", "shard_index")):
        raise ValueError(f"{manifest_path} already appears to be a sharded manifest and cannot be re-sharded")
    expected_model_revisions = {model: suite_model_revision(model) for model in SUITE_MODEL_ROSTER}
    if dict(manifest.get("model_revisions", {})) != expected_model_revisions:
        raise ValueError(f"{manifest_path} model revisions do not match the canonical suite roster")


def build_matrix_shards(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    profile: str = CANONICAL_PROFILE,
    shard_count: int = DEFAULT_SHARD_COUNT,
) -> list[dict[str, Any]]:
    _validate_canonical_manifest(manifest, profile=profile, manifest_path=manifest_path)
    if int(shard_count) <= 0:
        raise ValueError("shard_count must be positive")

    canonical_manifest = copy.deepcopy(manifest)
    canonical_runs = [dict(run) for run in canonical_manifest.get("runs", []) if str(run.get("profile", "")).strip() == profile]
    if not canonical_runs:
        raise ValueError(f"{manifest_path} does not contain any runs for profile '{profile}'")
    if len(canonical_runs) != len(canonical_manifest.get("runs", [])):
        raise ValueError(f"{manifest_path} contains runs that do not belong to profile '{profile}'")

    manifest_digest = _canonical_manifest_digest(canonical_manifest, manifest_path=manifest_path)
    canonical_model_revisions = dict(canonical_manifest.get("model_revisions", {}))
    indexed_runs = list(enumerate(canonical_runs))
    ordered_runs = sorted(indexed_runs, key=lambda item: _run_weight(item[1], manifest_index=item[0]))

    shard_buckets: list[list[dict[str, Any]]] = [[] for _ in range(int(shard_count))]
    shard_totals = [0 for _ in range(int(shard_count))]
    shard_run_counts = [0 for _ in range(int(shard_count))]

    for manifest_index, run in ordered_runs:
        shard_index = min(
            range(int(shard_count)),
            key=lambda idx: (shard_totals[idx], shard_run_counts[idx], idx),
        )
        cost = int(_expected_priority(run))
        payload = copy.deepcopy(run)
        shard_profile = _shard_profile(profile, shard_index + 1, int(shard_count))
        payload["profile"] = shard_profile
        payload["canonical_profile"] = profile
        payload["canonical_manifest"] = _relpath(manifest_path)
        payload["canonical_manifest_digest"] = manifest_digest
        payload["canonical_manifest_digest_algorithm"] = "sha256"
        payload["shard_profile"] = shard_profile
        payload["shard_index"] = shard_index + 1
        payload["shard_count"] = int(shard_count)
        payload["canonical_run_index"] = int(manifest_index)
        payload["shard_run_cost"] = cost
        shard_buckets[shard_index].append(payload)
        shard_totals[shard_index] += cost
        shard_run_counts[shard_index] += 1

    shard_manifests: list[dict[str, Any]] = []
    for shard_index, shard_runs in enumerate(shard_buckets):
        shard_profile = _shard_profile(profile, shard_index + 1, int(shard_count))
        shard_runs = sorted(
            shard_runs,
            key=lambda run: (
                int(run.get("canonical_run_index", 0) or 0),
                str(run.get("run_id", "")).strip(),
            ),
        )
        payload = copy.deepcopy(canonical_manifest)
        payload["profile"] = shard_profile
        payload["description"] = f"{str(canonical_manifest.get('description', '')).strip()} [shard {shard_index + 1}/{int(shard_count)}]"
        payload["execution_mode"] = "sharded_identical_execution_class"
        payload["canonical_profile"] = profile
        payload["canonical_manifest"] = _relpath(manifest_path)
        payload["canonical_manifest_digest"] = manifest_digest
        payload["canonical_manifest_digest_algorithm"] = "sha256"
        payload["canonical_model_revisions"] = canonical_model_revisions
        payload["shard_profile"] = shard_profile
        payload["shard_index"] = shard_index + 1
        payload["shard_count"] = int(shard_count)
        payload["shard_strategy"] = "weighted_greedy_by_priority"
        payload["shard_run_count"] = len(shard_runs)
        payload["shard_weight_total"] = int(shard_totals[shard_index])
        payload["shard_run_ids"] = [str(run["run_id"]) for run in shard_runs]
        payload["shard_canonical_run_indices"] = [int(run["canonical_run_index"]) for run in shard_runs]
        payload["model_revisions"] = canonical_model_revisions
        payload["runs"] = shard_runs
        shard_manifests.append(payload)
    return shard_manifests


def write_matrix_shards(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    output_dir: Path,
    profile: str = CANONICAL_PROFILE,
    shard_count: int = DEFAULT_SHARD_COUNT,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_manifests = build_matrix_shards(manifest, manifest_path=manifest_path, profile=profile, shard_count=shard_count)
    output_paths: list[Path] = []
    for shard_manifest in shard_manifests:
        shard_profile = str(shard_manifest["shard_profile"]).strip()
        output_path = output_dir / f"{shard_profile}.json"
        dump_json(output_path, shard_manifest)
        output_paths.append(output_path)
    return output_paths


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else (ROOT / args.manifest)
    manifest = _read_manifest(manifest_path)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir)
    output_paths = write_matrix_shards(
        manifest,
        manifest_path=manifest_path,
        output_dir=output_dir,
        profile=args.profile,
        shard_count=int(args.shards),
    )
    summary = {
        "canonical_manifest": _relpath(manifest_path),
        "canonical_profile": _normalized_profile(args.profile),
        "shard_count": int(args.shards),
        "output_dir": _relpath(output_dir),
        "shards": [
            {
                "shard_index": index + 1,
                "profile": _shard_profile(_normalized_profile(args.profile), index + 1, int(args.shards)),
                "path": _relpath(path),
            }
            for index, path in enumerate(output_paths)
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
