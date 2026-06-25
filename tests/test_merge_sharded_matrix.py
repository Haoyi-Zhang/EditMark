from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts import build_matrix_shards, export_full_run_tables, merge_sharded_matrix


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MANIFEST = ROOT / "configs" / "matrices" / "suite_all_models_methods.json"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run_dir(base_dir: Path, run_id: str) -> Path:
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "report.json", {"rows": [{"task_id": f"{run_id}-1"}]})
    _write_json(run_dir / "baseline_eval.json", {"status": "ok"})
    (run_dir / "run.log").write_text("[exit_code=0]\n", encoding="utf-8")
    (run_dir / "_resolved_config.yaml").write_text("benchmark: {}\n", encoding="utf-8")
    return run_dir


def _build_shard_inputs(
    tmp_path: Path,
    *,
    shard_count: int = 2,
    full_matrix_audit: dict[str, object] | None = None,
    code_snapshot_digest: str = "snapshot-digest",
) -> tuple[list[Path], list[Path]]:
    canonical_manifest = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
    default_full_matrix_audit = full_matrix_audit or {"status": "clean"}
    shard_manifest_dir = tmp_path / "matrix_shards"
    shard_paths = build_matrix_shards.write_matrix_shards(
        canonical_manifest,
        manifest_path=CANONICAL_MANIFEST,
        output_dir=shard_manifest_dir,
        profile="suite_all_models_methods",
        shard_count=shard_count,
    )
    canonical_manifest_digest = _sha256(CANONICAL_MANIFEST)
    canonical_model_revisions = canonical_manifest["model_revisions"]

    shard_indexes: list[Path] = []
    host_receipts: list[Path] = []
    for shard_manifest_path in shard_paths:
        shard_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
        shard_profile = str(shard_manifest["profile"])
        shard_matrix_root = tmp_path / "results" / "matrix" / shard_profile
        shard_index_path = shard_matrix_root / "matrix_index.json"
        receipt_path = tmp_path / "results" / "certifications" / shard_profile / "matrix_shard_readiness.json"

        runs: list[dict[str, object]] = []
        for run in shard_manifest["runs"]:
            run_id = str(run["run_id"])
            run_dir = _write_run_dir(shard_matrix_root, run_id)
            runs.append(
                {
                    "run_id": run_id,
                    "status": "success",
                    "reason": "",
                    "output_dir": str(run_dir),
                    "report_path": str(run_dir / "report.json"),
                    "baseline_eval_path": str(run_dir / "baseline_eval.json"),
                    "log_path": str(run_dir / "run.log"),
                    "resolved_config_path": str(run_dir / "_resolved_config.yaml"),
                }
            )

        _write_json(
            shard_index_path,
            {
                "schema_version": 1,
                "profile": shard_profile,
                "manifest": merge_sharded_matrix._repo_relpath(shard_manifest_path),
                "gpu_pool_mode": "shared",
                "run_count": len(runs),
                "completed_count": len(runs),
                "success_count": len(runs),
                "skipped_count": 0,
                "running_count": 0,
                "failed_count": 0,
                "pending_count": 0,
                "runs": runs,
            },
        )
        _write_json(
            receipt_path,
            {
                "schema_version": 2,
                "receipt_type": "matrix_shard_readiness",
                "status": "passed",
                "manifest": merge_sharded_matrix._repo_relpath(shard_manifest_path),
                "profile": shard_profile,
                "execution_mode": "sharded_identical_execution_class",
                "canonical_manifest": "configs/matrices/suite_all_models_methods.json",
                "canonical_profile": "suite_all_models_methods",
                "gpu_slots": 8,
                "gpu_pool_mode": "shared",
                "cpu_workers": 9,
                "retry_count": 1,
                "manifest_digests": {
                    "canonical_manifest": canonical_manifest_digest,
                    "manifest": _sha256(shard_manifest_path),
                },
                "code_snapshot_digest": code_snapshot_digest,
                "suite_model_revisions": canonical_model_revisions,
                "environment_receipt": {
                    "python_bin": "/example/posteditbench_env/posteditbench_release_clean/bin/python",
                    "environment_fingerprint": "identical-hosts",
                    "host_environment_fingerprint": f"host-{shard_profile}",
                    "execution_environment_fingerprint": "identical-execution-class",
                    "python_executable": "/example/posteditbench_env/posteditbench_release_clean/bin/python",
                    "cuda_visible_devices": "0,1,2,3,4,5,6,7",
                    "visible_gpu_count": 8,
                    "preflight_gpu_slots": 8,
                },
                "audits": {
                    "benchmark_audit": {"status": "ok"},
                    "full_matrix_audit": default_full_matrix_audit,
                },
            },
        )
        shard_indexes.append(shard_index_path)
        host_receipts.append(receipt_path)
    return shard_indexes, host_receipts


def test_merge_sharded_matrix_merges_canonical_suite_and_stays_exportable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_indexes, host_receipts = _build_shard_inputs(tmp_path, shard_count=2)
    output_index = tmp_path / "results" / "matrix" / "reviewer_sharded_inspection" / "merged" / "matrix_index.json"

    argv = [
        "merge_sharded_matrix.py",
        "--manifest",
        str(CANONICAL_MANIFEST),
        "--profile",
        "suite_all_models_methods",
        "--output-index",
        str(output_index),
    ]
    for shard_index in shard_indexes:
        argv.extend(["--shard-index", str(shard_index)])
    for receipt in host_receipts:
        argv.extend(["--host-receipt", str(receipt)])
    monkeypatch.setattr(sys, "argv", argv)

    assert merge_sharded_matrix.main() == 0

    payload = json.loads(output_index.read_text(encoding="utf-8"))
    assert payload["profile"] == "suite_all_models_methods"
    assert payload["manifest"] == "configs/matrices/suite_all_models_methods.json"
    assert payload["run_count"] == 140
    assert payload["success_count"] == 140
    assert payload["completed_count"] == 140
    assert payload["pending_count"] == 0
    assert payload["failed_count"] == 0
    assert payload["running_count"] == 0
    assert payload["execution_mode"] == "sharded_identical_execution_class"
    assert len(payload["shard_profiles"]) == 2
    assert payload["execution_environment_fingerprint"] == "identical-execution-class"
    assert payload["code_snapshot_digest"] == "snapshot-digest"
    assert len(payload["host_environment_fingerprints"]) == 2

    export_full_run_tables._require_canonical_suite_identity(
        payload,
        label="merged sharded matrix exact-value exports",
    )
    exportable_runs = export_full_run_tables._exportable_runs(payload, base_dir=ROOT)
    assert len(exportable_runs) == 140


def test_merge_sharded_matrix_rejects_canonical_publication_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_indexes, host_receipts = _build_shard_inputs(tmp_path, shard_count=2)

    argv = [
        "merge_sharded_matrix.py",
        "--manifest",
        str(CANONICAL_MANIFEST),
        "--profile",
        "suite_all_models_methods",
        "--output-index",
        str(ROOT / "results" / "matrix" / "suite_all_models_methods" / "matrix_index.json"),
    ]
    for shard_index in shard_indexes:
        argv.extend(["--shard-index", str(shard_index)])
    for receipt in host_receipts:
        argv.extend(["--host-receipt", str(receipt)])
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="inspection-only and must not overwrite"):
        merge_sharded_matrix.main()


def test_merge_sharded_matrix_rejects_shard_run_membership_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_indexes, host_receipts = _build_shard_inputs(tmp_path, shard_count=2)
    first_index = json.loads(shard_indexes[0].read_text(encoding="utf-8"))
    first_index["runs"] = list(first_index["runs"][1:])
    first_index["run_count"] = len(first_index["runs"])
    first_index["completed_count"] = len(first_index["runs"])
    first_index["success_count"] = len(first_index["runs"])
    _write_json(shard_indexes[0], first_index)

    argv = [
        "merge_sharded_matrix.py",
        "--manifest",
        str(CANONICAL_MANIFEST),
        "--profile",
        "suite_all_models_methods",
        "--output-index",
        str(tmp_path / "merged_index.json"),
    ]
    for shard_index in shard_indexes:
        argv.extend(["--shard-index", str(shard_index)])
    for receipt in host_receipts:
        argv.extend(["--host-receipt", str(receipt)])
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="run set does not match shard manifest membership"):
        merge_sharded_matrix.main()


def test_merge_sharded_matrix_rejects_receipt_manifest_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_indexes, host_receipts = _build_shard_inputs(tmp_path, shard_count=2)
    receipt = json.loads(host_receipts[0].read_text(encoding="utf-8"))
    receipt["manifest_digests"]["manifest"] = "bad-digest"
    _write_json(host_receipts[0], receipt)

    argv = [
        "merge_sharded_matrix.py",
        "--manifest",
        str(CANONICAL_MANIFEST),
        "--profile",
        "suite_all_models_methods",
        "--output-index",
        str(tmp_path / "merged_index.json"),
    ]
    for shard_index in shard_indexes:
        argv.extend(["--shard-index", str(shard_index)])
    for host_receipt in host_receipts:
        argv.extend(["--host-receipt", str(host_receipt)])
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="shard manifest digest mismatch"):
        merge_sharded_matrix.main()


def test_merge_sharded_matrix_rejects_environment_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_indexes, host_receipts = _build_shard_inputs(tmp_path, shard_count=2)
    second_receipt = json.loads(host_receipts[1].read_text(encoding="utf-8"))
    second_receipt["environment_receipt"]["execution_environment_fingerprint"] = "different-execution-class"
    _write_json(host_receipts[1], second_receipt)

    argv = [
        "merge_sharded_matrix.py",
        "--manifest",
        str(CANONICAL_MANIFEST),
        "--profile",
        "suite_all_models_methods",
        "--output-index",
        str(tmp_path / "merged_index.json"),
    ]
    for shard_index in shard_indexes:
        argv.extend(["--shard-index", str(shard_index)])
    for receipt in host_receipts:
        argv.extend(["--host-receipt", str(receipt)])
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="execution environment fingerprint"):
        merge_sharded_matrix.main()


def test_merge_sharded_matrix_rejects_code_snapshot_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_indexes, host_receipts = _build_shard_inputs(tmp_path, shard_count=2)
    second_receipt = json.loads(host_receipts[1].read_text(encoding="utf-8"))
    second_receipt["code_snapshot_digest"] = "different-snapshot"
    _write_json(host_receipts[1], second_receipt)

    argv = [
        "merge_sharded_matrix.py",
        "--manifest",
        str(CANONICAL_MANIFEST),
        "--profile",
        "suite_all_models_methods",
        "--output-index",
        str(tmp_path / "merged_index.json"),
    ]
    for shard_index in shard_indexes:
        argv.extend(["--shard-index", str(shard_index)])
    for receipt in host_receipts:
        argv.extend(["--host-receipt", str(receipt)])
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="code_snapshot_digest"):
        merge_sharded_matrix.main()


def test_merge_sharded_matrix_requires_code_snapshot_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_indexes, host_receipts = _build_shard_inputs(tmp_path, shard_count=2)
    first_receipt = json.loads(host_receipts[0].read_text(encoding="utf-8"))
    first_receipt.pop("code_snapshot_digest", None)
    _write_json(host_receipts[0], first_receipt)

    argv = [
        "merge_sharded_matrix.py",
        "--manifest",
        str(CANONICAL_MANIFEST),
        "--profile",
        "suite_all_models_methods",
        "--output-index",
        str(tmp_path / "merged_index.json"),
    ]
    for shard_index in shard_indexes:
        argv.extend(["--shard-index", str(shard_index)])
    for receipt in host_receipts:
        argv.extend(["--host-receipt", str(receipt)])
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="missing code_snapshot_digest"):
        merge_sharded_matrix.main()


def test_merge_sharded_matrix_requires_execution_environment_fingerprint_without_compatibility_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_indexes, host_receipts = _build_shard_inputs(tmp_path, shard_count=2)
    first_receipt = json.loads(host_receipts[0].read_text(encoding="utf-8"))
    first_receipt["environment_receipt"].pop("execution_environment_fingerprint", None)
    _write_json(host_receipts[0], first_receipt)

    argv = [
        "merge_sharded_matrix.py",
        "--manifest",
        str(CANONICAL_MANIFEST),
        "--profile",
        "suite_all_models_methods",
        "--output-index",
        str(tmp_path / "merged_index.json"),
    ]
    for shard_index in shard_indexes:
        argv.extend(["--shard-index", str(shard_index)])
    for receipt in host_receipts:
        argv.extend(["--host-receipt", str(receipt)])
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="missing environment_receipt.execution_environment_fingerprint"):
        merge_sharded_matrix.main()


def test_merge_sharded_matrix_rejects_compatibility_shard_manifest_execution_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_indexes, host_receipts = _build_shard_inputs(tmp_path, shard_count=2)
    shard_manifest_path = tmp_path / "matrix_shards" / "suite_all_models_methods_shard_01_of_02.json"
    shard_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
    shard_manifest["execution_mode"] = "sharded_identical_hosts"
    _write_json(shard_manifest_path, shard_manifest)

    first_receipt = json.loads(host_receipts[0].read_text(encoding="utf-8"))
    first_receipt["manifest_digests"]["manifest"] = _sha256(shard_manifest_path)
    _write_json(host_receipts[0], first_receipt)

    argv = [
        "merge_sharded_matrix.py",
        "--manifest",
        str(CANONICAL_MANIFEST),
        "--profile",
        "suite_all_models_methods",
        "--output-index",
        str(tmp_path / "merged_index.json"),
    ]
    for shard_index in shard_indexes:
        argv.extend(["--shard-index", str(shard_index)])
    for receipt in host_receipts:
        argv.extend(["--host-receipt", str(receipt)])
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="execution_mode must be sharded_identical_execution_class"):
        merge_sharded_matrix.main()


def test_merge_sharded_matrix_rejects_compatibility_receipt_execution_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_indexes, host_receipts = _build_shard_inputs(tmp_path, shard_count=2)
    first_receipt = json.loads(host_receipts[0].read_text(encoding="utf-8"))
    first_receipt["execution_mode"] = "sharded_identical_hosts"
    _write_json(host_receipts[0], first_receipt)

    argv = [
        "merge_sharded_matrix.py",
        "--manifest",
        str(CANONICAL_MANIFEST),
        "--profile",
        "suite_all_models_methods",
        "--output-index",
        str(tmp_path / "merged_index.json"),
    ]
    for shard_index in shard_indexes:
        argv.extend(["--shard-index", str(shard_index)])
    for receipt in host_receipts:
        argv.extend(["--host-receipt", str(receipt)])
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="execution_mode must be sharded_identical_execution_class"):
        merge_sharded_matrix.main()


def test_merge_sharded_matrix_accepts_shard_local_fairness_gap_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_indexes, host_receipts = _build_shard_inputs(
        tmp_path,
        shard_count=2,
        full_matrix_audit={
            "status": "has_issues",
            "issues": ["matrix fairness coverage is incomplete for one or more (model, benchmark) slices"],
            "missing_methods": [],
            "missing_model_roster": [],
            "missing_benchmark_roster": [],
            "missing_provider_modes": [],
            "missing_gpu_pools": [],
            "missing_slice_methods": [
                {
                    "benchmark": "HumanEval+",
                    "missing_methods": ["ewd_runtime", "kgw_runtime"],
                    "model": "Qwen/Qwen2.5-Coder-14B-Instruct",
                }
            ],
            "hf_model_smoke": [],
            "method_smoke": [{"status": "ok"}],
        },
    )

    argv = [
        "merge_sharded_matrix.py",
        "--manifest",
        str(CANONICAL_MANIFEST),
        "--profile",
        "suite_all_models_methods",
        "--output-index",
        str(tmp_path / "merged_index.json"),
    ]
    for shard_index in shard_indexes:
        argv.extend(["--shard-index", str(shard_index)])
    for receipt in host_receipts:
        argv.extend(["--host-receipt", str(receipt)])
    monkeypatch.setattr(sys, "argv", argv)

    assert merge_sharded_matrix.main() == 0


def test_merge_sharded_matrix_rejects_non_fairness_shard_audit_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_indexes, host_receipts = _build_shard_inputs(
        tmp_path,
        shard_count=2,
        full_matrix_audit={
            "status": "has_issues",
            "issues": ["hf cache mismatch"],
            "missing_methods": [],
            "missing_model_roster": [],
            "missing_benchmark_roster": [],
            "missing_provider_modes": [],
            "missing_gpu_pools": [],
            "hf_model_smoke": [],
            "method_smoke": [{"status": "ok"}],
        },
    )

    argv = [
        "merge_sharded_matrix.py",
        "--manifest",
        str(CANONICAL_MANIFEST),
        "--profile",
        "suite_all_models_methods",
        "--output-index",
        str(tmp_path / "merged_index.json"),
    ]
    for shard_index in shard_indexes:
        argv.extend(["--shard-index", str(shard_index)])
    for receipt in host_receipts:
        argv.extend(["--host-receipt", str(receipt)])
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="full matrix audit is not merge-safe"):
        merge_sharded_matrix.main()
