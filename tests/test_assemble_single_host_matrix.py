from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts import assemble_single_host_matrix


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


def _canonical_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "configs" / "matrices" / "suite_all_models_methods.json"
    _write_json(
        manifest_path,
        {
            "profile": "suite_all_models_methods",
            "model_revisions": {
                "model-a": "rev-a",
            },
            "runs": [
                {"run_id": "run-01", "profile": "suite_all_models_methods"},
                {"run_id": "run-02", "profile": "suite_all_models_methods"},
                {"run_id": "run-03", "profile": "suite_all_models_methods"},
            ],
        },
    )
    return manifest_path


def test_assemble_single_host_matrix_builds_canonical_index_from_multiple_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _canonical_manifest(tmp_path)
    canonical_digest = _sha256(manifest_path)

    source_a_dir = tmp_path / "results" / "matrix" / "source_a"
    source_b_dir = tmp_path / "results" / "matrix" / "source_b"
    run01_dir = _write_run_dir(source_a_dir, "run-01")
    run02_dir = _write_run_dir(source_b_dir, "run-02")
    run03_dir = _write_run_dir(source_b_dir, "run-03")

    source_a_index = source_a_dir / "matrix_index.json"
    source_b_index = source_b_dir / "matrix_index.json"

    _write_json(
        source_a_index,
        {
            "schema_version": 1,
            "profile": "source_a",
            "manifest": "results/matrix/source_a/source_a.json",
            "execution_mode": "single_host_canonical",
            "gpu_pool_mode": "shared",
            "canonical_manifest": str(manifest_path),
            "canonical_manifest_digest": canonical_digest,
            "canonical_model_revisions": {"model-a": "rev-a"},
            "code_snapshot_digest": "snapshot-digest",
            "execution_environment_fingerprint": "env-fp",
            "run_count": 1,
            "success_count": 1,
            "failed_count": 0,
            "running_count": 0,
            "pending_count": 0,
            "skipped_count": 0,
            "runs": [
                {
                    "run_id": "run-01",
                    "status": "success",
                    "report_path": "/remote/missing/report.json",
                    "baseline_eval_path": "/remote/missing/baseline_eval.json",
                    "log_path": "/remote/missing/run.log",
                    "resolved_config_path": "/remote/missing/_resolved_config.yaml",
                }
            ],
        },
    )
    _write_json(
        source_b_index,
        {
            "schema_version": 1,
            "profile": "source_b",
            "manifest": "results/matrix/source_b/source_b.json",
            "execution_mode": "single_host_canonical",
            "gpu_pool_mode": "shared",
            "canonical_manifest": str(manifest_path),
            "canonical_manifest_digest": canonical_digest,
            "canonical_model_revisions": {"model-a": "rev-a"},
            "code_snapshot_digest": "snapshot-digest",
            "execution_environment_fingerprint": "env-fp",
            "run_count": 2,
            "success_count": 2,
            "failed_count": 0,
            "running_count": 0,
            "pending_count": 0,
            "skipped_count": 0,
            "runs": [
                {
                    "run_id": "run-02",
                    "status": "success",
                    "output_dir": str(run02_dir),
                    "report_path": str(run02_dir / "report.json"),
                    "baseline_eval_path": str(run02_dir / "baseline_eval.json"),
                    "log_path": str(run02_dir / "run.log"),
                    "resolved_config_path": str(run02_dir / "_resolved_config.yaml"),
                },
                {
                    "run_id": "run-03",
                    "status": "success",
                    "output_dir": str(run03_dir),
                    "report_path": str(run03_dir / "report.json"),
                    "baseline_eval_path": str(run03_dir / "baseline_eval.json"),
                    "log_path": str(run03_dir / "run.log"),
                    "resolved_config_path": str(run03_dir / "_resolved_config.yaml"),
                },
            ],
        },
    )

    output_index = tmp_path / "results" / "matrix" / "suite_all_models_methods_assembled_inspection" / "matrix_index.json"
    argv = [
        "assemble_single_host_matrix.py",
        "--manifest",
        str(manifest_path),
        "--profile",
        "suite_all_models_methods",
        "--code-snapshot-digest",
        "snapshot-digest",
        "--execution-environment-fingerprint",
        "env-fp",
        "--input-index",
        str(source_a_index),
        "--input-index",
        str(source_b_index),
        "--stage-external-runs-under",
        "results/matrix/suite_all_models_methods_staged",
        "--output-index",
        str(output_index),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert assemble_single_host_matrix.main() == 0

    payload = json.loads(output_index.read_text(encoding="utf-8"))
    assert payload["profile"] == "suite_all_models_methods"
    assert Path(payload["manifest"]).resolve() == manifest_path.resolve()
    assert payload["execution_mode"] == "single_host_canonical"
    assert payload["run_count"] == 3
    assert payload["success_count"] == 3
    assert payload["failed_count"] == 0
    assert payload["code_snapshot_digest"] == "snapshot-digest"
    assert payload["execution_environment_fingerprint"] == "env-fp"
    assert payload["environment_fingerprint"] == "env-fp"
    assert payload["pending_count"] == 0
    assert [run["run_id"] for run in payload["runs"]] == ["run-01", "run-02", "run-03"]
    assert len(payload["assembly_source_indexes"]) == 2
    assert set(payload["assembly_source_execution_modes"]) == {"single_host_canonical"}
    staged_report = assemble_single_host_matrix.ROOT / Path(payload["runs"][0]["report_path"])
    assert staged_report.exists()
    assert json.loads(staged_report.read_text(encoding="utf-8")) == json.loads(
        (run01_dir / "report.json").read_text(encoding="utf-8")
    )


def test_assemble_single_host_matrix_rejects_formal_canonical_output_for_multi_segment_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(assemble_single_host_matrix, "ROOT", tmp_path)
    manifest_path = _canonical_manifest(tmp_path)
    canonical_digest = _sha256(manifest_path)
    source_a_dir = tmp_path / "results" / "matrix" / "source_a"
    source_b_dir = tmp_path / "results" / "matrix" / "source_b"
    run_a = _write_run_dir(source_a_dir, "run-01")
    run_b = _write_run_dir(source_b_dir, "run-02")
    source_a_index = source_a_dir / "matrix_index.json"
    source_b_index = source_b_dir / "matrix_index.json"
    for index_path, run_dir, run_id, profile in (
        (source_a_index, run_a, "run-01", "source_a"),
        (source_b_index, run_b, "run-02", "source_b"),
    ):
        _write_json(
            index_path,
            {
                "schema_version": 1,
                "profile": profile,
                "manifest": f"{profile}.json",
                "execution_mode": "single_host_canonical",
                "canonical_manifest": str(manifest_path),
                "canonical_manifest_digest": canonical_digest,
                "canonical_model_revisions": {"model-a": "rev-a"},
                "code_snapshot_digest": "snapshot-digest",
                "execution_environment_fingerprint": "env-fp",
                "run_count": 1,
                "success_count": 1,
                "failed_count": 0,
                "running_count": 0,
                "pending_count": 0,
                "skipped_count": 0,
                "runs": [
                    {
                        "run_id": run_id,
                        "status": "success",
                        "output_dir": str(run_dir),
                        "report_path": str(run_dir / "report.json"),
                        "baseline_eval_path": str(run_dir / "baseline_eval.json"),
                        "log_path": str(run_dir / "run.log"),
                        "resolved_config_path": str(run_dir / "_resolved_config.yaml"),
                    }
                ],
            },
        )

    output_index = tmp_path / "results" / "matrix" / "suite_all_models_methods" / "matrix_index.json"
    argv = [
        "assemble_single_host_matrix.py",
        "--manifest",
        str(manifest_path),
        "--profile",
        "suite_all_models_methods",
        "--code-snapshot-digest",
        "snapshot-digest",
        "--execution-environment-fingerprint",
        "env-fp",
        "--input-index",
        str(source_a_index),
        "--input-index",
        str(source_b_index),
        "--output-index",
        str(output_index),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="formal canonical release path is reserved for the one-shot run_full_matrix result"):
        assemble_single_host_matrix.main()


def test_assemble_single_host_matrix_rejects_duplicate_run_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _canonical_manifest(tmp_path)
    canonical_digest = _sha256(manifest_path)
    source_a_dir = tmp_path / "results" / "matrix" / "source_a"
    source_b_dir = tmp_path / "results" / "matrix" / "source_b"
    run_a = _write_run_dir(source_a_dir, "run-01")
    run_b = _write_run_dir(source_b_dir, "run-01")

    source_a_index = source_a_dir / "matrix_index.json"
    source_b_index = source_b_dir / "matrix_index.json"
    for index_path, run_dir, profile in (
        (source_a_index, run_a, "source_a"),
        (source_b_index, run_b, "source_b"),
    ):
        _write_json(
            index_path,
            {
                "schema_version": 1,
                "profile": profile,
                "manifest": f"{profile}.json",
                "execution_mode": "single_host_canonical",
                "canonical_manifest": str(manifest_path),
                "canonical_manifest_digest": canonical_digest,
                "run_count": 1,
                "success_count": 1,
                "failed_count": 0,
                "running_count": 0,
                "pending_count": 0,
                "skipped_count": 0,
                "runs": [
                    {
                        "run_id": "run-01",
                        "status": "success",
                        "output_dir": str(run_dir),
                        "report_path": str(run_dir / "report.json"),
                        "baseline_eval_path": str(run_dir / "baseline_eval.json"),
                        "log_path": str(run_dir / "run.log"),
                        "resolved_config_path": str(run_dir / "_resolved_config.yaml"),
                    }
                ],
            },
        )

    argv = [
        "assemble_single_host_matrix.py",
        "--manifest",
        str(manifest_path),
        "--profile",
        "suite_all_models_methods",
        "--input-index",
        str(source_a_index),
        "--input-index",
        str(source_b_index),
        "--stage-external-runs-under",
        "results/matrix/suite_all_models_methods_staged",
        "--output-index",
        str(tmp_path / "assembled.json"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="duplicate run_id across source indexes"):
        assemble_single_host_matrix.main()


def test_assemble_single_host_matrix_requires_publication_identity_when_sources_do_not_provide_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _canonical_manifest(tmp_path)
    canonical_digest = _sha256(manifest_path)
    source_a_dir = tmp_path / "results" / "matrix" / "source_a"
    source_b_dir = tmp_path / "results" / "matrix" / "source_b"
    source_c_dir = tmp_path / "results" / "matrix" / "source_c"
    run_a = _write_run_dir(source_a_dir, "run-01")
    run_b = _write_run_dir(source_b_dir, "run-02")
    run_c = _write_run_dir(source_c_dir, "run-03")
    source_a_index = source_a_dir / "matrix_index.json"
    source_b_index = source_b_dir / "matrix_index.json"
    source_c_index = source_c_dir / "matrix_index.json"
    for index_path, run_dir, run_id, profile in (
        (source_a_index, run_a, "run-01", "source_a"),
        (source_b_index, run_b, "run-02", "source_b"),
        (source_c_index, run_c, "run-03", "source_c"),
    ):
        _write_json(
            index_path,
            {
                "schema_version": 1,
                "profile": profile,
                "manifest": f"{profile}.json",
                "execution_mode": "single_host_canonical",
                "canonical_manifest": str(manifest_path),
                "canonical_manifest_digest": canonical_digest,
                "canonical_model_revisions": {"model-a": "rev-a"},
                "run_count": 1,
                "success_count": 1,
                "failed_count": 0,
                "running_count": 0,
                "pending_count": 0,
                "skipped_count": 0,
                "runs": [
                    {
                        "run_id": run_id,
                        "status": "success",
                        "output_dir": str(run_dir),
                        "report_path": str(run_dir / "report.json"),
                        "baseline_eval_path": str(run_dir / "baseline_eval.json"),
                        "log_path": str(run_dir / "run.log"),
                        "resolved_config_path": str(run_dir / "_resolved_config.yaml"),
                    }
                ],
            },
        )

    argv = [
        "assemble_single_host_matrix.py",
        "--manifest",
        str(manifest_path),
        "--profile",
        "suite_all_models_methods",
        "--input-index",
        str(source_a_index),
        "--input-index",
        str(source_b_index),
        "--input-index",
        str(source_c_index),
        "--stage-external-runs-under",
        "results/matrix/suite_all_models_methods_staged",
        "--output-index",
        str(tmp_path / "assembled.json"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="missing code_snapshot_digest"):
        assemble_single_host_matrix.main()


def test_assemble_single_host_matrix_requires_source_execution_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _canonical_manifest(tmp_path)
    canonical_digest = _sha256(manifest_path)
    source_dir = tmp_path / "results" / "matrix" / "source_a"
    run_dir = _write_run_dir(source_dir, "run-01")
    source_index = source_dir / "matrix_index.json"
    _write_json(
        source_index,
        {
            "schema_version": 1,
            "profile": "source_a",
            "manifest": "source_a.json",
            "canonical_manifest": str(manifest_path),
            "canonical_manifest_digest": canonical_digest,
            "canonical_model_revisions": {"model-a": "rev-a"},
            "run_count": 1,
            "success_count": 1,
            "failed_count": 0,
            "running_count": 0,
            "pending_count": 0,
            "skipped_count": 0,
            "runs": [
                {
                    "run_id": "run-01",
                    "status": "success",
                    "output_dir": str(run_dir),
                    "report_path": str(run_dir / "report.json"),
                    "baseline_eval_path": str(run_dir / "baseline_eval.json"),
                    "log_path": str(run_dir / "run.log"),
                    "resolved_config_path": str(run_dir / "_resolved_config.yaml"),
                }
            ],
        },
    )
    argv = [
        "assemble_single_host_matrix.py",
        "--manifest",
        str(manifest_path),
        "--profile",
        "suite_all_models_methods",
        "--code-snapshot-digest",
        "snapshot-digest",
        "--execution-environment-fingerprint",
        "env-fp",
        "--input-index",
        str(source_index),
        "--output-index",
        str(tmp_path / "assembled.json"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="missing execution_mode"):
        assemble_single_host_matrix.main()
