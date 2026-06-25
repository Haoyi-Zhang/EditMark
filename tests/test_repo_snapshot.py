from __future__ import annotations

import pytest
from pathlib import Path

from scripts import _repo_snapshot


def test_repo_snapshot_sha256_excludes_runtime_outputs_and_pycache(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "keep.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "results" / "matrix").mkdir(parents=True)
    (tmp_path / "results" / "matrix" / "should_skip.json").write_text("skip\n", encoding="utf-8")
    (tmp_path / "results" / "matrix_shards").mkdir(parents=True)
    (tmp_path / "results" / "matrix_shards" / "should_skip.json").write_text("skip\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "skip.pyc").write_bytes(b"pyc")

    digest_before = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    (tmp_path / "results" / "matrix" / "should_skip.json").write_text("changed\n", encoding="utf-8")
    (tmp_path / "results" / "matrix_shards" / "should_skip.json").write_text("changed\n", encoding="utf-8")
    (tmp_path / "__pycache__" / "skip.pyc").write_bytes(b"changed")
    digest_after_ignored_changes = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    assert digest_after_ignored_changes == digest_before

    (tmp_path / "scripts" / "keep.py").write_text("print('changed')\n", encoding="utf-8")
    digest_after_real_change = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    assert digest_after_real_change != digest_before


def test_repo_snapshot_sha256_excludes_coordination_runtime_state(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "keep.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / ".coordination" / "external" / "STONE.gitcheckout" / ".git" / "logs").mkdir(parents=True)
    git_log = tmp_path / ".coordination" / "external" / "STONE.gitcheckout" / ".git" / "logs" / "HEAD"
    git_log.write_text("first\n", encoding="utf-8")

    digest_before = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    git_log.write_text("changed\n", encoding="utf-8")
    digest_after_coordination_change = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    assert digest_after_coordination_change == digest_before


def test_repo_snapshot_sha256_excludes_environment_figure_and_table_exports(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "keep.py").write_text("print('ok')\n", encoding="utf-8")
    for rel in (
        "results/environment/suite_all_models_methods/environment.json",
        "results/figures/suite_all_models_methods/figure.png",
        "results/tables/suite_all_models_methods/table.csv",
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("first\n", encoding="utf-8")

    digest_before = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    for rel in (
        "results/environment/suite_all_models_methods/environment.json",
        "results/figures/suite_all_models_methods/figure.png",
        "results/tables/suite_all_models_methods/table.csv",
    ):
        (tmp_path / rel).write_text("changed\n", encoding="utf-8")

    digest_after_generated_artifact_change = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    assert digest_after_generated_artifact_change == digest_before


def test_repo_snapshot_sha256_excludes_launcher_runtime_state(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "keep.py").write_text("print('ok')\n", encoding="utf-8")
    launcher_log = tmp_path / "results" / "launchers" / "suite_all_models_methods" / "full_run.launch.log"
    launcher_log.parent.mkdir(parents=True, exist_ok=True)
    launcher_log.write_text("first\n", encoding="utf-8")

    digest_before = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    launcher_log.write_text("changed\n", encoding="utf-8")
    digest_after_launcher_change = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    assert digest_after_launcher_change == digest_before


def test_repo_snapshot_sha256_excludes_archive_and_release_bundle_runtime_state(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "keep.py").write_text("print('ok')\n", encoding="utf-8")
    archive_file = tmp_path / "results" / "archive" / "previous_run.tar"
    bundle_file = tmp_path / "results" / "release_bundle" / "bundle.manifest.json"
    archive_file.parent.mkdir(parents=True, exist_ok=True)
    bundle_file.parent.mkdir(parents=True, exist_ok=True)
    archive_file.write_text("first\n", encoding="utf-8")
    bundle_file.write_text("{\"first\": true}\n", encoding="utf-8")

    digest_before = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    archive_file.write_text("changed\n", encoding="utf-8")
    bundle_file.write_text("{\"changed\": true}\n", encoding="utf-8")
    digest_after_runtime_artifact_change = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    assert digest_after_runtime_artifact_change == digest_before


def test_repo_snapshot_sha256_excludes_release_scratch_and_review_outputs(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "keep.py").write_text("print('ok')\n", encoding="utf-8")
    scratch_paths = (
        "results/tmp/final_release_refresh.log",
        "results/test_release_bundle/report.json",
        "results/fetched_suite/download.json",
        "results/article_preflight/preflight.json",
        "_review_outputs/metric_scan.md",
        "_remote_preview_figures/overview.png",
    )
    for rel in scratch_paths:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("first\n", encoding="utf-8")

    digest_before = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    for rel in scratch_paths:
        (tmp_path / rel).write_text("changed\n", encoding="utf-8")
    digest_after_scratch_change = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    assert digest_after_scratch_change == digest_before


def test_repo_snapshot_sha256_excludes_certification_runtime_state(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "keep.py").write_text("print('ok')\n", encoding="utf-8")
    gate_file = tmp_path / "results" / "certifications" / "suite_all_models_methods" / "suite_precheck_gate.json"
    gate_file.parent.mkdir(parents=True, exist_ok=True)
    gate_file.write_text('{"status": "passed"}\n', encoding="utf-8")

    digest_before = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    gate_file.write_text('{"status": "failed"}\n', encoding="utf-8")
    digest_after_certification_change = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    assert digest_after_certification_change == digest_before


def test_repo_snapshot_sha256_excludes_repo_local_venv_runtime_state(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "keep.py").write_text("print('ok')\n", encoding="utf-8")
    venv_python = tmp_path / ".venv" / "posteditbench_release" / "bin" / "python"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("first\n", encoding="utf-8")

    digest_before = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    venv_python.write_text("changed\n", encoding="utf-8")
    digest_after_venv_change = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    assert digest_after_venv_change == digest_before


def test_repo_snapshot_sha256_excludes_external_checkout_model_cache_and_nested_git(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "keep.py").write_text("print('ok')\n", encoding="utf-8")
    nested_git = tmp_path / ".tmp_pytest_bundle" / "repo" / ".git" / "HEAD"
    nested_git.parent.mkdir(parents=True, exist_ok=True)
    nested_git.write_text("first\n", encoding="utf-8")
    ext_file = tmp_path / "external_checkout" / "STONE-watermarking" / "README.md"
    ext_file.parent.mkdir(parents=True, exist_ok=True)
    ext_file.write_text("first\n", encoding="utf-8")
    target_cache = tmp_path.parent / f"{tmp_path.name}_cache_target"
    target_cache.mkdir(parents=True, exist_ok=True)
    (target_cache / "blob.bin").write_bytes(b"first")
    try:
        (tmp_path / "model_cache").symlink_to(target_cache, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available in this environment")

    digest_before = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    nested_git.write_text("changed\n", encoding="utf-8")
    ext_file.write_text("changed\n", encoding="utf-8")
    (target_cache / "blob.bin").write_bytes(b"changed")
    digest_after_ignored_change = _repo_snapshot.repo_snapshot_sha256(tmp_path)

    assert digest_after_ignored_change == digest_before
