from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from scripts import clean_suite_outputs


ROOT = Path(__file__).resolve().parents[1]


def _ensure_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stale\n", encoding="utf-8")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".keep").write_text("stale\n", encoding="utf-8")


def test_clean_suite_outputs_removes_active_suite_paths_and_recreates_skeleton(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    targets = [
        repo_root / "results" / "matrix" / "suite_canary_heavy",
        repo_root / "results" / "matrix" / "model_invocation_smoke",
        repo_root / "results" / "matrix" / "suite_all_models_methods",
        repo_root / "results" / "matrix_shards" / "suite_all_models_methods",
        repo_root / "results" / "matrix" / "suite_all_models_methods_shard_01_of_02",
        repo_root / "results" / "matrix" / "suite_all_models_methods_shard_02_of_02",
        repo_root / "results" / "matrix" / "suite_reviewer_subset_a",
        repo_root / "results" / "matrix" / "suite_reviewer_subset_b",
        repo_root / "results" / "figures" / "suite_precheck",
        repo_root / "results" / "figures" / "suite_all_models_methods_shard_01_of_02",
        repo_root / "results" / "tables" / "suite_all_models_methods_shard_01_of_02",
        repo_root / "results" / "environment" / "suite_all_models_methods_shard_01_of_02",
        repo_root / "results" / "audits" / "suite_all_models_methods_shard_01_of_02",
        repo_root / "results" / "certifications" / "suite_all_models_methods",
        repo_root / "results" / "certifications" / "suite_all_models_methods_shard_01_of_02",
        repo_root / "results" / "launchers" / "suite_all_models_methods",
        repo_root / "results" / "release_bundle",
        repo_root / "results" / "release_bundle_candidate",
        repo_root / "results" / "tmp" / "release_bundle_candidate",
    ]
    file_targets = [
        repo_root / "results" / "environment" / "runtime_environment.json",
        repo_root / "results" / "environment" / "runtime_environment.md",
        repo_root / "results" / "certifications" / "remote_preflight_receipt.json",
        repo_root / "results" / "certifications" / "suite_precheck_gate.json",
        repo_root / "results" / "certifications" / "suite_precheck.nohup.log",
        repo_root / "results" / "certifications" / "suite_precheck.live.log",
        repo_root / "results" / "matrix" / "subsets" / "suite_reviewer_subset_a.json",
        repo_root / "results" / "matrix" / "subsets" / "suite_reviewer_subset_b.json",
        repo_root / "results" / "environment" / "suite_reviewer_subset_a_runtime_environment.json",
        repo_root / "results" / "environment" / "suite_reviewer_subset_a_runtime_environment.md",
        repo_root / "results" / "environment" / "suite_reviewer_subset_b_runtime_environment.json",
        repo_root / "results" / "environment" / "suite_reviewer_subset_b_runtime_environment.md",
        repo_root / "results" / "certifications" / "subset_a.log",
        repo_root / "results" / "certifications" / "subset_b.log",
        repo_root / "results" / "certifications" / "suite_all_models_methods" / "full_run.launch.log",
        repo_root / "results" / "certifications" / "suite_all_models_methods" / "full_run.launch.status",
        repo_root / "results" / "certifications" / "suite_all_models_methods" / "full_run.launch.pid",
        repo_root / "results" / "certifications" / "suite_all_models_methods" / "full_run.launch.sh",
        repo_root / "results" / "tmp" / "bundle_staging" / "artifact.bin",
    ]
    try:
        monkeypatch.setattr(clean_suite_outputs, "ROOT", repo_root)
        monkeypatch.setattr(clean_suite_outputs, "_active_cleanup_conflicts", lambda: [])
        for path in targets:
            _ensure_dir(path)
        for path in file_targets:
            _ensure_file(path)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "clean_suite_outputs.py",
                "--include-full-matrix",
                "--include-release-bundle",
            ],
        )

        assert clean_suite_outputs.main() == 0
        for path in targets:
            assert not path.exists()
        for path in file_targets:
            if path.name in {"runtime_environment.json", "runtime_environment.md"}:
                assert path.exists()
            else:
                assert not path.exists()
        assert (repo_root / "results" / "matrix").exists()
        assert (repo_root / "results" / "figures").exists()
        assert (repo_root / "results" / "certifications").exists()
        placeholder_payload = (repo_root / "results" / "environment" / "runtime_environment.json").read_text(encoding="utf-8")
        assert "sanitized_placeholder" in placeholder_payload
    finally:
        if repo_root.exists():
            shutil.rmtree(repo_root)


def test_clean_suite_outputs_does_not_reset_full_exports_without_include_full_matrix(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    figures_dir = repo_root / "results" / "figures" / "suite_all_models_methods"
    tables_dir = repo_root / "results" / "tables" / "suite_all_models_methods"
    environment_json = repo_root / "results" / "environment" / "runtime_environment.json"
    environment_md = repo_root / "results" / "environment" / "runtime_environment.md"
    try:
        monkeypatch.setattr(clean_suite_outputs, "ROOT", repo_root)
        monkeypatch.setattr(clean_suite_outputs, "_active_cleanup_conflicts", lambda: [])
        _ensure_dir(repo_root / "results" / "matrix" / "suite_reviewer_subset_a")
        _ensure_dir(figures_dir)
        _ensure_dir(tables_dir)
        _ensure_file(environment_json)
        _ensure_file(environment_md)
        (figures_dir / "custom.txt").write_text("keep\n", encoding="utf-8")
        (tables_dir / "custom.txt").write_text("keep\n", encoding="utf-8")

        monkeypatch.setattr(sys, "argv", ["clean_suite_outputs.py"])
        assert clean_suite_outputs.main() == 0

        assert (figures_dir / "custom.txt").exists()
        assert (tables_dir / "custom.txt").exists()
        assert environment_json.read_text(encoding="utf-8") == "stale\n"
        assert environment_md.read_text(encoding="utf-8") == "stale\n"
    finally:
        if repo_root.exists():
            shutil.rmtree(repo_root)


def test_clean_suite_outputs_preserves_launcher_artifacts_when_requested(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    launcher_dir = repo_root / "results" / "launchers" / "suite_all_models_methods"
    monkeypatch.setattr(clean_suite_outputs, "ROOT", repo_root)
    monkeypatch.setattr(clean_suite_outputs, "_active_cleanup_conflicts", lambda: [])
    _ensure_dir(launcher_dir)

    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "clean_suite_outputs.py",
                "--include-full-matrix",
                "--preserve-launcher-artifacts",
            ],
        )
        assert clean_suite_outputs.main() == 0
        assert launcher_dir.exists()
    finally:
        if repo_root.exists():
            shutil.rmtree(repo_root)


def test_clean_suite_outputs_preserves_precheck_artifacts_when_requested(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    stage_a_dir = repo_root / "results" / "matrix" / "suite_canary_heavy"
    stage_b_dir = repo_root / "results" / "matrix" / "model_invocation_smoke"
    gate_json = repo_root / "results" / "certifications" / "suite_precheck_gate.json"
    audit_dir = repo_root / "results" / "audits"
    full_matrix_dir = repo_root / "results" / "matrix" / "suite_all_models_methods"
    monkeypatch.setattr(clean_suite_outputs, "ROOT", repo_root)
    monkeypatch.setattr(clean_suite_outputs, "_active_cleanup_conflicts", lambda: [])
    _ensure_dir(stage_a_dir)
    _ensure_dir(stage_b_dir)
    _ensure_dir(audit_dir)
    _ensure_file(gate_json)
    _ensure_dir(full_matrix_dir)

    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "clean_suite_outputs.py",
                "--include-full-matrix",
                "--preserve-precheck-artifacts",
            ],
        )
        assert clean_suite_outputs.main() == 0
        assert stage_a_dir.exists()
        assert stage_b_dir.exists()
        assert gate_json.exists()
        assert audit_dir.exists()
        assert not full_matrix_dir.exists()
    finally:
        if repo_root.exists():
            shutil.rmtree(repo_root)


def test_clean_suite_outputs_removes_explicit_extra_paths_within_repo_boundary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    extra_file = repo_root / "tmp" / "js4_redownload_qwen7b.log"
    extra_dir = repo_root / "tmp" / "relay_stage"
    try:
        monkeypatch.setattr(clean_suite_outputs, "ROOT", repo_root)
        monkeypatch.setattr(clean_suite_outputs, "_active_cleanup_conflicts", lambda: [])
        _ensure_dir(repo_root / "results")
        extra_file.parent.mkdir(parents=True, exist_ok=True)
        extra_file.write_text("stale\n", encoding="utf-8")
        _ensure_dir(extra_dir)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "clean_suite_outputs.py",
                "--extra-path",
                str(extra_file),
                "--extra-path",
                str(extra_dir),
            ],
        )
        assert clean_suite_outputs.main() == 0

        assert not extra_file.exists()
        assert not extra_dir.exists()
    finally:
        if repo_root.exists():
            shutil.rmtree(repo_root)
        if extra_file.exists():
            extra_file.unlink()
        if extra_dir.exists():
            shutil.rmtree(extra_dir)


def test_clean_suite_outputs_rejects_extra_paths_outside_repo_boundary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    outside_path = tmp_path / "outside.log"
    monkeypatch.setattr(clean_suite_outputs, "ROOT", repo_root)
    monkeypatch.setattr(clean_suite_outputs, "_active_cleanup_conflicts", lambda: [])
    _ensure_dir(repo_root / "results")
    outside_path.write_text("stale\n", encoding="utf-8")

    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "clean_suite_outputs.py",
                "--extra-path",
                str(outside_path),
            ],
        )
        try:
            clean_suite_outputs.main()
        except ValueError as exc:
            assert "extra cleanup path must stay under one of" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("expected outside extra-path cleanup to be rejected")
    finally:
        if repo_root.exists():
            shutil.rmtree(repo_root)
        if outside_path.exists():
            outside_path.unlink()


def test_clean_suite_outputs_rejects_extra_paths_under_protected_repo_assets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    monkeypatch.setattr(clean_suite_outputs, "ROOT", repo_root)
    monkeypatch.setattr(clean_suite_outputs, "_active_cleanup_conflicts", lambda: [])
    _ensure_dir(repo_root / "results")
    _ensure_file(repo_root / "README.md")

    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "clean_suite_outputs.py",
                "--extra-path",
                "README.md",
            ],
        )
        with pytest.raises(ValueError, match="extra cleanup path must stay under one of"):
            clean_suite_outputs.main()
    finally:
        if repo_root.exists():
            shutil.rmtree(repo_root)


def test_clean_suite_outputs_rejects_extra_paths_through_symlinked_temp_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    external_tmp = tmp_path / "external-tmp"
    external_tmp.mkdir(parents=True, exist_ok=True)
    symlinked_tmp = repo_root / "tmp"
    repo_root.mkdir(parents=True, exist_ok=True)
    try:
        symlinked_tmp.symlink_to(external_tmp, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available in this environment")

    monkeypatch.setattr(clean_suite_outputs, "ROOT", repo_root)
    monkeypatch.setattr(clean_suite_outputs, "_active_cleanup_conflicts", lambda: [])

    with pytest.raises(ValueError, match="symlinked temp roots"):
        clean_suite_outputs._resolve_extra_path("tmp/relay_stage")


def test_clean_suite_outputs_refuses_to_run_when_active_suite_processes_exist(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    monkeypatch.setattr(clean_suite_outputs, "ROOT", repo_root)
    monkeypatch.setattr(clean_suite_outputs, "_active_cleanup_conflicts", lambda: ["123 run_preflight.sh /repo"])
    _ensure_dir(repo_root / "results")

    try:
        monkeypatch.setattr(sys, "argv", ["clean_suite_outputs.py", "--include-full-matrix"])
        with pytest.raises(RuntimeError, match="active suite processes"):
            clean_suite_outputs.main()
    finally:
        if repo_root.exists():
            shutil.rmtree(repo_root)


def test_clean_suite_outputs_rejects_symlinked_full_export_targets(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    external = tmp_path / "external-figures"
    external.mkdir(parents=True, exist_ok=True)
    target = repo_root / "results" / "figures" / "suite_all_models_methods"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available in this environment")

    monkeypatch.setattr(clean_suite_outputs, "ROOT", repo_root)
    monkeypatch.setattr(clean_suite_outputs, "_active_cleanup_conflicts", lambda: [])

    try:
        monkeypatch.setattr(sys, "argv", ["clean_suite_outputs.py", "--include-full-matrix"])
        with pytest.raises(RuntimeError, match="symlinked path components"):
            clean_suite_outputs.main()
    finally:
        if target.exists() or target.is_symlink():
            target.unlink()
        if repo_root.exists():
            shutil.rmtree(repo_root)


def test_reset_export_target_rejects_symlinked_placeholder_leaf(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    directory = repo_root / "results" / "figures" / "suite_all_models_methods"
    external_readme = tmp_path / "outside-readme.md"
    external_readme.write_text("outside\n", encoding="utf-8")
    directory.mkdir(parents=True, exist_ok=True)
    placeholder = directory / "README.md"
    try:
        placeholder.symlink_to(external_readme)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available in this environment")

    monkeypatch.setattr(clean_suite_outputs, "ROOT", repo_root)

    with pytest.raises(RuntimeError, match="symlinked files"):
        clean_suite_outputs._reset_export_target(
            directory,
            placeholder_name="README.md",
            placeholder_text="placeholder",
        )


def test_reset_environment_placeholders_rejects_symlinked_leaf(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    environment_dir = repo_root / "results" / "environment"
    external_json = tmp_path / "outside-environment.json"
    external_json.write_text("outside\n", encoding="utf-8")
    environment_dir.mkdir(parents=True, exist_ok=True)
    placeholder = environment_dir / "runtime_environment.json"
    try:
        placeholder.symlink_to(external_json)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available in this environment")

    monkeypatch.setattr(clean_suite_outputs, "ROOT", repo_root)

    with pytest.raises(RuntimeError, match="symlinked files"):
        clean_suite_outputs._reset_environment_placeholders()


def test_active_cleanup_conflicts_ignores_parent_wrapper_and_unrelated_relative_commands(monkeypatch) -> None:
    repo_root = ROOT.resolve()
    current_pid = 2001
    parent_pid = 2000
    grandparent_pid = 1999
    process_rows = [
        (current_pid, parent_pid, f"python {repo_root}/scripts/clean_suite_outputs.py --include-full-matrix"),
        (parent_pid, grandparent_pid, f"bash {repo_root}/scripts/remote/run_suite_matrix.sh --run-full"),
        (grandparent_pid, 1, f"bash {repo_root}/results/launchers/suite_all_models_methods/full_run.launch.sh"),
        (3000, 1, "bash scripts/remote/run_matrix_shard.sh --manifest configs/matrices/suite_all_models_methods_shard_01_of_02.json"),
    ]

    monkeypatch.setattr(clean_suite_outputs, "ROOT", repo_root)
    monkeypatch.setattr(clean_suite_outputs.os, "name", "posix")
    monkeypatch.setattr(clean_suite_outputs.os, "getpid", lambda: current_pid)
    monkeypatch.setattr(clean_suite_outputs, "_load_process_rows", lambda: process_rows)
    monkeypatch.setattr(clean_suite_outputs, "_process_cwd", lambda pid: None)

    conflicts = clean_suite_outputs._active_cleanup_conflicts()

    assert conflicts == []


def test_active_cleanup_conflicts_detects_relative_repo_processes_by_cwd(monkeypatch) -> None:
    repo_root = ROOT.resolve()
    current_pid = 2001
    process_rows = [
        (current_pid, 2000, f"python {repo_root}/scripts/clean_suite_outputs.py --include-full-matrix"),
        (3000, 1, "bash scripts/remote/run_matrix_shard.sh --manifest configs/matrices/suite_all_models_methods_shard_01_of_02.json"),
    ]

    monkeypatch.setattr(clean_suite_outputs, "ROOT", repo_root)
    monkeypatch.setattr(clean_suite_outputs.os, "name", "posix")
    monkeypatch.setattr(clean_suite_outputs.os, "getpid", lambda: current_pid)
    monkeypatch.setattr(clean_suite_outputs, "_load_process_rows", lambda: process_rows)
    monkeypatch.setattr(
        clean_suite_outputs,
        "_process_cwd",
        lambda pid: repo_root if pid == 3000 else None,
    )

    conflicts = clean_suite_outputs._active_cleanup_conflicts()

    assert len(conflicts) == 1
    assert conflicts[0].startswith("3000 ")
    assert "run_matrix_shard.sh" in conflicts[0]


def test_active_cleanup_conflicts_detects_same_repo_processes_by_repo_root(monkeypatch) -> None:
    repo_root = ROOT.resolve()
    current_pid = 2001
    parent_pid = 2000
    process_rows = [
        (current_pid, parent_pid, f"python {repo_root}/scripts/clean_suite_outputs.py --include-full-matrix"),
        (
            3001,
            1,
            f"bash {repo_root}/scripts/remote/run_matrix_shard.sh --manifest {repo_root}/results/matrix_shards/suite_all_models_methods/suite_all_models_methods_shard_01_of_02.json",
        ),
    ]

    monkeypatch.setattr(clean_suite_outputs, "ROOT", repo_root)
    monkeypatch.setattr(clean_suite_outputs.os, "name", "posix")
    monkeypatch.setattr(clean_suite_outputs.os, "getpid", lambda: current_pid)
    monkeypatch.setattr(clean_suite_outputs, "_load_process_rows", lambda: process_rows)
    monkeypatch.setattr(clean_suite_outputs, "_process_cwd", lambda pid: None)

    conflicts = clean_suite_outputs._active_cleanup_conflicts()

    assert len(conflicts) == 1
    assert str(repo_root) in conflicts[0]
    assert "run_matrix_shard.sh" in conflicts[0]


def test_active_cleanup_conflicts_detects_windows_same_repo_processes_by_command_line(monkeypatch) -> None:
    repo_root = ROOT.resolve()
    current_pid = 2001
    process_rows = [
        (current_pid, 2000, f"python {repo_root}/scripts/clean_suite_outputs.py --include-full-matrix"),
        (3001, 1, f"python {repo_root}/scripts/remote/run_preflight.sh --formal-full-only"),
    ]

    monkeypatch.setattr(clean_suite_outputs, "ROOT", repo_root)
    monkeypatch.setattr(clean_suite_outputs.os, "name", "nt")
    monkeypatch.setattr(clean_suite_outputs.os, "getpid", lambda: current_pid)
    monkeypatch.setattr(clean_suite_outputs, "_load_process_rows", lambda: process_rows)
    monkeypatch.setattr(clean_suite_outputs, "_process_cwd", lambda pid: None)

    conflicts = clean_suite_outputs._active_cleanup_conflicts()

    assert len(conflicts) == 1
    assert conflicts[0].startswith("3001 ")
    assert "run_preflight.sh" in conflicts[0]


def test_active_cleanup_conflicts_detects_windows_repo_relative_suite_commands(monkeypatch) -> None:
    repo_root = ROOT.resolve()
    current_pid = 2001
    process_rows = [
        (current_pid, 2000, f"python {repo_root}/scripts/clean_suite_outputs.py --include-full-matrix"),
        (3001, 1, "python scripts/remote/run_preflight.sh --formal-full-only"),
    ]

    monkeypatch.setattr(clean_suite_outputs, "ROOT", repo_root)
    monkeypatch.setattr(clean_suite_outputs.os, "name", "nt")
    monkeypatch.setattr(clean_suite_outputs.os, "getpid", lambda: current_pid)
    monkeypatch.setattr(clean_suite_outputs, "_load_process_rows", lambda: process_rows)
    monkeypatch.setattr(clean_suite_outputs, "_process_cwd", lambda pid: None)

    conflicts = clean_suite_outputs._active_cleanup_conflicts()

    assert len(conflicts) == 1
    assert conflicts[0].startswith("3001 ")
    assert "run_preflight.sh" in conflicts[0]


def test_active_paths_include_repo_preflight_lock_for_full_cleanup() -> None:
    paths = clean_suite_outputs._active_paths(
        include_full_matrix=True,
        include_release_bundle=False,
        preserve_precheck_artifacts=False,
        preserve_launcher_artifacts=False,
    )

    assert clean_suite_outputs.ROOT / "results" / "launchers" / ".remote_preflight.lock" in paths


def test_clean_suite_outputs_help_mentions_external_relay_cleanup() -> None:
    script = (ROOT / "scripts" / "clean_suite_outputs.py").read_text(encoding="utf-8")
    assert "relay/smoke artifacts staged inside the repo" in script


def test_environment_placeholder_refresh_command_matches_formal_single_host_contract() -> None:
    script = (ROOT / "scripts" / "clean_suite_outputs.py").read_text(encoding="utf-8")
    assert "--execution-mode single_host_canonical" in script
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in script
