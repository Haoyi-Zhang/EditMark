from __future__ import annotations

from pathlib import Path
import sys

import pytest

from scripts import reviewer_workflow


def test_regenerate_fails_cleanly_when_matrix_index_is_missing(tmp_path: Path) -> None:
    matrix_index = tmp_path / "missing_matrix_index.json"

    with pytest.raises(SystemExit, match="Matrix index not found for regeneration"):
        reviewer_workflow.main(
            [
                "regenerate",
                "--matrix-index",
                str(matrix_index),
            ]
        )


def test_regenerate_uses_table_export_then_materialized_redraw(monkeypatch, tmp_path: Path) -> None:
    matrix_index = tmp_path / "matrix_index.json"
    matrix_index.write_text("{}", encoding="utf-8")
    figure_dir = tmp_path / "figures"
    table_dir = tmp_path / "tables"
    commands: list[list[str]] = []

    monkeypatch.setattr(reviewer_workflow, "_ensure_matrix_index_exists", lambda _path: None)
    monkeypatch.setattr(reviewer_workflow, "_ensure_regenerate_output_contract", lambda *_args: None)
    monkeypatch.setattr(reviewer_workflow, "_run", lambda command, log_path=None: commands.append(list(command)))

    reviewer_workflow._regenerate_outputs(
        sys.executable,
        matrix_index,
        figure_dir,
        table_dir,
        require_times_new_roman=False,
    )

    assert commands[0][1].endswith("refresh_report_metadata.py")
    assert commands[1][1].endswith("export_full_run_tables.py")
    assert commands[2][1].endswith("render_materialized_summary_figures.py")
    assert commands[3][1].endswith("export_dataset_statistics.py")
    assert "--export-identity" in commands[2]


def test_regenerate_rejects_canonical_outputs_for_non_single_host_matrix(tmp_path: Path) -> None:
    matrix_index = tmp_path / "matrix_index.json"
    matrix_index.write_text(
        '{"manifest":"configs/matrices/suite_all_models_methods.json","profile":"suite_all_models_methods","run_count":140,"success_count":140,"failed_count":0,"execution_mode":"sharded_identical_execution_class"}\n',
        encoding="utf-8",
    )

    canonical_figure_dir = tmp_path / "figures"
    canonical_table_dir = tmp_path / "tables"

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(reviewer_workflow, "_default_matrix_index", lambda: matrix_index)
        monkeypatch.setattr(reviewer_workflow, "_default_figure_dir", lambda: canonical_figure_dir)
        monkeypatch.setattr(reviewer_workflow, "_default_table_dir", lambda: canonical_table_dir)
        with pytest.raises(SystemExit, match="execution_mode=single_host_canonical"):
            reviewer_workflow._ensure_regenerate_output_contract(
                matrix_index,
                canonical_figure_dir,
                canonical_table_dir,
            )


def test_subset_dry_run_prints_build_audit_capture_and_run_commands(capsys) -> None:
    assert (
        reviewer_workflow.main(
            [
                "subset",
                "--models",
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "--methods",
                "sweet_runtime",
                "--sources",
                "crafted_original",
                "--python",
                sys.executable,
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "build_suite_manifests.py" in output
    assert "audit_benchmarks.py --manifest" in output
    assert "--matrix-profile suite_reviewer_subset" in output
    assert "--profile suite_reviewer_subset" in output
    assert "audit_full_matrix.py" in output
    assert "--strict-hf-cache" in output
    assert "--skip-hf-access" in output
    assert "capture_environment.py" in output
    assert "run_full_matrix.py" in output
    assert "Execution artifacts:" in output
    assert "Matrix index: results/matrix/suite_reviewer_subset/matrix_index.json" in output
    assert "Monitor command:" in output
    assert "monitor_matrix.py" in output


def test_subset_dry_run_surfaces_limit_override(capsys) -> None:
    assert (
        reviewer_workflow.main(
            [
                "subset",
                "--models",
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                "--methods",
                "kgw_runtime",
                "--sources",
                "humaneval_plus",
                "--limit",
                "8",
                "--python",
                sys.executable,
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "build_suite_manifests.py" in output
    assert "--limit 8" in output


def test_subset_dry_run_accepts_benchmark_source_alias(capsys) -> None:
    assert (
        reviewer_workflow.main(
            [
                "subset",
                "--models",
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                "--methods",
                "kgw_runtime",
                "--benchmark-source",
                "humaneval_plus",
                "--python",
                sys.executable,
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "--sources humaneval_plus" in output


def test_subset_no_run_dry_run_omits_execution_artifacts(capsys) -> None:
    assert reviewer_workflow.main(["subset", "--python", sys.executable, "--dry-run", "--no-run"]) == 0

    output = capsys.readouterr().out
    assert "build_suite_manifests.py" in output
    assert "capture_environment.py" not in output
    assert "run_full_matrix.py" not in output
    assert "Execution artifacts:" not in output
    assert "Build-only subset dry-run" in output


def test_full_dry_run_prints_build_audit_capture_and_run_commands(capsys) -> None:
    assert reviewer_workflow.main(["full", "--python", sys.executable, "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "run_formal_single_host_full.sh" in output
    assert "--gpu-slots 8" in output
    assert "--gpu-pool-mode shared" in output
    assert "--cpu-workers 9" in output
    assert "--retry-count 1" in output
    assert "--command-timeout-seconds 259200" in output
    assert "--skip-hf-access" in output
    assert "build_suite_manifests.py" not in output
    assert "audit_full_matrix.py" not in output
    assert "capture_environment.py" not in output
    assert "run_full_matrix.py" not in output
    assert "--fail-fast" not in output
    assert "Execution artifacts:" in output
    assert "Matrix index: results/matrix/suite_all_models_methods/matrix_index.json" in output
    assert "Monitor command:" in output
    assert "monitor_matrix.py" in output


def test_full_dry_run_rejects_resume_for_formal_single_host_path() -> None:
    with pytest.raises(SystemExit, match="does not support --resume"):
        reviewer_workflow.main(["full", "--python", sys.executable, "--dry-run", "--resume"])


def test_full_dry_run_treats_repo_relative_default_manifest_as_canonical(capsys) -> None:
    assert (
        reviewer_workflow.main(
            [
                "full",
                "--python",
                sys.executable,
                "--dry-run",
                "--manifest",
                "configs/matrices/suite_all_models_methods.json",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "run_formal_single_host_full.sh" in output
    assert "--manifest" not in output
    assert "suite_all_models_methods.json" in output


def test_full_dry_run_with_existing_custom_manifest_does_not_rebuild_it(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "custom_full_manifest.json"
    manifest.write_text('{"profile":"custom_full"}\n', encoding="utf-8")

    assert reviewer_workflow.main(
        [
            "full",
            "--python",
            sys.executable,
            "--dry-run",
            "--manifest",
            str(manifest),
            "--profile",
            "custom_full",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "build_suite_manifests.py" not in output
    assert f"--manifest {manifest}" in output
    assert "--profile custom_full" in output


def test_full_rejects_manifest_profile_mismatch(tmp_path: Path) -> None:
    manifest = tmp_path / "custom_full_manifest.json"
    manifest.write_text('{"profile":"custom_full"}\n', encoding="utf-8")

    with pytest.raises(SystemExit, match="Manifest/profile mismatch for full reviewer workflow"):
        reviewer_workflow.main(
            [
                "full",
                "--python",
                sys.executable,
                "--dry-run",
                "--manifest",
                str(manifest),
                "--profile",
                "different_profile",
            ]
        )


def test_full_rejects_missing_custom_manifest() -> None:
    with pytest.raises(SystemExit, match="Custom full-manifest runs require an existing manifest file"):
        reviewer_workflow.main(
            [
                "full",
                "--python",
                sys.executable,
                "--dry-run",
                "--manifest",
                str(Path("missing_custom_manifest.json")),
                "--profile",
                "custom_full",
            ]
        )


def test_browse_mentions_release_slice_and_placeholder_state(capsys) -> None:
    assert reviewer_workflow.main(["browse"]) == 0

    output = capsys.readouterr().out
    assert "results/figures/dataset_statistics/release_slice_composition.png" in output
    assert "results/tables/dataset_statistics/release_slice_language_breakdown.csv" in output
    assert "results/tables/dataset_statistics/dataset_task_category_breakdown.csv" in output
    assert "results/tables/dataset_statistics/dataset_family_breakdown.csv" in output
    assert "suite_all_models_methods_score_decomposition.png" in output
    assert "suite_all_models_methods_detection_vs_utility.png" in output
    assert "suite_all_models_methods_method_master_leaderboard.csv" in output
    assert "suite_all_models_methods_utility_robustness_summary.csv" in output
    figures_dir = reviewer_workflow.ROOT / "results" / "figures" / "suite_all_models_methods"
    tables_dir = reviewer_workflow.ROOT / "results" / "tables" / "suite_all_models_methods"
    figure_status = reviewer_workflow._status_label(figures_dir, expect_rerun_backed=True)
    table_status = reviewer_workflow._status_label(tables_dir, expect_rerun_backed=True)
    assert f"{figures_dir} [{figure_status}]" in output
    assert f"{tables_dir} [{table_status}]" in output
    assert "publication-facing review packet is rooted in the canonical suite_all_models_methods figure and table directories" in output
    assert "tracked figure roster is intentionally narrow" in output
    assert "Qwen/Qwen2.5-Coder-14B-Instruct @" in output


def test_full_dry_run_can_probe_hf_access(capsys) -> None:
    assert reviewer_workflow.main(["full", "--python", sys.executable, "--dry-run", "--probe-hf-access"]) == 0

    output = capsys.readouterr().out
    assert "run_formal_single_host_full.sh" in output
    assert "--skip-hf-access" not in output


def test_python_bin_accepts_repo_local_windows_venv(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "PostEditBench"
    windows_python = repo_root / ".venv" / "Scripts" / "python.exe"
    windows_python.parent.mkdir(parents=True, exist_ok=True)
    windows_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(reviewer_workflow, "ROOT", repo_root)
    monkeypatch.setattr(reviewer_workflow.sys, "executable", str(tmp_path / "system_python.exe"))
    monkeypatch.delenv("PYTHON_BIN", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)

    assert reviewer_workflow._python_bin(None) == str(windows_python)


def test_full_dry_run_uses_remote_wrapper_for_canonical_posix(monkeypatch, capsys) -> None:
    monkeypatch.setattr(reviewer_workflow.os, "name", "posix")

    assert reviewer_workflow.main(["full", "--python", sys.executable, "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "run_formal_single_host_full.sh" in output
    assert "run_full_matrix.py" not in output


def test_subset_dry_run_surfaces_resume_when_requested(capsys) -> None:
    assert reviewer_workflow.main(["subset", "--python", sys.executable, "--dry-run", "--resume"]) == 0

    output = capsys.readouterr().out
    assert "run_full_matrix.py" in output
    assert "--resume" in output


def test_subset_run_prints_execution_artifacts_before_matrix_launch(monkeypatch, capsys) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs) -> None:
        commands.append(list(command))
        print("+ " + " ".join(command))

    def fake_capture(*args, **kwargs) -> None:
        print("capture_environment")

    monkeypatch.setattr(reviewer_workflow, "_run", fake_run)
    monkeypatch.setattr(reviewer_workflow, "_capture_environment", fake_capture)

    assert (
        reviewer_workflow.main(
            [
                "subset",
                "--models",
                "Qwen/Qwen2.5-Coder-14B-Instruct",
                "--methods",
                "sweet_runtime",
                "--sources",
                "crafted_original",
                "--python",
                sys.executable,
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Execution artifacts:" in output
    assert "run_full_matrix.py" in output
    assert output.index("Execution artifacts:") < output.index("run_full_matrix.py")


def test_subset_dry_run_keeps_fail_fast_by_default(capsys) -> None:
    assert reviewer_workflow.main(["subset", "--python", sys.executable, "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "run_full_matrix.py" in output
    assert "--fail-fast" in output
