from __future__ import annotations

from contextlib import contextmanager
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tests._stone_test_helpers import create_runtime_checkout, update_manifest
from scripts import validate_release_bundle


ROOT = Path(__file__).resolve().parents[1]
BASH_RELEASE_TESTS_ENABLED = shutil.which("bash") is not None and os.name != "nt"
_BASELINE_ENV = {
    "stone_runtime": ("POSTEDITBENCH_STONE_UPSTREAM_ROOT", "POSTEDITBENCH_STONE_UPSTREAM_MANIFEST"),
    "sweet_runtime": ("POSTEDITBENCH_SWEET_UPSTREAM_ROOT", "POSTEDITBENCH_SWEET_UPSTREAM_MANIFEST"),
    "ewd_runtime": ("POSTEDITBENCH_EWD_UPSTREAM_ROOT", "POSTEDITBENCH_EWD_UPSTREAM_MANIFEST"),
    "kgw_runtime": ("POSTEDITBENCH_KGW_UPSTREAM_ROOT", "POSTEDITBENCH_KGW_UPSTREAM_MANIFEST"),
}
_REQUIRED_SUMMARY_EXPORTS = {
    "results/tables/suite_all_models_methods/suite_all_models_methods_method_master_leaderboard.csv": "column\nvalue\n",
    "results/tables/suite_all_models_methods/suite_all_models_methods_method_model_leaderboard.csv": "column\nvalue\n",
    "results/tables/suite_all_models_methods/suite_all_models_methods_model_method_functional_quality.csv": "column\nvalue\n",
    "results/tables/suite_all_models_methods/suite_all_models_methods_utility_robustness_summary.csv": "column\nvalue\n",
    "results/tables/suite_all_models_methods/suite_all_models_methods_model_method_timing.csv": "column\nvalue\n",
    "results/tables/suite_all_models_methods/timing_summary.csv": "column\nvalue\n",
    "results/tables/suite_all_models_methods/suite_all_models_methods_export_identity.json": "{\n  \"status\": \"synthetic_test_surface\"\n}\n",
    "results/figures/suite_all_models_methods/suite_all_models_methods_score_decomposition.png": "synthetic png placeholder\n",
    "results/figures/suite_all_models_methods/suite_all_models_methods_score_decomposition.pdf": "%PDF-1.4\nsynthetic pdf placeholder\n",
    "results/figures/suite_all_models_methods/suite_all_models_methods_score_decomposition.json": "[{\"row_role\":\"synthetic_figure_sidecar\"}]\n",
    "results/figures/suite_all_models_methods/suite_all_models_methods_score_decomposition.csv": "key,value\nplaceholder,1\n",
    "results/figures/suite_all_models_methods/suite_all_models_methods_detection_vs_utility.png": "synthetic png placeholder\n",
    "results/figures/suite_all_models_methods/suite_all_models_methods_detection_vs_utility.pdf": "%PDF-1.4\nsynthetic pdf placeholder\n",
    "results/figures/suite_all_models_methods/suite_all_models_methods_detection_vs_utility.json": "[{\"row_role\":\"synthetic_figure_sidecar\"}]\n",
    "results/figures/suite_all_models_methods/suite_all_models_methods_detection_vs_utility.csv": "key,value\nplaceholder,1\n",
}


def _on_rm_error(func, path, exc_info):
    target = Path(path)
    if target.exists():
        target.chmod(stat.S_IWRITE)
        func(path)


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=_on_rm_error)


def _publish_ready_environment_payload() -> dict[str, object]:
    return {
        "label": "formal_execution_host",
        "platform": {
            "system": "Linux",
            "release": "6.8.0",
            "version": "#1 SMP PREEMPT_DYNAMIC",
            "machine": "x86_64",
        },
        "python": {
            "executable": "/opt/posteditbench/.venv/bin/python",
            "version": "3.11.9",
        },
        "packages": {
            "torch": "2.5.1",
            "transformers": "4.46.3",
            "numpy": "2.1.3",
            "pandas": "2.2.3",
        },
        "tools": {
            "git": {"stdout": "git version 2.43.0"},
            "nvidia-smi": {"stdout": "NVIDIA-SMI 550.54.15"},
        },
        "gpu": {
            "count": 8,
            "visible_gpu_count": 8,
            "cuda_visible_devices": "0,1,2,3,4,5,6,7",
            "driver_version": "550.54.15",
            "devices": [
                {"name": "NVIDIA A40", "driver_version": "550.54.15", "memory_total": "46068 MiB"}
            ],
            "visible_devices": [
                {"name": "NVIDIA A40", "driver_version": "550.54.15", "memory_total": "46068 MiB"}
            ],
        },
        "execution": {
            "execution_mode": "single_host_canonical",
            "cuda_visible_devices": "0,1,2,3,4,5,6,7",
            "visible_gpu_count": 8,
            "code_snapshot_digest": "a" * 64,
            "execution_environment_fingerprint": "b" * 64,
        },
    }


def _publish_ready_environment_markdown() -> str:
    return (
        "# Environment Capture\n\n"
        "- Label: `formal_execution_host`\n"
        "- System: `Linux`\n"
        "- Release: `6.8.0`\n"
        "- Version: `#1 SMP PREEMPT_DYNAMIC`\n"
        "- Machine: `x86_64`\n"
        "- Python executable: `/opt/posteditbench/.venv/bin/python`\n"
        "- Python version: `3.11.9`\n"
        "- Execution mode: `single_host_canonical`\n"
        "- GPU count (physical): `8`\n"
        "- GPU count (visible execution class): `8`\n"
        "- CUDA_VISIBLE_DEVICES: `0,1,2,3,4,5,6,7`\n"
        "- Code snapshot digest: `" + ("a" * 64) + "`\n"
        "- Execution environment fingerprint: `" + ("b" * 64) + "`\n"
        "- GPU driver version: `550.54.15`\n"
        "- CUDA version (torch build): `12.4`\n"
        "- CUDA version (nvidia-smi): `12.4`\n\n"
        "## Package Versions\n"
        "- `torch`: `2.5.1`\n"
        "- `transformers`: `4.46.3`\n"
        "- `numpy`: `2.1.3`\n"
        "- `pandas`: `2.2.3`\n\n"
        "## GPU Devices\n"
        "- `NVIDIA A40` | driver `550.54.15` | memory `46068 MiB`\n\n"
        "## Toolchain Checks\n"
        "- `git`: `ok`\n"
        "  - stdout: `git version 2.43.0`\n"
        "- `nvidia-smi`: `ok`\n"
        "  - stdout: `NVIDIA-SMI 550.54.15`\n"
    )


@contextmanager
def _temporary_publish_ready_environment_capture():
    json_path = ROOT / "results" / "environment" / "runtime_environment.json"
    md_path = ROOT / "results" / "environment" / "runtime_environment.md"
    original_json = json_path.read_text(encoding="utf-8")
    original_md = md_path.read_text(encoding="utf-8")
    try:
        json_path.write_text(
            json.dumps(_publish_ready_environment_payload(), indent=2) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(_publish_ready_environment_markdown(), encoding="utf-8")
        yield
    finally:
        json_path.write_text(original_json, encoding="utf-8")
        md_path.write_text(original_md, encoding="utf-8")


@contextmanager
def _temporary_required_summary_exports():
    created: list[Path] = []
    try:
        for relative, content in _REQUIRED_SUMMARY_EXPORTS.items():
            path = ROOT / relative
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            created.append(path)
        yield
    finally:
        for path in reversed(created):
            if path.exists():
                path.unlink()
        for path in reversed(created):
            parent = path.parent
            while parent != ROOT and parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent


def _run_bundle_script(
    output_relative: Path,
    *,
    expect_success: bool = True,
    preclean: bool = True,
) -> tuple[Path, subprocess.CompletedProcess[str]]:
    output_dir = ROOT / output_relative
    if preclean and output_dir.exists():
        _remove_tree(output_dir)
    with _temporary_required_summary_exports():
        env = os.environ.copy()
        env["PYTHON_BIN"] = sys.executable
        env["POSTEDITBENCH_ALLOW_DIRTY_BUNDLE"] = "1"
        bash = shutil.which("bash") or "bash"
        completed = subprocess.run(
            [bash, "scripts/package_zenodo.sh", output_relative.as_posix()],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    if expect_success:
        assert completed.returncode == 0, completed.stderr or completed.stdout
    return output_dir, completed


def _prepare_external_runtime_suite(tmp_path: Path, monkeypatch) -> None:
    for method, (root_env, manifest_env) in _BASELINE_ENV.items():
        checkout, manifest, _ = create_runtime_checkout(
            ROOT,
            method,
            relative_path=f"external_checkout/{method}.pytest-external",
            manifest_root=tmp_path / "manifests",
            include_license=(method == "kgw_runtime"),
            license_status="redistributable" if method == "kgw_runtime" else "unverified",
        )
        update_manifest(
            manifest,
            checkout_root=f"third_party/{Path(checkout).name}",
            external_root=f"external_checkout/{method}.pytest-external",
            public_external_root=f"external_checkout/{method}.pytest-external",
        )
        monkeypatch.setenv(root_env, str(checkout))
        monkeypatch.setenv(manifest_env, str(manifest))


def test_run_bundle_script_can_preserve_existing_output_for_cleanup_regression(monkeypatch):
    output_relative = Path("results/test_release_bundle_helper_preserve")
    output_dir = ROOT / output_relative
    try:
        _remove_tree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stale.txt").write_text("stale\n", encoding="utf-8")

        def _fake_run(*_args, **_kwargs):
            return subprocess.CompletedProcess(args=["bash"], returncode=1, stdout="", stderr="expected failure")

        monkeypatch.setattr(shutil, "which", lambda _name: "bash")
        monkeypatch.setattr(subprocess, "run", _fake_run)

        returned_dir, completed = _run_bundle_script(output_relative, expect_success=False, preclean=False)

        assert returned_dir == output_dir
        assert completed.returncode == 1
        assert output_dir.exists()
        assert (output_dir / "stale.txt").exists()
    finally:
        _remove_tree(output_dir)


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_records_policy_exclusions(tmp_path: Path, monkeypatch):
    _prepare_external_runtime_suite(tmp_path, monkeypatch)
    output_relative = Path("results/test_release_bundle_pytest")
    output_dir = ROOT / output_relative
    try:
        _remove_tree(output_dir)
        with _temporary_publish_ready_environment_capture():
            output_dir, _completed = _run_bundle_script(output_relative)

        manifest_text = (output_dir / "MANIFEST.txt").read_text(encoding="utf-8")
        excluded_text = (output_dir / "EXCLUDED.txt").read_text(encoding="utf-8")
        bundle_manifest = json.loads((output_dir / "bundle.manifest.json").read_text(encoding="utf-8"))
        provenance = json.loads((output_dir / "baseline_provenance.json").read_text(encoding="utf-8"))

        assert str(ROOT).replace("\\", "/") not in json.dumps(bundle_manifest)
        assert bundle_manifest["bundle_root"] == "results/test_release_bundle_pytest"
        assert set(provenance) == {"stone", "sweet", "ewd", "kgw"}
        assert "results/export_schema.json" in bundle_manifest["included"]
        assert "results/environment/runtime_environment.json" in bundle_manifest["included"]
        assert "results/environment/runtime_environment.md" in bundle_manifest["included"]
        assert "results/audits" not in manifest_text
        for relative_path in (
            ".git",
            "configs/archive",
            "paper",
            "proposal.md",
            "data/interim",
            "results/runs",
            "results/article_preflight",
            "scripts/archive_suite_outputs.py",
        ):
            if (ROOT / relative_path).exists():
                assert f"policy\t{relative_path}" in excluded_text
    finally:
        _remove_tree(output_dir)


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_writes_four_baseline_provenance_entries(tmp_path: Path, monkeypatch):
    _prepare_external_runtime_suite(tmp_path, monkeypatch)
    output_relative = Path("results/test_release_bundle_four_baselines")
    output_dir = ROOT / output_relative
    try:
        _remove_tree(output_dir)
        with _temporary_publish_ready_environment_capture():
            output_dir, _completed = _run_bundle_script(output_relative)
        bundle_manifest = json.loads((output_dir / "bundle.manifest.json").read_text(encoding="utf-8"))
        provenance = json.loads((output_dir / "baseline_provenance.json").read_text(encoding="utf-8"))

        assert set(provenance) == {"stone", "sweet", "ewd", "kgw"}
        assert set(bundle_manifest["baseline_provenance_map"]) == {"stone", "sweet", "ewd", "kgw"}
        assert all(entry["origin"] == "external_checkout" for entry in provenance.values())
        assert all(entry["checkout_valid"] is True for entry in provenance.values())
    finally:
        _remove_tree(output_dir)


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_rejects_local_dirty_external_checkout_state(tmp_path: Path, monkeypatch):
    _prepare_external_runtime_suite(tmp_path, monkeypatch)
    for method, (root_env, _manifest_env) in _BASELINE_ENV.items():
        checkout = Path(os.environ[root_env])
        target = checkout / "runtime_dirty_marker.txt"
        target.write_text("dirty\n", encoding="utf-8")
    output_relative = Path("results/test_release_bundle_dirty_external")
    output_dir = ROOT / output_relative
    try:
        _remove_tree(output_dir)
        with _temporary_publish_ready_environment_capture():
            output_dir, completed = _run_bundle_script(output_relative, expect_success=False)
        assert completed.returncode != 0
        assert "external_unverified" in (completed.stderr or completed.stdout).lower() or "uncommitted changes" in (completed.stderr or completed.stdout).lower()
        assert not output_dir.exists()
    finally:
        _remove_tree(output_dir)


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_removes_stale_output_on_early_failure(tmp_path: Path, monkeypatch):
    _prepare_external_runtime_suite(tmp_path, monkeypatch)
    output_relative = Path("results/test_release_bundle_stale_output_cleanup")
    output_dir = ROOT / output_relative
    try:
        _remove_tree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stale.txt").write_text("stale\n", encoding="utf-8")

        output_dir, completed = _run_bundle_script(output_relative, expect_success=False, preclean=False)
        assert completed.returncode != 0
        assert not output_dir.exists()
    finally:
        _remove_tree(output_dir)


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_rejects_placeholder_environment_capture(tmp_path: Path, monkeypatch):
    _prepare_external_runtime_suite(tmp_path, monkeypatch)
    output_relative = Path("results/test_release_bundle_placeholder_environment")
    output_dir = ROOT / output_relative
    try:
        _remove_tree(output_dir)
        output_dir, completed = _run_bundle_script(output_relative, expect_success=False)
        assert completed.returncode != 0
        combined_output = (completed.stderr or completed.stdout).lower()
        assert (
            "placeholder runtime environment capture" in combined_output
            or "fixed cuda_visible_devices=0,1,2,3,4,5,6,7" in combined_output
            or "visible_gpu_count=8" in combined_output
        )
        assert not output_dir.exists()
    finally:
        _remove_tree(output_dir)


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_rejects_symlinked_vendored_checkout_root(tmp_path: Path, monkeypatch):
    output_relative = Path("results/test_release_bundle_symlinked_vendored")
    output_dir = ROOT / output_relative
    vendored_root = ROOT / "third_party" / "lm-watermarking"
    vendored_backup = vendored_root.with_name("lm-watermarking.pytest-backup")
    if vendored_backup.exists():
        _remove_tree(vendored_backup)
    if vendored_root.exists():
        shutil.move(str(vendored_root), str(vendored_backup))
    try:
        _prepare_external_runtime_suite(tmp_path, monkeypatch)
        external_checkout, manifest, _commit = create_runtime_checkout(
            ROOT,
            "kgw_runtime",
            relative_path="external_checkout/kgw_runtime.symlinked",
            manifest_root=tmp_path / "manifests",
            include_license=True,
            license_status="redistributable",
        )
        update_manifest(
            manifest,
            checkout_root="third_party/lm-watermarking",
            public_external_root="external_checkout/kgw_runtime.symlinked",
        )
        try:
            vendored_root.symlink_to(external_checkout, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("directory symlink creation is not available in this environment")
        monkeypatch.setenv("POSTEDITBENCH_KGW_UPSTREAM_ROOT", str(external_checkout))
        monkeypatch.setenv("POSTEDITBENCH_KGW_UPSTREAM_MANIFEST", str(manifest))

        with _temporary_publish_ready_environment_capture():
            output_dir, completed = _run_bundle_script(output_relative, expect_success=False)
        assert completed.returncode != 0
        combined_output = (completed.stderr or "") + "\n" + (completed.stdout or "")
        assert "symlinked path components" in combined_output.lower()
        assert not output_dir.exists()
    finally:
        _remove_tree(output_dir)
        if vendored_root.is_symlink():
            vendored_root.unlink()
        elif vendored_root.exists():
            _remove_tree(vendored_root)
        if vendored_backup.exists():
            shutil.move(str(vendored_backup), str(vendored_root))


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_rejects_forbidden_nested_vendored_paths(tmp_path: Path, monkeypatch):
    output_relative = Path("results/test_release_bundle_nested_vendored_forbidden")
    output_dir = ROOT / output_relative
    vendored_root = ROOT / "third_party" / "lm-watermarking"
    vendored_backup = vendored_root.with_name("lm-watermarking.pytest-backup")
    if vendored_backup.exists():
        _remove_tree(vendored_backup)
    if vendored_root.exists():
        shutil.move(str(vendored_root), str(vendored_backup))
    try:
        _prepare_external_runtime_suite(tmp_path, monkeypatch)
        checkout, manifest, _commit = create_runtime_checkout(
            ROOT,
            "kgw_runtime",
            relative_path="third_party/lm-watermarking",
            manifest_root=tmp_path / "manifests",
            include_license=True,
            license_status="redistributable",
        )
        forbidden_file = checkout / "results" / "audits" / "trace.json"
        forbidden_file.parent.mkdir(parents=True, exist_ok=True)
        forbidden_file.write_text("{}\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=checkout, check=True, stdin=subprocess.DEVNULL)
        subprocess.run(
            ["git", "commit", "-m", "add forbidden nested residue"],
            cwd=checkout,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        update_manifest(
            manifest,
            checkout_root="third_party/lm-watermarking",
            public_external_root="external_checkout/kgw_runtime.pytest-external",
        )
        monkeypatch.setenv("POSTEDITBENCH_KGW_UPSTREAM_ROOT", str(checkout))
        monkeypatch.setenv("POSTEDITBENCH_KGW_UPSTREAM_MANIFEST", str(manifest))

        with _temporary_publish_ready_environment_capture():
            output_dir, completed = _run_bundle_script(output_relative, expect_success=False)
        assert completed.returncode != 0
        combined_output = (completed.stderr or "") + "\n" + (completed.stdout or "")
        assert "forbidden tracked path" in combined_output.lower()
        assert not output_dir.exists()
    finally:
        _remove_tree(output_dir)
        if vendored_root.exists():
            _remove_tree(vendored_root)
        if vendored_backup.exists():
            shutil.move(str(vendored_backup), str(vendored_root))


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_allows_hidden_vendored_leaf_file(tmp_path: Path, monkeypatch):
    output_relative = Path("results/test_release_bundle_hidden_vendored_leaf")
    output_dir = ROOT / output_relative
    vendored_root = ROOT / "third_party" / "lm-watermarking"
    vendored_backup = vendored_root.with_name("lm-watermarking.pytest-backup")
    if vendored_backup.exists():
        _remove_tree(vendored_backup)
    if vendored_root.exists():
        shutil.move(str(vendored_root), str(vendored_backup))
    try:
        _prepare_external_runtime_suite(tmp_path, monkeypatch)
        checkout, manifest, _commit = create_runtime_checkout(
            ROOT,
            "kgw_runtime",
            relative_path="third_party/lm-watermarking",
            manifest_root=tmp_path / "manifests",
            include_license=True,
            license_status="redistributable",
        )
        hidden_leaf = checkout / ".gitignore"
        hidden_leaf.write_text("*.pyc\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=checkout, check=True, stdin=subprocess.DEVNULL)
        subprocess.run(
            ["git", "commit", "-m", "add hidden vendored leaf"],
            cwd=checkout,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        update_manifest(
            manifest,
            checkout_root="third_party/lm-watermarking",
            public_external_root="external_checkout/kgw_runtime.pytest-external",
        )
        monkeypatch.setenv("POSTEDITBENCH_KGW_UPSTREAM_ROOT", str(checkout))
        monkeypatch.setenv("POSTEDITBENCH_KGW_UPSTREAM_MANIFEST", str(manifest))

        with _temporary_publish_ready_environment_capture():
            output_dir, completed = _run_bundle_script(output_relative)
        assert completed.returncode == 0
        manifest_entries = set((output_dir / "MANIFEST.txt").read_text(encoding="utf-8").splitlines())
        assert "third_party/lm-watermarking/.gitignore" in manifest_entries
    finally:
        _remove_tree(output_dir)
        if vendored_root.exists():
            _remove_tree(vendored_root)
        if vendored_backup.exists():
            shutil.move(str(vendored_backup), str(vendored_root))


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_cleans_output_after_vendored_staging_failure(tmp_path: Path, monkeypatch):
    output_relative = Path("results/test_release_bundle_staging_cleanup")
    output_dir = ROOT / output_relative
    vendored_root = ROOT / "third_party" / "lm-watermarking"
    vendored_backup = vendored_root.with_name("lm-watermarking.pytest-backup")
    if vendored_backup.exists():
        _remove_tree(vendored_backup)
    if vendored_root.exists():
        shutil.move(str(vendored_root), str(vendored_backup))
    try:
        _prepare_external_runtime_suite(tmp_path, monkeypatch)
        checkout, manifest, _commit = create_runtime_checkout(
            ROOT,
            "kgw_runtime",
            relative_path="third_party/lm-watermarking",
            manifest_root=tmp_path / "manifests",
            include_license=True,
            license_status="redistributable",
        )
        tracked_symlink = checkout / "tracked_symlink.py"
        target = checkout / "extended_watermark_processor.py"
        try:
            tracked_symlink.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation is not available in this environment")
        subprocess.run(["git", "add", "-A"], cwd=checkout, check=True, stdin=subprocess.DEVNULL)
        subprocess.run(
            ["git", "commit", "-m", "add tracked symlink"],
            cwd=checkout,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        update_manifest(
            manifest,
            checkout_root="third_party/lm-watermarking",
            public_external_root="external_checkout/kgw_runtime.pytest-external",
        )
        monkeypatch.setenv("POSTEDITBENCH_KGW_UPSTREAM_ROOT", str(checkout))
        monkeypatch.setenv("POSTEDITBENCH_KGW_UPSTREAM_MANIFEST", str(manifest))

        with _temporary_publish_ready_environment_capture():
            output_dir, completed = _run_bundle_script(output_relative, expect_success=False)
        assert completed.returncode != 0
        combined_output = (completed.stderr or "") + "\n" + (completed.stdout or "")
        assert "symlinked tracked file" in combined_output.lower()
        assert not output_dir.exists()
    finally:
        _remove_tree(output_dir)
        if vendored_root.exists():
            _remove_tree(vendored_root)
        if vendored_backup.exists():
            shutil.move(str(vendored_backup), str(vendored_root))


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_bundle_passes_release_validator(tmp_path: Path, monkeypatch):
    _prepare_external_runtime_suite(tmp_path, monkeypatch)
    output_relative = Path("results/test_release_bundle_validator")
    output_dir = ROOT / output_relative
    try:
        _remove_tree(output_dir)
        with _temporary_publish_ready_environment_capture():
            output_dir, _completed = _run_bundle_script(output_relative)
        previous_argv = sys.argv[:]
        try:
            sys.argv = ["validate_release_bundle.py", "--bundle", str(output_dir)]
            assert validate_release_bundle.main() == 0
        finally:
            sys.argv = previous_argv
    finally:
        _remove_tree(output_dir)


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_includes_redistributable_vendored_checkout(tmp_path: Path, monkeypatch):
    output_relative = Path("results/test_release_bundle_vendored_kgw")
    output_dir = ROOT / output_relative
    vendored_root = ROOT / "third_party" / "lm-watermarking"
    vendored_backup = vendored_root.with_name("lm-watermarking.pytest-backup")
    if vendored_backup.exists():
        _remove_tree(vendored_backup)
    if vendored_root.exists():
        shutil.move(str(vendored_root), str(vendored_backup))
    try:
        _prepare_external_runtime_suite(tmp_path, monkeypatch)
        checkout, manifest, commit = create_runtime_checkout(
            ROOT,
            "kgw_runtime",
            relative_path="third_party/lm-watermarking",
            manifest_root=tmp_path / "manifests",
            include_license=True,
            license_status="redistributable",
        )
        update_manifest(
            manifest,
            checkout_root="third_party/lm-watermarking",
            external_root="external_checkout/kgw_runtime.pytest-external",
        )
        monkeypatch.setenv("POSTEDITBENCH_KGW_UPSTREAM_ROOT", str(checkout))
        monkeypatch.setenv("POSTEDITBENCH_KGW_UPSTREAM_MANIFEST", str(manifest))

        with _temporary_publish_ready_environment_capture():
            output_dir, _completed = _run_bundle_script(output_relative)
        manifest_entries = set((output_dir / "MANIFEST.txt").read_text(encoding="utf-8").splitlines())
        provenance = json.loads((output_dir / "baseline_provenance.json").read_text(encoding="utf-8"))

        assert any(entry.startswith("third_party/lm-watermarking/") for entry in manifest_entries)
        assert provenance["kgw"]["origin"] == "vendored_snapshot"
        assert provenance["kgw"]["bundle_eligible"] is True
        assert provenance["kgw"]["upstream_commit"] == commit
        assert sorted(provenance["kgw"]["vendored_files"]) == sorted(
            entry for entry in manifest_entries if entry.startswith("third_party/lm-watermarking/")
        )
    finally:
        _remove_tree(output_dir)
        _remove_tree(vendored_root)
        if vendored_backup.exists():
            shutil.move(str(vendored_backup), str(vendored_root))


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_rejects_unverified_vendored_checkout(tmp_path: Path, monkeypatch):
    output_relative = Path("results/test_release_bundle_unverified_vendored")
    output_dir = ROOT / output_relative
    vendored_root = ROOT / "third_party" / "sweet-watermark"
    vendored_backup = vendored_root.with_name("sweet-watermark.pytest-backup")
    if vendored_backup.exists():
        _remove_tree(vendored_backup)
    if vendored_root.exists():
        shutil.move(str(vendored_root), str(vendored_backup))
    try:
        _prepare_external_runtime_suite(tmp_path, monkeypatch)
        checkout, manifest, _ = create_runtime_checkout(
            ROOT,
            "sweet_runtime",
            relative_path="third_party/sweet-watermark",
            manifest_root=tmp_path / "manifests",
            include_license=False,
            license_status="unverified",
        )
        update_manifest(manifest, checkout_root="third_party/sweet-watermark")
        monkeypatch.setenv("POSTEDITBENCH_SWEET_UPSTREAM_ROOT", str(checkout))
        monkeypatch.setenv("POSTEDITBENCH_SWEET_UPSTREAM_MANIFEST", str(manifest))

        previous = os.getcwd()
        os.chdir(ROOT)
        try:
            with _temporary_publish_ready_environment_capture(), _temporary_required_summary_exports():
                env = os.environ.copy()
                env["PYTHON_BIN"] = sys.executable
                env["POSTEDITBENCH_ALLOW_DIRTY_BUNDLE"] = "1"
                bash = shutil.which("bash") or "bash"
                completed = subprocess.run(
                    [bash, "scripts/package_zenodo.sh", output_relative.as_posix()],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            assert completed.returncode != 0
            assert "redistributable license" in (completed.stderr or completed.stdout).lower() or "vendored_unverified" in (completed.stderr or completed.stdout).lower()
        finally:
            os.chdir(previous)
    finally:
        _remove_tree(output_dir)
        _remove_tree(vendored_root)
        if vendored_backup.exists():
            shutil.move(str(vendored_backup), str(vendored_root))


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_rejects_mixed_state_vendored_checkout_metadata(tmp_path: Path, monkeypatch):
    output_relative = Path("results/test_release_bundle_mixed_state_vendored")
    output_dir = ROOT / output_relative
    vendored_root = ROOT / "third_party" / "lm-watermarking"
    vendored_backup = vendored_root.with_name("lm-watermarking.pytest-backup")
    if vendored_backup.exists():
        _remove_tree(vendored_backup)
    if vendored_root.exists():
        shutil.move(str(vendored_root), str(vendored_backup))
    try:
        _prepare_external_runtime_suite(tmp_path, monkeypatch)
        vendored_root.mkdir(parents=True, exist_ok=True)
        (vendored_root / "LICENSE.md").write_text("redistributable\n", encoding="utf-8")
        kgw_manifest = Path(os.environ["POSTEDITBENCH_KGW_UPSTREAM_MANIFEST"])
        update_manifest(
            kgw_manifest,
            checkout_root="third_party/lm-watermarking",
            public_external_root="external_checkout/kgw_runtime.pytest-external",
        )

        with _temporary_publish_ready_environment_capture():
            output_dir, completed = _run_bundle_script(output_relative, expect_success=False)
        assert completed.returncode != 0
        combined_output = (completed.stderr or "") + "\n" + (completed.stdout or "")
        assert "vendored_unverified" in combined_output.lower() or "verified checkout selected" in combined_output.lower()
        assert not output_dir.exists()
    finally:
        _remove_tree(output_dir)
        _remove_tree(vendored_root)
        if vendored_backup.exists():
            shutil.move(str(vendored_backup), str(vendored_root))


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_ignores_untracked_vendored_checkout_residue(tmp_path: Path, monkeypatch):
    output_relative = Path("results/test_release_bundle_vendored_tracked_only")
    output_dir = ROOT / output_relative
    vendored_root = ROOT / "third_party" / "lm-watermarking"
    vendored_backup = vendored_root.with_name("lm-watermarking.pytest-backup")
    if vendored_backup.exists():
        _remove_tree(vendored_backup)
    if vendored_root.exists():
        shutil.move(str(vendored_root), str(vendored_backup))
    try:
        _prepare_external_runtime_suite(tmp_path, monkeypatch)
        checkout, manifest, _commit = create_runtime_checkout(
            ROOT,
            "kgw_runtime",
            relative_path="third_party/lm-watermarking",
            manifest_root=tmp_path / "manifests",
            include_license=True,
            license_status="redistributable",
        )
        update_manifest(
            manifest,
            checkout_root="third_party/lm-watermarking",
            public_external_root="external_checkout/kgw_runtime.pytest-external",
        )
        monkeypatch.setenv("POSTEDITBENCH_KGW_UPSTREAM_ROOT", str(checkout))
        monkeypatch.setenv("POSTEDITBENCH_KGW_UPSTREAM_MANIFEST", str(manifest))
        (checkout / "local_residue.txt").write_text("should stay local\n", encoding="utf-8")

        with _temporary_publish_ready_environment_capture():
            output_dir, _completed = _run_bundle_script(output_relative)

        manifest_entries = set((output_dir / "MANIFEST.txt").read_text(encoding="utf-8").splitlines())
        provenance = json.loads((output_dir / "baseline_provenance.json").read_text(encoding="utf-8"))
        assert "third_party/lm-watermarking/local_residue.txt" not in manifest_entries
        assert "third_party/lm-watermarking/local_residue.txt" not in provenance["kgw"]["vendored_files"]
    finally:
        _remove_tree(output_dir)
        _remove_tree(vendored_root)
        if vendored_backup.exists():
            shutil.move(str(vendored_backup), str(vendored_root))


def test_public_environment_capture_files_are_sanitized() -> None:
    md = (ROOT / "results/environment/runtime_environment.md").read_text(encoding="utf-8")
    payload = json.loads((ROOT / "results/environment/runtime_environment.json").read_text(encoding="utf-8"))
    serialized_payload = json.dumps(payload, sort_keys=True)
    freeze = (ROOT / "results/environment/release_pip_freeze.txt").read_text(encoding="utf-8")

    forbidden_public_markers = (
        "local_pre_run_preview",
        "C:" + "\\Users\\Administrator",
        "NVIDIA GeForce RTX 3070 Ti",
        "private-cloud" + ".example.invalid",
        "example-" + "ssh-port",
        "example-" + "password-fragment",
        "/private" + "/execution-root",
        "github" + "_pat_",
        "gh" + "p_",
        "BEGIN " + "OPENSSH PRIVATE KEY",
        "BEGIN " + "RSA PRIVATE KEY",
    )
    for marker in forbidden_public_markers:
        assert marker not in md
        assert marker not in serialized_payload
        assert marker not in freeze

    assert payload["label"] == "formal-single-host-full"
    assert payload["host"] == {"hostname": "execution-host", "fqdn": "execution-host"}
    assert payload["python"]["executable"] == "<release-python>/python"
    assert payload["execution"]["execution_mode"] == "single_host_canonical"
    assert payload["execution"]["cuda_visible_devices"] == "0,1,2,3,4,5,6,7"
    assert payload["execution"]["visible_gpu_count"] == 8
    assert payload["gpu"]["count"] == 8
    assert payload["gpu"]["visible_gpu_count"] == 8
    assert all(device["name"] == "NVIDIA A800-SXM4-40GB" for device in payload["gpu"]["devices"])
    assert "torch==2.6.0+cu124" in freeze
    assert "transformers==4.57.6" in freeze
    assert "numpy==2.2.6" in freeze
