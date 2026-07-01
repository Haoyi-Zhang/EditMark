from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import export_publish_repo, refresh_public_snapshots


def test_export_publish_repo_clean_tree_guard_rejects_untracked_files(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# PostEditBench\n", encoding="utf-8")

    monkeypatch.setattr(export_publish_repo, "ROOT", root)
    monkeypatch.setattr(export_publish_repo, "_git_lines", lambda *args: ["?? untracked.secret"] if args[:2] == ("status", "--short") else [])

    with pytest.raises(SystemExit, match="clean tracked working tree"):
        export_publish_repo._ensure_clean_tree()


def test_export_publish_repo_copies_only_tracked_files(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    tracked = root / "README.md"
    tracked.write_text("# PostEditBench\n", encoding="utf-8")
    hidden = root / "notes.secret"
    hidden.write_text("do not ship\n", encoding="utf-8")

    output = tmp_path / "publish"
    monkeypatch.setattr(export_publish_repo, "ROOT", root)
    monkeypatch.setattr(export_publish_repo.release_bundle_contract, "tracked_bundle_surface", lambda repo_root: {"README.md"})
    export_publish_repo._copy_tracked_files(output)

    assert (output / "README.md").exists()
    assert not (output / "notes.secret").exists()


def test_export_publish_repo_rejects_preexisting_output_directory(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    output = tmp_path / "publish"
    output.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(export_publish_repo, "ROOT", root)
    monkeypatch.setattr(export_publish_repo, "_ensure_clean_tree", lambda: None)
    monkeypatch.setattr(export_publish_repo, "_validate_required_public_assets", lambda bundle_root: None)
    monkeypatch.setattr(export_publish_repo, "_ensure_real_environment_capture", lambda bundle_root: None)
    monkeypatch.setattr(export_publish_repo, "_ensure_release_summary_provenance", lambda bundle_root: None)
    monkeypatch.setattr(export_publish_repo, "_copy_tracked_files", lambda bundle_root: None)
    monkeypatch.setattr(export_publish_repo, "_validate_public_snapshot_identity_markers", lambda bundle_root: None)
    monkeypatch.setattr(export_publish_repo.check_zero_compatibility_name, "scan_tree", lambda bundle_root: [])
    monkeypatch.setattr(export_publish_repo, "_init_git", lambda *args, **kwargs: None)

    previous_argv = sys.argv[:]
    try:
        sys.argv = ["export_publish_repo.py", "--output", str(output)]
        with pytest.raises(SystemExit, match="fresh nonexistent directory"):
            export_publish_repo.main()
    finally:
        sys.argv = previous_argv


def test_export_publish_repo_validates_required_public_assets(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# PostEditBench\n", encoding="utf-8")

    monkeypatch.setattr(export_publish_repo, "ROOT", root)
    monkeypatch.setattr(export_publish_repo, "REQUIRED_PUBLIC_FILES", (Path("README.md"), Path("LICENSE")))

    with pytest.raises(SystemExit, match="missing required public assets"):
        export_publish_repo._validate_required_public_assets(root)


def test_export_publish_repo_identity_scan_rejects_leaked_metadata(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "publish"
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("contact author@example.invalid\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="identity-marker findings"):
        export_publish_repo._validate_public_snapshot_identity_markers(root)


def test_export_publish_repo_identity_scan_skips_test_fixtures(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "publish"
    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "fixture_test.py").write_text("CONTACT = 'author@example.invalid'\n", encoding="utf-8")

    export_publish_repo._validate_public_snapshot_identity_markers(root)


def test_export_publish_repo_rejects_invalid_release_summary_provenance(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "publish"
    root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        export_publish_repo.validate_release_bundle,
        "_summary_export_identity_issues",
        lambda bundle_root: ["summary export identity does not prove one-shot provenance"],
    )

    with pytest.raises(SystemExit, match="validated release summary provenance"):
        export_publish_repo._ensure_release_summary_provenance(root)


def test_export_publish_repo_reuses_environment_validator(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "publish"
    root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        export_publish_repo.validate_release_bundle,
        "_environment_capture_issues",
        lambda bundle_root: ["runtime environment JSON execution_environment_fingerprint must match the structured capture"],
    )

    with pytest.raises(SystemExit, match="validated environment capture"):
        export_publish_repo._ensure_real_environment_capture(root)


def test_refresh_public_snapshots_defaults_to_release_sources(monkeypatch, tmp_path: Path) -> None:
    release_root = tmp_path / "data" / "release" / "sources"
    release_root.mkdir(parents=True, exist_ok=True)
    target = release_root / "suite_humaneval_plus_release.normalized.jsonl"
    target.write_text(
        json.dumps(
            {
                "source_path": str(tmp_path / ".coordination" / "external" / "hidden.jsonl"),
                "official_problem_file": str(tmp_path / ".coordination" / "external" / "problem.json"),
                "public_source": "HumanEval+",
                "dataset": "HumanEval+",
                "source_group": "humaneval_plus",
                "language": "python",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(refresh_public_snapshots, "RELEASE_SOURCES_ROOT", release_root)
    previous_argv = sys.argv[:]
    try:
        sys.argv = ["refresh_public_snapshots.py"]
        assert refresh_public_snapshots.main() == 0
    finally:
        sys.argv = previous_argv

    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["source_path"] == "hidden.jsonl"
    assert rows[0]["official_problem_file"] == "problem.json"
