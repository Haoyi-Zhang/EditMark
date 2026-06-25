from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import verify_hf_model_clean_state


def test_verify_hf_model_clean_state_passes_when_model_is_absent(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_hf_model_clean_state.py",
            "--model",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--cache-dir",
            str(tmp_path / "hf_cache"),
            "--report",
            str(report_path),
        ],
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", ""))

    assert verify_hf_model_clean_state.main() == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["issues"] == []


def test_verify_hf_model_clean_state_fails_when_residuals_exist(tmp_path: Path, monkeypatch) -> None:
    cache_root = tmp_path / "hf_cache"
    hub_entry = cache_root / "hub" / "models--Qwen--Qwen2.5-Coder-7B-Instruct"
    hub_entry.mkdir(parents=True, exist_ok=True)
    incomplete = cache_root / "hub" / "models--Qwen--Qwen2.5-Coder-7B-Instruct" / "blob.incomplete"
    incomplete.write_text("partial", encoding="utf-8")
    lock_dir = cache_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    (lock_dir / "models--Qwen--Qwen2.5-Coder-7B-Instruct.lock").write_text("locked", encoding="utf-8")
    extra_file = tmp_path / "js4_redownload_qwen7b.log"
    extra_file.write_text("stale", encoding="utf-8")
    report_path = tmp_path / "report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_hf_model_clean_state.py",
            "--model",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--cache-dir",
            str(cache_root),
            "--process-pattern",
            "js4_redownload_qwen7b",
            "--extra-path",
            str(extra_file),
            "--report",
            str(report_path),
        ],
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "123 1 python js4_redownload_qwen7b\n", ""),
    )

    assert verify_hf_model_clean_state.main() == 1
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert any("hub cache entry still exists" in issue for issue in payload["issues"])
    assert any("lock files still exist" in issue for issue in payload["issues"])
    assert any("incomplete downloads still exist" in issue for issue in payload["issues"])
    assert any("matching process still running" in issue for issue in payload["issues"])
    assert any("extra paths still exist" in issue for issue in payload["issues"])


def test_verify_hf_model_clean_state_fails_when_hub_lock_residual_exists(tmp_path: Path, monkeypatch) -> None:
    cache_root = tmp_path / "hf_cache"
    hub_lock_dir = cache_root / "hub" / ".locks"
    hub_lock_dir.mkdir(parents=True, exist_ok=True)
    (hub_lock_dir / "models--Qwen--Qwen2.5-Coder-7B-Instruct.lock").write_text("locked", encoding="utf-8")
    report_path = tmp_path / "report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_hf_model_clean_state.py",
            "--model",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--cache-dir",
            str(cache_root),
            "--report",
            str(report_path),
        ],
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", ""))

    assert verify_hf_model_clean_state.main() == 1
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert any("lock files still exist" in issue for issue in payload["issues"])


def test_verify_hf_model_clean_state_fails_when_artifact_prefix_residuals_exist(tmp_path: Path, monkeypatch) -> None:
    cache_root = tmp_path / "hf_cache"
    report_path = tmp_path / "report.json"
    artifact_prefix = tmp_path / "js4_redownload_qwen7b"
    (tmp_path / "js4_redownload_qwen7b.py").write_text("print('stale')\n", encoding="utf-8")
    (tmp_path / "js4_redownload_qwen7b.status").write_text("stale", encoding="utf-8")
    (tmp_path / "js4_redownload_qwen7b.pid").write_text("123", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_hf_model_clean_state.py",
            "--model",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--cache-dir",
            str(cache_root),
            "--artifact-prefix",
            str(artifact_prefix),
            "--report",
            str(report_path),
        ],
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", ""))

    assert verify_hf_model_clean_state.main() == 1
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert sorted(payload["existing_artifact_family_paths"]) == sorted(
        [
            str(tmp_path / "js4_redownload_qwen7b.py"),
            str(tmp_path / "js4_redownload_qwen7b.pid"),
            str(tmp_path / "js4_redownload_qwen7b.status"),
        ]
    )
    assert any("artifact-prefix residuals still exist" in issue for issue in payload["issues"])


def test_verify_hf_model_clean_state_fails_on_broken_symlink_residual(tmp_path: Path, monkeypatch) -> None:
    cache_root = tmp_path / "hf_cache"
    broken_root = cache_root / "models--Qwen--Qwen2.5-Coder-7B-Instruct"
    broken_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        broken_root.symlink_to(tmp_path / "missing_target", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    report_path = tmp_path / "report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_hf_model_clean_state.py",
            "--model",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--cache-dir",
            str(cache_root),
            "--report",
            str(report_path),
        ],
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", ""))

    assert verify_hf_model_clean_state.main() == 1
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert any("root cache entry still exists" in issue for issue in payload["issues"])


def test_verify_hf_model_clean_state_ignores_its_own_process_match(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "report.json"

    def fake_run(*args, **kwargs):
        line = f"{__import__('os').getpid()} 1 python js4_redownload_qwen7b\n"
        return subprocess.CompletedProcess(args[0], 0, line, "")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_hf_model_clean_state.py",
            "--model",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--cache-dir",
            str(tmp_path / "hf_cache"),
            "--process-pattern",
            "js4_redownload_qwen7b",
            "--report",
            str(report_path),
        ],
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert verify_hf_model_clean_state.main() == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
