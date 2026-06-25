from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import prepare_hf_model_relay_target


def test_prepare_hf_model_relay_target_passes_when_target_is_clean(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "relay_gate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_hf_model_relay_target.py",
            "--model",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--cache-dir",
            str(tmp_path / "hf_cache"),
            "--report",
            str(report_path),
        ],
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", ""))

    assert prepare_hf_model_relay_target.main() == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["gate_type"] == "hf_model_relay_target_clean_state"


def test_prepare_hf_model_relay_target_fails_on_artifact_prefix_residue(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "relay_gate.json"
    artifact_prefix = tmp_path / "js4_redownload_qwen7b"
    (tmp_path / "js4_redownload_qwen7b.py").write_text("print('stale')\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_hf_model_relay_target.py",
            "--model",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--cache-dir",
            str(tmp_path / "hf_cache"),
            "--artifact-prefix",
            str(artifact_prefix),
            "--report",
            str(report_path),
        ],
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", ""))

    assert prepare_hf_model_relay_target.main() == 1
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["gate_type"] == "hf_model_relay_target_clean_state"
    assert any("artifact-prefix residuals still exist" in issue for issue in payload["issues"])
