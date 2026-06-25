from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import validate_single_host_launch_receipt


def test_validate_single_host_launch_receipt_accepts_matching_gate(monkeypatch, capsys) -> None:
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=Path("results/matrix"),
        precheck_gate=Path("results/certifications/suite_precheck_gate.json"),
        skip_hf_access=True,
    )
    monkeypatch.setattr(validate_single_host_launch_receipt, "parse_args", lambda: args)
    monkeypatch.setattr(
        validate_single_host_launch_receipt.certify_suite_precheck,
        "_load_matching_launch_receipt",
        lambda parsed_args, *, output_root, gate_path: {"status": "ready"},
    )

    assert validate_single_host_launch_receipt.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["reason"] == "launch_time_post_precheck_receipt_match"
    assert payload["precheck_gate"].replace("\\", "/").endswith("results/certifications/suite_precheck_gate.json")


def test_validate_single_host_launch_receipt_rejects_missing_gate(monkeypatch) -> None:
    args = SimpleNamespace(
        python_bin="python",
        full_manifest=Path("configs/matrices/suite_all_models_methods.json"),
        full_profile="suite_all_models_methods",
        stage_a_manifest=Path("configs/matrices/suite_canary_heavy.json"),
        stage_a_profile="suite_canary_heavy",
        stage_b_manifest=Path("configs/matrices/model_invocation_smoke.json"),
        stage_b_profile="model_invocation_smoke",
        output_root=Path("results/matrix"),
        precheck_gate=Path("results/certifications/suite_precheck_gate.json"),
        skip_hf_access=True,
    )
    monkeypatch.setattr(validate_single_host_launch_receipt, "parse_args", lambda: args)
    monkeypatch.setattr(
        validate_single_host_launch_receipt.certify_suite_precheck,
        "_load_matching_launch_receipt",
        lambda parsed_args, *, output_root, gate_path: None,
    )

    with pytest.raises(SystemExit, match="post-precheck receipt validation failed"):
        validate_single_host_launch_receipt.main()
