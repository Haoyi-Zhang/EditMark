from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts import render_materialized_summary_figures


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _method_rows() -> list[dict[str, object]]:
    return [
        {
            "method": method,
            "origin": "upstream",
            "score_version": "posteditbench-suite-v7-structural-unsupported-generalization",
            "generalization_status": "unsupported" if method != "STONE" else "supported_zero",
            "generalization": None if method != "STONE" else 0.0,
            "headline_generalization": 0.5 if method != "STONE" else 0.05,
            "gate": 0.9,
            "headline_core_score": 0.2,
            "detection_separability": 0.8,
            "robustness": 0.0 if method != "STONE" else 0.1,
            "utility": 0.6,
            "stealth_conditioned": 0.7,
            "efficiency_conditioned": 0.8,
            "PostEditScore": 0.09 if method != "STONE" else 0.0045,
        }
        for method in ("STONE", "SWEET", "EWD", "KGW")
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_identity(table_dir: Path) -> Path:
    required_tables = (
        "method_summary.json",
        "suite_all_models_methods_method_master_leaderboard.json",
        "suite_all_models_methods_method_model_leaderboard.json",
        "suite_all_models_methods_model_method_functional_quality.json",
    )
    identity = {
        "artifact_role": "suite_all_models_methods_release_summary_export_identity",
        "schema_version": 1,
        "manifest": "configs/matrices/suite_all_models_methods.json",
        "profile": "suite_all_models_methods",
        "canonical_manifest_digest": render_materialized_summary_figures.rpf._CANONICAL_SUITE_MANIFEST_DIGEST,
        "execution_mode": "single_host_canonical",
        "run_count": 140,
        "success_count": 140,
        "failed_count": 0,
        "score_version": "posteditbench-suite-v7-structural-unsupported-generalization",
        "model_roster": [
            {
                "model": "Qwen/Qwen2.5-Coder-14B-Instruct",
                "model_revision": "aedcc2d42b622764e023cf882b6652e646b95671",
                "model_slug": "qwen25_14b",
                "model_family": "qwen25",
            },
            {
                "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "model_revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
                "model_slug": "qwen25_7b",
                "model_family": "qwen25",
            },
            {
                "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "model_revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a",
                "model_slug": "qwen25_1p5b",
                "model_family": "qwen25",
            },
            {
                "model": "bigcode/starcoder2-7b",
                "model_revision": "bb9afde76d7945da5745592525db122d4d729eb1",
                "model_slug": "starcoder2_7b",
                "model_family": "starcoder2",
            },
            {
                "model": "deepseek-ai/deepseek-coder-6.7b-instruct",
                "model_revision": "e5d64addd26a6a1db0f9b863abf6ee3141936807",
                "model_slug": "deepseek_coder_6p7b",
                "model_family": "deepseek_coder",
            },
        ],
        "required_table_hashes": {
            name: _sha256(table_dir / name)
            for name in required_tables
        },
        "figure_stems": [
            "suite_all_models_methods_score_decomposition",
            "suite_all_models_methods_detection_vs_utility",
        ],
    }
    identity_path = table_dir / "suite_all_models_methods_export_identity.json"
    _write_json(identity_path, identity)
    return identity_path


def _prepare_tables(
    tmp_path: Path,
    *,
    method_summary_rows: list[dict[str, object]] | None = None,
    method_master_rows: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    table_dir = tmp_path / "tables"
    summary_rows = method_summary_rows or _method_rows()
    master_rows = method_master_rows or _method_rows()
    _write_json(table_dir / "method_summary.json", summary_rows)
    _write_json(table_dir / "suite_all_models_methods_method_master_leaderboard.json", master_rows)
    _write_json(table_dir / "suite_all_models_methods_method_model_leaderboard.json", master_rows)
    _write_json(table_dir / "suite_all_models_methods_model_method_functional_quality.json", master_rows)
    identity_path = _write_identity(table_dir)
    return table_dir, identity_path


def test_materialized_redraw_requires_matching_export_identity(tmp_path: Path, monkeypatch) -> None:
    table_dir, identity_path = _prepare_tables(tmp_path)
    output_dir = tmp_path / "figures"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_materialized_summary_figures.py",
            "--table-dir",
            str(table_dir),
            "--output-dir",
            str(output_dir),
            "--export-identity",
            str(identity_path),
            "--allow-font-fallback",
        ],
    )

    render_materialized_summary_figures.main()

    assert (output_dir / "suite_all_models_methods_score_decomposition.png").exists()
    assert (output_dir / "suite_all_models_methods_detection_vs_utility.png").exists()
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    recorded_hashes = identity["required_figure_hashes"]
    assert recorded_hashes["suite_all_models_methods_score_decomposition.png"] == _sha256(
        output_dir / "suite_all_models_methods_score_decomposition.png"
    )
    assert recorded_hashes["suite_all_models_methods_detection_vs_utility.png"] == _sha256(
        output_dir / "suite_all_models_methods_detection_vs_utility.png"
    )


def test_materialized_redraw_rejects_hash_mismatch(tmp_path: Path, monkeypatch) -> None:
    table_dir, identity_path = _prepare_tables(tmp_path)
    _write_json(table_dir / "method_summary.json", _method_rows()[:-1])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_materialized_summary_figures.py",
            "--table-dir",
            str(table_dir),
            "--output-dir",
            str(tmp_path / "figures"),
            "--export-identity",
            str(identity_path),
            "--allow-font-fallback",
        ],
    )

    with pytest.raises(SystemExit, match="does not match the recorded export identity"):
        render_materialized_summary_figures.main()


def test_materialized_redraw_uses_method_master_leaderboard_surface(tmp_path: Path, monkeypatch) -> None:
    summary_rows = _method_rows()
    master_rows = [dict(row, utility=round(float(row["utility"]) + 0.1, 3)) for row in _method_rows()]
    table_dir, identity_path = _prepare_tables(
        tmp_path,
        method_summary_rows=summary_rows,
        method_master_rows=master_rows,
    )
    captured: dict[str, list[dict[str, object]]] = {}

    monkeypatch.setattr(
        render_materialized_summary_figures.rpf,
        "plot_score_decomposition",
        lambda _plt, rows, **_kwargs: captured.setdefault("score_rows", list(rows)) or [],
    )
    monkeypatch.setattr(
        render_materialized_summary_figures.rpf,
        "plot_detection_vs_utility",
        lambda _plt, rows, **_kwargs: captured.setdefault("frontier_rows", list(rows)) or [],
    )
    monkeypatch.setattr(
        render_materialized_summary_figures.rpf,
        "configure_matplotlib",
        lambda **_kwargs: (None, object()),
    )
    monkeypatch.setattr(
        render_materialized_summary_figures,
        "_update_export_identity_figure_hashes",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_materialized_summary_figures.py",
            "--table-dir",
            str(table_dir),
            "--output-dir",
            str(tmp_path / "figures"),
            "--export-identity",
            str(identity_path),
            "--allow-font-fallback",
        ],
    )

    render_materialized_summary_figures.main()

    assert captured["score_rows"] == master_rows
    assert captured["frontier_rows"] == master_rows
