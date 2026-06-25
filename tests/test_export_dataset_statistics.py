from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import export_dataset_statistics


def test_export_dataset_statistics_writes_expected_outputs(tmp_path: Path, monkeypatch) -> None:
    table_dir = tmp_path / "tables"
    figure_dir = tmp_path / "figures"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_dataset_statistics.py",
            "--table-dir",
            str(table_dir),
            "--figure-dir",
            str(figure_dir),
            "--allow-font-fallback",
        ],
    )

    assert export_dataset_statistics.main() == 0

    expected_tables = {
        "release_slice_summary.json",
        "benchmark_definition_summary.json",
        "release_slice_language_breakdown.json",
        "release_source_manifest_index.json",
        "dataset_task_category_breakdown.json",
        "dataset_family_breakdown.json",
        "dataset_statistics_manifest.json",
    }
    assert expected_tables.issubset({path.name for path in table_dir.iterdir()})

    expected_figures = {
        "release_slice_composition.pdf",
        "evaluation_dimensions_overview.pdf",
    }
    assert expected_figures.issubset({path.name for path in figure_dir.iterdir()})

    manifest = json.loads((table_dir / "dataset_statistics_manifest.json").read_text(encoding="utf-8"))
    assert manifest["active_sources"] == 7
    assert manifest["manual_review_files"] == [
        "benchmark_definition_summary.csv",
        "release_slice_language_breakdown.csv",
        "dataset_task_category_breakdown.csv",
        "dataset_family_breakdown.csv",
        "release_source_manifest_index.csv",
        "release_slice_composition.png",
        "evaluation_dimensions_overview.png",
    ]

    benchmark_definition = json.loads((table_dir / "benchmark_definition_summary.json").read_text(encoding="utf-8"))
    mbxp = next(row for row in benchmark_definition if row["slug"] == "mbxp_5lang")
    assert mbxp["source"] == "MBXP-5lang (5-language balanced slice)"
    assert mbxp["active_release_size"] == 200
    assert mbxp["execution_slice"] == "python/cpp/java/javascript/go"
    assert mbxp["scored_in_aggregate"] is True
    assert "smoke-overlay support" in mbxp["sampling_rule"]

    manifest_index = json.loads((table_dir / "release_source_manifest_index.json").read_text(encoding="utf-8"))
    mbxp_manifest = next(row for row in manifest_index if row["slug"] == "mbxp_5lang")
    assert mbxp_manifest["record_count"] == 200
    assert mbxp_manifest["executed_release_count"] == 200
    assert mbxp_manifest["canonical_reference_count"] < 200
    assert mbxp_manifest["smoke_overlay_reference_count"] > 0
    assert mbxp_manifest["reference_support_mode"] == "mixed_canonical_and_smoke_overlay"

    task_breakdown = json.loads((table_dir / "dataset_task_category_breakdown.json").read_text(encoding="utf-8"))
    assert task_breakdown
    assert {row["analysis_view"] for row in task_breakdown} == {"crafted_only"}
