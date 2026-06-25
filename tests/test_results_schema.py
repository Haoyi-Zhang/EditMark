from __future__ import annotations

import json
from pathlib import Path


def test_results_schema_covers_baseline_provenance_fields():
    schema = json.loads(Path("results/schema.json").read_text(encoding="utf-8"))

    assert schema["record"]["baseline_family"] == "string"
    assert schema["record"]["baseline_origin"] == "string"
    assert schema["record"]["baseline_upstream_commit"] == "string"
    assert schema["record"]["evaluation_track"] == "string"
    assert schema["summary"]["by_baseline_family"] == "object"
    assert schema["summary"]["by_evaluation_track"] == "object"
    assert schema["summary"]["baseline_families"] == "array"
    assert schema["summary"]["baseline_origins"] == "array"
    assert schema["summary"]["baseline_upstream_commits"] == "array"
    assert schema["summary"]["evaluation_tracks"] == "array"
    assert schema["summary"]["paper_primary_track"] == "string"
    assert schema["summary"]["paper_track_ready"] == "boolean"
    assert schema["summary"]["watermarked_functional_metrics"] == "object"
    assert schema["summary"]["runtime_validation_basis"] == "string"
    assert schema["summary"]["runtime_validation_annotations_available"] == "boolean"
    assert schema["summary"]["clean_reference_compile_rate"] == "number|null"
    assert schema["summary"]["clean_reference_pass_rate"] == "number|null"
    assert schema["scorecard"]["PostEditScore"] == "number"
    assert schema["scorecard"]["headline_core_score"] == "number"
    assert schema["scorecard"]["generalization"] == "number|null"
    assert schema["scorecard"]["headline_generalization"] == "number"
    assert schema["scorecard"]["generalization_status"] == "string"
    assert schema["record"]["attack_outcome_score"] == "number"
    assert schema["summary"]["mean_attack_outcome_score"] == "number"
    assert schema["scorecard"]["raw_robustness_strict"] == "number|null"
    assert schema["scorecard"]["robustness_support_rate"] == "number"
    assert schema["scorecard"]["stress_robustness"] == "number|null"
    assert schema["scorecard"]["raw_utility_strict"] == "number|null"
    assert schema["scorecard"]["utility_support_rate"] == "number"
    assert schema["scorecard"]["raw_generalization_strict"] == "number|null"
    assert schema["scorecard"]["raw_core_score_strict"] == "number"
    assert schema["scorecard"]["raw_composite_strict"] == "number"
    assert schema["scorecard_contract"]["headline_formula"] == "PostEditScore = gate * geometric_mean(headline_core_score, headline_generalization)"
