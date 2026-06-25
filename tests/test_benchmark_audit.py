from __future__ import annotations

import json
from pathlib import Path

from scripts import audit_benchmarks


def test_benchmark_audit_accepts_release_crafted_snapshot(tmp_path: Path) -> None:
    result = audit_benchmarks.audit_one(Path("data/release/sources/crafted_original_release.normalized.jsonl"))
    assert result["status"] == "ok"
    assert result["path"] == "data/release/sources/crafted_original_release.normalized.jsonl"


def test_benchmark_audit_defaults_to_suite_profile() -> None:
    inputs = audit_benchmarks._profile_inputs("")
    normalized = {path.as_posix() for path in inputs}

    def _has_suffix(suffix: str) -> bool:
        return any(path.endswith(suffix) for path in normalized)

    assert _has_suffix("/data/release/sources/suite_humaneval_plus_release.normalized.jsonl")
    assert _has_suffix("/data/release/sources/suite_mbpp_plus_release.normalized.jsonl")
    assert _has_suffix("/data/release/sources/suite_humanevalx_release.normalized.jsonl")
    assert _has_suffix("/data/release/sources/suite_mbxp_release.normalized.jsonl")
    assert _has_suffix("/data/release/sources/crafted_original_release.normalized.jsonl")
    assert _has_suffix("/data/release/sources/crafted_translation_release.normalized.jsonl")
    assert _has_suffix("/data/release/sources/crafted_stress_release.normalized.jsonl")


def test_benchmark_audit_flags_duplicate_task_ids(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "broken.normalized.jsonl"
    benchmark_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": "dup",
                        "language": "python",
                        "validation_supported": True,
                        "record_kind": "crafted_benchmark",
                        "source_group": "crafted_original",
                        "origin_type": "crafted_original",
                        "family_id": "fam_001",
                        "difficulty": "easy",
                        "reference_kind": "canonical",
                        "semantic_contract": "x",
                        "category": "strings/parsing",
                        "validation_backend": "python_exec",
                        "source_digest": "digest-a",
                    }
                ),
                json.dumps(
                    {
                        "task_id": "dup",
                        "language": "cpp",
                        "validation_supported": True,
                        "record_kind": "crafted_benchmark",
                        "source_group": "crafted_original",
                        "origin_type": "crafted_original",
                        "family_id": "fam_001",
                        "difficulty": "medium",
                        "reference_kind": "canonical",
                        "semantic_contract": "y",
                        "category": "strings/parsing",
                        "validation_backend": "cpp_exec",
                        "source_digest": "digest-b",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "task_count": 2,
                "observed_languages": ["python", "cpp"],
                "reference_kind_counts": {"canonical": 2},
                "category_counts": {"strings/parsing": 1},
                "template_family_counts": {"strings": 1},
                "task_count_per_family": 5,
                "family_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = audit_benchmarks.audit_one(benchmark_path)
    assert result["status"] == "failed"
    assert any("duplicate task_id" in failure for failure in result["failures"])


def test_benchmark_audit_manifest_discovers_active_prepared_benchmarks(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "prepared.normalized.jsonl"
    benchmark_path.write_text(
        json.dumps(
            {
                "task_id": "ok-1",
                "language": "python",
                "validation_supported": True,
                "record_kind": "smoke_fixture",
                "source_group": "smoke_synthetic",
                "origin_type": "smoke",
                "reference_kind": "canonical",
                "source_digest": "digest-a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "record_count": 1,
                "observed_languages": ["python"],
                "reference_kind_counts": {"canonical": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "tiny.json"
    config_path.write_text(
        json.dumps(
            {
                "benchmark": {"prepared_output": str(benchmark_path)},
                "provider": {"mode": "offline_mock", "parameters": {}},
                "watermark": {"scheme": "comment", "strength": 0.5},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "suite_all_models_methods",
                "runs": [
                    {
                        "run_id": "one",
                        "profile": "suite_all_models_methods",
                        "config": str(config_path),
                        "config_overrides": {
                            "paths": {"prepared_benchmark": str(benchmark_path)},
                            "benchmark": {"prepared_output": str(benchmark_path)},
                        },
                        "resource": "cpu",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    inputs = audit_benchmarks._matrix_inputs(manifest_path, profile="suite_all_models_methods")

    assert inputs == [benchmark_path.resolve()]


def test_benchmark_audit_flags_runtime_unavailable_collection_with_runtime_metrics(tmp_path: Path) -> None:
    collection_path = tmp_path / "collection.normalized.jsonl"
    collection_path.write_text(
        json.dumps(
            {
                "task_id": "ok-1",
                "language": "python",
                "validation_supported": True,
                "record_kind": "public_benchmark",
                "source_group": "public_humaneval_plus",
                "origin_type": "public",
                "reference_kind": "canonical",
                "source_digest": "digest-a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    collection_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "record_count": 1,
                "collection_name": "public_multilingual_core",
                "observed_languages": ["python"],
                "reference_kind_counts": {"canonical": 1},
                "language_counts": {"python": 1},
                "source_group_counts": {"public_humaneval_plus": 1},
                "origin_type_counts": {"public": 1},
                "family_count": 0,
                "source_manifests": [{}],
                "coverage": {
                    "runtime_validation_basis": "unavailable",
                    "runtime_validation_annotations_available": True,
                    "runtime_semantic_validation_rate": 1.0,
                    "runtime_semantic_validation_language_rate": 1.0,
                    "semantic_validation_rate": 1.0,
                    "semantic_validation_language_rate": 1.0,
                    "clean_reference_compile_rate": 1.0,
                    "clean_reference_pass_rate": 1.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = audit_benchmarks.audit_one(collection_path)

    assert result["status"] == "failed"
    assert any("runtime-unavailable collection must leave coverage.runtime_semantic_validation_rate = null" in failure for failure in result["failures"])

