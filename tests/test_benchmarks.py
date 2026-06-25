from __future__ import annotations

import json

import pytest

from posteditbench.benchmarks import DEFAULT_NORMALIZED_BENCHMARK, build_benchmark_manifest, load_benchmark_corpus
from posteditbench.models import BenchmarkExample
from posteditbench.validation import SemanticValidationResult


def test_load_benchmark_corpus_uses_normalized_fixture():
    examples = load_benchmark_corpus(DEFAULT_NORMALIZED_BENCHMARK, count=4, seed=7)

    assert len(examples) == 4
    assert examples[0].example_id == "synthetic_py_sort_unique"
    assert examples[0].metadata["validation_mode"] == "python_exec"
    assert examples[0].metadata["validation_supported"] is True
    assert examples[3].language == "javascript"
    assert examples[3].metadata["validation_supported"] is False
    assert examples[0].metadata["source_path"].endswith("benchmark.normalized.jsonl")


def test_benchmark_manifest_measures_cross_lingual_coverage():
    examples = load_benchmark_corpus(DEFAULT_NORMALIZED_BENCHMARK, count=8, seed=7)
    manifest = build_benchmark_manifest(
        examples,
        source_path=DEFAULT_NORMALIZED_BENCHMARK,
        claimed_languages=["python", "javascript", "java", "rust"],
    )

    assert manifest.record_count == 8
    assert manifest.coverage["claimed_language_count"] == 4
    assert manifest.coverage["observed_coverage_rate"] == pytest.approx(1.0)
    assert manifest.coverage["semantic_validation_rate"] == pytest.approx(0.5)
    assert manifest.coverage["declared_semantic_validation_rate"] == pytest.approx(0.5)
    assert manifest.coverage["runtime_semantic_validation_rate"] == pytest.approx(0.5)
    assert manifest.coverage["clean_reference_compile_rate"] == pytest.approx(0.5)
    assert manifest.coverage["clean_reference_pass_rate"] == pytest.approx(0.5)
    assert manifest.coverage["semantic_validation_language_rate"] == pytest.approx(0.25)
    assert manifest.language_summary["python"]["validation_available_count"] == 4
    assert manifest.language_summary["python"]["runtime_validation_available_count"] == 4
    assert "javascript" in manifest.coverage["unvalidated_languages"]


def test_load_benchmark_corpus_filters_reference_kinds(tmp_path):
    benchmark_path = tmp_path / "reference-kinds.jsonl"
    rows = [
        {
            "task_id": "canonical-1",
            "dataset": "synthetic",
            "language": "python",
            "prompt": "Return 1.",
            "reference_solution": "def f():\n    return 1\n",
            "execution_tests": ["assert f() == 1"],
            "reference_kind": "canonical",
        },
        {
            "task_id": "overlay-1",
            "dataset": "synthetic",
            "language": "python",
            "prompt": "Return 2.",
            "reference_solution": "def g():\n    return 2\n",
            "execution_tests": ["assert g() == 2"],
            "reference_kind": "smoke_overlay",
        },
    ]
    benchmark_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    examples = load_benchmark_corpus(benchmark_path, include_reference_kinds=["canonical"])

    assert len(examples) == 1
    assert examples[0].example_id == "canonical-1"


def test_benchmark_manifest_uses_runtime_validation_for_semantic_validation_alias(monkeypatch) -> None:
    examples = [
        BenchmarkExample(
            example_id="python-1",
            language="python",
            prompt="Return 1.",
            reference_solution="def f():\n    return 1\n",
            execution_tests=("assert f() == 1",),
            metadata={},
        ),
        BenchmarkExample(
            example_id="javascript-1",
            language="javascript",
            prompt="Return 1.",
            reference_solution="function f() { return 1; }\n",
            execution_tests=(),
            metadata={},
        ),
    ]

    def _unavailable(example: BenchmarkExample, _source: str) -> SemanticValidationResult:
        return SemanticValidationResult(
            example_id=example.example_id,
            language=example.language,
            available=False,
            passed=None,
            metadata={},
        )

    monkeypatch.setattr("posteditbench.benchmarks.validate_semantics", _unavailable)

    manifest = build_benchmark_manifest(
        examples,
        source_path=DEFAULT_NORMALIZED_BENCHMARK,
        claimed_languages=["python", "javascript"],
    )

    assert manifest.coverage["declared_semantic_validation_rate"] == pytest.approx(0.5)
    assert manifest.coverage["runtime_semantic_validation_rate"] == pytest.approx(0.0)
    assert manifest.coverage["semantic_validation_rate"] == pytest.approx(0.0)
    assert manifest.coverage["semantic_validation_language_rate"] == pytest.approx(0.0)
    assert manifest.coverage["unvalidated_languages"] == ["python", "javascript"]
