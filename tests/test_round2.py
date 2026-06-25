from __future__ import annotations

from dataclasses import replace
import signal

import pytest

from posteditbench.models import BenchmarkExample
from posteditbench.attacks.registry import build_attack_bundle
from posteditbench.models import BenchmarkRow
from posteditbench.pipeline import run_experiment
from posteditbench.report import summarize_rows
from posteditbench.validation import executable_validation_source, validate_semantics, visible_evaluation_source
from posteditbench.watermarks.registry import build_watermark_bundle


def test_run_experiment_marks_semantics_and_budget_curves(sample_config):
    result = run_experiment(sample_config)

    semantic_rows = [row for row in result.report.rows if row.semantic_validation_available]
    assert semantic_rows
    preserving_rows = [row for row in semantic_rows if row.semantic_preserving is True]
    assert preserving_rows
    assert all(row.semantic_preserving is True for row in result.report.rows if row.attack_name == "budgeted_adaptive")
    assert result.report.summary["semantic_validation_rate"] == pytest.approx(1.0)
    assert result.report.summary["semantic_preservation_rate"] == pytest.approx(1.0)
    assert "budgeted_adaptive" in result.report.summary["budget_curves"]
    assert result.report.summary["benchmark_manifest"]["record_count"] == len(result.examples)
    assert result.report.summary["by_language"]["python"]["semantic_validation_rate"] == pytest.approx(1.0)


def test_report_separates_declared_and_runtime_validation_by_language():
    rows = [
        BenchmarkRow(
            example_id="python-1",
            attack_name="attack",
            language="python",
            semantic_validation_available=False,
            metadata={"example_metadata": {"validation_supported": True}},
        ),
        BenchmarkRow(
            example_id="java-1",
            attack_name="attack",
            language="java",
            semantic_validation_available=True,
            semantic_preserving=True,
            metadata={"example_metadata": {"validation_supported": False}},
        ),
    ]

    summary = summarize_rows(rows, benchmark_manifest={"claimed_languages": ["python", "java"], "coverage": {}})

    assert summary["semantic_validation_by_language"] == summary["runtime_semantic_validation_by_language"]
    assert summary["declared_semantic_validation_by_language"]["python"]["declared_semantic_validation_rate"] == 1.0
    assert summary["declared_semantic_validation_by_language"]["java"]["declared_semantic_validation_rate"] == 0.0
    assert summary["runtime_semantic_validation_by_language"]["python"]["semantic_validation_rate"] == 0.0
    assert summary["runtime_semantic_validation_by_language"]["java"]["semantic_validation_rate"] == 1.0


def test_report_preserves_runtime_validation_unavailable_without_collapsing_to_zero() -> None:
    rows = [
        BenchmarkRow(
            example_id="python-1",
            attack_name="attack",
            language="python",
            semantic_validation_available=False,
            metadata={"example_metadata": {"validation_supported": True}},
        ),
        BenchmarkRow(
            example_id="java-1",
            attack_name="attack",
            language="java",
            semantic_validation_available=True,
            semantic_preserving=True,
            metadata={"example_metadata": {"validation_supported": True}},
        ),
    ]

    summary = summarize_rows(
        rows,
        benchmark_manifest={
            "claimed_languages": ["python", "java"],
            "coverage": {
                "runtime_validation_basis": "unavailable",
                "runtime_validation_annotations_available": False,
                "runtime_unvalidated_languages": [],
                "clean_reference_compile_rate": None,
                "clean_reference_pass_rate": None,
            },
        },
    )

    assert summary["runtime_validation_basis"] == "unavailable"
    assert summary["runtime_validation_annotations_available"] is False
    assert summary["clean_reference_compile_rate"] is None
    assert summary["clean_reference_pass_rate"] is None
    assert summary["coverage_gaps"]["runtime_unvalidated_languages"] == ["python"]


def test_structural_watermark_survives_comment_noise_but_not_control_flow_flatten(sample_example, sample_spec):
    bundle = build_watermark_bundle("structural_flow", allow_internal=True)
    watermarked = bundle.embed(sample_example, sample_spec)
    clean_result = bundle.detect(watermarked, sample_spec, example_id=sample_example.example_id)

    noisy = build_attack_bundle("comment_strip").apply(watermarked.source, seed=11)
    noisy_result = bundle.detect(noisy.source, sample_spec, example_id=sample_example.example_id)

    flattened = build_attack_bundle("control_flow_flatten").apply(
        watermarked.source,
        seed=11,
        context={"language": sample_example.language},
    )
    flattened_result = bundle.detect(flattened.source, sample_spec, example_id=sample_example.example_id)

    assert clean_result.detected is True
    assert noisy_result.detected is True
    assert flattened.changed is True
    assert flattened_result.score < clean_result.score
    assert flattened_result.detected is False or flattened_result.score == pytest.approx(0.0)
    assert watermarked.metadata["depth"] >= 2


def test_control_flow_flatten_attack_preserves_python_semantics(sample_example, sample_spec):
    structural = build_watermark_bundle("structural_flow", allow_internal=True)
    watermarked = structural.embed(sample_example, sample_spec)
    attack = build_attack_bundle("control_flow_flatten")
    outcome = attack.apply(watermarked.source, seed=13, context={"language": sample_example.language})

    validation = validate_semantics(sample_example, outcome.source)

    assert outcome.changed is True
    assert validation.available is True
    assert validation.passed is True


def test_budgeted_adaptive_attack_uses_context_and_budget_curve(sample_example, sample_spec):
    watermark = build_watermark_bundle("kgw", allow_internal=True)
    watermarked = watermark.embed(sample_example, sample_spec)
    attack = build_attack_bundle("budgeted_adaptive")

    outcome = attack.apply(
        watermarked.source,
        seed=13,
        context={
            "detector": lambda candidate: watermark.detect(candidate, sample_spec, example_id=sample_example.example_id).score,
            "quality": lambda candidate: 1.0 - (0.0 if candidate == watermarked.source else 0.1),
            "validate": lambda candidate: bool(validate_semantics(sample_example, candidate).passed),
            "config": {"budget": 2, "min_quality": 0.2},
        },
    )

    assert outcome.changed is True
    assert outcome.metadata["budget"] == 2
    assert outcome.metadata["budget_curve"][0]["budget"] == 0
    assert len(outcome.metadata["budget_curve"]) >= 2
    assert outcome.metadata["final_detector_score"] <= outcome.metadata["budget_curve"][0]["detector_score"]


def test_comment_strip_marks_malformed_python_as_unsupported():
    attack = build_attack_bundle("comment_strip")
    source = (
        "# headline\n"
        "def demo(value):\n"
        "    if value:\n"
        "        return value\n"
        "      return 0  # trailing\n"
    )

    outcome = attack.apply(source, seed=7, metadata={"language": "python"}, context={"language": "python"})

    assert outcome.changed is False
    assert outcome.source == source
    assert outcome.notes == ("python_comment_strip_parse_failed",)
    assert outcome.metadata["supported"] is False
    assert outcome.metadata["unsupported_reason"] == "python_comment_strip_parse_failed"


def test_comment_strip_keeps_comment_free_python_supported():
    attack = build_attack_bundle("comment_strip")
    source = (
        "def demo(value):\n"
        "    if value:\n"
        "        return value\n"
        "    return 0\n"
    )

    outcome = attack.apply(source, seed=7, metadata={"language": "python"}, context={"language": "python"})

    assert outcome.changed is False
    assert outcome.source == source
    assert outcome.notes == ("no_comment_content",)
    assert outcome.metadata["supported"] is True
    assert "unsupported_reason" not in outcome.metadata


def test_budgeted_adaptive_survives_malformed_python_comment_strip_candidate():
    attack = build_attack_bundle("budgeted_adaptive")
    source = (
        "# headline\n"
        "def demo(value):\n"
        "    if value:\n"
        "        return value\n"
        "      return 0  # trailing\n"
    )

    outcome = attack.apply(
        source,
        seed=7,
        metadata={"language": "python"},
        context={
            "language": "python",
            "detector": lambda candidate: 1.0 if candidate == source else 0.0,
            "quality": lambda candidate: 1.0,
            "validate": lambda candidate: True,
            "config": {"budget": 1, "candidate_order": ["comment_strip"]},
        },
    )

    assert outcome.changed is False
    assert outcome.metadata["supported"] is False
    assert outcome.metadata["unsupported_reason"] == "comment_strip:python_comment_strip_parse_failed"
    assert outcome.metadata["selected_steps"] == []
    assert outcome.metadata["budget_curve"] == [
        {
            "budget": 0,
            "detector_score": 1.0,
            "quality_score": 1.0,
            "semantic_preserving": True,
            "step_name": "start",
        }
    ]


def test_validate_semantics_fails_closed_on_toolchain_version_mismatch(monkeypatch):
    monkeypatch.setattr(
        "posteditbench.validation.inspect_local_toolchain",
        lambda language: {
            "verified": False,
            "runner_image": "node:20",
            "language_version": "20",
            "tools": [{"tool": "node", "version": "18.20.0", "expected_prefix": "20", "verified": False}],
            "issues": ["tool_version_mismatch:node:18.20.0!=20"],
        },
    )
    example = BenchmarkExample(
        example_id="js-1",
        language="javascript",
        prompt="Write sumArray(values).",
        reference_solution="function sumArray(values) { return values.reduce((a, b) => a + b, 0); }",
        execution_tests=("if (sumArray([1, 2, 3]) !== 6) { throw new Error('bad'); }",),
        metadata={
            "validation_mode": "docker_remote",
            "evaluation_backend": "docker_remote",
            "runner_image": "node:20",
            "language_family": "ecmascript",
            "claimed_languages": ["javascript"],
        },
    )

    result = validate_semantics(example, example.reference_solution)

    assert result.available is False
    assert result.metadata["reason"] == "toolchain_version_mismatch"
    assert result.metadata["toolchain_verified"] is False
    assert any("tool_version_mismatch" in failure for failure in result.failures)


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM timeout path is only available on POSIX runtimes")
def test_validate_semantics_times_out_for_non_terminating_python(monkeypatch):
    monkeypatch.setenv("POSTEDITBENCH_PYTHON_VALIDATION_TIMEOUT_SECONDS", "0.1")
    example = BenchmarkExample(
        example_id="python-timeout-1",
        language="python",
        prompt="Write loop_forever().",
        reference_solution="def loop_forever():\n    return 1\n",
        execution_tests=("while True:\n    pass\n",),
        metadata={},
    )

    result = validate_semantics(example, example.reference_solution)

    assert result.available is True
    assert result.passed is False
    assert result.metadata["error_kind"] == "timeout"
    assert result.metadata["compile_success"] is True
    assert any("TimeoutExpired" in failure for failure in result.failures)


def test_executable_validation_source_composes_prompt_prefix_python_completion() -> None:
    prompt = (
        "from typing import List\n\n\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        "    \"\"\"Return whether any pair is within threshold.\"\"\""
    )
    example = BenchmarkExample(
        example_id="humaneval-prefix-1",
        language="python",
        prompt=prompt,
        reference_solution=(
            "from typing import List\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "    \"\"\"Return whether any pair is within threshold.\"\"\"\n"
            "    sorted_numbers = sorted(numbers)\n"
            "    for i in range(len(sorted_numbers) - 1):\n"
            "        if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:\n"
            "            return True\n"
            "    return False\n"
        ),
        execution_tests=(
            "assert has_close_elements([1.0, 2.0, 2.1], 0.2) is True\n"
            "assert has_close_elements([1.0, 2.0, 3.0], 0.2) is False"
        ),
        metadata={"adapter_name": "human-eval-plus", "public_source": "humaneval_plus"},
    )

    completion_only = (
        "\n    sorted_numbers = sorted(numbers)\n"
        "    for i in range(len(sorted_numbers) - 1):\n"
        "        if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:\n"
        "            return True\n"
        "    return False\n"
    )

    executable = executable_validation_source(example, completion_only)
    compiled = compile(executable, "<posteditbench-test>", "exec")

    assert "def has_close_elements" in executable
    assert "sorted_numbers = sorted(numbers)" in executable
    assert compiled is not None


def test_visible_evaluation_source_strips_prompt_prefix_for_humaneval_reference() -> None:
    prompt = (
        "from typing import List\n\n\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        "    \"\"\"Return whether any pair is within threshold.\"\"\""
    )
    full_reference = (
        "from typing import List\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        "    \"\"\"Return whether any pair is within threshold.\"\"\"\n"
        "    sorted_numbers = sorted(numbers)\n"
        "    for i in range(len(sorted_numbers) - 1):\n"
        "        if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:\n"
        "            return True\n"
        "    return False\n"
    )
    example = BenchmarkExample(
        example_id="humaneval-prefix-2",
        language="python",
        prompt=prompt,
        reference_solution=full_reference,
        execution_tests=("assert True",),
        metadata={"adapter_name": "human-eval-plus", "public_source": "humaneval_plus"},
    )

    visible = visible_evaluation_source(example, full_reference)

    assert visible.lstrip().startswith("sorted_numbers = sorted(numbers)")
    assert "def has_close_elements" not in visible


def test_visible_evaluation_source_extracts_fenced_python_code_block() -> None:
    prompt = (
        "from typing import List\n\n\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        "    \"\"\"Return whether any pair is within threshold.\"\"\""
    )
    example = BenchmarkExample(
        example_id="humaneval-fenced-1",
        language="python",
        prompt=prompt,
        reference_solution="",
        execution_tests=("assert True",),
        metadata={"adapter_name": "human-eval-plus", "public_source": "humaneval_plus"},
    )
    fenced_completion = (
        "Here is a Python solution.\n\n"
        "```python\n"
        "    sorted_numbers = sorted(numbers)\n"
        "    for i in range(len(sorted_numbers) - 1):\n"
        "        if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:\n"
        "            return True\n"
        "    return False\n"
        "```"
    )

    visible = visible_evaluation_source(example, fenced_completion)

    assert visible.lstrip().startswith("sorted_numbers = sorted(numbers)")
    assert "```" not in visible
    assert "Here is a Python solution" not in visible


def test_executable_validation_source_trims_explanatory_prefix_before_standalone_code() -> None:
    prompt = (
        "from typing import List\n\n\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        "    \"\"\"Return whether any pair is within threshold.\"\"\""
    )
    example = BenchmarkExample(
        example_id="humaneval-standalone-1",
        language="python",
        prompt=prompt,
        reference_solution="",
        execution_tests=("assert True",),
        metadata={"adapter_name": "human-eval-plus", "public_source": "humaneval_plus"},
    )
    standalone_completion = (
        "Sure, here is the implementation:\n\n"
        "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n"
        "    sorted_numbers = sorted(numbers)\n"
        "    for i in range(len(sorted_numbers) - 1):\n"
        "        if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:\n"
        "            return True\n"
        "    return False\n"
    )

    executable = executable_validation_source(example, standalone_completion)

    assert executable.lstrip().startswith("def has_close_elements")
    assert "Sure, here is the implementation" not in executable
    assert compile(executable, "<posteditbench-standalone>", "exec") is not None


def test_executable_validation_source_accepts_dedented_mbxp_standalone_reference() -> None:
    prompt = (
        "#include <bits/stdc++.h>\n"
        "using namespace std;\n\n"
        "vector<int> smallNnum(vector<int> list1, int n) {"
    )
    standalone_reference = (
        "    #include <bits/stdc++.h>\n"
        "    using namespace std;\n\n"
        "    vector<int> smallNnum(vector<int> list1, int n) {\n"
        "        return {10, 20};\n"
        "    }\n"
    )
    example = BenchmarkExample(
        example_id="mbxp-dedented-1",
        language="cpp",
        prompt=prompt,
        reference_solution=standalone_reference,
        execution_tests=("int main() { return 0; }",),
        metadata={"adapter_name": "mbxp-5lang", "public_source": "mbxp_5lang"},
    )

    executable = executable_validation_source(example, standalone_reference)

    assert executable.lstrip().startswith("#include <bits/stdc++.h>")
    assert executable.count("#include <bits/stdc++.h>") == 1
    assert "vector<int> smallNnum" in executable


def test_run_experiment_rejects_unknown_registry_names(sample_config):
    with pytest.raises(ValueError, match="unknown attack names"):
        run_experiment(replace(sample_config, attacks=("comment_strip", "unknown_attack")))

    with pytest.raises(ValueError, match="unknown watermark scheme"):
        run_experiment(replace(sample_config, watermark_name="not_a_registry"))

