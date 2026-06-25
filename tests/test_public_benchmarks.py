from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from posteditbench.benchmarks import build_benchmark_manifest, load_benchmark_corpus, normalize_benchmark_record
from posteditbench.config import build_experiment_config, load_config
from posteditbench.pipeline import run_experiment
from posteditbench.public_benchmarks import get_public_source_spec
from posteditbench.providers import build_provider


def _python_public_exec(source: str, tests: tuple[str, ...]) -> tuple[bool, bool]:
    namespace = {
        "__builtins__": __builtins__,
        "bisect": __import__("bisect"),
        "collections": __import__("collections"),
        "datetime": __import__("datetime"),
        "decimal": __import__("decimal"),
        "fractions": __import__("fractions"),
        "functools": __import__("functools"),
        "heapq": __import__("heapq"),
        "itertools": __import__("itertools"),
        "math": __import__("math"),
        "operator": __import__("operator"),
        "pathlib": __import__("pathlib"),
        "random": __import__("random"),
        "re": __import__("re"),
        "statistics": __import__("statistics"),
        "string": __import__("string"),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        exec(compile(source, "<test-public>", "exec"), namespace, namespace)
        for index, test in enumerate(tests):
            exec(compile(test, f"<test-public-{index}>", "exec"), namespace, namespace)
    return True, True


@pytest.mark.parametrize(
    "relative_path, expected_count, dataset_label, public_source",
    [
        ("data/release/sources/suite_humaneval_plus_release.normalized.jsonl", 164, "HumanEval+", "humaneval_plus"),
        ("data/release/sources/suite_mbpp_plus_release.normalized.jsonl", 378, "MBPP+", "mbpp_plus"),
        (
            "data/release/sources/suite_humanevalx_release.normalized.jsonl",
            200,
            "HumanEval-X (5-language balanced slice)",
            "humaneval_x",
        ),
        (
            "data/release/sources/suite_mbxp_release.normalized.jsonl",
            200,
            "MBXP-5lang (5-language balanced slice)",
            "mbxp_5lang",
        ),
    ],
)
def test_release_sources_load_and_preserve_public_provenance(
    relative_path: str,
    expected_count: int,
    dataset_label: str,
    public_source: str,
) -> None:
    path = Path(relative_path)
    examples = load_benchmark_corpus(path, count=1)
    assert len(examples) == 1
    assert examples[0].metadata["dataset"] == dataset_label
    assert examples[0].metadata["public_source"] == public_source
    assert examples[0].metadata["reference_kind"] in {"canonical", "smoke_overlay"}
    if examples[0].language == "python" and examples[0].metadata["validation_supported"]:
        assert _python_public_exec(examples[0].reference_solution, examples[0].execution_tests) == (True, True)

    manifest = build_benchmark_manifest(examples, source_path=path, claimed_languages=[examples[0].language])
    assert manifest.record_count == 1
    assert manifest.as_dict()["source_group_counts"]
    assert manifest.datasets

    sample_manifest = load_benchmark_corpus(path, count=expected_count)
    assert len(sample_manifest) == expected_count


@pytest.mark.parametrize(
    "relative_path, task_id",
    [
        ("data/release/sources/suite_humanevalx_release.normalized.jsonl", "Python/0"),
        ("data/release/sources/suite_mbxp_release.normalized.jsonl", "MBCPP/1"),
    ],
)
def test_known_release_examples_validate_clean_reference(relative_path: str, task_id: str) -> None:
    path = Path(relative_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    row = next(item for item in rows if str(item.get("task_id")) == task_id)
    example = normalize_benchmark_record(row, rows.index(row), path)
    assert example.metadata["reference_kind"] in {"canonical", "smoke_overlay"}
    if example.language == "python" and example.metadata["validation_supported"]:
        assert _python_public_exec(example.reference_solution, example.execution_tests) == (True, True)


@pytest.mark.parametrize(
    "relative_path",
    [
        "data/release/sources/suite_humaneval_plus_release.normalized.jsonl",
        "data/release/sources/suite_mbpp_plus_release.normalized.jsonl",
        "data/release/sources/suite_humanevalx_release.normalized.jsonl",
        "data/release/sources/suite_mbxp_release.normalized.jsonl",
    ],
)
def test_release_source_python_rows_validate_clean_reference(relative_path: str) -> None:
    path = Path(relative_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    failing: list[str] = []
    for index, row in enumerate(rows):
        if str(row.get("language", "")).lower() != "python":
            continue
        if not bool(row.get("validation_supported")):
            continue
        example = normalize_benchmark_record(row, index, path)
        try:
            _python_public_exec(example.reference_solution, example.execution_tests)
        except Exception as exc:
            failing.append(f"{example.example_id}:{exc.__class__.__name__}:{exc}")
    assert not failing, f"unexpected python clean-reference failures in {relative_path}: {failing[:5]}"


@pytest.mark.parametrize(
    "manifest_path, expected_revision",
    [
        ("data/release/sources/suite_humanevalx_release.normalized.manifest.json", "2838420b7b4492cf3d16bce5320e26e65960c9e2"),
        ("data/release/sources/suite_mbxp_release.normalized.manifest.json", "e09974f990eeaf0c0e8f2b5eaff4be66effb2c86"),
    ],
)
def test_multilingual_release_manifests_record_hardened_provenance(manifest_path: str, expected_revision: str) -> None:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["sample_ids_path"]
    sample_ids_path = Path(__file__).resolve().parents[1] / manifest["sample_ids_path"].replace("\\", "/")
    assert sample_ids_path.exists()
    assert manifest["record_count"] >= 200
    assert all(item.get("source_sha256") for item in manifest["source_manifests"])
    assert all(item.get("source_revision") == expected_revision for item in manifest["source_manifests"])


def test_class_eval_is_supported_but_not_part_of_default_core() -> None:
    spec = get_public_source_spec("class_eval")
    assert spec.source_kind == "git_checkout"
    assert spec.included_in_core is False
    assert spec.artifact_policy == "restricted_noncommercial"
    assert spec.validate_python_references is False


def test_public_runner_defaults_to_offline_mock_and_reports_calibration() -> None:
    source = load_config(Path("tests/fixtures/configs/public_humaneval.yaml"))
    benchmark = dict(source.raw["benchmark"])
    benchmark["limit"] = 2
    config = build_experiment_config(source, benchmark=benchmark)
    result = run_experiment(config)

    assert result.config.provider_mode == "offline_mock"
    assert result.report.summary["provider_mode"] == "offline_mock"
    assert result.report.summary["validation_scope"] == "python_first"
    assert "detection_calibration" in result.report.summary
    assert result.report.summary["detection_calibration"]["thresholds"]
    assert result.report.summary["benchmark_manifest"]["splits"] == ["test"]
    assert result.report.summary["semantic_validation_rate"] == pytest.approx(1.0)


def test_offline_mock_provider_is_deterministic() -> None:
    example = load_benchmark_corpus(Path("data/release/sources/suite_mbpp_plus_release.normalized.jsonl"), count=1)[0]
    provider = build_provider("offline_mock", {})
    assert provider.generate(example) == example.reference_solution
