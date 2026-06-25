from __future__ import annotations

import json
from dataclasses import replace

import pytest

from posteditbench.models import BenchmarkRow, DetectionResult, ExperimentConfig, WatermarkSpec, WatermarkedSnippet
from posteditbench.pipeline import run_experiment
from posteditbench.report import build_report
from posteditbench.validation import SemanticValidationResult
from posteditbench.watermarks.base import WatermarkBundle


def test_run_experiment_builds_report(sample_config):
    result = run_experiment(sample_config)

    assert result.config.seed == sample_config.seed
    assert len(result.examples) == sample_config.corpus_size
    assert len(result.report.rows) == sample_config.corpus_size * len(sample_config.attacks)
    assert result.report.summary["rows"] == float(len(result.report.rows))
    assert "attack_breakdown" in result.report.summary
    assert "by_attack" in result.report.summary
    assert "by_language" in result.report.summary
    assert "semantic_attack_robustness" in result.report.summary
    assert "budget_curves" in result.report.summary
    assert "benchmark_manifest" in result.report.summary
    assert result.report.summary["clean_retention"] >= 0.0
    assert result.report.summary["mean_watermark_retention"] >= 0.0
    assert result.report.summary["mean_robustness_score"] >= 0.0
    assert result.report.summary["semantic_validation_rate"] >= 0.0
    assert result.report.summary["claimed_languages"]


def test_build_report_prefers_effective_provider_mode_over_config_default() -> None:
    report = build_report(
        ExperimentConfig(provider_mode="offline_mock", watermark_name="stone_runtime"),
        [
            BenchmarkRow(
                example_id="ex-1",
                attack_name="comment_strip",
                evaluation_track="generation_time",
                watermark_scheme="stone_runtime",
                method_origin="upstream",
                model_label="bigcode/starcoder2-7b",
                metadata={"provider_mode": "watermark_runtime", "example_metadata": {"validation_supported": True}},
            )
        ],
        benchmark_manifest={},
    )

    assert report.summary["configured_provider_mode"] == "offline_mock"
    assert report.summary["provider_mode"] == "watermark_runtime"
    assert report.summary["provider_modes"] == ["watermark_runtime"]


def test_run_experiment_blinds_provider_from_reference_solution(monkeypatch, sample_config) -> None:
    captured = {}

    class FakeProvider:
        name = "fake"

        def generate(self, example, *, seed: int = 0) -> str:
            captured["example"] = example
            return "def reverse_text(text):\n    return text[::-1]\n"

    def fake_validation(example, source):
        return SemanticValidationResult(
            example_id=example.example_id,
            language=example.language,
            available=True,
            passed=True,
            failures=(),
            metadata={"compile_success": True, "error_kind": ""},
        )

    monkeypatch.setattr("posteditbench.pipeline.orchestrator.build_provider", lambda mode, parameters: FakeProvider())
    monkeypatch.setattr("posteditbench.pipeline.orchestrator.validate_semantics", fake_validation)

    config = ExperimentConfig(
        **{
            **sample_config.as_dict(),
            "provider_mode": "local_command",
            "provider_parameters": {"command": "ignored"},
            "corpus_size": 1,
            "attacks": ("comment_strip",),
        }
    )

    run_experiment(config)

    blinded_example = captured["example"]
    assert blinded_example.reference_solution == ""
    assert blinded_example.reference_tests == ()
    assert blinded_example.execution_tests == ()
    assert blinded_example.prompt


def test_run_experiment_writes_incremental_progress_artifacts(sample_config, tmp_path) -> None:
    output_path = tmp_path / "report.json"
    config = replace(
        sample_config,
        corpus_size=2,
        attacks=("comment_strip",),
        output_path=str(output_path),
    )

    result = run_experiment(config)

    progress_path = output_path.with_name("progress.json")
    partial_rows_path = output_path.with_name("partial_rows.jsonl")
    partial_report_path = output_path.with_name("partial_report.json")

    assert progress_path.exists()
    assert partial_rows_path.exists()
    assert partial_report_path.exists()

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    partial_report = json.loads(partial_report_path.read_text(encoding="utf-8"))
    partial_rows = [line for line in partial_rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert progress["status"] == "completed"
    assert progress["rows_completed"] == len(result.report.rows)
    assert partial_report["summary"]["record_count"] == len(result.report.rows)
    assert partial_report["report_state"] == "partial"
    assert partial_report["completed_examples"] == config.corpus_size
    assert partial_report["total_examples"] == config.corpus_size
    assert partial_report["current_stage"] == "report_ready"
    assert len(partial_rows) == len(result.report.rows)


def test_run_experiment_survives_malformed_python_comment_strip(monkeypatch, sample_config, sample_example, tmp_path) -> None:
    class FakeProvider:
        name = "fake"

        def generate(self, example, *, seed: int = 0) -> str:
            return "def factorial(n):\n    return n\n"

    class FakeEmbedder:
        name = "fake_runtime"

        def embed(self, example, spec):
            return WatermarkedSnippet(
                example_id=example.example_id,
                language=example.language,
                source=(
                    "# headline\n"
                    "def factorial(n):\n"
                    "    if n <= 1:\n"
                    "        return 1\n"
                    "      return n  # trailing\n"
                ),
                watermark=spec,
                metadata={"baseline_family": "runtime_official"},
            )

    class FakeDetector:
        name = "fake_runtime"

        def detect(self, source, spec, *, example_id=""):
            return DetectionResult(
                example_id=example_id,
                method=self.name,
                score=1.0,
                detected=True,
                threshold=0.5,
                metadata={},
            )

    def fake_validation(example, source):
        return SemanticValidationResult(
            example_id=example.example_id,
            language=example.language,
            available=True,
            passed=True,
            failures=(),
            metadata={"compile_success": True, "error_kind": ""},
        )

    fake_bundle = WatermarkBundle(
        name="fake_runtime",
        embedder=FakeEmbedder(),
        detector=FakeDetector(),
    )

    monkeypatch.setattr("posteditbench.pipeline.orchestrator.generate_corpus", lambda *args, **kwargs: [sample_example])
    monkeypatch.setattr("posteditbench.pipeline.orchestrator.build_provider", lambda mode, parameters: FakeProvider())
    monkeypatch.setattr("posteditbench.pipeline.orchestrator.build_watermark_bundle", lambda name: fake_bundle)
    monkeypatch.setattr("posteditbench.pipeline.orchestrator.validate_experiment_config", lambda config: [])
    monkeypatch.setattr("posteditbench.pipeline.orchestrator.validate_semantics", fake_validation)

    output_path = tmp_path / "malformed_python_report.json"
    config = ExperimentConfig(
        **{
            **sample_config.as_dict(),
            "watermark_name": "fake_runtime",
            "provider_mode": "offline_mock",
            "corpus_size": 1,
            "attacks": ("comment_strip",),
            "output_path": str(output_path),
        }
    )

    result = run_experiment(config)

    assert len(result.report.rows) == 1
    row = result.report.rows[0]
    assert row.attack_name == "comment_strip"
    assert row.status == "attack-unsupported"
    assert "attack_unsupported" in row.comment
    assert "semantic_validation_passed" not in row.comment
    assert row.metadata["attack_metadata"]["supported"] is False
    assert row.metadata["attack_metadata"]["unsupported_reason"] == "python_comment_strip_parse_failed"
    assert output_path.exists()


def test_runtime_detection_snippets_preserve_generation_prompt(monkeypatch, sample_config, sample_example, tmp_path) -> None:
    observed: list[dict[str, object]] = []

    class FakeRuntimePreparer:
        name = "fake_runtime"

        def prepare(self, example, spec):
            metadata = {
                **dict(example.metadata),
                "provider_mode": "watermark_runtime",
                "generation_prompt": example.prompt,
                "baseline_family": "runtime_official",
            }
            return replace(example, reference_solution="def factorial(n):\n    return 2\n", metadata=metadata)

    class FakeRuntimeEmbedder:
        name = "fake_runtime"

        def embed(self, example, spec):
            return WatermarkedSnippet(
                example_id=example.example_id,
                language=example.language,
                source="def factorial(n):\n    return 3\n",
                watermark=spec,
                metadata={"generation_prompt": example.prompt},
            )

    class FakeRuntimeDetector:
        name = "fake_runtime"

        def detect(self, source, spec, *, example_id=""):
            assert isinstance(source, WatermarkedSnippet)
            observed.append(dict(source.metadata))
            prompt = str(source.metadata.get("generation_prompt", ""))
            return DetectionResult(
                example_id=example_id,
                method=self.name,
                score=1.0,
                detected=True,
                threshold=0.5,
                metadata={"prompt_available": bool(prompt)},
            )

    fake_bundle = WatermarkBundle(
        name="fake_runtime",
        embedder=FakeRuntimeEmbedder(),
        detector=FakeRuntimeDetector(),
        preparer=FakeRuntimePreparer(),
    )

    def fake_validation(example, source):
        return SemanticValidationResult(
            example_id=example.example_id,
            language=example.language,
            available=True,
            passed=True,
            failures=(),
            metadata={"compile_success": True, "error_kind": ""},
        )

    monkeypatch.setattr("posteditbench.pipeline.orchestrator.generate_corpus", lambda *args, **kwargs: [sample_example])
    monkeypatch.setattr("posteditbench.pipeline.orchestrator.build_watermark_bundle", lambda name: fake_bundle)
    monkeypatch.setattr("posteditbench.pipeline.orchestrator.validate_experiment_config", lambda config: [])
    monkeypatch.setattr("posteditbench.pipeline.orchestrator.validate_semantics", fake_validation)

    config = ExperimentConfig(
        **{
            **sample_config.as_dict(),
            "watermark_name": "fake_runtime",
            "provider_mode": "offline_mock",
            "corpus_size": 1,
            "attacks": ("budgeted_adaptive",),
            "output_path": str(tmp_path / "runtime_prompt_report.json"),
        }
    )

    run_experiment(config)

    assert observed
    assert len(observed) >= 3
    assert all(payload.get("generation_prompt") == sample_example.prompt for payload in observed)


def test_run_experiment_writes_partial_rows_with_tensor_detection_metadata(
    monkeypatch, sample_config, sample_example, tmp_path
) -> None:
    import torch

    class FakeRuntimePreparer:
        name = "fake_runtime"

        def prepare(self, example, spec):
            metadata = {
                **dict(example.metadata),
                "provider_mode": "watermark_runtime",
                "generation_prompt": example.prompt,
                "baseline_family": "runtime_official",
            }
            return replace(example, reference_solution="def factorial(n):\n    return 2\n", metadata=metadata)

    class FakeRuntimeEmbedder:
        name = "fake_runtime"

        def embed(self, example, spec):
            return WatermarkedSnippet(
                example_id=example.example_id,
                language=example.language,
                source="def factorial(n):\n    return 3\n",
                watermark=spec,
                metadata={"generation_prompt": example.prompt},
            )

    class FakeRuntimeDetector:
        name = "fake_runtime"

        def detect(self, source, spec, *, example_id=""):
            return DetectionResult(
                example_id=example_id,
                method=self.name,
                score=1.0,
                detected=True,
                threshold=0.5,
                metadata={
                    "payload": {
                        "prediction": torch.tensor(True),
                        "scores": torch.tensor([1.0, 2.0]),
                    }
                },
            )

    fake_bundle = WatermarkBundle(
        name="fake_runtime",
        embedder=FakeRuntimeEmbedder(),
        detector=FakeRuntimeDetector(),
        preparer=FakeRuntimePreparer(),
    )

    def fake_validation(example, source):
        return SemanticValidationResult(
            example_id=example.example_id,
            language=example.language,
            available=True,
            passed=True,
            failures=(),
            metadata={"compile_success": True, "error_kind": ""},
        )

    monkeypatch.setattr("posteditbench.pipeline.orchestrator.generate_corpus", lambda *args, **kwargs: [sample_example])
    monkeypatch.setattr("posteditbench.pipeline.orchestrator.build_watermark_bundle", lambda name: fake_bundle)
    monkeypatch.setattr("posteditbench.pipeline.orchestrator.validate_experiment_config", lambda config: [])
    monkeypatch.setattr("posteditbench.pipeline.orchestrator.validate_semantics", fake_validation)

    output_path = tmp_path / "runtime_tensor_report.json"
    config = ExperimentConfig(
        **{
            **sample_config.as_dict(),
            "watermark_name": "fake_runtime",
            "provider_mode": "offline_mock",
            "corpus_size": 1,
            "attacks": ("comment_strip",),
            "output_path": str(output_path),
        }
    )

    run_experiment(config)

    partial_rows_path = output_path.with_name("partial_rows.jsonl")
    lines = [line for line in partial_rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["metadata"]["clean_detection_metadata"]["payload"]["scores"] == [1.0, 2.0]


def test_runtime_detection_internal_errors_do_not_count_as_available_negative_controls(
    monkeypatch, sample_config, sample_example, tmp_path
) -> None:
    class FakeRuntimePreparer:
        name = "fake_runtime"

        def prepare(self, example, spec):
            metadata = {
                **dict(example.metadata),
                "provider_mode": "watermark_runtime",
                "generation_prompt": example.prompt,
                "baseline_family": "runtime_official",
            }
            return replace(example, reference_solution="func solve() int { return 2 }\n", metadata=metadata)

    class FakeRuntimeEmbedder:
        name = "fake_runtime"

        def embed(self, example, spec):
            return WatermarkedSnippet(
                example_id=example.example_id,
                language=example.language,
                source="func solve() int { return 3 }\n",
                watermark=spec,
                metadata={"generation_prompt": example.prompt},
            )

    class FakeRuntimeDetector:
        name = "fake_runtime"

        def detect(self, source, spec, *, example_id=""):
            return DetectionResult(
                example_id=example_id,
                method=self.name,
                score=0.0,
                detected=False,
                threshold=0.5,
                metadata={
                    "payload": {
                        "runtime_detection_error_kind": "runtime_detector_internal_error",
                        "runtime_detection_available": False,
                    }
                },
            )

    fake_bundle = WatermarkBundle(
        name="fake_runtime",
        embedder=FakeRuntimeEmbedder(),
        detector=FakeRuntimeDetector(),
        preparer=FakeRuntimePreparer(),
    )

    def fake_validation(example, source):
        return SemanticValidationResult(
            example_id=example.example_id,
            language=example.language,
            available=True,
            passed=True,
            failures=(),
            metadata={"compile_success": True, "error_kind": ""},
        )

    monkeypatch.setattr("posteditbench.pipeline.orchestrator.generate_corpus", lambda *args, **kwargs: [sample_example])
    monkeypatch.setattr("posteditbench.pipeline.orchestrator.build_watermark_bundle", lambda name: fake_bundle)
    monkeypatch.setattr("posteditbench.pipeline.orchestrator.validate_experiment_config", lambda config: [])
    monkeypatch.setattr("posteditbench.pipeline.orchestrator.validate_semantics", fake_validation)

    output_path = tmp_path / "runtime_detection_error_report.json"
    config = ExperimentConfig(
        **{
            **sample_config.as_dict(),
            "watermark_name": "fake_runtime",
            "provider_mode": "offline_mock",
            "corpus_size": 1,
            "attacks": ("comment_strip",),
            "output_path": str(output_path),
        }
    )

    result = run_experiment(config)

    row = result.report.rows[0]
    negative_controls = dict(row.metadata)["negative_controls"]
    assert negative_controls["human_reference"]["available"] is False
    assert negative_controls["clean_generation"]["available"] is False
    assert dict(row.metadata)["watermarked_detection"]["available"] is False
    assert result.report.summary["scorecard"]["negative_control_support_rate"] == pytest.approx(0.0)
