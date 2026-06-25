from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from ..attacks.registry import build_attack_bundle
from ..benchmarks import DEFAULT_NORMALIZED_BENCHMARK, build_benchmark_manifest
from ..config import build_experiment_config, load_config, validate_experiment_config
from ..metrics import overall_quality_score, stealth_score
from ..models import BenchmarkExample, BenchmarkReport, BenchmarkRow, ExperimentConfig, WatermarkSpec, WatermarkedSnippet
from ..report import build_report
from ..utils import ensure_parent, line_count, stable_hash, tokenize
from ..providers import build_provider
from ..validation import validate_semantics, visible_evaluation_source
from ..watermarks.registry import build_watermark_bundle, internal_watermarks, watermark_origin
from .generator import generate_corpus


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    config: ExperimentConfig
    examples: tuple[BenchmarkExample, ...]
    report: BenchmarkReport
    benchmark_manifest: Mapping[str, Any] = field(default_factory=dict)


def _progress_enabled() -> bool:
    value = str(os.environ.get("POSTEDITBENCH_PROGRESS_LOG", "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _log_progress(message: str) -> None:
    if not _progress_enabled():
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    sys.stderr.write(f"[progress {timestamp}] {message}\n")
    sys.stderr.flush()


def _build_spec(config: ExperimentConfig) -> WatermarkSpec:
    watermark_metadata = dict(config.metadata.get("watermark", {})) if isinstance(config.metadata, dict) else {}
    extra_parameters = {
        str(key): value
        for key, value in watermark_metadata.items()
        if key not in {"scheme", "secret", "payload", "strength"}
    }
    return WatermarkSpec(
        name=config.watermark_name,
        secret=config.watermark_secret,
        payload=config.watermark_payload,
        strength=config.watermark_strength,
        parameters={"threshold": 0.5, "seed": config.seed, **extra_parameters},
    )


def _select_benchmark_path(config: ExperimentConfig) -> Path:
    candidates = [
        config.corpus_parameters.get("prepared_output"),
        config.corpus_parameters.get("benchmark_path"),
        config.corpus_parameters.get("source"),
    ]
    for candidate in candidates:
        if candidate:
            return Path(str(candidate))
    return DEFAULT_NORMALIZED_BENCHMARK


def _status_label(*, attacked_detected: bool, quality_score: float, semantic_available: bool, semantic_preserving: bool | None) -> str:
    if semantic_available and semantic_preserving is False:
        return "semantic-failed"
    if attacked_detected and quality_score >= 0.8:
        return "stable"
    return "needs-review"


def _clean_samples_per_task(config: ExperimentConfig, *, runtime_watermark: bool) -> int:
    if runtime_watermark:
        return 1
    reporting = {}
    if isinstance(config.metadata, dict):
        reporting = dict(config.metadata.get("reporting", {}) or {})
    requested = reporting.get("clean_samples_per_task", config.corpus_parameters.get("clean_samples_per_task", 1))
    try:
        return max(1, int(requested))
    except Exception:
        return 1


def _model_label(config: ExperimentConfig, spec: WatermarkSpec, *, provider_mode: str) -> str:
    if provider_mode == "local_hf":
        model_name = str(config.provider_parameters.get("model", "")).strip()
        return model_name or "local_hf"
    if provider_mode == "watermark_runtime":
        model_name = str(spec.parameters.get("model_name", "")).strip()
        return model_name or "runtime_model"
    if provider_mode == "offline_mock":
        return "reference_oracle"
    if provider_mode == "local_command":
        command = str(config.provider_parameters.get("command", "")).strip()
        return command or "local_command"
    return provider_mode or "unspecified"


def _evaluation_track(*, provider_mode: str, watermark_uses_internal_generation: bool) -> str:
    normalized = str(provider_mode).strip().lower()
    if watermark_uses_internal_generation or normalized in {"watermark_runtime", "local_hf", "local_command"}:
        return "generation_time"
    return "reference_code"


def _summarize_clean_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        return {
            "sample_count": 0,
            "compile_success_rate": 0.0,
            "test_pass_rate": 0.0,
            "error_taxonomy": {},
        }
    compile_successes = [1.0 if trial.get("compile_success") else 0.0 for trial in trials if trial.get("compile_success") is not None]
    passed = [1.0 if trial.get("passed") else 0.0 for trial in trials if trial.get("passed") is not None]
    taxonomy: dict[str, int] = {}
    for trial in trials:
        key = str(trial.get("error_kind", "")).strip() or "passed"
        taxonomy[key] = taxonomy.get(key, 0) + 1
    return {
        "sample_count": len(trials),
        "compile_success_rate": round(sum(compile_successes) / max(1, len(compile_successes)), 4) if compile_successes else 0.0,
        "test_pass_rate": round(sum(passed) / max(1, len(passed)), 4) if passed else 0.0,
        "error_taxonomy": taxonomy,
    }


def _representative_clean_validation(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        return {
            "available": False,
            "passed": None,
            "failures": [],
            "metadata": {"compile_success": None, "error_kind": ""},
        }
    first = dict(trials[0])
    return {
        "available": bool(first.get("available")),
        "passed": first.get("passed"),
        "failures": list(first.get("failures", [])),
        "metadata": {
            "compile_success": first.get("compile_success"),
            "error_kind": first.get("error_kind", ""),
        },
    }


def _round_stage_seconds(value: float) -> float:
    return round(max(0.0, float(value)), 6)


def _standardized_token_count(text: str) -> int:
    return len(tokenize(str(text or "")))


def _standardized_line_count(text: str) -> int:
    return line_count(str(text or ""))


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    with candidate.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            handle.write("\n")


def _append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    with candidate.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            handle.write("\n")


def _progress_artifact_paths(config: ExperimentConfig) -> dict[str, Path]:
    if not config.output_path:
        return {}
    report_path = Path(config.output_path)
    return {
        "progress": report_path.with_name("progress.json"),
        "partial_rows": report_path.with_name("partial_rows.jsonl"),
        "partial_report": report_path.with_name("partial_report.json"),
    }


def _reset_progress_artifacts(config: ExperimentConfig) -> dict[str, Path]:
    paths = _progress_artifact_paths(config)
    for path in paths.values():
        if path.exists():
            path.unlink()
    return paths


def _write_progress_state(
    config: ExperimentConfig,
    *,
    progress_paths: dict[str, Path],
    stage: str,
    example_index: int,
    total_examples: int,
    rows_completed: int,
    latest_example_id: str = "",
    latest_attack: str = "",
    status: str = "running",
) -> None:
    if not progress_paths:
        return
    payload = {
        "status": status,
        "stage": stage,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "report_path": config.output_path or "",
        "partial_report_path": str(progress_paths.get("partial_report", "")),
        "partial_rows_path": str(progress_paths.get("partial_rows", "")),
        "example_index": int(example_index),
        "total_examples": int(total_examples),
        "rows_completed": int(rows_completed),
        "latest_example_id": latest_example_id,
        "latest_attack": latest_attack,
    }
    progress_paths["progress"].write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_partial_report(
    config: ExperimentConfig,
    *,
    progress_paths: dict[str, Path],
    rows: list[BenchmarkRow],
    benchmark_manifest: dict[str, Any],
    example_index: int,
    total_examples: int,
    stage: str,
) -> None:
    if not progress_paths:
        return
    partial_report_path = progress_paths["partial_report"]
    partial_config = replace(config, output_path=str(partial_report_path))
    partial_report = build_report(
        partial_config,
        rows,
        output_path=str(partial_report_path),
        benchmark_manifest=benchmark_manifest,
    )
    payload = json.loads(partial_report.to_json())
    payload["report_state"] = "partial"
    payload["completed_examples"] = int(example_index)
    payload["total_examples"] = int(total_examples)
    payload["current_stage"] = stage
    partial_report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _emit_sensitive_baseline_eval_payloads() -> bool:
    return str(os.environ.get("POSTEDITBENCH_ENABLE_BASELINE_EVAL_PAYLOADS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _baseline_eval_payload_output_path(config: ExperimentConfig) -> Path | None:
    explicit = str(os.environ.get("POSTEDITBENCH_BASELINE_EVAL_PAYLOAD_PATH", "")).strip()
    if explicit:
        return Path(explicit)
    if config.output_path:
        return Path(config.output_path).with_name("baseline_eval_payloads.private.jsonl")
    return None


def _provider_visible_example(example: BenchmarkExample, *, provider_mode: str) -> BenchmarkExample:
    if str(provider_mode).strip().lower() == "offline_mock":
        return example
    sanitized_metadata = {
        "provider_visible_fields": ["example_id", "language", "prompt"],
        "provider_blind_eval": True,
    }
    return replace(
        example,
        reference_solution="",
        reference_tests=(),
        execution_tests=(),
        metadata=sanitized_metadata,
    )


def _detection_snippet(
    example: BenchmarkExample,
    source: str,
    spec: WatermarkSpec,
    *metadata_sources: Mapping[str, Any] | None,
) -> WatermarkedSnippet:
    metadata: dict[str, Any] = dict(example.metadata)
    for source_metadata in metadata_sources:
        if source_metadata:
            metadata.update(dict(source_metadata))
    metadata.setdefault("generation_prompt", example.prompt)
    metadata.setdefault("prompt", example.prompt)
    return WatermarkedSnippet(
        example_id=example.example_id,
        language=example.language,
        source=source,
        watermark=spec,
        metadata=metadata,
    )


def run_experiment(config: ExperimentConfig | dict[str, Any] | str | Path | None = None, **overrides: Any) -> BenchmarkRun:
    if isinstance(config, (str, Path)):
        config = build_experiment_config(load_config(config), **overrides)
    elif isinstance(config, dict) or config is None:
        config = build_experiment_config(config, **overrides)
    elif overrides:
        raw = config.as_dict()
        raw.update(overrides)
        config = build_experiment_config(raw)

    issues = validate_experiment_config(config)
    if issues:
        raise ValueError("; ".join(issues))

    benchmark_path = _select_benchmark_path(config)
    _log_progress(
        "stage=benchmark_prepare "
        f"benchmark={benchmark_path} corpus_size={config.corpus_size} "
        f"provider_mode={config.provider_mode} watermark={config.watermark_name} "
        f"attacks={','.join(config.attacks)}"
    )
    requested_languages = config.corpus_parameters.get("languages")
    reference_kinds = config.corpus_parameters.get("include_reference_kinds")
    examples = tuple(
        generate_corpus(
            config.corpus_size,
            seed=config.seed,
            language=requested_languages or None,
            include_reference_kinds=reference_kinds if isinstance(reference_kinds, (list, tuple)) else None,
            prompt_prefix=str(config.corpus_parameters.get("prompt_prefix", "")),
            benchmark_path=benchmark_path,
        )
    )
    benchmark_manifest = build_benchmark_manifest(
        examples,
        source_path=benchmark_path,
        claimed_languages=requested_languages if isinstance(requested_languages, (list, tuple)) else None,
    )
    _log_progress(f"stage=benchmark_ready examples={len(examples)}")
    if config.watermark_name in set(internal_watermarks()):
        watermark = build_watermark_bundle(config.watermark_name, allow_internal=True)
    else:
        watermark = build_watermark_bundle(config.watermark_name)
    spec = _build_spec(config)
    provider = None if watermark.uses_internal_generation else build_provider(config.provider_mode, dict(config.provider_parameters))
    attack_bundles = tuple(build_attack_bundle(name) for name in config.attacks)
    _log_progress(
        "stage=runtime_ready "
        f"uses_internal_generation={watermark.uses_internal_generation} "
        f"attack_count={len(attack_bundles)}"
    )
    rows: list[BenchmarkRow] = []
    baseline_eval_records: list[dict[str, Any]] = []
    baseline_eval_payloads: list[dict[str, Any]] = []
    progress_paths = _reset_progress_artifacts(config)
    _write_progress_state(
        config,
        progress_paths=progress_paths,
        stage="runtime_ready",
        example_index=0,
        total_examples=len(examples),
        rows_completed=0,
        status="running",
    )

    total_examples = len(examples)
    for example_index, example in enumerate(examples, start=1):
        _log_progress(
            f"stage=example_start index={example_index}/{total_examples} "
            f"example_id={example.example_id} language={example.language}"
        )
        _write_progress_state(
            config,
            progress_paths=progress_paths,
            stage="example_start",
            example_index=example_index,
            total_examples=total_examples,
            rows_completed=len(rows),
            latest_example_id=example.example_id,
            status="running",
        )
        clean_generation_seconds = 0.0
        watermarked_generation_seconds = 0.0
        clean_validation_seconds = 0.0
        attacked_validation_seconds_by_attack: dict[str, float] = {}
        attacked_detection_seconds_by_attack: dict[str, float] = {}
        attack_seconds_by_attack: dict[str, float] = {}

        prepare_started = time.perf_counter()
        prepared_example = watermark.prepare_example(example, spec)
        prepared_elapsed = time.perf_counter() - prepare_started
        clean_trials: list[dict[str, Any]] = []
        clean_candidates: list[BenchmarkExample] = []
        provider_mode = config.provider_mode
        if watermark.uses_internal_generation:
            clean_generation_seconds += prepared_elapsed
            working_example = prepared_example
            provider_mode = str(working_example.metadata.get("provider_mode", "watermark_runtime"))
            clean_validation_started = time.perf_counter()
            clean_validation = validate_semantics(working_example, working_example.reference_solution)
            clean_validation_seconds += time.perf_counter() - clean_validation_started
            clean_trials.append(
                {
                    "sample_index": 0,
                    "source_digest": stable_hash(working_example.reference_solution),
                    "available": clean_validation.available,
                    "passed": clean_validation.passed,
                    "compile_success": clean_validation.metadata.get("compile_success"),
                    "error_kind": clean_validation.metadata.get("error_kind", ""),
                    "failures": list(clean_validation.failures),
                }
            )
        else:
            assert provider is not None
            sample_count = _clean_samples_per_task(config, runtime_watermark=False)
            provider_request_example = _provider_visible_example(example, provider_mode=config.provider_mode)
            for sample_index in range(sample_count):
                provider_error = ""
                provider_completion = ""
                try:
                    clean_generation_started = time.perf_counter()
                    completion = provider.generate(provider_request_example, seed=config.seed + sample_index)
                    clean_generation_seconds += time.perf_counter() - clean_generation_started
                    provider_completion = str(completion or "")
                except Exception as exc:  # pragma: no cover - provider/runtime dependent
                    clean_generation_seconds += time.perf_counter() - clean_generation_started
                    provider_error = f"{exc.__class__.__name__}: {exc}"
                    provider_completion = ""
                candidate = replace(example, reference_solution=provider_completion)
                clean_candidates.append(candidate)
                validation_started = time.perf_counter()
                validation = validate_semantics(candidate, candidate.reference_solution)
                clean_validation_seconds += time.perf_counter() - validation_started
                clean_trials.append(
                    {
                        "sample_index": sample_index,
                        "source_digest": stable_hash(candidate.reference_solution),
                        "generation_succeeded": bool(provider_completion.strip()) and not provider_error,
                        "generation_error": provider_error,
                        "available": validation.available,
                        "passed": validation.passed,
                        "compile_success": validation.metadata.get("compile_success"),
                        "error_kind": validation.metadata.get("error_kind", ""),
                        "failures": list(validation.failures),
                    }
                )
            working_example = clean_candidates[0] if clean_candidates else replace(example, reference_solution=example.reference_solution)
        clean_summary = _summarize_clean_trials(clean_trials)
        clean_validation_payload = _representative_clean_validation(clean_trials)
        _log_progress(
            f"stage=clean_ready index={example_index}/{total_examples} "
            f"example_id={example.example_id} clean_trials={len(clean_trials)} "
            f"provider_mode={provider_mode}"
        )
        _write_progress_state(
            config,
            progress_paths=progress_paths,
            stage="clean_ready",
            example_index=example_index,
            total_examples=total_examples,
            rows_completed=len(rows),
            latest_example_id=example.example_id,
            status="running",
        )
        working_metadata = dict(working_example.metadata)
        working_metadata["provider_mode"] = provider_mode
        working_metadata["provider_completion_digest"] = stable_hash(working_example.reference_solution)
        working_metadata["provider_generation_succeeded"] = bool(str(working_example.reference_solution).strip())
        working_metadata["validation_scope"] = config.validation_scope
        working_metadata["clean_functional_trials"] = clean_trials
        working_metadata["clean_functional_summary"] = clean_summary
        working_example = replace(working_example, metadata=working_metadata)

        watermarked_generation_started = time.perf_counter()
        watermarked = watermark.embed(working_example, spec)
        watermarked_generation_seconds = time.perf_counter() - watermarked_generation_started

        detection_cache: dict[str, DetectionResult] = {}
        validation_cache: dict[str, Any] = {}
        quality_cache: dict[str, float] = {}
        stage_timing_totals = {
            "negative_control_detection_seconds": 0.0,
            "watermarked_detection_seconds": 0.0,
            "watermarked_validation_seconds": 0.0,
        }

        def _source_text(candidate: str | WatermarkedSnippet) -> str:
            if isinstance(candidate, WatermarkedSnippet):
                return candidate.source
            return str(candidate)

        def _detection_available(result: DetectionResult) -> bool:
            payload = result.metadata.get("payload", {}) if isinstance(result.metadata, Mapping) else {}
            if not isinstance(payload, Mapping):
                return True
            return bool(payload.get("runtime_detection_available", True))

        def _cached_detect(candidate: str | WatermarkedSnippet, *, timing_bucket: str = "") -> DetectionResult:
            source_text = _source_text(candidate)
            cached = detection_cache.get(source_text)
            if cached is not None:
                return cached
            detect_started = time.perf_counter()
            detected = watermark.detect(candidate, spec, example_id=example.example_id)
            detect_elapsed = time.perf_counter() - detect_started
            if timing_bucket == "negative_control":
                stage_timing_totals["negative_control_detection_seconds"] += detect_elapsed
            elif timing_bucket == "watermarked":
                stage_timing_totals["watermarked_detection_seconds"] += detect_elapsed
            elif timing_bucket.startswith("attacked:"):
                attack_name = timing_bucket.split(":", 1)[1]
                attacked_detection_seconds_by_attack[attack_name] = (
                    attacked_detection_seconds_by_attack.get(attack_name, 0.0) + detect_elapsed
                )
            detection_cache[source_text] = detected
            return detected

        def _cached_validate(candidate: str, *, timing_bucket: str = "") -> Any:
            cached = validation_cache.get(candidate)
            if cached is not None:
                return cached
            validation_started = time.perf_counter()
            validation = validate_semantics(working_example, candidate)
            validation_elapsed = time.perf_counter() - validation_started
            if timing_bucket == "watermarked":
                stage_timing_totals["watermarked_validation_seconds"] += validation_elapsed
            elif timing_bucket.startswith("attacked:"):
                attack_name = timing_bucket.split(":", 1)[1]
                attacked_validation_seconds_by_attack[attack_name] = (
                    attacked_validation_seconds_by_attack.get(attack_name, 0.0) + validation_elapsed
                )
            validation_cache[candidate] = validation
            return validation

        def _cached_quality(candidate: str) -> float:
            cached = quality_cache.get(candidate)
            if cached is not None:
                return cached
            score = overall_quality_score(watermarked.source, candidate)
            quality_cache[candidate] = score
            return score

        def _candidate_detection_score(candidate: str | WatermarkedSnippet) -> float:
            if isinstance(candidate, WatermarkedSnippet):
                return _cached_detect(candidate).score
            visible_source = visible_evaluation_source(working_example, str(candidate or ""))
            return _cached_detect(_detection_snippet(working_example, visible_source, spec, watermarked.metadata)).score

        human_visible_source = visible_evaluation_source(example, example.reference_solution)
        clean_visible_source = visible_evaluation_source(working_example, working_example.reference_solution)
        watermarked_visible_source = visible_evaluation_source(working_example, watermarked.source)

        watermarked_detection = _cached_detect(
            _detection_snippet(working_example, watermarked_visible_source, spec, watermarked.metadata),
            timing_bucket="watermarked",
        )
        watermarked_validation = _cached_validate(watermarked.source, timing_bucket="watermarked")
        _log_progress(
            f"stage=watermark_ready index={example_index}/{total_examples} "
            f"example_id={example.example_id} baseline_family="
            f"{str(working_example.metadata.get('baseline_family', watermarked.metadata.get('baseline_family', '')))}"
        )
        _write_progress_state(
            config,
            progress_paths=progress_paths,
            stage="watermark_ready",
            example_index=example_index,
            total_examples=total_examples,
            rows_completed=len(rows),
            latest_example_id=example.example_id,
            status="running",
        )
        example_metadata = dict(working_example.metadata)
        watermarked_metadata = dict(watermarked.metadata)
        method_origin_label = watermark_origin(watermark.name)
        model_label = _model_label(config, spec, provider_mode=provider_mode)
        evaluation_track = _evaluation_track(
            provider_mode=provider_mode,
            watermark_uses_internal_generation=watermark.uses_internal_generation,
        )
        baseline_family = str(example_metadata.get("baseline_family", watermarked_metadata.get("baseline_family", "")))
        baseline_origin = str(example_metadata.get("baseline_origin", watermarked_metadata.get("baseline_origin", "")))
        baseline_upstream_commit = str(
            example_metadata.get("baseline_upstream_commit", watermarked_metadata.get("baseline_upstream_commit", ""))
        )
        human_detection = _cached_detect(
            _detection_snippet(example, human_visible_source, spec),
            timing_bucket="negative_control",
        )
        clean_reference_available = bool(str(clean_visible_source).strip()) and clean_visible_source != human_visible_source
        clean_reference_detection = (
            _cached_detect(_detection_snippet(working_example, clean_visible_source, spec), timing_bucket="negative_control")
            if clean_reference_available
            else None
        )
        negative_controls = {
            "human_reference": {
                "applicable": True,
                "available": _detection_available(human_detection),
                "source_digest": stable_hash(example.reference_solution),
                "score": human_detection.score,
                "detected": human_detection.detected,
                "threshold": human_detection.threshold,
                "metadata": dict(human_detection.metadata),
            },
            "clean_generation": {
                "applicable": clean_reference_available,
                "available": clean_reference_detection is not None and _detection_available(clean_reference_detection),
                "source_digest": stable_hash(working_example.reference_solution) if clean_reference_detection is not None else "",
                "score": clean_reference_detection.score if clean_reference_detection is not None else 0.0,
                "detected": clean_reference_detection.detected if clean_reference_detection is not None else False,
                "threshold": clean_reference_detection.threshold if clean_reference_detection is not None else float(spec.parameters.get("threshold", 0.5)),
                "metadata": dict(clean_reference_detection.metadata) if clean_reference_detection is not None else {},
            },
        }
        if baseline_family:
            baseline_eval_records.append(
                {
                    "example_id": example.example_id,
                    "task_id": str(example_metadata.get("task_id", example.example_id)),
                    "dataset": str(example_metadata.get("dataset", "unknown")),
                    "source_group": str(example_metadata.get("source_group", "")),
                    "language": working_example.language,
                    "baseline_family": baseline_family,
                    "baseline_origin": baseline_origin,
                    "baseline_upstream_commit": baseline_upstream_commit,
                    "watermark_scheme": watermark.name,
                    "evaluation_track": evaluation_track,
                    "model_label": model_label,
                    "prompt_digest": stable_hash(working_example.prompt),
                    "human_reference_digest": stable_hash(human_visible_source),
                    "clean_reference_digest": stable_hash(clean_visible_source),
                    "watermarked_source_digest": stable_hash(watermarked_visible_source),
                    "prompt_char_length": len(working_example.prompt),
                    "human_reference_char_length": len(human_visible_source),
                    "clean_reference_char_length": len(clean_visible_source),
                    "watermarked_source_char_length": len(watermarked_visible_source),
                    "human_detect_score": human_detection.score,
                    "human_detected": human_detection.detected,
                    "human_detect_available": _detection_available(human_detection),
                    "clean_reference_detect_score": clean_reference_detection.score if clean_reference_detection is not None else 0.0,
                    "clean_reference_detected": clean_reference_detection.detected if clean_reference_detection is not None else False,
                    "clean_reference_detect_available": (
                        _detection_available(clean_reference_detection) if clean_reference_detection is not None else False
                    ),
                    "watermarked_detect_score": watermarked_detection.score,
                    "watermarked_detected": watermarked_detection.detected,
                    "watermarked_detect_available": _detection_available(watermarked_detection),
                    "watermarked_validation": watermarked_validation.as_dict(),
                    "provider_mode": provider_mode,
                }
            )
            if _emit_sensitive_baseline_eval_payloads():
                baseline_eval_payloads.append(
                    {
                        "example_id": example.example_id,
                        "task_id": str(example_metadata.get("task_id", example.example_id)),
                        "prompt": working_example.prompt,
                        "human_reference_solution": example.reference_solution,
                        "clean_reference_solution": working_example.reference_solution,
                        "watermarked_source": watermarked.source,
                    }
                )

        def _semantic_gate(candidate: str, _example: BenchmarkExample = working_example) -> bool | None:
            validation = _cached_validate(candidate)
            return validation.passed if validation.available else None

        for attack_bundle in attack_bundles:
            _log_progress(
                f"stage=attack_start index={example_index}/{total_examples} "
                f"example_id={example.example_id} attack={attack_bundle.name}"
            )
            _write_progress_state(
                config,
                progress_paths=progress_paths,
                stage="attack_start",
                example_index=example_index,
                total_examples=total_examples,
                rows_completed=len(rows),
                latest_example_id=example.example_id,
                latest_attack=attack_bundle.name,
                status="running",
            )
            attack_context = {
                "detector": _candidate_detection_score,
                "quality": lambda candidate: _cached_quality(candidate),
                "validate": _semantic_gate,
                "config": config.attack_parameters.get(attack_bundle.name, {}),
                "language": working_example.language,
                "example_id": example.example_id,
            }
            attack_started = time.perf_counter()
            outcome = attack_bundle.apply(
                watermarked.source,
                seed=config.seed,
                metadata={
                    "example_id": example.example_id,
                    "attack": attack_bundle.name,
                    "language": example.language,
                },
                context=attack_context,
            )
            attack_seconds_by_attack[attack_bundle.name] = time.perf_counter() - attack_started
            attacked_visible_source = visible_evaluation_source(working_example, outcome.source)
            attacked_detection = _cached_detect(
                _detection_snippet(working_example, attacked_visible_source, spec, watermarked.metadata, outcome.metadata),
                timing_bucket=f"attacked:{attack_bundle.name}",
            )
            attacked_validation = _cached_validate(outcome.source, timing_bucket=f"attacked:{attack_bundle.name}")
            quality_score = _cached_quality(outcome.source)
            watermark_retention = 0.0
            if watermarked_detection.score > 0:
                watermark_retention = min(1.0, attacked_detection.score / watermarked_detection.score)
            robustness_score = max(0.0, min(1.0, attacked_detection.score * quality_score))
            _log_progress(
                f"stage=attack_done index={example_index}/{total_examples} "
                f"example_id={example.example_id} attack={attack_bundle.name} "
                f"detected={attacked_detection.detected} "
                f"semantic={'na' if not attacked_validation.available else attacked_validation.passed}"
            )
            comment_bits = list(outcome.notes)
            attack_metadata = dict(outcome.metadata)
            attack_supported = bool(attack_metadata.get("supported", True))
            unsupported_reason = str(attack_metadata.get("unsupported_reason", "")).strip()
            if not attack_supported:
                comment_bits.insert(0, "attack_unsupported")
                if unsupported_reason and unsupported_reason not in comment_bits:
                    comment_bits.append(unsupported_reason)
            else:
                if attacked_validation.available:
                    if attacked_validation.passed is False:
                        comment_bits.append("semantic_validation_failed")
                    else:
                        comment_bits.append("semantic_validation_passed")
                else:
                    comment_bits.append(str(attacked_validation.metadata.get("reason", "semantic_validation_unavailable")))
            row_attack_seconds = attack_seconds_by_attack.get(attack_bundle.name, 0.0)
            row_attacked_validation_seconds = attacked_validation_seconds_by_attack.get(attack_bundle.name, 0.0)
            row_attacked_detection_seconds = attacked_detection_seconds_by_attack.get(attack_bundle.name, 0.0)
            shared_validation_seconds = clean_validation_seconds + stage_timing_totals["watermarked_validation_seconds"]
            shared_detection_seconds = (
                stage_timing_totals["negative_control_detection_seconds"]
                + stage_timing_totals["watermarked_detection_seconds"]
            )
            row_validation_seconds = shared_validation_seconds + row_attacked_validation_seconds
            row_detection_seconds = shared_detection_seconds + row_attacked_detection_seconds
            row_total_example_seconds = (
                clean_generation_seconds
                + watermarked_generation_seconds
                + row_attack_seconds
                + row_validation_seconds
                + row_detection_seconds
            )
            clean_source_text = working_example.reference_solution
            watermarked_source_text = watermarked.source
            attacked_source_text = outcome.source
            rows.append(
                BenchmarkRow(
                    example_id=example.example_id,
                    attack_name=attack_bundle.name,
                    task_id=str(example_metadata.get("task_id", example.example_id)),
                    dataset=str(example_metadata.get("dataset", "unknown")),
                    language=working_example.language,
                    task_category=str(example_metadata.get("category", "")),
                    reference_kind=str(example_metadata.get("reference_kind", "canonical")),
                    method_origin=method_origin_label,
                    evaluation_track=evaluation_track,
                    model_label=model_label,
                    baseline_family=baseline_family,
                    baseline_origin=baseline_origin,
                    baseline_upstream_commit=baseline_upstream_commit,
                    source_group=str(example_metadata.get("source_group", "")),
                    origin_type=str(example_metadata.get("origin_type", "")),
                    family_id=str(example_metadata.get("family_id", "")),
                    difficulty=str(example_metadata.get("difficulty", "")),
                    attack_severity=float(getattr(attack_bundle, "severity", 0.0)),
                    watermark_scheme=watermark.name,
                    watermark_strength=spec.strength,
                    prompt_digest=str(example_metadata.get("prompt_digest", stable_hash(working_example.prompt))),
                    clean_score=clean_reference_detection.score if clean_reference_detection is not None else 0.0,
                    watermarked_score=watermarked_detection.score,
                    attacked_score=attacked_detection.score,
                    clean_detected=clean_reference_detection.detected if clean_reference_detection is not None else False,
                    watermarked_detected=watermarked_detection.detected,
                    attacked_detected=attacked_detection.detected,
                    quality_score=quality_score,
                    stealth_score=stealth_score(working_example.reference_solution, watermarked.source),
                    mutation_distance=1.0 - quality_score,
                    watermark_retention=watermark_retention,
                    robustness_score=robustness_score,
                    semantic_validation_available=attacked_validation.available,
                    semantic_preserving=attacked_validation.passed if attacked_validation.available else None,
                    status=(
                        "attack-unsupported"
                        if not attack_supported
                        else _status_label(
                            attacked_detected=attacked_detection.detected,
                            quality_score=quality_score,
                            semantic_available=attacked_validation.available,
                            semantic_preserving=attacked_validation.passed if attacked_validation.available else None,
                        )
                    ),
                    comment="; ".join(comment_bits),
                    notes=outcome.notes,
                    metadata={
                        "clean_evidence": list(clean_reference_detection.evidence) if clean_reference_detection is not None else [],
                        "watermarked_evidence": list(watermarked_detection.evidence),
                        "attacked_evidence": list(attacked_detection.evidence),
                        "clean_validation": clean_validation_payload,
                        "watermarked_validation": watermarked_validation.as_dict(),
                        "attacked_validation": attacked_validation.as_dict(),
                        "clean_functional_trials": clean_trials,
                        "clean_functional_summary": clean_summary,
                        "watermark_metadata": watermarked_metadata,
                        "negative_controls": negative_controls,
                        "clean_detection": {
                            "available": (
                                _detection_available(clean_reference_detection) if clean_reference_detection is not None else False
                            ),
                            "score": clean_reference_detection.score if clean_reference_detection is not None else 0.0,
                            "detected": clean_reference_detection.detected if clean_reference_detection is not None else False,
                            "threshold": (
                                clean_reference_detection.threshold
                                if clean_reference_detection is not None
                                else float(spec.parameters.get("threshold", 0.5))
                            ),
                            "evidence": list(clean_reference_detection.evidence) if clean_reference_detection is not None else [],
                        },
                        "clean_detection_metadata": dict(clean_reference_detection.metadata) if clean_reference_detection is not None else {},
                        "watermarked_detection": {
                            "available": _detection_available(watermarked_detection),
                            "score": watermarked_detection.score,
                            "detected": watermarked_detection.detected,
                            "threshold": watermarked_detection.threshold,
                            "evidence": list(watermarked_detection.evidence),
                        },
                        "attacked_detection": {
                            "available": _detection_available(attacked_detection),
                            "score": attacked_detection.score,
                            "detected": attacked_detection.detected,
                            "threshold": attacked_detection.threshold,
                            "evidence": list(attacked_detection.evidence),
                        },
                        "watermarked_detection_metadata": dict(watermarked_detection.metadata),
                        "attacked_detection_metadata": dict(attacked_detection.metadata),
                        "attack_metadata": attack_metadata,
                        "stage_timing": {
                            "timing_version": 1,
                            "clean_generation_seconds": _round_stage_seconds(clean_generation_seconds),
                            "watermarked_generation_seconds": _round_stage_seconds(watermarked_generation_seconds),
                            "attack_seconds": _round_stage_seconds(row_attack_seconds),
                            "validation_seconds": _round_stage_seconds(row_validation_seconds),
                            "detection_seconds": _round_stage_seconds(row_detection_seconds),
                            "total_example_seconds": _round_stage_seconds(row_total_example_seconds),
                            "clean_generation_standardized_token_count": _standardized_token_count(clean_source_text),
                            "watermarked_generation_standardized_token_count": _standardized_token_count(watermarked_source_text),
                            "attacked_standardized_token_count": _standardized_token_count(attacked_source_text),
                            "clean_generation_line_count": _standardized_line_count(clean_source_text),
                            "watermarked_line_count": _standardized_line_count(watermarked_source_text),
                            "attacked_line_count": _standardized_line_count(attacked_source_text),
                            "shared_stage_components": {
                                "clean_generation_seconds": _round_stage_seconds(clean_generation_seconds),
                                "watermarked_generation_seconds": _round_stage_seconds(watermarked_generation_seconds),
                                "clean_validation_seconds": _round_stage_seconds(clean_validation_seconds),
                                "watermarked_validation_seconds": _round_stage_seconds(stage_timing_totals["watermarked_validation_seconds"]),
                                "negative_control_detection_seconds": _round_stage_seconds(stage_timing_totals["negative_control_detection_seconds"]),
                                "watermarked_detection_seconds": _round_stage_seconds(stage_timing_totals["watermarked_detection_seconds"]),
                            },
                            "row_stage_components": {
                                "attack_seconds": _round_stage_seconds(row_attack_seconds),
                                "attacked_validation_seconds": _round_stage_seconds(row_attacked_validation_seconds),
                                "attacked_detection_seconds": _round_stage_seconds(row_attacked_detection_seconds),
                            },
                        },
                        "example_metadata": example_metadata,
                        "benchmark_manifest": benchmark_manifest.as_dict(),
                        "provider_mode": provider_mode,
                    },
                )
            )
            _append_jsonl(progress_paths.get("partial_rows", ""), [rows[-1].as_dict()] if progress_paths else [])
        _log_progress(
            f"stage=example_done index={example_index}/{total_examples} "
            f"example_id={example.example_id} accumulated_rows={len(rows)}"
        )
        _write_partial_report(
            config,
            progress_paths=progress_paths,
            rows=rows,
            benchmark_manifest=benchmark_manifest.as_dict(),
            example_index=example_index,
            total_examples=total_examples,
            stage="example_done",
        )
        _write_progress_state(
            config,
            progress_paths=progress_paths,
            stage="example_done",
            example_index=example_index,
            total_examples=total_examples,
            rows_completed=len(rows),
            latest_example_id=example.example_id,
            status="running",
        )

    report = build_report(
        config,
        rows,
        output_path=config.output_path,
        benchmark_manifest=benchmark_manifest.as_dict(),
    )
    _log_progress(f"stage=report_ready rows={len(rows)} output={config.output_path or ''}")
    _write_partial_report(
        config,
        progress_paths=progress_paths,
        rows=rows,
        benchmark_manifest=benchmark_manifest.as_dict(),
        example_index=total_examples,
        total_examples=total_examples,
        stage="report_ready",
    )

    if config.output_path:
        ensure_parent(config.output_path).write_text(report.to_json(), encoding="utf-8")
        if baseline_eval_records:
            _write_jsonl(Path(config.output_path).with_name("baseline_eval_records.jsonl"), baseline_eval_records)
        payload_path = _baseline_eval_payload_output_path(config)
        if baseline_eval_payloads and payload_path is not None:
            _write_jsonl(payload_path, baseline_eval_payloads)
    _log_progress(f"stage=complete rows={len(rows)} examples={len(examples)}")
    _write_progress_state(
        config,
        progress_paths=progress_paths,
        stage="complete",
        example_index=total_examples,
        total_examples=total_examples,
        rows_completed=len(rows),
        status="completed",
    )

    return BenchmarkRun(config=config, examples=examples, report=report, benchmark_manifest=benchmark_manifest.as_dict())
