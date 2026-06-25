from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .language_support import (
    default_evaluation_backend,
    language_family,
    normalize_language_name,
    source_relative_to,
    supports_execution,
    validation_mode,
)
from .models import BenchmarkExample
from .utils import stable_hash
from .validation import validate_semantics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NORMALIZED_BENCHMARK = PROJECT_ROOT / "data" / "fixtures" / "benchmark.normalized.jsonl"


def _release_source_path(source: str | Path | None) -> str:
    if source is None:
        return ""
    return source_relative_to(PROJECT_ROOT, Path(source))


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    source_path: str
    record_count: int
    datasets: tuple[str, ...]
    splits: tuple[str, ...]
    observed_languages: tuple[str, ...]
    claimed_languages: tuple[str, ...]
    validation_supported_languages: tuple[str, ...]
    runtime_validation_supported_languages: tuple[str, ...]
    language_summary: Mapping[str, Any]
    coverage: Mapping[str, Any]
    source_digest: str
    source_group_counts: Mapping[str, int] = None
    origin_type_counts: Mapping[str, int] = None
    difficulty_counts: Mapping[str, int] = None
    reference_kind_counts: Mapping[str, int] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "record_count": self.record_count,
            "datasets": list(self.datasets),
            "splits": list(self.splits),
            "observed_languages": list(self.observed_languages),
            "claimed_languages": list(self.claimed_languages),
            "validation_supported_languages": list(self.validation_supported_languages),
            "runtime_validation_supported_languages": list(self.runtime_validation_supported_languages),
            "language_summary": {key: dict(value) for key, value in self.language_summary.items()},
            "coverage": dict(self.coverage),
            "source_group_counts": dict(self.source_group_counts or {}),
            "origin_type_counts": dict(self.origin_type_counts or {}),
            "difficulty_counts": dict(self.difficulty_counts or {}),
            "reference_kind_counts": dict(self.reference_kind_counts or {}),
            "source_digest": self.source_digest,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"{path} must contain JSON objects on each line")
            rows.append(payload)
    return rows


def _normalize_sequence(values: object, *, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if values is None:
        return default
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Sequence):
        return default
    return tuple(str(item).strip() for item in values if str(item).strip())


def _normalize_reference_kinds(values: object) -> tuple[str, ...]:
    normalized = _normalize_sequence(values)
    return tuple(str(item).strip().lower() for item in normalized if str(item).strip())


def _validation_supported(language: str, execution_tests: tuple[str, ...], evaluation_backend: str) -> bool:
    return supports_execution(language, execution_tests, backend=evaluation_backend)


def normalize_benchmark_record(record: Mapping[str, Any], index: int, source: Path) -> BenchmarkExample:
    prompt = str(record.get("prompt", "")).strip()
    if not prompt:
        raise ValueError(f"Benchmark record at index {index} in {source} is missing a prompt")

    language = normalize_language_name(str(record.get("language", "python")).strip())
    task_id = str(record.get("task_id") or record.get("id") or f"{source.stem}-{index:03d}")
    dataset = str(record.get("dataset", source.stem)).strip() or source.stem
    reference_solution = str(record.get("reference_solution", record.get("canonical_solution", ""))).rstrip()
    tags = _normalize_sequence(record.get("tags"))
    execution_tests = _normalize_sequence(record.get("execution_tests"))
    reference_tests = _normalize_sequence(record.get("reference_tests"), default=execution_tests)
    if not reference_tests:
        reference_tests = execution_tests
    claimed_languages = _normalize_sequence(record.get("claimed_languages"), default=(language,))
    evaluation_backend = str(record.get("evaluation_backend", default_evaluation_backend(language))).strip().lower()

    metadata: dict[str, Any] = {
        "task_id": task_id,
        "dataset": dataset,
        "source_path": _release_source_path(source),
        "source_index": index,
        "source_digest": stable_hash(f"{task_id}|{prompt}|{reference_solution}"),
        "prompt_digest": stable_hash(prompt),
        "solution_digest": stable_hash(reference_solution),
        "tags": list(tags),
        "language_family": language_family(language),
        "validation_mode": validation_mode(language),
        "evaluation_backend": evaluation_backend,
        "validation_supported": _validation_supported(language, execution_tests, evaluation_backend),
        "claimed_languages": list(claimed_languages),
    }
    for key in (
        "source_url",
        "source_revision",
        "source_sha256",
        "split",
        "license_note",
        "adapter_name",
        "validation_scope",
        "public_source",
        "record_kind",
        "entry_point",
        "contract",
        "stress_suite",
        "stress_base_input_count",
        "stress_plus_input_count",
        "source_group",
        "origin_type",
        "family_id",
        "category",
        "template_family",
        "difficulty",
        "validation_backend",
        "runner_image",
        "official_problem_file",
        "language_version",
        "reference_kind",
        "smoke_completion_available",
        "canonical_available",
        "example_test",
        "description",
        "semantic_contract",
        "translation_anchor_language",
        "class_level_task",
        "method_count",
        "artifact_policy",
        "included_in_core",
    ):
        if record.get(key) is not None:
            metadata[key] = record.get(key)
    if record.get("expected_behavior") is not None:
        metadata["expected_behavior"] = str(record.get("expected_behavior", ""))
    if record.get("notes") is not None:
        metadata["notes"] = str(record.get("notes", ""))
    if record.get("stress_tests") is not None:
        metadata["stress_tests"] = list(_normalize_sequence(record.get("stress_tests")))
    if record.get("metamorphic_tests") is not None:
        metadata["metamorphic_tests"] = list(_normalize_sequence(record.get("metamorphic_tests")))

    return BenchmarkExample(
        example_id=task_id,
        language=language,
        prompt=prompt,
        reference_solution=reference_solution,
        reference_tests=reference_tests,
        execution_tests=execution_tests,
        metadata=metadata,
    )


def _group_examples_by_language(examples: Sequence[BenchmarkExample]) -> dict[str, list[BenchmarkExample]]:
    grouped: dict[str, list[BenchmarkExample]] = defaultdict(list)
    for example in examples:
        grouped[example.language.lower()].append(example)
    return grouped


def _language_order(examples: Sequence[BenchmarkExample], languages: Sequence[str] | None) -> list[str]:
    if languages:
        return [normalize_language_name(str(language).lower()) for language in languages if str(language).strip()]
    seen: list[str] = []
    for example in examples:
        language = example.language.lower()
        if language not in seen:
            seen.append(language)
    return seen


def _select_examples(
    examples: Sequence[BenchmarkExample],
    *,
    count: int | None,
    languages: Sequence[str] | None,
    seed: int,
) -> list[BenchmarkExample]:
    if not examples:
        return []

    grouped = _group_examples_by_language(examples)
    if languages is None:
        pool = list(examples)
        if count is None or count >= len(pool):
            return pool
        return pool[:count]

    ordered_languages = _language_order(examples, languages)
    available = [language for language in ordered_languages if language in grouped]
    pool = [example for language in available for example in grouped[language]]

    if count is None or count >= len(pool):
        return pool

    if not ordered_languages:
        return pool[:count]

    ranked_groups: dict[str, list[BenchmarkExample]] = {}
    for language in ordered_languages:
        language_examples = grouped.get(language, [])
        ranked_groups[language] = sorted(
            language_examples,
            key=lambda example: (
                stable_hash(f"{seed}:{example.example_id}"),
                example.example_id,
            ),
        )

    selected: list[BenchmarkExample] = []
    offsets = {language: 0 for language in ordered_languages}
    while len(selected) < count:
        advanced = False
        for language in ordered_languages:
            bucket = ranked_groups.get(language, [])
            offset = offsets[language]
            if offset < len(bucket):
                selected.append(bucket[offset])
                offsets[language] = offset + 1
                advanced = True
                if len(selected) >= count:
                    break
        if not advanced:
            break
    return selected


def load_benchmark_corpus(
    path: str | Path | None = None,
    *,
    count: int | None = None,
    seed: int = 7,
    languages: str | Sequence[str] | None = None,
    include_reference_kinds: Sequence[str] | None = None,
    prompt_prefix: str = "",
) -> list[BenchmarkExample]:
    path = Path(path) if path is not None else DEFAULT_NORMALIZED_BENCHMARK
    if not path.exists():
        raise FileNotFoundError(path)

    raw_rows = _read_jsonl(path)
    examples = [normalize_benchmark_record(row, index, path) for index, row in enumerate(raw_rows)]
    reference_kinds = {item.lower() for item in include_reference_kinds or [] if str(item).strip()}
    if reference_kinds:
        examples = [
            example
            for example in examples
            if str(example.metadata.get("reference_kind", "canonical")).strip().lower() in reference_kinds
        ]

    normalized_languages: list[str] | None
    if languages is None:
        normalized_languages = None
    elif isinstance(languages, str):
        normalized_languages = [normalize_language_name(languages)]
    else:
        normalized_languages = [normalize_language_name(str(language)) for language in languages if str(language).strip()]

    selected = _select_examples(examples, count=count, languages=normalized_languages, seed=seed)
    if prompt_prefix:
        selected = [
            BenchmarkExample(
                example_id=example.example_id,
                language=example.language,
                prompt=f"{prompt_prefix}{example.prompt}",
                reference_solution=example.reference_solution,
                reference_tests=example.reference_tests,
                execution_tests=example.execution_tests,
                metadata=example.metadata,
            )
            for example in selected
        ]
    return selected


def build_benchmark_manifest(
    examples: Sequence[BenchmarkExample],
    *,
    source_path: str | Path | None = None,
    claimed_languages: Sequence[str] | None = None,
) -> BenchmarkManifest:
    dataset_names = sorted(
        {
            str(example.metadata.get("dataset", "")).strip() or "unknown"
            for example in examples
        }
    )
    split_names = sorted(
        {
            str(example.metadata.get("split", "")).strip() or "unspecified"
            for example in examples
        }
    )
    observed_languages = []
    for example in examples:
        language = example.language.lower()
        if language not in observed_languages:
            observed_languages.append(language)

    claimed = [normalize_language_name(str(language)) for language in claimed_languages or observed_languages if str(language).strip()]
    if not claimed:
        claimed = list(observed_languages)

    language_summary: dict[str, dict[str, Any]] = {}
    validation_supported_languages: list[str] = []
    runtime_validation_supported_languages: list[str] = []
    validation_results = {
        example.example_id: validate_semantics(example, example.reference_solution)
        for example in examples
    }
    for language in observed_languages:
        grouped = [example for example in examples if example.language.lower() == language]
        supported = [
            example
            for example in grouped
            if _validation_supported(
                example.language,
                tuple(example.execution_tests),
                str(example.metadata.get("evaluation_backend", default_evaluation_backend(example.language))),
            )
        ]
        runtime_supported = [
            example
            for example in grouped
            if validation_results[example.example_id].available
        ]
        if supported:
            validation_supported_languages.append(language)
        if runtime_supported:
            runtime_validation_supported_languages.append(language)
        language_summary[language] = {
            "count": len(grouped),
            "share": round(len(grouped) / len(examples), 4) if examples else 0.0,
            "validation_available_count": len(supported),
            "validation_available_rate": round(len(supported) / len(grouped), 4) if grouped else 0.0,
            "declared_validation_available_count": len(supported),
            "declared_validation_available_rate": round(len(supported) / len(grouped), 4) if grouped else 0.0,
            "runtime_validation_available_count": len(runtime_supported),
            "runtime_validation_available_rate": round(len(runtime_supported) / len(grouped), 4) if grouped else 0.0,
            "language_family": language_family(language),
            "validation_mode": validation_mode(language),
        }

    observed_set = set(observed_languages)
    claimed_set = set(claimed)
    validation_set = set(validation_supported_languages)
    declared_validation_rate = round(
        sum(
            1
            for example in examples
            if _validation_supported(
                example.language,
                tuple(example.execution_tests),
                str(example.metadata.get("evaluation_backend", default_evaluation_backend(example.language))),
            )
        )
        / len(examples),
        4,
    ) if examples else 0.0
    runtime_validation_rate = round(
        sum(1 for result in validation_results.values() if result.available) / len(examples),
        4,
    ) if examples else 0.0
    runtime_validation_language_rate = round(
        len(set(runtime_validation_supported_languages) & claimed_set) / len(claimed_set),
        4,
    ) if claimed_set else 1.0
    compile_success_rate = round(
        sum(1 for result in validation_results.values() if result.metadata.get("compile_success") is True) / len(examples),
        4,
    ) if examples else 0.0
    pass_rate = round(
        sum(1 for result in validation_results.values() if result.passed is True) / len(examples),
        4,
    ) if examples else 0.0
    coverage = {
        "observed_language_count": len(observed_languages),
        "claimed_language_count": len(claimed),
        "observed_coverage_rate": round(len(observed_set & claimed_set) / len(claimed_set), 4) if claimed_set else 1.0,
        "declared_semantic_validation_rate": declared_validation_rate,
        "declared_semantic_validation_language_rate": round(len(validation_set & claimed_set) / len(claimed_set), 4) if claimed_set else 1.0,
        "runtime_semantic_validation_rate": runtime_validation_rate,
        "runtime_semantic_validation_language_rate": runtime_validation_language_rate,
        "semantic_validation_rate": runtime_validation_rate,
        "semantic_validation_language_rate": runtime_validation_language_rate,
        "clean_reference_compile_rate": compile_success_rate,
        "clean_reference_pass_rate": pass_rate,
        "missing_claimed_languages": [language for language in claimed if language not in observed_set],
        "declared_unvalidated_languages": [language for language in observed_languages if language not in validation_set],
        "runtime_unvalidated_languages": [
            language for language in observed_languages if language not in set(runtime_validation_supported_languages)
        ],
        "unvalidated_languages": [
            language for language in observed_languages if language not in set(runtime_validation_supported_languages)
        ],
    }
    source_group_counts: dict[str, int] = {}
    origin_type_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    reference_kind_counts: dict[str, int] = {}
    for example in examples:
        source_group = str(example.metadata.get("source_group", "")).strip()
        if source_group:
            source_group_counts[source_group] = source_group_counts.get(source_group, 0) + 1
        origin_type = str(example.metadata.get("origin_type", "")).strip()
        if origin_type:
            origin_type_counts[origin_type] = origin_type_counts.get(origin_type, 0) + 1
        difficulty = str(example.metadata.get("difficulty", "")).strip()
        if difficulty:
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        reference_kind = str(example.metadata.get("reference_kind", "canonical")).strip().lower()
        if reference_kind:
            reference_kind_counts[reference_kind] = reference_kind_counts.get(reference_kind, 0) + 1
    source_digest = stable_hash("|".join(example.example_id for example in examples)) if examples else ""
    return BenchmarkManifest(
        source_path=_release_source_path(source_path),
        record_count=len(examples),
        datasets=tuple(dataset_names),
        splits=tuple(split_names),
        observed_languages=tuple(observed_languages),
        claimed_languages=tuple(claimed),
        validation_supported_languages=tuple(validation_supported_languages),
        runtime_validation_supported_languages=tuple(runtime_validation_supported_languages),
        language_summary=language_summary,
        coverage=coverage,
        source_group_counts=source_group_counts,
        origin_type_counts=origin_type_counts,
        difficulty_counts=difficulty_counts,
        reference_kind_counts=reference_kind_counts,
        source_digest=source_digest,
    )
