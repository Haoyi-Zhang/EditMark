from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import (
    DEFAULT_FIXTURE,
    DEFAULT_INTERIM_DIR,
    digest_text,
    dump_json,
    ensure_dir,
    load_config,
    load_json_or_jsonl,
    normalize_text,
    read_jsonl,
    write_jsonl,
)
from posteditbench.collections import compose_benchmark_collection, write_composed_collection
from posteditbench.crafted_benchmarks import write_crafted_benchmark
from posteditbench.language_support import (
    default_evaluation_backend,
    language_family,
    language_version,
    normalize_language_name,
    runner_image,
    source_relative_to,
    supports_execution,
    validation_mode,
)
from posteditbench.public_benchmarks import prepare_public_benchmark, resolve_public_source_name


SAMPLE_TASKS: list[dict[str, Any]] = [
    {
        "task_id": "synthetic_py_sort_unique",
        "dataset": "synthetic",
        "language": "python",
        "prompt": "Write a function unique_sorted(values) that returns the unique integers sorted from smallest to largest.",
        "reference_solution": "def unique_sorted(values):\n    return sorted(set(values))\n",
        "tags": ["dedup", "sorting"],
        "execution_tests": [
            "assert unique_sorted([3, 1, 2, 3]) == [1, 2, 3]",
            "assert unique_sorted([]) == []",
        ],
        "difficulty": "easy",
    },
    {
        "task_id": "synthetic_py_count_vowels",
        "dataset": "synthetic",
        "language": "python",
        "prompt": "Write a function count_vowels(text) that counts the vowels in the given string.",
        "reference_solution": "def count_vowels(text):\n    return sum(1 for char in text.lower() if char in 'aeiou')\n",
        "tags": ["string", "counting"],
        "execution_tests": [
            "assert count_vowels('hello') == 2",
            "assert count_vowels('') == 0",
        ],
        "difficulty": "easy",
    },
    {
        "task_id": "synthetic_py_flatten",
        "dataset": "synthetic",
        "language": "python",
        "prompt": "Write a function flatten_once(items) that flattens one level of nested lists.",
        "reference_solution": "def flatten_once(items):\n    out = []\n    for item in items:\n        if isinstance(item, list):\n            out.extend(item)\n        else:\n            out.append(item)\n    return out\n",
        "tags": ["lists", "nested"],
        "execution_tests": [
            "assert flatten_once([1, [2, 3], 4]) == [1, 2, 3, 4]",
            "assert flatten_once([]) == []",
        ],
        "difficulty": "medium",
    },
    {
        "task_id": "synthetic_js_sum_array",
        "dataset": "synthetic",
        "language": "javascript",
        "prompt": "Write a function sumArray(values) that returns the sum of numeric values in an array.",
        "reference_solution": "function sumArray(values) {\n  return values.reduce((total, value) => total + value, 0);\n}\n",
        "tags": ["aggregation", "javascript"],
        "execution_tests": [],
        "difficulty": "easy",
    },
    {
        "task_id": "synthetic_js_slugify",
        "dataset": "synthetic",
        "language": "javascript",
        "prompt": "Write a function slugify(title) that lowercases text and replaces runs of whitespace with hyphens.",
        "reference_solution": "function slugify(title) {\n  return title.trim().toLowerCase().replace(/\\s+/g, '-');\n}\n",
        "tags": ["string", "normalization"],
        "execution_tests": [],
        "difficulty": "easy",
    },
    {
        "task_id": "synthetic_java_merge_intervals",
        "dataset": "synthetic",
        "language": "java",
        "prompt": "Write a method mergeIntervals that merges overlapping inclusive integer intervals.",
        "reference_solution": "List<int[]> mergeIntervals(List<int[]> intervals) {\n  return intervals;\n}\n",
        "tags": ["intervals", "java"],
        "execution_tests": [],
        "difficulty": "medium",
    },
    {
        "task_id": "synthetic_go_reverse_words",
        "dataset": "synthetic",
        "language": "go",
        "prompt": "Write a function reverseWords(text) that reverses the order of space-separated words.",
        "reference_solution": "package main\n\nimport \"strings\"\n\nfunc reverseWords(text string) string {\n  parts := strings.Fields(text)\n  for i, j := 0, len(parts)-1; i < j; i, j = i+1, j-1 {\n    parts[i], parts[j] = parts[j], parts[i]\n  }\n  return strings.Join(parts, \" \")\n}\n",
        "tags": ["go", "strings"],
        "execution_tests": [],
        "difficulty": "easy",
    },
    {
        "task_id": "synthetic_py_reverse_words",
        "dataset": "synthetic",
        "language": "python",
        "prompt": "Write a function reverse_words(sentence) that reverses the order of whitespace-separated words.",
        "reference_solution": "def reverse_words(sentence):\n    return ' '.join(reversed(sentence.split()))\n",
        "tags": ["strings", "tokenization"],
        "execution_tests": [
            "assert reverse_words('one two three') == 'three two one'",
            "assert reverse_words('single') == 'single'",
        ],
        "difficulty": "easy",
    },
]


def normalize_task(task: dict[str, Any], index: int, source: Path) -> dict[str, Any]:
    prompt = normalize_text(str(task.get("prompt", "")))
    reference = str(task.get("reference_solution", task.get("canonical_solution", ""))).rstrip()
    if not prompt:
        raise ValueError(f"Task at index {index} is missing a prompt")
    task_id = str(task.get("task_id") or task.get("id") or f"task_{index:04d}")
    dataset = str(task.get("dataset", "synthetic"))
    language = normalize_language_name(str(task.get("language", "python")))
    tags = task.get("tags", [])
    if not isinstance(tags, list):
        tags = [str(tags)]
    execution = task.get("execution_tests", [])
    execution_tests = [str(test) for test in execution] if isinstance(execution, (list, tuple)) else []
    reference_tests = task.get("reference_tests", execution_tests)
    if isinstance(reference_tests, (list, tuple)):
        reference_tests = [str(test) for test in reference_tests]
    else:
        reference_tests = list(execution_tests)
    evaluation_backend = default_evaluation_backend(language)
    claimed_languages = task.get("claimed_languages")
    if isinstance(claimed_languages, (list, tuple)):
        claimed = [normalize_language_name(str(item)) for item in claimed_languages if str(item).strip()]
    else:
        claimed = [language]
    return {
        "task_id": task_id,
        "dataset": dataset,
        "language": language,
        "prompt": prompt,
        "reference_solution": reference,
        "tags": [str(tag) for tag in tags],
        "reference_tests": reference_tests,
        "execution_tests": execution_tests,
        "claimed_languages": claimed,
        "language_family": language_family(language),
        "validation_mode": validation_mode(language),
        "validation_supported": supports_execution(language, execution_tests, backend=evaluation_backend),
        "source": str(source.name),
        "source_digest": digest_text(prompt + reference + task_id),
        "prompt_digest": digest_text(prompt),
        "solution_digest": digest_text(reference),
        "expected_behavior": str(task.get("expected_behavior", "")),
        "notes": str(task.get("notes", "")),
        "record_kind": "smoke_fixture",
        "source_group": "smoke_synthetic",
        "origin_type": "smoke",
        "family_id": str(task.get("family_id") or task_id),
        "difficulty": str(task.get("difficulty", "easy")),
        "evaluation_backend": evaluation_backend,
        "runner_image": runner_image(language),
        "official_problem_file": str(source),
        "language_version": language_version(language),
        "reference_kind": "canonical",
    }


def _count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "")).strip().lower()
        if value:
            counts[value] = counts.get(value, 0) + 1
    return counts


def _reference_kind_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = _count_by_key(rows, "reference_kind")
    if "canonical" not in counts:
        counts["canonical"] = 0
    return dict(sorted(counts.items()))


def _coverage_from_rows(rows: list[dict[str, Any]], configured_languages: list[str]) -> dict[str, Any]:
    languages = sorted({row["language"] for row in rows})
    claimed = configured_languages or languages
    validation_supported_languages = sorted({row["language"] for row in rows if row.get("validation_supported")})
    runtime_annotation_available = any("clean_reference_validation_available" in row for row in rows)
    compile_annotation_available = any("clean_reference_compile_success" in row for row in rows)
    pass_annotation_available = any("clean_reference_passed" in row for row in rows)
    runtime_validation_supported_languages = (
        sorted({row["language"] for row in rows if row.get("clean_reference_validation_available")})
        if runtime_annotation_available
        else []
    )
    clean_reference_compile_rate = (
        round(sum(1 for row in rows if row.get("clean_reference_compile_success") is True) / max(1, len(rows)), 4)
        if compile_annotation_available
        else None
    )
    clean_reference_pass_rate = (
        round(sum(1 for row in rows if row.get("clean_reference_passed") is True) / max(1, len(rows)), 4)
        if pass_annotation_available
        else None
    )
    return {
        "observed_language_count": len(languages),
        "claimed_language_count": len(claimed),
        "observed_coverage_rate": round(len(set(languages) & set(claimed)) / max(1, len(claimed)), 4),
        "declared_semantic_validation_rate": round(sum(1 for row in rows if row.get("validation_supported")) / max(1, len(rows)), 4),
        "declared_semantic_validation_language_rate": round(len(set(validation_supported_languages)) / max(1, len(claimed)), 4),
        "runtime_semantic_validation_rate": (
            round(sum(1 for row in rows if row.get("clean_reference_validation_available")) / max(1, len(rows)), 4)
            if runtime_annotation_available
            else None
        ),
        "runtime_semantic_validation_language_rate": (
            round(len(set(runtime_validation_supported_languages)) / max(1, len(claimed)), 4)
            if runtime_annotation_available
            else None
        ),
        "semantic_validation_rate": (
            round(sum(1 for row in rows if row.get("clean_reference_validation_available")) / max(1, len(rows)), 4)
            if runtime_annotation_available
            else None
        ),
        "semantic_validation_language_rate": (
            round(len(set(runtime_validation_supported_languages)) / max(1, len(claimed)), 4)
            if runtime_annotation_available
            else None
        ),
        "clean_reference_compile_rate": clean_reference_compile_rate,
        "clean_reference_pass_rate": clean_reference_pass_rate,
        "runtime_validation_basis": "row_annotations" if runtime_annotation_available else "unavailable",
        "runtime_validation_annotations_available": runtime_annotation_available,
        "missing_claimed_languages": sorted(set(claimed) - set(languages)),
        "declared_unvalidated_languages": sorted({row["language"] for row in rows if not row.get("validation_supported")}),
        "runtime_unvalidated_languages": (
            sorted({row["language"] for row in rows if not row.get("clean_reference_validation_available")})
            if runtime_annotation_available
            else []
        ),
        "unvalidated_languages": (
            sorted({row["language"] for row in rows if not row.get("clean_reference_validation_available")})
            if runtime_annotation_available
            else []
        ),
    }


def _write_manifest(output: Path, manifest: dict[str, Any]) -> None:
    dump_json(output.with_suffix(".manifest.json"), manifest)


def _augment_manifest_with_reference_kinds(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl(output)
    rows = _sanitize_release_rows(rows)
    write_jsonl(output, rows)
    counts = _reference_kind_counts(rows)
    updated = dict(manifest)
    updated["reference_kind_counts"] = counts
    updated["reference_kind_total"] = sum(counts.values())
    updated["canonical_reference_count"] = counts.get("canonical", 0)
    updated["smoke_overlay_reference_count"] = counts.get("smoke_overlay", 0)
    _write_manifest(output, updated)
    return updated


def _scrub_release_path(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    candidate = source_relative_to(ROOT, Path(text))
    if Path(candidate).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", candidate):
        return Path(candidate).name
    if ".coordination" in candidate.replace("/", "\\").lower():
        return Path(candidate).name
    return candidate.replace("/", "\\")


def _sanitize_release_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item.get("source_path") is not None:
            item["source_path"] = _scrub_release_path(item.get("source_path"))
        if item.get("official_problem_file") is not None:
            item["official_problem_file"] = _scrub_release_path(item.get("official_problem_file"))
        sanitized.append(item)
    return sanitized


def bootstrap_fixture(path: Path) -> None:
    ensure_dir(path.parent)
    write_jsonl(path, SAMPLE_TASKS)


def _resolve_output_path(args: argparse.Namespace, benchmark_cfg: dict[str, Any], fallback: Path) -> Path:
    return args.output or Path(benchmark_cfg.get("prepared_output", fallback))


def _collection_input(entry: Any, benchmark_cfg: dict[str, Any], fetch: bool, *, force: bool = False) -> Path:
    if isinstance(entry, str):
        return Path(entry)
    if not isinstance(entry, dict):
        raise TypeError("collection_sources entries must be strings or objects")
    kind = str(entry.get("type", "path")).lower()
    path = Path(str(entry.get("path", ""))) if entry.get("path") else None
    if kind == "public":
        name = resolve_public_source_name(str(entry.get("name", "")))
        if path is None:
            path = DEFAULT_INTERIM_DIR.parent / "public" / name / "normalized.jsonl"
        if force or not path.exists() or not path.with_suffix(".manifest.json").exists():
            cache_dir = Path(benchmark_cfg.get("cache_dir", DEFAULT_INTERIM_DIR.parent / "public" / "_cache"))
            prepare_public_benchmark(name, output_path=path, fetch=fetch, cache_dir=cache_dir)
        return path
    if kind == "crafted":
        name = str(entry.get("name", "")).strip() or "crafted_original"
        if path is None:
            path = DEFAULT_INTERIM_DIR / "crafted" / f"{name}.normalized.jsonl"
        if force or not path.exists() or not path.with_suffix(".manifest.json").exists():
            write_crafted_benchmark(name, output_path=path)
        return path
    if path is None:
        raise ValueError("path-based collection source requires a path")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare normalized benchmark fixtures for PostEditBench.")
    parser.add_argument("--config", type=Path, default=None, help="JSON-compatible YAML config file.")
    parser.add_argument("--source", type=Path, default=None, help="Raw JSONL or JSON benchmark file.")
    parser.add_argument("--output", type=Path, default=None, help="Destination JSONL path for the normalized benchmark.")
    parser.add_argument("--bootstrap-fixture", action="store_true", help="Create the default synthetic fixture when the source is missing.")
    parser.add_argument("--public-source", type=str, default=None, help="Fetch and normalize a pinned public benchmark source.")
    parser.add_argument("--crafted-source", type=str, default=None, help="Generate a crafted benchmark source.")
    parser.add_argument("--compose", action="store_true", help="Compose multiple prepared sources into a unified collection.")
    parser.add_argument("--collection-input", action="append", default=None, help="Prepared normalized JSONL paths to compose.")
    parser.add_argument("--fetch", action="store_true", help="Allow downloading a public benchmark source if the cached archive is absent.")
    parser.add_argument("--force", action="store_true", help="Force regeneration even when a prepared snapshot already exists.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of tasks after normalization.")
    return parser.parse_args()


def _write_smoke_manifest(output: Path, rows: list[dict[str, Any]], configured_languages: list[str]) -> None:
    languages = sorted({row["language"] for row in rows})
    manifest = {
        "schema_version": 2,
        "source": str(output),
        "output": str(output),
        "record_count": len(rows),
        "datasets": sorted({row["dataset"] for row in rows}),
        "observed_languages": languages,
        "claimed_languages": configured_languages or sorted({language for row in rows for language in row.get("claimed_languages", [])}),
        "validation_supported_languages": sorted({row["language"] for row in rows if row.get("validation_supported")}),
        "language_summary": {
            language: {
                "count": sum(1 for row in rows if row["language"] == language),
                "validation_available_count": sum(1 for row in rows if row["language"] == language and row.get("validation_supported")),
                "validation_available_rate": round(
                    sum(1 for row in rows if row["language"] == language and row.get("validation_supported"))
                    / max(1, sum(1 for row in rows if row["language"] == language)),
                    4,
                ),
                "language_family": language_family(language),
                "validation_mode": validation_mode(language),
            }
            for language in languages
        },
        "coverage": {
            "observed_language_count": len(languages),
            "claimed_language_count": len(configured_languages or languages),
            "observed_coverage_rate": round(len(set(languages) & set(configured_languages or languages)) / max(1, len(configured_languages or languages)), 4),
            "semantic_validation_rate": round(sum(1 for row in rows if row.get("validation_supported")) / max(1, len(rows)), 4),
            "semantic_validation_language_rate": round(len({row["language"] for row in rows if row.get("validation_supported")}) / max(1, len(configured_languages or languages)), 4),
            "missing_claimed_languages": sorted(set(configured_languages or languages) - set(languages)),
            "unvalidated_languages": sorted({row["language"] for row in rows if not row.get("validation_supported")}),
        },
        "source_group_counts": {"smoke_synthetic": len(rows)},
        "origin_type_counts": {"smoke": len(rows)},
        "difficulty_counts": {
            difficulty: sum(1 for row in rows if row.get("difficulty") == difficulty)
            for difficulty in sorted({str(row.get("difficulty", "")).lower() for row in rows})
        },
        "reference_kind_counts": _reference_kind_counts(rows),
        "source_digest": digest_text("\n".join(row["task_id"] for row in rows)),
    }
    _write_manifest(output, manifest)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    benchmark_cfg = config.get("benchmark", {})
    configured_languages = [normalize_language_name(str(language)) for language in benchmark_cfg.get("languages", []) if str(language).strip()]

    public_source = args.public_source or benchmark_cfg.get("public_source")
    if public_source:
        resolved = resolve_public_source_name(str(public_source))
        output = _resolve_output_path(args, benchmark_cfg, DEFAULT_INTERIM_DIR.parent / "public" / resolved / "normalized.jsonl")
        cache_dir = Path(benchmark_cfg.get("cache_dir", DEFAULT_INTERIM_DIR.parent / "public" / "_cache"))
        manifest_path = output.with_suffix(".manifest.json")
        if output.exists() and manifest_path.exists() and not (args.force or args.fetch or benchmark_cfg.get("fetch", False)):
            rows = _sanitize_release_rows(read_jsonl(output))
            write_jsonl(output, rows)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = _augment_manifest_with_reference_kinds(output, manifest)
            print(f"Using cached public benchmark snapshot at {output}")
            print(f"{manifest['dataset_label']}: {manifest['task_count']} tasks, split={manifest.get('split', 'test')}")
            return 0
        manifest = prepare_public_benchmark(resolved, output_path=output, fetch=bool(args.fetch or benchmark_cfg.get("fetch", False)), cache_dir=cache_dir)
        manifest = _augment_manifest_with_reference_kinds(output, manifest)
        print(f"Wrote public benchmark snapshot to {output}")
        print(f"{manifest['dataset_label']}: {manifest['task_count']} tasks, split={manifest.get('split', 'test')}")
        return 0

    crafted_source = args.crafted_source or benchmark_cfg.get("crafted_source")
    if crafted_source:
        output = _resolve_output_path(args, benchmark_cfg, DEFAULT_INTERIM_DIR / "crafted" / f"{crafted_source}.normalized.jsonl")
        manifest = write_crafted_benchmark(str(crafted_source), output_path=output)
        manifest = _augment_manifest_with_reference_kinds(output, manifest)
        print(f"Wrote crafted benchmark snapshot to {output}")
        print(f"{manifest['benchmark']}: {manifest['task_count']} tasks, families={manifest['family_count']}")
        return 0

    collection_sources = benchmark_cfg.get("collection_sources") or []
    if args.compose or collection_sources or args.collection_input:
        raw_inputs = list(args.collection_input or [])
        raw_inputs.extend(collection_sources)
        resolved_inputs = [
            _collection_input(
                entry,
                benchmark_cfg,
                bool(args.fetch or benchmark_cfg.get("fetch", False)),
                force=bool(args.force),
            )
            for entry in raw_inputs
        ]
        output = _resolve_output_path(args, benchmark_cfg, DEFAULT_INTERIM_DIR / "collections" / f"{benchmark_cfg.get('collection_name', 'collection')}.normalized.jsonl")
        include_reference_kinds = benchmark_cfg.get("include_reference_kinds")
        allowed_reference_kinds = {
            str(item).lower()
            for item in include_reference_kinds or []
            if str(item).strip()
        }
        rows, manifest = compose_benchmark_collection(
            resolved_inputs,
            include_languages=benchmark_cfg.get("include_languages") or benchmark_cfg.get("languages"),
            include_source_groups=benchmark_cfg.get("include_source_groups"),
            include_origin_types=benchmark_cfg.get("include_origin_types"),
            include_difficulties=benchmark_cfg.get("include_difficulties"),
            include_reference_kinds=include_reference_kinds,
            quota_per_language=benchmark_cfg.get("quota_per_language"),
            quota_per_source_group=benchmark_cfg.get("quota_per_source_group"),
            balance_languages=benchmark_cfg.get("language_balance"),
            limit=args.limit if args.limit is not None else benchmark_cfg.get("limit"),
        )
        if allowed_reference_kinds:
            filtered_rows = [row for row in rows if str(row.get("reference_kind", "canonical")).lower() in allowed_reference_kinds]
            if not filtered_rows:
                raise ValueError(
                    f"reference kind filter {sorted(allowed_reference_kinds)} removed all rows from {output}"
                )
            rows = filtered_rows
            manifest = dict(manifest)
            manifest["include_reference_kinds"] = sorted(allowed_reference_kinds)
            manifest["record_count"] = len(rows)
            manifest["observed_languages"] = sorted({row["language"] for row in rows})
            manifest["claimed_languages"] = configured_languages or list(manifest.get("language_counts", {}).keys())
            manifest["validation_supported_languages"] = sorted({row["language"] for row in rows if row.get("validation_supported")})
            manifest["runtime_validation_supported_languages"] = sorted(
                {row["language"] for row in rows if row.get("clean_reference_validation_available")}
            )
            manifest["language_counts"] = _count_by_key(rows, "language")
            manifest["source_group_counts"] = _count_by_key(rows, "source_group")
            manifest["origin_type_counts"] = _count_by_key(rows, "origin_type")
            manifest["difficulty_counts"] = _count_by_key(rows, "difficulty")
            manifest["family_count"] = len({str(row.get("family_id", "")).strip() for row in rows if str(row.get("family_id", "")).strip()})
            manifest["family_counts"] = _count_by_key(rows, "family_id")
            manifest["reference_kind_counts"] = _reference_kind_counts(rows)
            manifest["coverage"] = _coverage_from_rows(rows, configured_languages)
        rows = _sanitize_release_rows(rows)
        manifest.update(
            {
                "collection_name": str(benchmark_cfg.get("collection_name", output.stem)),
                "claimed_languages": configured_languages or list(manifest.get("language_counts", {}).keys()),
                "reference_kind_counts": _reference_kind_counts(rows),
            }
        )
        write_composed_collection(output, rows, manifest)
        print(f"Wrote composed benchmark collection to {output}")
        print(f"records={manifest['record_count']} families={manifest['family_count']}")
        return 0

    source = args.source or Path(benchmark_cfg.get("source", DEFAULT_FIXTURE))
    if not source.exists():
        if args.bootstrap_fixture:
            bootstrap_fixture(source)
        else:
            raise FileNotFoundError(f"Source benchmark {source} does not exist. Use --bootstrap-fixture to create the sample fixture.")

    output = _resolve_output_path(args, benchmark_cfg, DEFAULT_INTERIM_DIR / "benchmark.normalized.jsonl")
    raw_rows = load_json_or_jsonl(source)
    normalized_rows = [normalize_task(row, index, source) for index, row in enumerate(raw_rows)]
    limit = args.limit if args.limit is not None else benchmark_cfg.get("limit")
    if limit is not None:
        normalized_rows = normalized_rows[: int(limit)]
    ensure_dir(output.parent)
    write_jsonl(output, normalized_rows)
    _write_smoke_manifest(output, normalized_rows, configured_languages)
    print(f"Wrote {len(normalized_rows)} normalized tasks to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
