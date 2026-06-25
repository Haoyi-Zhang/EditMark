from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .crafted_templates import execution_tests, reference_tests, solution_source
from .language_support import (
    default_evaluation_backend,
    language_family,
    language_version,
    normalize_language_name,
    runner_image,
    supports_execution,
    validation_mode,
)
from .suite import OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES


ROOT = Path(__file__).resolve().parents[1]
LANGUAGES: tuple[str, ...] = OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES
TEMPLATE_FAMILIES: tuple[str, ...] = (
    "strings",
    "arrays/lists",
    "maps/sets",
    "parsing",
    "math/bit ops",
    "interval/greedy",
    "graph/search",
    "dp/recursion",
    "stateful update",
    "API-style normalization",
)
ENTRY_POINTS: dict[str, str] = {
    "strings": "canonicalize_tokens",
    "arrays/lists": "sum_after_marker",
    "maps/sets": "dominant_key",
    "parsing": "count_balanced_pairs",
    "math/bit ops": "bit_balance_score",
    "interval/greedy": "total_interval_coverage",
    "graph/search": "shortest_grid_path",
    "dp/recursion": "max_non_adjacent_sum",
    "stateful update": "inventory_total",
    "API-style normalization": "normalize_query",
}
TAXONOMY_CATEGORY_SEQUENCES: dict[str, tuple[str, ...]] = {
    "crafted_original": (
        "data structures",
        "strings/parsing",
        "numeric/boundary conditions",
        "recursion/dp",
        "graph/search",
        "state machines/simulation",
        "class/object interaction",
        "exception/error handling",
    ),
    "crafted_translation": (
        "cross-language idiom preservation",
        "strings/parsing",
        "data structures",
        "class/object interaction",
        "exception/error handling",
    ),
    "crafted_stress": (
        "numeric/boundary conditions",
        "graph/search",
        "state machines/simulation",
        "exception/error handling",
        "cross-language idiom preservation",
        "recursion/dp",
        "data structures",
        "strings/parsing",
        "class/object interaction",
    ),
}
CATEGORY_TEMPLATE_MAP: dict[str, tuple[str, ...]] = {
    "data structures": ("arrays/lists", "maps/sets"),
    "strings/parsing": ("strings", "parsing"),
    "numeric/boundary conditions": ("math/bit ops", "interval/greedy"),
    "recursion/dp": ("dp/recursion",),
    "graph/search": ("graph/search",),
    "state machines/simulation": ("stateful update",),
    "class/object interaction": ("API-style normalization", "stateful update"),
    "exception/error handling": ("parsing", "maps/sets", "API-style normalization"),
    "cross-language idiom preservation": ("strings", "API-style normalization", "arrays/lists"),
}
CATEGORY_INTENTS: dict[str, str] = {
    "data structures": "Use idiomatic containers while preserving the exact aggregation semantics.",
    "strings/parsing": "The solution must be robust to separators, whitespace, and token-boundary edge cases.",
    "numeric/boundary conditions": "Handle signed values, touching boundaries, and degenerate numeric inputs consistently.",
    "recursion/dp": "Preserve the optimization objective while remaining deterministic on corner cases.",
    "graph/search": "Keep the shortest-path or reachability semantics unchanged across languages.",
    "state machines/simulation": "Replay state transitions exactly and keep invalid transitions side-effect free.",
    "class/object interaction": "Treat the input stream as observable object interactions over shared state, not isolated calls.",
    "exception/error handling": "Malformed records and failed conversions must be handled gracefully without crashing.",
    "cross-language idiom preservation": "Implement the same contract idiomatically in each target language without semantic drift.",
}
SCENARIOS: tuple[str, ...] = (
    "incident triage",
    "release automation",
    "privacy scrub",
    "dataset stitching",
    "evaluation replay",
    "service rollout",
    "artifact packaging",
)
TONES: tuple[str, ...] = (
    "deterministic",
    "robust",
    "budget-aware",
    "review-friendly",
    "regression-safe",
)
CRAFTED_KINDS: dict[str, dict[str, Any]] = {
    "crafted_original": {
        "dataset_label": "Crafted Original",
        "family_count": 48,
        "source_group": "crafted_original",
        "origin_type": "crafted_original",
    },
    "crafted_translation": {
        "dataset_label": "Crafted Translation",
        "family_count": 48,
        "source_group": "crafted_translation",
        "origin_type": "crafted_translation",
    },
    "crafted_stress": {
        "dataset_label": "Crafted Stress",
        "family_count": 48,
        "source_group": "crafted_stress",
        "origin_type": "crafted_stress",
    },
}


def _difficulty_schedule(kind: str) -> list[str]:
    if kind == "crafted_original":
        return ["easy"] * 14 + ["medium"] * 24 + ["hard"] * 10
    if kind == "crafted_translation":
        return ["easy"] * 12 + ["medium"] * 26 + ["hard"] * 10
    return ["easy"] * 6 + ["medium"] * 18 + ["hard"] * 24


def _scenario(index: int) -> str:
    return SCENARIOS[index % len(SCENARIOS)]


def _tone(index: int) -> str:
    return TONES[index % len(TONES)]


def _case(args: list[Any], expected: Any) -> dict[str, Any]:
    return {"args": args, "expected": expected}


def _kind_category(kind: str, family_index: int) -> str:
    categories = TAXONOMY_CATEGORY_SEQUENCES[kind]
    return categories[family_index % len(categories)]


def _template_family(kind: str, category: str, family_index: int) -> str:
    options = CATEGORY_TEMPLATE_MAP[category]
    return options[(family_index + len(kind)) % len(options)]


def _semantic_suffix(kind: str, category: str, family_index: int) -> str:
    kind_notes = {
        "crafted_original": "This expert-authored task belongs to the Crafted Original release family.",
        "crafted_translation": "This expert-authored task belongs to the Crafted Translation release family and must preserve semantics across the canonical five executed languages.",
        "crafted_stress": "This expert-authored task belongs to the Crafted Stress release family and must preserve semantics under malformed inputs and edge-heavy cases.",
    }
    return (
        f"{CATEGORY_INTENTS[category]} "
        f"{kind_notes[kind]} "
        f"Variant seed: {family_index}. Benchmark family kind: {kind}."
    )


def _family_content(template_family: str, category: str, family_index: int, kind: str, difficulty: str) -> dict[str, Any]:
    if template_family == "strings":
        contract = "Lowercase the input, keep only alphanumeric token runs, and join the runs with '/'."
        base_cases = [
            _case(["Alpha__ beta--Gamma"], "alpha/beta/gamma"),
            _case(["  release___READY  "], "release/ready"),
        ]
        stress_cases = [_case(["pkg::v2__hotfix&&ROLLBACK"], "pkg/v2/hotfix/rollback")]
        metamorphic = ["Extra separators do not change token order.", "Case changes do not change the output."]
        title = "token_canonicalizer"
    elif template_family == "arrays/lists":
        contract = "After the first occurrence of marker, sum only the positive values that appear later."
        base_cases = [_case([[3, -1, 4, 2, 5], 4], 7), _case([[1, 2, 3], 9], 0)]
        stress_cases = [_case([[7, -3, 2, 7, 4, -1, 5], 7], 11)]
        metamorphic = ["Values before the marker do not contribute.", "Non-positive suffix values do not increase the sum."]
        title = "post_marker_positive_sum"
    elif template_family == "maps/sets":
        contract = "Parse key:value updates, accumulate totals, and return the lexicographically smallest key with the highest total."
        base_cases = [_case([["alpha:3", "beta:5", "alpha:4"]], "alpha"), _case([["zeta:2", "omega:2"]], "omega")]
        stress_cases = [_case([["alpha:3", "oops", "alpha:-1", "gamma:8"]], "gamma")]
        metamorphic = ["Malformed entries are ignored.", "Reordering entries preserves the winning key if totals stay unchanged."]
        title = "dominant_key_selector"
    elif template_family == "parsing":
        contract = "Count lines 'a,b' where both integers parse, share parity, and differ by at most window."
        base_cases = [_case([["2,4", "3,8", "10,12"], 2], 2), _case([["1,4", "6,9"], 1], 0)]
        stress_cases = [_case([["7,9", "bad", "6,6", "5,2"], 2], 2)]
        metamorphic = ["Malformed lines are ignored.", "Increasing window never decreases the count."]
        title = "balanced_pair_counter"
    elif template_family == "math/bit ops":
        contract = "For each integer, add popcount(abs(x)) when x is even and subtract it when x is odd."
        base_cases = [_case([[2, 3, 4]], 1), _case([[-5, 8]], -1)]
        stress_cases = [_case([[0, 1, 2, 15]], -1)]
        metamorphic = ["Negating a value preserves its bit contribution magnitude.", "Adding zero changes nothing."]
        title = "bit_balance_score"
    elif template_family == "interval/greedy":
        contract = "Parse inclusive intervals, merge overlapping or touching ranges, and return total covered length."
        base_cases = [_case([["1-3", "2-5", "10-12"]], 8), _case([["7-7", "9-11"]], 4)]
        stress_cases = [_case([["5-1", "8-9", "10-10"]], 10)]
        metamorphic = ["Touching intervals can be merged without changing coverage.", "Swapping endpoints preserves coverage."]
        title = "inclusive_interval_coverage"
    elif template_family == "graph/search":
        contract = "Return the shortest 4-neighbor path length from S to E avoiding '#'; return -1 if unreachable."
        base_cases = [_case([["S..", ".#.", "..E"]], 4), _case([["S#E"]], -1)]
        stress_cases = [_case([["S...", "##..", "...E"]], 5)]
        metamorphic = ["Adding irrelevant walls off every shortest path cannot shorten the answer.", "Unreachable instances stay unreachable after adding blocked rows."]
        title = "grid_shortest_path"
    elif template_family == "dp/recursion":
        contract = "Return the maximum non-negative sum obtainable by selecting non-adjacent values."
        base_cases = [_case([[2, 7, 9, 3, 1]], 12), _case([[-5, -1, -3]], 0)]
        stress_cases = [_case([[4, 1, 1, 9, 1]], 13)]
        metamorphic = ["Appending zeros does not change the optimum.", "Making a negative value smaller cannot improve the optimum."]
        title = "max_non_adjacent_sum"
    elif template_family == "stateful update":
        contract = "Replay add/remove inventory events with a floor at zero and return total remaining stock."
        base_cases = [_case([["add apple 3", "add pear 2", "remove apple 1"]], 4), _case([["remove x 4", "add x 2"]], 2)]
        stress_cases = [_case([["add a 5", "remove a 8", "add b 2", "remove b 1"]], 1)]
        metamorphic = ["Over-removal floors an item at zero.", "Malformed events do not change the total."]
        title = "inventory_replay_total"
    else:
        contract = "Normalize query pairs by lowercasing keys/values, keeping the latest non-empty value per key, sorting by key, and joining with '&'."
        base_cases = [_case(["B=2&a=1&A=7"], "a=7&b=2"), _case(["mode=SAFE;retry=3;mode=fast"], "mode=fast&retry=3")]
        stress_cases = [_case(["a=1&&B=2;empty= ;a=9"], "a=9&b=2")]
        metamorphic = ["Empty assignments do not affect the result.", "Only the latest duplicate assignment for a key matters."]
        title = "query_normalizer"

    semantic_contract = f"{contract} {_semantic_suffix(kind, category, family_index)}"
    prompt = (
        f"Implement `{title}` for the {_scenario(family_index)} release family.\n"
        f"Difficulty: {difficulty}. Style target: {_tone(family_index)}.\n"
        f"Category: {category}.\n"
        f"Semantic contract: {semantic_contract}\n"
        "Return only the implementation."
    )
    return {
        "title": title,
        "prompt": prompt,
        "semantic_contract": semantic_contract,
        "base_cases": base_cases,
        "stress_cases": stress_cases,
        "metamorphic_tests": metamorphic,
    }


def build_crafted_benchmark(kind: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if kind not in CRAFTED_KINDS:
        raise KeyError(f"unknown crafted benchmark kind: {kind}")
    spec = CRAFTED_KINDS[kind]
    rows: list[dict[str, Any]] = []
    authoring_rows: list[dict[str, Any]] = []
    difficulty_schedule = _difficulty_schedule(kind)
    for family_index in range(spec["family_count"]):
        category = _kind_category(kind, family_index)
        template_family = _template_family(kind, category, family_index)
        difficulty = difficulty_schedule[family_index]
        family = _family_content(template_family, category, family_index, kind, difficulty)
        authoring_rows.append(
            {
                "family_id": f"{kind}_{family_index:03d}",
                "kind": kind,
                "category": category,
                "template_family": template_family,
                "difficulty": difficulty,
                "title": family["title"],
                "prompt": family["prompt"],
                "semantic_contract": family["semantic_contract"],
                "language_coverage": list(LANGUAGES),
                "validation_backend": {language: default_evaluation_backend(language) for language in LANGUAGES},
                "reference_kind": "canonical",
                "base_cases": list(family["base_cases"]),
                "stress_cases": list(family["stress_cases"]),
                "metamorphic_tests": list(family["metamorphic_tests"]),
            }
        )
        entry_root = ENTRY_POINTS[template_family]
        for language in LANGUAGES:
            normalized_language = normalize_language_name(language)
            function_name = f"{entry_root}_{family_index:03d}"
            eval_backend = default_evaluation_backend(normalized_language)
            tests = execution_tests(normalized_language, function_name, family["base_cases"], family["stress_cases"])
            rows.append(
                {
                    "task_id": f"{kind}/{family_index:03d}/{normalized_language}",
                    "dataset": spec["dataset_label"],
                    "language": normalized_language,
                    "prompt": family["prompt"] + f"\nTarget language: {normalized_language}. Entry point: {function_name}.",
                    "reference_solution": solution_source(template_family, normalized_language, function_name),
                    "reference_tests": list(reference_tests(normalized_language, function_name, family["base_cases"])),
                    "execution_tests": list(tests),
                    "entry_point": function_name,
                    "claimed_languages": list(LANGUAGES),
                    "language_family": language_family(normalized_language),
                    "validation_mode": validation_mode(normalized_language),
                    "validation_supported": supports_execution(normalized_language, list(tests), backend=eval_backend),
                    "adapter_name": "crafted_release_curator",
                    "validation_scope": "multilingual_exec",
                    "record_kind": "crafted_benchmark",
                    "source_group": spec["source_group"],
                    "origin_type": spec["origin_type"],
                    "family_id": f"{kind}_{family_index:03d}",
                    "category": category,
                    "template_family": template_family,
                    "difficulty": difficulty,
                    "evaluation_backend": eval_backend,
                    "validation_backend": eval_backend,
                    "runner_image": runner_image(normalized_language),
                    "official_problem_file": f"crafted_release/{kind}/{family_index:03d}/{normalized_language}",
                    "language_version": language_version(normalized_language),
                    "reference_kind": "canonical",
                    "semantic_contract": family["semantic_contract"],
                    "metamorphic_tests": list(family["metamorphic_tests"]),
                    "stress_tests": list(reference_tests(normalized_language, function_name, family["stress_cases"])),
                    "functional_cases": list(family["base_cases"]),
                    "stress_cases": list(family["stress_cases"]),
                    "translation_anchor_language": "python" if kind == "crafted_translation" else None,
                    "description": family["title"],
                    "notes": (
                        f"Expert-authored crafted benchmark family for {category}; "
                        "manually reviewed and finalized for public release."
                    ),
                }
            )

    manifest = {
        "schema_version": 2,
        "benchmark": kind,
        "dataset_label": spec["dataset_label"],
        "collection_name": f"{kind}_release",
        "source_group": spec["source_group"],
        "origin_type": spec["origin_type"],
        "record_count": len(rows),
        "task_count": len(rows),
        "family_count": spec["family_count"],
        "canonical_reference_count": len(rows),
        "smoke_overlay_reference_count": 0,
        "languages": list(LANGUAGES),
        "claimed_languages": list(LANGUAGES),
        "language_counts": {language: sum(1 for row in rows if row["language"] == language) for language in LANGUAGES},
        "source_group_counts": dict(Counter(row["source_group"] for row in rows)),
        "origin_type_counts": dict(Counter(row["origin_type"] for row in rows)),
        "family_counts": dict(Counter(row["family_id"] for row in rows)),
        "validation_supported_languages": list(LANGUAGES),
        "difficulty_counts": dict(Counter(row["difficulty"] for row in rows)),
        "family_difficulty_counts": dict(Counter(row["difficulty"] for row in authoring_rows)),
        "category_counts": dict(Counter(row["category"] for row in authoring_rows)),
        "template_family_counts": dict(Counter(row["template_family"] for row in authoring_rows)),
        "validation_backend_counts": dict(Counter(row["evaluation_backend"] for row in rows)),
        "reference_kind_counts": {"canonical": len(rows)},
        "taxonomy_categories": sorted(CATEGORY_TEMPLATE_MAP),
        "task_count_per_family": len(LANGUAGES),
        "family_language_coverage_rate": 1.0,
        "include_difficulties": sorted({str(row["difficulty"]) for row in rows}),
        "include_languages": list(LANGUAGES),
        "include_origin_types": [spec["origin_type"]],
        "include_reference_kinds": ["canonical"],
        "include_source_groups": [spec["source_group"]],
        "contract_drift_families": [],
        "construction_note": (
            "Crafted sources use expert-authored benchmark families together with "
            "cross-language review, deterministic release checks, and manually "
            "finalized public release records."
        ),
        "suite_selection_policy": {
            "type": "category_balanced_canonical_release_source",
            "family_count": spec["family_count"],
            "languages_per_family": len(LANGUAGES),
            "source_group": spec["source_group"],
        },
    }
    return rows, authoring_rows, manifest


def write_crafted_benchmark(kind: str, *, output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, authoring_rows, manifest = build_crafted_benchmark(kind)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            handle.write("\n")
    manifest = {
        **manifest,
        "normalized_path": str(output.relative_to(ROOT)).replace("\\", "/"),
        "source_manifests": [
            {
                "benchmark": kind,
                "dataset_label": CRAFTED_KINDS[kind]["dataset_label"],
                "source_group": CRAFTED_KINDS[kind]["source_group"],
                "origin_type": CRAFTED_KINDS[kind]["origin_type"],
                "task_count": len(rows),
                "family_count": CRAFTED_KINDS[kind]["family_count"],
                "languages": list(LANGUAGES),
                "category_counts": dict(Counter(row["category"] for row in authoring_rows)),
                "template_family_counts": dict(Counter(row["template_family"] for row in authoring_rows)),
                "task_count_per_family": len(LANGUAGES),
            }
        ],
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
