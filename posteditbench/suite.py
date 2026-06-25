from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SuiteModelSpec:
    name: str
    revision: str
    slug: str
    family: str


CANONICAL_SUITE_MODELS: tuple[SuiteModelSpec, ...] = (
    SuiteModelSpec(
        name="Qwen/Qwen2.5-Coder-14B-Instruct",
        revision="aedcc2d42b622764e023cf882b6652e646b95671",
        slug="qwen25_14b",
        family="qwen25",
    ),
    SuiteModelSpec(
        name="Qwen/Qwen2.5-Coder-7B-Instruct",
        revision="c03e6d358207e414f1eca0bb1891e29f1db0e242",
        slug="qwen25_7b",
        family="qwen25",
    ),
    SuiteModelSpec(
        name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        revision="2e1fd397ee46e1388853d2af2c993145b0f1098a",
        slug="qwen25_1p5b",
        family="qwen25",
    ),
    SuiteModelSpec(
        name="bigcode/starcoder2-7b",
        revision="bb9afde76d7945da5745592525db122d4d729eb1",
        slug="starcoder2_7b",
        family="starcoder2",
    ),
    SuiteModelSpec(
        name="deepseek-ai/deepseek-coder-6.7b-instruct",
        revision="e5d64addd26a6a1db0f9b863abf6ee3141936807",
        slug="deepseek_coder_6p7b",
        family="deepseek_coder",
    ),
)
SUITE_MODEL_REVISIONS: dict[str, str] = {spec.name: spec.revision for spec in CANONICAL_SUITE_MODELS}
_MODEL_BY_NAME: dict[str, SuiteModelSpec] = {spec.name: spec for spec in CANONICAL_SUITE_MODELS}


OFFICIAL_BASELINE_ROSTER: tuple[str, ...] = (
    "stone_runtime",
    "sweet_runtime",
    "ewd_runtime",
    "kgw_runtime",
)
OFFICIAL_RUNTIME_BASELINES: tuple[str, ...] = OFFICIAL_BASELINE_ROSTER
OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES: tuple[str, ...] = (
    "python",
    "cpp",
    "java",
    "javascript",
    "go",
)

PAPER_MODEL_ROSTER: tuple[str, ...] = (
    *(spec.name for spec in CANONICAL_SUITE_MODELS),
)
SUITE_MODEL_ROSTER: tuple[str, ...] = PAPER_MODEL_ROSTER

# Active release-suite sizing policy:
# - keep HumanEval+ complete as the public Python anchor
# - keep MBPP+ complete as the public Python stress-oriented benchmark
# - keep multilingual sources on deterministic five-language balanced slices
# - keep the crafted sources as the single canonical executed five-language release version
ACTIVE_SUITE_LIMITS: dict[str, int] = {
    "humaneval_plus": 164,
    "mbpp_plus": 378,
    "humaneval_x": 200,
    "mbxp_5lang": 200,
    "crafted_original": 240,
    "crafted_translation": 240,
    "crafted_stress": 240,
}

HEAVY_PRECHECK_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
SUITE_MODEL_SLUGS: dict[str, str] = {
    spec.name: spec.slug for spec in CANONICAL_SUITE_MODELS
}
SUITE_MODEL_FAMILIES: dict[str, str] = {
    spec.name: spec.family for spec in CANONICAL_SUITE_MODELS
}
SUITE_MODEL_FAMILY_ORDER: tuple[str, ...] = (
    "qwen25",
    "starcoder2",
    "deepseek_coder",
)


@dataclass(frozen=True, slots=True)
class SuiteSourceSpec:
    slug: str
    dataset_label: str
    source_group: str
    prepared_benchmark: str
    prepared_output: str
    validation_scope: str
    languages: tuple[str, ...]
    base_template: str
    collection_name: str = ""
    public_source: str = ""
    include_reference_kinds: tuple[str, ...] = ("canonical",)
    full_limit: int = 0
    stage_a_limit: int = 0
    stage_b_limit: int = 0
    aggregate_score: bool = True


SUITE_INVENTORY_SOURCES: tuple[SuiteSourceSpec, ...] = (
    SuiteSourceSpec(
        slug="humaneval_plus",
        dataset_label="HumanEval+",
        source_group="public_humaneval_plus",
        prepared_benchmark="data/release/sources/suite_humaneval_plus_release.normalized.jsonl",
        prepared_output="data/release/sources/suite_humaneval_plus_release.normalized.jsonl",
        validation_scope="python_first",
        languages=("python",),
        base_template="humaneval_plus",
        collection_name="suite_humaneval_plus_release",
        public_source="humaneval_plus",
        full_limit=164,
        stage_a_limit=12,
        stage_b_limit=2,
    ),
    SuiteSourceSpec(
        slug="mbpp_plus",
        dataset_label="MBPP+",
        source_group="public_mbpp_plus",
        prepared_benchmark="data/release/sources/suite_mbpp_plus_release.normalized.jsonl",
        prepared_output="data/release/sources/suite_mbpp_plus_release.normalized.jsonl",
        validation_scope="python_first",
        languages=("python",),
        base_template="mbpp_plus",
        collection_name="suite_mbpp_plus_release",
        public_source="mbpp_plus",
        full_limit=ACTIVE_SUITE_LIMITS["mbpp_plus"],
        stage_a_limit=12,
        stage_b_limit=2,
    ),
    SuiteSourceSpec(
        slug="humaneval_x",
        dataset_label="HumanEval-X (5-language balanced slice)",
        source_group="public_humaneval_x",
        prepared_benchmark="data/release/sources/suite_humanevalx_release.normalized.jsonl",
        prepared_output="data/release/sources/suite_humanevalx_release.normalized.jsonl",
        validation_scope="multilingual_exec",
        languages=OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES,
        base_template="humaneval_plus",
        collection_name="suite_humanevalx_release",
        full_limit=ACTIVE_SUITE_LIMITS["humaneval_x"],
        stage_a_limit=15,
        stage_b_limit=2,
    ),
    SuiteSourceSpec(
        slug="mbxp_5lang",
        dataset_label="MBXP-5lang (5-language balanced slice)",
        source_group="public_mbxp_5lang",
        prepared_benchmark="data/release/sources/suite_mbxp_release.normalized.jsonl",
        prepared_output="data/release/sources/suite_mbxp_release.normalized.jsonl",
        validation_scope="multilingual_exec",
        languages=OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES,
        base_template="humaneval_plus",
        collection_name="suite_mbxp_release",
        include_reference_kinds=("canonical", "smoke_overlay"),
        full_limit=ACTIVE_SUITE_LIMITS["mbxp_5lang"],
        stage_a_limit=15,
        stage_b_limit=2,
    ),
    SuiteSourceSpec(
        slug="crafted_original",
        dataset_label="Crafted Original",
        source_group="crafted_original",
        prepared_benchmark="data/release/sources/crafted_original_release.normalized.jsonl",
        prepared_output="data/release/sources/crafted_original_release.normalized.jsonl",
        validation_scope="multilingual_exec",
        languages=OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES,
        base_template="humaneval_plus",
        collection_name="crafted_original_release",
        include_reference_kinds=(),
        full_limit=ACTIVE_SUITE_LIMITS["crafted_original"],
        stage_a_limit=15,
        stage_b_limit=2,
    ),
    SuiteSourceSpec(
        slug="crafted_translation",
        dataset_label="Crafted Translation",
        source_group="crafted_translation",
        prepared_benchmark="data/release/sources/crafted_translation_release.normalized.jsonl",
        prepared_output="data/release/sources/crafted_translation_release.normalized.jsonl",
        validation_scope="multilingual_exec",
        languages=OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES,
        base_template="humaneval_plus",
        collection_name="crafted_translation_release",
        include_reference_kinds=(),
        full_limit=ACTIVE_SUITE_LIMITS["crafted_translation"],
        stage_a_limit=15,
        stage_b_limit=2,
    ),
    SuiteSourceSpec(
        slug="crafted_stress",
        dataset_label="Crafted Stress",
        source_group="crafted_stress",
        prepared_benchmark="data/release/sources/crafted_stress_release.normalized.jsonl",
        prepared_output="data/release/sources/crafted_stress_release.normalized.jsonl",
        validation_scope="multilingual_exec",
        languages=OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES,
        base_template="humaneval_plus",
        collection_name="crafted_stress_release",
        full_limit=ACTIVE_SUITE_LIMITS["crafted_stress"],
        stage_a_limit=15,
        stage_b_limit=2,
    ),
)

SUITE_AGGREGATE_SOURCES: tuple[SuiteSourceSpec, ...] = tuple(
    source for source in SUITE_INVENTORY_SOURCES if source.aggregate_score
)
SUITE_MULTILINGUAL_AGGREGATE_SOURCES: tuple[SuiteSourceSpec, ...] = tuple(
    source for source in SUITE_AGGREGATE_SOURCES if len(tuple(source.languages)) > 1
)
SUITE_ATOMIC_SOURCE_ORDER: tuple[str, ...] = tuple(source.slug for source in SUITE_AGGREGATE_SOURCES)
SUITE_ATOMIC_SOURCE_LABELS: dict[str, str] = {
    source.slug: source.dataset_label for source in SUITE_AGGREGATE_SOURCES
}

SUITE_INVENTORY_SOURCE_GROUPS: tuple[str, ...] = tuple(source.source_group for source in SUITE_INVENTORY_SOURCES)
SUITE_AGGREGATE_SOURCE_GROUPS: tuple[str, ...] = tuple(source.source_group for source in SUITE_AGGREGATE_SOURCES)
SUITE_MULTILINGUAL_AGGREGATE_SOURCE_GROUPS: tuple[str, ...] = tuple(
    source.source_group for source in SUITE_MULTILINGUAL_AGGREGATE_SOURCES
)
SUITE_INVENTORY_DATASETS: tuple[str, ...] = tuple(source.dataset_label for source in SUITE_INVENTORY_SOURCES)
SUITE_AGGREGATE_DATASETS: tuple[str, ...] = tuple(source.dataset_label for source in SUITE_AGGREGATE_SOURCES)

_SOURCE_BY_GROUP = {source.source_group: source for source in SUITE_INVENTORY_SOURCES}
_SOURCE_BY_SLUG = {source.slug: source for source in SUITE_INVENTORY_SOURCES}


def normalize_source_group(value: str | None) -> str:
    return str(value or "").strip().lower()


def suite_model_spec(model_name: str | None) -> SuiteModelSpec | None:
    return _MODEL_BY_NAME.get(str(model_name or "").strip())


def suite_model_revision(model_name: str | None) -> str:
    spec = suite_model_spec(model_name)
    return spec.revision if spec is not None else ""


def resolve_model_revision(
    model_name: str | None,
    requested_revision: str | None = None,
    *,
    require_canonical: bool = False,
) -> str:
    explicit = str(requested_revision or "").strip()
    spec = suite_model_spec(model_name)
    if spec is not None:
        if explicit and explicit != spec.revision:
            normalized = str(model_name or "").strip()
            raise ValueError(
                f"model '{normalized}' is pinned to revision '{spec.revision}', not '{explicit}'"
            )
        return spec.revision
    if explicit:
        return explicit
    if require_canonical:
        normalized = str(model_name or "").strip()
        raise ValueError(
            f"model '{normalized}' is not part of the canonical pinned roster; provide an explicit revision"
        )
    return ""


def require_pinned_model_revision(model_name: str | None, requested_revision: str | None = None) -> str:
    return resolve_model_revision(model_name, requested_revision, require_canonical=True)


def suite_source_by_group(source_group: str | None) -> SuiteSourceSpec | None:
    return _SOURCE_BY_GROUP.get(normalize_source_group(source_group))


def suite_source_by_slug(slug: str | None) -> SuiteSourceSpec | None:
    return _SOURCE_BY_SLUG.get(str(slug or "").strip())


def is_suite_inventory_source_group(source_group: str | None) -> bool:
    return suite_source_by_group(source_group) is not None


def is_suite_aggregate_source_group(source_group: str | None) -> bool:
    spec = suite_source_by_group(source_group)
    return bool(spec and spec.aggregate_score)


def suite_benchmark_roster() -> list[str]:
    return [source.dataset_label for source in SUITE_AGGREGATE_SOURCES]


def suite_experiment_languages(source: SuiteSourceSpec) -> tuple[str, ...]:
    if source.validation_scope != "multilingual_exec":
        return source.languages
    return tuple(
        language
        for language in source.languages
        if language in OFFICIAL_RUNTIME_COMMON_MULTILINGUAL_LANGUAGES
    )


def model_family_for_label(model_label: str | None) -> str:
    return SUITE_MODEL_FAMILIES.get(str(model_label or "").strip(), "unspecified")


def is_multilingual_aggregate_source_group(source_group: str | None) -> bool:
    return normalize_source_group(source_group) in {
        normalize_source_group(group) for group in SUITE_MULTILINGUAL_AGGREGATE_SOURCE_GROUPS
    }
