from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .attacks.registry import available_attacks
from .baselines import runtime_family_names
from .models import ExperimentConfig
from .providers import summarize_provider_configuration, validate_provider_configuration


_FALLBACK_KNOWN_WATERMARKS = tuple(
    dict.fromkeys(
        (
            *runtime_family_names(),
        )
    )
)


@dataclass(frozen=True, slots=True)
class ConfigSource:
    path: Path | None
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path | None = None) -> ConfigSource:
    if path is None:
        return ConfigSource(path=None, raw={})
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError("configuration file must contain a mapping")
    return ConfigSource(path=path, raw=data)


def _count_records(path: str | Path | None) -> int:
    if path is None:
        return 0
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return 0
    try:
        text = candidate.read_text(encoding="utf-8")
    except Exception:
        return 0
    if candidate.suffix.lower() == ".jsonl":
        return sum(1 for line in text.splitlines() if line.strip())
    try:
        payload = yaml.safe_load(text)
    except Exception:
        return 0
    if isinstance(payload, list):
        return len(payload)
    return 0


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _fallback_known_watermarks() -> set[str]:
    return {str(name).strip().lower() for name in _FALLBACK_KNOWN_WATERMARKS}


def _known_watermarks_payload() -> tuple[set[str], str | None]:
    try:
        from .watermarks.registry import all_watermarks as _all_watermarks
    except Exception as exc:
        return _fallback_known_watermarks(), (
            "watermark registry import failed; using fallback watermark roster "
            f"({exc.__class__.__name__}: {exc})"
        )
    try:
        return {str(name).strip().lower() for name in _all_watermarks()}, None
    except Exception as exc:
        return _fallback_known_watermarks(), (
            "watermark registry enumeration failed; using fallback watermark roster "
            f"({exc.__class__.__name__}: {exc})"
        )


def _deep_merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def merge_config_source(source: ConfigSource | dict[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if isinstance(source, ConfigSource):
        raw.update(source.raw)
    elif isinstance(source, dict):
        raw.update(source)
    return _deep_merge_dicts(raw, dict(overrides))


def build_experiment_config(source: ConfigSource | dict[str, Any] | None = None, **overrides: Any) -> ExperimentConfig:
    raw = merge_config_source(source, **overrides)

    metadata_section = raw.get("metadata") or {}
    project = raw.get("project") or metadata_section.get("project") or {}
    benchmark = raw.get("benchmark") or metadata_section.get("benchmark") or {}
    watermark = raw.get("watermark") or {}
    provider = raw.get("provider") or metadata_section.get("provider") or {}
    attack_section = raw.get("attacks") or {}
    paths = raw.get("paths") or metadata_section.get("paths") or {}
    reporting = raw.get("reporting") or metadata_section.get("reporting") or {}

    attacks = raw.get("attacks", ["comment_strip", "identifier_rename", "whitespace_normalize"])
    if isinstance(attack_section, dict) and attack_section.get("include") is not None:
        attacks = attack_section.get("include", attacks)
    if isinstance(attacks, list):
        attacks = tuple(str(item).lower() for item in attacks)
    elif isinstance(attacks, tuple):
        attacks = tuple(str(item).lower() for item in attacks)
    else:
        attacks = (str(attacks).lower(),)

    attack_parameters = raw.get("attack_parameters") or {}
    corpus_parameters = raw.get("corpus_parameters") or {}

    if isinstance(watermark, dict):
        watermark_name = watermark.get("scheme", raw.get("watermark_name", "stone_runtime"))
        watermark_secret = watermark.get("secret", raw.get("watermark_secret", "redacted"))
        watermark_payload = watermark.get("payload", raw.get("watermark_payload", "wm"))
        watermark_strength = watermark.get("strength", raw.get("watermark_strength", 1.0))
    else:
        watermark_name = raw.get("watermark_name", "stone_runtime")
        watermark_secret = raw.get("watermark_secret", "redacted")
        watermark_payload = raw.get("watermark_payload", "wm")
        watermark_strength = raw.get("watermark_strength", 1.0)

    if isinstance(project, dict):
        seed = project.get("seed", raw.get("seed", 7))
    else:
        seed = raw.get("seed", 7)

    if isinstance(benchmark, dict):
        if "limit" in benchmark:
            benchmark_limit = benchmark.get("limit")
        elif "corpus_size" in raw:
            benchmark_limit = raw.get("corpus_size")
        else:
            benchmark_limit = None
        language = benchmark.get("language", raw.get("language", "python"))
        prepared_benchmark = paths.get("prepared_benchmark") or benchmark.get("prepared_output") or benchmark.get("source")
        limit_mode = "sample"
        if benchmark_limit is None:
            limit_mode = "full"
            corpus_size = _count_records(prepared_benchmark) or _count_records(benchmark.get("source")) or None
        else:
            corpus_size = int(benchmark_limit)
        benchmark_parameters = {str(key): value for key, value in benchmark.items()}
        benchmark_parameters.update(
            {
                "source": benchmark.get("source"),
                "prepared_output": benchmark.get("prepared_output"),
                "prepared_benchmark": prepared_benchmark,
                "benchmark_path": prepared_benchmark,
                "limit_mode": limit_mode,
                "dataset_label": benchmark.get("dataset_label"),
                "public_source": benchmark.get("public_source"),
                "split": benchmark.get("split"),
                "source_url": benchmark.get("source_url"),
                "source_revision": benchmark.get("source_revision"),
                "source_sha256": benchmark.get("source_sha256"),
                "license_note": benchmark.get("license_note"),
                "validation_scope": benchmark.get("validation_scope"),
                "stress_suite": benchmark.get("stress_suite"),
                "include_reference_kinds": (
                    list(benchmark.get("include_reference_kinds", []))
                    if isinstance(benchmark.get("include_reference_kinds"), list)
                    else benchmark.get("include_reference_kinds")
                ),
                "languages": list(benchmark.get("languages", [])) if isinstance(benchmark.get("languages"), list) else [],
            }
        )
    else:
        corpus_size = raw.get("corpus_size") if "corpus_size" in raw else None
        language = raw.get("language", "python")
        benchmark_parameters = {}

    if isinstance(provider, dict):
        provider_mode = provider.get("mode", raw.get("provider_mode", "offline_mock"))
        provider_parameters = provider.get("parameters", raw.get("provider_parameters", {}))
        if not isinstance(provider_parameters, dict):
            provider_parameters = {}
    else:
        provider_mode = raw.get("provider_mode", "offline_mock")
        provider_parameters = raw.get("provider_parameters", {})
        if not isinstance(provider_parameters, dict):
            provider_parameters = {}

    validation_scope = raw.get("validation_scope")
    if validation_scope is None and isinstance(benchmark, dict):
        validation_scope = benchmark.get("validation_scope")
    if validation_scope is None:
        validation_scope = "python_first"

    return ExperimentConfig(
        seed=int(seed),
        corpus_size=None if corpus_size is None else int(corpus_size),
        language=str(language),
        watermark_name=str(watermark_name).lower(),
        watermark_secret=str(watermark_secret),
        watermark_payload=str(watermark_payload),
        watermark_strength=float(watermark_strength),
        attacks=tuple(attacks),
        attack_parameters={str(k): dict(v) for k, v in attack_parameters.items()},
        corpus_parameters={**benchmark_parameters, **dict(corpus_parameters)},
        provider_mode=str(provider_mode).lower(),
        provider_parameters={str(k): v for k, v in dict(provider_parameters).items()},
        validation_scope=str(validation_scope),
        output_path=raw.get("output_path") or paths.get("output_path"),
        metadata={
            **dict(raw.get("metadata", {})),
            "project": dict(project) if isinstance(project, dict) else {},
            "benchmark": dict(benchmark) if isinstance(benchmark, dict) else {},
            "watermark": dict(watermark) if isinstance(watermark, dict) else {},
            "provider": dict(provider) if isinstance(provider, dict) else {},
            "provider_summary": summarize_provider_configuration(str(provider_mode).lower(), provider_parameters),
            "paths": dict(paths) if isinstance(paths, dict) else {},
            "reporting": dict(reporting) if isinstance(reporting, dict) else {},
        },
    )


def validate_experiment_config(config: ExperimentConfig) -> list[str]:
    issues: list[str] = []
    known_watermarks, registry_issue = _known_watermarks_payload()
    known_attacks = set(available_attacks())
    from .providers import available_providers
    from .baselines.stone_family.common import stone_family_checkout_available, validate_checkout as validate_runtime_checkout
    from .baselines.stone_family.official_runtime import runtime_compatibility_profile_name, runtime_compatibility_profiles

    runtime_watermarks = set(runtime_family_names())

    known_providers = set(available_providers())
    if registry_issue:
        issues.append(registry_issue)
    if config.watermark_name not in known_watermarks:
        issues.append(
            f"unknown watermark scheme '{config.watermark_name}'"
            f" (available: {sorted(known_watermarks)})"
        )
    unknown_attacks = [name for name in config.attacks if name not in known_attacks]
    if unknown_attacks:
        issues.append(
            f"unknown attack names {sorted(set(unknown_attacks))}"
            f" (available: {sorted(known_attacks)})"
        )
    provider_issues = validate_provider_configuration(config.provider_mode, config.provider_parameters)
    if config.provider_mode not in known_providers and config.provider_mode != "openai_compatible":
        issues.append(
            f"unknown provider mode '{config.provider_mode}'"
            f" (available: {sorted(known_providers)})"
        )
    issues.extend(provider_issues)
    if config.watermark_name in runtime_watermarks:
        watermark_metadata = dict(config.metadata.get("watermark", {})) if isinstance(config.metadata, dict) else {}
        model_name = str(watermark_metadata.get("model_name", "")).strip()
        if not model_name:
            issues.append(f"runtime watermark '{config.watermark_name}' requires watermark.model_name")
        compatibility_profile = str(watermark_metadata.get("compatibility_profile", "auto")).strip().lower()
        allowed_profiles = {"auto", *runtime_compatibility_profiles()}
        if compatibility_profile and compatibility_profile not in allowed_profiles:
            issues.append(
                f"runtime watermark '{config.watermark_name}' uses unsupported watermark.compatibility_profile "
                f"'{compatibility_profile}' (expected one of {sorted(allowed_profiles)})"
            )
        if model_name:
            inferred_profile = runtime_compatibility_profile_name(model_name)
            if compatibility_profile not in {"", "auto", inferred_profile}:
                issues.append(
                    f"runtime watermark '{config.watermark_name}' uses watermark.compatibility_profile "
                    f"'{compatibility_profile}' but model '{model_name}' resolves to '{inferred_profile}'"
                )
            try:
                if int(watermark_metadata.get("max_new_tokens", 0) or 0) <= 0:
                    issues.append(
                        f"runtime watermark '{config.watermark_name}' requires watermark.max_new_tokens to be a positive integer"
                    )
            except Exception:
                issues.append(
                    f"runtime watermark '{config.watermark_name}' requires watermark.max_new_tokens to be a positive integer"
                )
        checkout_issues = validate_runtime_checkout(config.watermark_name)
        issues.extend(checkout_issues)
        if not checkout_issues and not stone_family_checkout_available(config.watermark_name):
            issues.append(
                f"runtime watermark '{config.watermark_name}' requires a valid official upstream checkout"
            )
    benchmark_path = config.corpus_parameters.get("prepared_benchmark") or config.corpus_parameters.get("benchmark_path")
    if benchmark_path and not Path(str(benchmark_path)).exists():
        issues.append(f"missing benchmark corpus '{benchmark_path}'")
    return issues


def dump_config(config: ExperimentConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.as_dict(), sort_keys=False), encoding="utf-8")
    return path
