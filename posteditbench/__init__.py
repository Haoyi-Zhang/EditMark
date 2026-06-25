from __future__ import annotations

from .config import build_experiment_config, dump_config, load_config, validate_experiment_config
from .benchmarks import build_benchmark_manifest, load_benchmark_corpus
from .providers import available_providers, build_provider
from .public_benchmarks import available_public_sources, prepare_public_benchmark
from .models import (
    AttackOutcome,
    BenchmarkExample,
    BenchmarkReport,
    BenchmarkRow,
    BudgetCurvePoint,
    DetectionResult,
    ExperimentConfig,
    WatermarkSpec,
    WatermarkedSnippet,
)

__all__ = [
    "AttackOutcome",
    "BenchmarkExample",
    "BenchmarkReport",
    "BenchmarkRow",
    "BenchmarkRun",
    "BudgetCurvePoint",
    "DetectionResult",
    "ExperimentConfig",
    "available_providers",
    "available_public_sources",
    "WatermarkSpec",
    "WatermarkedSnippet",
    "build_benchmark_manifest",
    "build_experiment_config",
    "build_provider",
    "dump_config",
    "generate_corpus",
    "load_config",
    "load_benchmark_corpus",
    "prepare_public_benchmark",
    "run_experiment",
    "validate_experiment_config",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in {"BenchmarkRun", "generate_corpus", "run_experiment"}:
        from .pipeline import BenchmarkRun, generate_corpus, run_experiment

        exported = {
            "BenchmarkRun": BenchmarkRun,
            "generate_corpus": generate_corpus,
            "run_experiment": run_experiment,
        }
        globals().update(exported)
        return exported[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
