from __future__ import annotations

from typing import Any

from ..benchmarks import build_benchmark_manifest, load_benchmark_corpus

__all__ = [
    "BenchmarkRun",
    "build_benchmark_manifest",
    "generate_corpus",
    "load_benchmark_corpus",
    "run_experiment",
]


def __getattr__(name: str) -> Any:
    if name in {"BenchmarkRun", "generate_corpus", "run_experiment"}:
        from .generator import generate_corpus
        from .orchestrator import BenchmarkRun, run_experiment

        exports = {
            "BenchmarkRun": BenchmarkRun,
            "generate_corpus": generate_corpus,
            "run_experiment": run_experiment,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
