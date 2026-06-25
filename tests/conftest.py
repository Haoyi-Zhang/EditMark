from __future__ import annotations

import os
import sys

import pytest

from posteditbench.models import BenchmarkExample, ExperimentConfig, WatermarkSpec


# Keep the cloud authoritative repo free of machine-specific bytecode residue
# while tests import project modules directly from the working tree.
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True


@pytest.fixture
def sample_example() -> BenchmarkExample:
    return BenchmarkExample(
        example_id="sample-python-1",
        language="python",
        prompt="Write a function `factorial(n)` that returns the factorial of `n`.",
        reference_solution=(
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        ),
        reference_tests=("assert factorial(5) == 120",),
        execution_tests=("assert factorial(4) == 24",),
        metadata={"dataset": "fixture", "validation_supported": True},
    )


@pytest.fixture
def sample_spec() -> WatermarkSpec:
    return WatermarkSpec(
        name="kgw",
        secret="redacted",
        payload="wm",
        strength=1.0,
        parameters={"threshold": 0.5},
    )


@pytest.fixture
def sample_config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=7,
        corpus_size=3,
        language="python",
        watermark_name="kgw",
        watermark_secret="redacted",
        watermark_payload="wm",
        watermark_strength=1.0,
        attacks=("comment_strip", "identifier_rename", "budgeted_adaptive"),
        provider_mode="offline_mock",
        validation_scope="python_first",
        corpus_parameters={"dataset_label": "fixture", "validation_supported": True},
        metadata={"benchmark": {"dataset_label": "fixture"}},
    )
