from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from posteditbench.config import build_experiment_config, load_config, validate_experiment_config

from tests._stone_test_helpers import create_runtime_checkout


RUNTIME_CONFIGS = (
    "configs/public_humaneval_plus_stone_runtime.yaml",
    "configs/public_humaneval_plus_sweet_runtime.yaml",
    "configs/public_humaneval_plus_ewd_runtime.yaml",
    "configs/public_humaneval_plus_kgw_runtime.yaml",
    "configs/public_mbpp_plus_stone_runtime.yaml",
    "configs/public_mbpp_plus_sweet_runtime.yaml",
    "configs/public_mbpp_plus_ewd_runtime.yaml",
    "configs/public_mbpp_plus_kgw_runtime.yaml",
)


def _prepare_runtime_checkout(monkeypatch, tmp_path):
    for method in ("stone_runtime", "sweet_runtime", "ewd_runtime", "kgw_runtime"):
        checkout, manifest, _ = create_runtime_checkout(tmp_path, method)
        prefix = method.split("_", 1)[0].upper()
        monkeypatch.setenv(f"POSTEDITBENCH_{prefix}_UPSTREAM_ROOT", str(checkout))
        monkeypatch.setenv(f"POSTEDITBENCH_{prefix}_UPSTREAM_MANIFEST", str(manifest))


def test_runtime_configs_validate_cleanly(monkeypatch, tmp_path):
    _prepare_runtime_checkout(monkeypatch, tmp_path)
    for relative_path in RUNTIME_CONFIGS:
        config = build_experiment_config(load_config(Path(relative_path)))
        assert validate_experiment_config(config) == []


def test_runtime_configs_use_local_cache_only():
    for relative_path in RUNTIME_CONFIGS:
        config = build_experiment_config(load_config(Path(relative_path)))
        parameters = dict(config.metadata.get("watermark", {}))
        assert parameters.get("cache_dir") == "model_cache/huggingface"
        assert parameters.get("local_files_only") is True
        assert parameters.get("dtype") == "float16"


def test_validate_experiment_config_surfaces_registry_failures(monkeypatch, tmp_path):
    benchmark_path = tmp_path / "benchmark.normalized.jsonl"
    benchmark_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "posteditbench.config._known_watermarks_payload",
        lambda: ({"stone_runtime"}, "watermark registry import failed; using fallback watermark roster (RuntimeError: boom)"),
    )
    config = build_experiment_config(
        {
            "benchmark": {"prepared_output": str(benchmark_path)},
            "watermark": {"scheme": "stone_runtime", "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct", "max_new_tokens": 64},
        }
    )

    issues = validate_experiment_config(config)

    assert any("watermark registry import failed" in issue for issue in issues)


def test_run_experiment_dry_run_preserves_runtime_watermark_metadata(monkeypatch, tmp_path):
    _prepare_runtime_checkout(monkeypatch, tmp_path)

    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "public_humaneval_plus_stone_runtime.yaml"
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "runtime-dry-run"),
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(root / "scripts" / "run_experiment.py"), run_name="__main__")

    assert exit_info.value.code == 0


def test_runtime_watermark_requires_model_name(monkeypatch, tmp_path):
    checkout, manifest, _ = create_runtime_checkout(tmp_path, "stone_runtime")
    monkeypatch.setenv("POSTEDITBENCH_STONE_UPSTREAM_ROOT", str(checkout))
    monkeypatch.setenv("POSTEDITBENCH_STONE_UPSTREAM_MANIFEST", str(manifest))
    config = build_experiment_config(
        {
            "benchmark": {"prepared_output": "data/fixtures/benchmark.normalized.jsonl"},
            "watermark": {"scheme": "stone_runtime"},
        }
    )
    issues = validate_experiment_config(config)
    assert any("requires watermark.model_name" in issue for issue in issues)


def test_runtime_watermark_requires_upstream_checkout(monkeypatch, tmp_path):
    benchmark_path = tmp_path / "benchmark.normalized.jsonl"
    benchmark_path.write_text("", encoding="utf-8")
    monkeypatch.setattr("posteditbench.baselines.stone_family.common._workspace_root", lambda: tmp_path)
    config = build_experiment_config(
        {
            "benchmark": {"prepared_output": str(benchmark_path)},
            "watermark": {"scheme": "stone_runtime", "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct"},
        }
    )
    issues = validate_experiment_config(config)
    assert any("local official checkout" in issue for issue in issues)
    assert sum("valid official upstream checkout" in issue for issue in issues) == 0


def test_runtime_watermark_rejects_unknown_compatibility_profile(monkeypatch, tmp_path):
    checkout, manifest, _ = create_runtime_checkout(tmp_path, "sweet_runtime")
    benchmark_path = tmp_path / "benchmark.normalized.jsonl"
    benchmark_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("POSTEDITBENCH_SWEET_UPSTREAM_ROOT", str(checkout))
    monkeypatch.setenv("POSTEDITBENCH_SWEET_UPSTREAM_MANIFEST", str(manifest))
    config = build_experiment_config(
        {
            "benchmark": {"prepared_output": str(benchmark_path)},
            "watermark": {
                "scheme": "sweet_runtime",
                "model_name": "deepseek-ai/deepseek-coder-6.7b-instruct",
                "compatibility_profile": "unknown_profile",
                "max_new_tokens": 256,
            },
        }
    )
    issues = validate_experiment_config(config)
    assert any("compatibility_profile" in issue for issue in issues)


def test_runtime_watermark_accepts_auto_compatibility_profile(monkeypatch, tmp_path):
    checkout, manifest, _ = create_runtime_checkout(tmp_path, "sweet_runtime")
    benchmark_path = tmp_path / "benchmark.normalized.jsonl"
    benchmark_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("POSTEDITBENCH_SWEET_UPSTREAM_ROOT", str(checkout))
    monkeypatch.setenv("POSTEDITBENCH_SWEET_UPSTREAM_MANIFEST", str(manifest))
    config = build_experiment_config(
        {
            "benchmark": {"prepared_output": str(benchmark_path)},
            "watermark": {
                "scheme": "sweet_runtime",
                "model_name": "deepseek-ai/deepseek-coder-6.7b-instruct",
                "compatibility_profile": "auto",
                "max_new_tokens": 256,
            },
        }
    )
    assert validate_experiment_config(config) == []


def test_full_configs_use_full_limit_semantics():
    expected_sizes = {
        "tests/fixtures/configs/full_humaneval.yaml": 164,
        "tests/fixtures/configs/full_mbpp.yaml": 378,
    }
    for relative_path, expected_size in expected_sizes.items():
        config = build_experiment_config(load_config(Path(relative_path)))
        assert config.corpus_parameters["limit_mode"] == "full"
        assert config.corpus_size == expected_size
