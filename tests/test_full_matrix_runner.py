from __future__ import annotations

import json
import importlib.util
import os
import runpy
import socket
import threading
import sys
import subprocess
import time
from pathlib import Path

import pytest


def _write_manifest(path: Path, runs: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "suite_all_models_methods",
                "runs": runs,
                "excluded_configs": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_runner_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "run_full_matrix.py"
    spec = importlib.util.spec_from_file_location("run_full_matrix_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_full_matrix_gpu_pool_executors_are_split() -> None:
    module = _load_runner_module()
    slot_mapping = module._slot_mapping(4, "split")
    specs = module._executor_specs(slot_mapping, cpu_workers=6)

    assert specs["cpu"] == 6
    assert specs["gpu:runtime"] == 2
    assert specs["gpu:local_hf"] == 2

    runtime_run = module.MatrixRun(
        run_id="runtime_run",
        config_path=Path("configs/public_humaneval_plus_stone_runtime.yaml"),
        resource="gpu",
        output_dir=Path("results/runtime_run"),
        report_path=Path("results/runtime_run/report.json"),
        log_path=Path("results/runtime_run/run.log"),
        gpu_pool="runtime",
    )
    local_hf_run = module.MatrixRun(
        run_id="local_hf_run",
        config_path=Path("tests/fixtures/configs/public_humaneval_plus_local_hf_qwen25_7b.yaml"),
        resource="gpu",
        output_dir=Path("results/local_hf_run"),
        report_path=Path("results/local_hf_run/report.json"),
        log_path=Path("results/local_hf_run/run.log"),
        gpu_pool="local_hf",
    )

    assert module._executor_key_for(runtime_run, slot_mapping) == "gpu:runtime"
    assert module._executor_key_for(local_hf_run, slot_mapping) == "gpu:local_hf"


def test_run_full_matrix_shared_gpu_pool_collapses_gpu_work() -> None:
    module = _load_runner_module()
    slot_mapping = module._slot_mapping(8, "shared")
    specs = module._executor_specs(slot_mapping, cpu_workers=6)

    assert slot_mapping == {"default": list(range(8))}
    assert specs["cpu"] == 6
    assert specs["gpu:default"] == 8

    runtime_run = module.MatrixRun(
        run_id="runtime_run",
        config_path=Path("configs/public_humaneval_plus_stone_runtime.yaml"),
        resource="gpu",
        output_dir=Path("results/runtime_run"),
        report_path=Path("results/runtime_run/report.json"),
        log_path=Path("results/runtime_run/run.log"),
        gpu_pool="runtime",
    )
    local_hf_run = module.MatrixRun(
        run_id="local_hf_run",
        config_path=Path("tests/fixtures/configs/public_humaneval_plus_local_hf_qwen25_7b.yaml"),
        resource="gpu",
        output_dir=Path("results/local_hf_run"),
        report_path=Path("results/local_hf_run/report.json"),
        log_path=Path("results/local_hf_run/run.log"),
        gpu_pool="local_hf",
    )

    assert module._executor_key_for(runtime_run, slot_mapping) == "gpu:default"
    assert module._executor_key_for(local_hf_run, slot_mapping) == "gpu:default"


def test_run_full_matrix_marks_local_cache_runs_as_offline_assets() -> None:
    module = _load_runner_module()
    runtime_run = module.MatrixRun(
        run_id="runtime_run",
        config_path=Path("configs/public_humaneval_plus_stone_runtime.yaml"),
        resource="gpu",
        output_dir=Path("results/runtime_run"),
        report_path=Path("results/runtime_run/report.json"),
        log_path=Path("results/runtime_run/run.log"),
        gpu_pool="runtime",
    )
    local_hf_run = module.MatrixRun(
        run_id="local_hf_run",
        config_path=Path("tests/fixtures/configs/public_humaneval_plus_local_hf_qwen25_7b.yaml"),
        resource="gpu",
        output_dir=Path("results/local_hf_run"),
        report_path=Path("results/local_hf_run/report.json"),
        log_path=Path("results/local_hf_run/run.log"),
        gpu_pool="local_hf",
    )

    assert module._requires_offline_hf_assets(runtime_run) is True
    assert module._requires_offline_hf_assets(local_hf_run) is True


def test_run_full_matrix_orders_longer_gpu_work_first(monkeypatch) -> None:
    module = _load_runner_module()
    runs = [
        module.MatrixRun(
            run_id="humaneval_local_small",
            config_path=Path("tests/fixtures/configs/public_humaneval_plus_local_hf_qwen25_1p5b.yaml"),
            resource="gpu",
            output_dir=Path("results/a"),
            report_path=Path("results/a/report.json"),
            log_path=Path("results/a/run.log"),
            gpu_pool="local_hf",
        ),
        module.MatrixRun(
            run_id="mbpp_runtime",
            config_path=Path("configs/public_mbpp_plus_stone_runtime.yaml"),
            resource="gpu",
            output_dir=Path("results/b"),
            report_path=Path("results/b/report.json"),
            log_path=Path("results/b/run.log"),
            gpu_pool="runtime",
            baseline_eval=True,
        ),
        module.MatrixRun(
            run_id="big_cpu",
            config_path=Path("tests/fixtures/configs/unified_multilingual_full.yaml"),
            resource="cpu",
            output_dir=Path("results/c"),
            report_path=Path("results/c/report.json"),
            log_path=Path("results/c/run.log"),
        ),
    ]

    metadata = {
        "humaneval_local_small": {
            "corpus_size": 164,
            "provider_mode": "local_hf",
            "watermark_name": "kgw",
            "provider_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "runtime_model": "",
            "effective_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "benchmark_label": "HumanEval+",
            "benchmark_path": "data/release/sources/suite_humaneval_plus_release.normalized.jsonl",
        },
        "mbpp_runtime": {
            "corpus_size": 378,
            "provider_mode": "offline_mock",
            "watermark_name": "stone_runtime",
            "provider_model": "",
            "runtime_model": "bigcode/starcoder2-7b",
            "effective_model": "bigcode/starcoder2-7b",
            "benchmark_label": "MBPP+",
            "benchmark_path": "data/release/sources/suite_mbpp_plus_release.normalized.jsonl",
        },
        "big_cpu": {
            "corpus_size": 7629,
            "provider_mode": "offline_mock",
            "watermark_name": "kgw",
            "provider_model": "",
            "runtime_model": "",
            "effective_model": "",
            "benchmark_label": "Unified Multilingual Full",
            "benchmark_path": "data/collections/unified_multilingual_full.jsonl",
        },
    }
    path_to_run_id = {run.config_path: run.run_id for run in runs}

    def fake_load_run_metadata(config_path: Path):
        return metadata[path_to_run_id[Path(config_path)]]

    monkeypatch.setattr(module, "_load_run_metadata", fake_load_run_metadata)

    ordered = module._ordered_runs(runs)

    assert [run.run_id for run in ordered] == ["mbpp_runtime", "humaneval_local_small", "big_cpu"]


def test_run_full_matrix_interleaves_gpu_models_heavy_first(monkeypatch) -> None:
    module = _load_runner_module()
    runs = [
        module.MatrixRun(
            run_id="q14_ewd",
            config_path=Path("configs/public_humaneval_plus_ewd_runtime.yaml"),
            resource="gpu",
            output_dir=Path("results/q14_ewd"),
            report_path=Path("results/q14_ewd/report.json"),
            log_path=Path("results/q14_ewd/run.log"),
            gpu_pool="runtime",
        ),
        module.MatrixRun(
            run_id="q14_kgw",
            config_path=Path("configs/public_humaneval_plus_kgw_runtime.yaml"),
            resource="gpu",
            output_dir=Path("results/q14_kgw"),
            report_path=Path("results/q14_kgw/report.json"),
            log_path=Path("results/q14_kgw/run.log"),
            gpu_pool="runtime",
        ),
        module.MatrixRun(
            run_id="deepseek_ewd",
            config_path=Path("configs/public_humaneval_plus_ewd_runtime.yaml"),
            resource="gpu",
            output_dir=Path("results/deepseek_ewd"),
            report_path=Path("results/deepseek_ewd/report.json"),
            log_path=Path("results/deepseek_ewd/run.log"),
            gpu_pool="runtime",
        ),
        module.MatrixRun(
            run_id="star7_ewd",
            config_path=Path("configs/public_humaneval_plus_ewd_runtime.yaml"),
            resource="gpu",
            output_dir=Path("results/star7_ewd"),
            report_path=Path("results/star7_ewd/report.json"),
            log_path=Path("results/star7_ewd/run.log"),
            gpu_pool="runtime",
        ),
    ]
    metadata = {
        "q14_ewd": {
            "corpus_size": 600,
            "provider_mode": "offline_mock",
            "watermark_name": "ewd_runtime",
            "provider_model": "",
            "runtime_model": "Qwen/Qwen2.5-Coder-14B-Instruct",
            "effective_model": "Qwen/Qwen2.5-Coder-14B-Instruct",
            "benchmark_label": "Crafted Original",
            "benchmark_path": "data/release/sources/crafted_original_release.normalized.jsonl",
        },
        "q14_kgw": {
            "corpus_size": 600,
            "provider_mode": "offline_mock",
            "watermark_name": "kgw_runtime",
            "provider_model": "",
            "runtime_model": "Qwen/Qwen2.5-Coder-14B-Instruct",
            "effective_model": "Qwen/Qwen2.5-Coder-14B-Instruct",
            "benchmark_label": "Crafted Original",
            "benchmark_path": "data/release/sources/crafted_original_release.normalized.jsonl",
        },
        "deepseek_ewd": {
            "corpus_size": 600,
            "provider_mode": "offline_mock",
            "watermark_name": "ewd_runtime",
            "provider_model": "",
            "runtime_model": "deepseek-ai/deepseek-coder-6.7b-instruct",
            "effective_model": "deepseek-ai/deepseek-coder-6.7b-instruct",
            "benchmark_label": "Crafted Original",
            "benchmark_path": "data/release/sources/crafted_original_release.normalized.jsonl",
        },
        "star7_ewd": {
            "corpus_size": 600,
            "provider_mode": "offline_mock",
            "watermark_name": "ewd_runtime",
            "provider_model": "",
            "runtime_model": "bigcode/starcoder2-7b",
            "effective_model": "bigcode/starcoder2-7b",
            "benchmark_label": "Crafted Original",
            "benchmark_path": "data/release/sources/crafted_original_release.normalized.jsonl",
        },
    }

    ordered = module._ordered_runs(runs, metadata_map=metadata)

    assert [run.run_id for run in ordered] == ["q14_ewd", "deepseek_ewd", "star7_ewd", "q14_kgw"]


def test_run_full_matrix_dry_run_writes_index(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "matrix.json"
    _write_manifest(
        manifest_path,
        [
            {"run_id": "cpu_run", "profile": "suite_all_models_methods", "config": "configs/debug.yaml", "resource": "cpu"},
            {"run_id": "gpu_run", "profile": "suite_all_models_methods", "config": "tests/fixtures/configs/public_humaneval_plus_local_hf_qwen25_7b.yaml", "resource": "gpu", "gpu_pool": "local_hf"},
        ],
    )
    output_root = tmp_path / "matrix_out"
    root = Path(__file__).resolve().parents[1]

    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--gpu-slots",
            "4",
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit) as dry_exit:
        runpy.run_path(str(root / "scripts" / "run_full_matrix.py"), run_name="__main__")
    assert dry_exit.value.code == 0

    index_path = output_root / "suite_all_models_methods" / "matrix_index.dry_run.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["run_count"] == 2
    assert payload["gpu_pool_mode"] == "shared"
    assert payload["gpu_slot_mapping"]["default"] == [0, 1, 2, 3]
    assert payload["runs"][0]["priority"] == 0


def test_run_full_matrix_dry_run_resolves_config_overrides(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "matrix.json"
    _write_manifest(
        manifest_path,
        [
            {
                "run_id": "override_run",
                "profile": "suite_all_models_methods",
                "config": "tests/fixtures/configs/public_humaneval_plus_local_hf_qwen25_7b.yaml",
                "config_overrides": {
                    "project": {"name": "override-run"},
                    "provider": {"parameters": {"model": "deepseek-ai/deepseek-coder-6.7b-instruct"}},
                    "watermark": {"scheme": "comment", "strength": 0.52},
                },
                "resource": "gpu",
                "gpu_pool": "local_hf",
            }
        ],
    )
    output_root = tmp_path / "matrix_out"
    root = Path(__file__).resolve().parents[1]

    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--gpu-slots",
            "2",
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit) as dry_exit:
        runpy.run_path(str(root / "scripts" / "run_full_matrix.py"), run_name="__main__")
    assert dry_exit.value.code == 0

    index_path = output_root / "suite_all_models_methods" / "matrix_index.dry_run.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    run = payload["runs"][0]
    assert run["provider_model"] == "deepseek-ai/deepseek-coder-6.7b-instruct"
    assert run["effective_model"] == "deepseek-ai/deepseek-coder-6.7b-instruct"
    assert run["watermark_name"] == "comment"
    assert run["config_overrides"]["provider"]["parameters"]["model"] == "deepseek-ai/deepseek-coder-6.7b-instruct"


def test_run_full_matrix_dry_run_surfaces_runtime_effective_model(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "matrix.json"
    _write_manifest(
        manifest_path,
        [
            {
                "run_id": "runtime_override",
                "profile": "suite_all_models_methods",
                "config": "configs/public_humaneval_plus_stone_runtime.yaml",
                "config_overrides": {
                    "project": {"name": "runtime-override"},
                    "watermark": {"model_name": "Qwen/Qwen2.5-Coder-7B-Instruct"},
                },
                "resource": "gpu",
                "gpu_pool": "runtime",
                "baseline_eval": True,
            }
        ],
    )
    output_root = tmp_path / "matrix_out"
    root = Path(__file__).resolve().parents[1]

    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--gpu-slots",
            "2",
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit) as dry_exit:
        runpy.run_path(str(root / "scripts" / "run_full_matrix.py"), run_name="__main__")
    assert dry_exit.value.code == 0

    index_path = output_root / "suite_all_models_methods" / "matrix_index.dry_run.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    run = payload["runs"][0]
    assert run["provider_model"] == ""
    assert run["runtime_model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert run["effective_model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"


def test_run_full_matrix_dry_run_does_not_overwrite_live_index(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "matrix.json"
    _write_manifest(
        manifest_path,
        [{"run_id": "cpu_run", "profile": "suite_all_models_methods", "config": "configs/debug.yaml", "resource": "cpu"}],
    )
    output_root = tmp_path / "matrix_out"
    live_index = output_root / "suite_all_models_methods" / "matrix_index.json"
    live_index.parent.mkdir(parents=True, exist_ok=True)
    live_index.write_text(json.dumps({"status": "running"}) + "\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]

    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit) as dry_exit:
        runpy.run_path(str(root / "scripts" / "run_full_matrix.py"), run_name="__main__")
    assert dry_exit.value.code == 0
    assert json.loads(live_index.read_text(encoding="utf-8"))["status"] == "running"
    assert (live_index.parent / "matrix_index.dry_run.json").exists()


def test_run_full_matrix_fail_fast_skips_queued_runs(tmp_path, monkeypatch) -> None:
    module = _load_runner_module()
    manifest_path = tmp_path / "matrix.json"
    _write_manifest(
        manifest_path,
        [
            {
                "run_id": "first_cpu",
                "profile": "suite_all_models_methods",
                "config": "configs/debug.yaml",
                "resource": "cpu",
                "priority": 10,
            },
            {
                "run_id": "second_cpu",
                "profile": "suite_all_models_methods",
                "config": "configs/debug.yaml",
                "resource": "cpu",
                "priority": 1,
            },
        ],
    )
    output_root = tmp_path / "matrix_out"
    root = Path(__file__).resolve().parents[1]

    call_results = iter([1, 0])

    monkeypatch.setattr(module, "_execute_subprocess", lambda *args, **kwargs: next(call_results))
    monkeypatch.setattr(
        module,
        "_load_run_metadata_for_matrix",
        lambda run: {
            "corpus_size": 1,
            "provider_mode": "offline_mock",
            "watermark_name": "stone_runtime",
            "provider_model": "",
            "runtime_model": "bigcode/starcoder2-7b",
            "effective_model": "bigcode/starcoder2-7b",
            "benchmark_label": "Debug",
            "benchmark_path": "data/interim/benchmark.normalized.jsonl",
        },
    )

    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--cpu-workers",
            "1",
            "--fail-fast",
        ],
    )

    exit_code = module.main()
    assert exit_code == 1

    index_path = output_root / "suite_all_models_methods" / "matrix_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    by_run = {run["run_id"]: run for run in payload["runs"]}
    assert by_run["first_cpu"]["status"] == "failed"
    assert by_run["second_cpu"]["status"] == "skipped"
    assert by_run["second_cpu"]["reason"] == "fail_fast_stop_requested"
    assert payload["stop_requested"] is True


def test_run_full_matrix_records_invalid_metadata_without_crashing(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "matrix.json"
    _write_manifest(
        manifest_path,
        [
            {
                "run_id": "broken_run",
                "profile": "suite_all_models_methods",
                "config": str(tmp_path / "missing.yaml"),
                "resource": "cpu",
            },
        ],
    )
    output_root = tmp_path / "matrix_out"
    root = Path(__file__).resolve().parents[1]

    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(root / "scripts" / "run_full_matrix.py"), run_name="__main__")
    assert exit_info.value.code == 1

    index_path = output_root / "suite_all_models_methods" / "matrix_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    broken = next(item for item in payload["runs"] if item["run_id"] == "broken_run")
    assert broken["status"] == "failed"
    assert broken["reason"] == "invalid_run_metadata"


def test_run_full_matrix_rejects_api_resources(tmp_path) -> None:
    manifest_path = tmp_path / "matrix.json"
    _write_manifest(
        manifest_path,
        [
            {
                "run_id": "api_run",
                "profile": "suite_all_models_methods",
                "config": "configs/public_humaneval_plus_stone_runtime.yaml",
                "resource": "api",
            },
        ],
    )

    module = _load_runner_module()
    with pytest.raises(ValueError, match="API support removed"):
        module._load_matrix_runs(manifest_path, "suite_all_models_methods", tmp_path / "matrix_out")


def test_run_full_matrix_marks_subprocess_timeout(tmp_path, monkeypatch) -> None:
    module = _load_runner_module()
    log_path = tmp_path / "run.log"
    script_path = tmp_path / "sleepy.py"
    script_path.write_text(
        "import time\nprint('partial', flush=True)\ntime.sleep(2)\n",
        encoding="utf-8",
    )

    exit_code = module._execute_subprocess(
        [sys.executable, str(script_path)],
        env={},
        log_path=log_path,
        timeout_seconds=1,
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert exit_code == 124
    assert "partial" in log_text
    assert "[timeout_seconds=1]" in log_text
    assert "[exit_code=124]" in log_text


def test_run_full_matrix_streams_logs_while_child_is_running(tmp_path) -> None:
    module = _load_runner_module()
    log_path = tmp_path / "run.log"
    script_path = tmp_path / "slow_writer.py"
    script_path.write_text(
        "\n".join(
            [
                "import sys",
                "import time",
                "print('stage=start', flush=True)",
                "sys.stderr.write('stage=stderr\\n')",
                "sys.stderr.flush()",
                "time.sleep(0.5)",
                "print('stage=done', flush=True)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result: dict[str, int] = {}

    def runner() -> None:
        result["exit_code"] = module._execute_subprocess(
            [sys.executable, str(script_path)],
            env={},
            log_path=log_path,
            timeout_seconds=10,
        )

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()

    deadline = time.time() + 2.0
    observed = ""
    while time.time() < deadline:
        if log_path.exists():
            observed = log_path.read_text(encoding="utf-8")
            if "stage=start" in observed and "stage=stderr" in observed and "stage=done" not in observed:
                break
        time.sleep(0.05)

    thread.join(timeout=5.0)
    assert result["exit_code"] == 0
    final_log = log_path.read_text(encoding="utf-8")
    assert "stage=start" in observed or "stage=start" in final_log
    assert "stage=stderr" in observed or "stage=stderr" in final_log
    assert "stage=done" in final_log
    assert "[exit_code=0]" in final_log


def test_run_full_matrix_resume_marks_existing_report_as_skipped(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "tiny.json"
    config_path.write_text(
        json.dumps(
            {
                "project": {"name": "tiny-matrix-run", "seed": 77},
                "benchmark": {
                    "source": "data/release/sources/suite_humaneval_plus_release.normalized.jsonl",
                    "prepared_output": "data/release/sources/suite_humaneval_plus_release.normalized.jsonl",
                    "limit": 1,
                    "include_reference_kinds": ["canonical"],
                },
                "provider": {"mode": "offline_mock", "parameters": {}},
                "watermark": {"scheme": "comment", "strength": 0.5},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "matrix.json"
    _write_manifest(
        manifest_path,
        [{"run_id": "tiny_run", "profile": "suite_all_models_methods", "config": str(config_path), "resource": "cpu"}],
    )
    output_root = tmp_path / "matrix_out"
    root = Path(__file__).resolve().parents[1]

    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--cpu-workers",
            "1",
            "--retry-count",
            "0",
        ],
    )
    with pytest.raises(SystemExit) as first_exit:
        runpy.run_path(str(root / "scripts" / "run_full_matrix.py"), run_name="__main__")
    assert first_exit.value.code == 0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--cpu-workers",
            "1",
            "--retry-count",
            "0",
            "--resume",
        ],
    )
    with pytest.raises(SystemExit) as second_exit:
        runpy.run_path(str(root / "scripts" / "run_full_matrix.py"), run_name="__main__")
    assert second_exit.value.code == 0

    index_path = output_root / "suite_all_models_methods" / "matrix_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    tiny_run = next(item for item in payload["runs"] if item["run_id"] == "tiny_run")
    assert tiny_run["status"] == "success"
    assert tiny_run["reason"] == "resume_existing_report"


def test_run_full_matrix_resume_reruns_when_report_config_is_stale(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "tiny.json"
    config_path.write_text(
        json.dumps(
            {
                "project": {"name": "tiny-matrix-run", "seed": 77},
                "benchmark": {
                    "source": "data/release/sources/suite_humaneval_plus_release.normalized.jsonl",
                    "prepared_output": "data/release/sources/suite_humaneval_plus_release.normalized.jsonl",
                    "limit": 1,
                    "include_reference_kinds": ["canonical"],
                },
                "provider": {"mode": "offline_mock", "parameters": {}},
                "watermark": {"scheme": "comment", "strength": 0.5},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "matrix.json"
    _write_manifest(
        manifest_path,
        [{"run_id": "tiny_run", "profile": "suite_all_models_methods", "config": str(config_path), "resource": "cpu"}],
    )
    output_root = tmp_path / "matrix_out"
    root = Path(__file__).resolve().parents[1]

    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--cpu-workers",
            "1",
            "--retry-count",
            "0",
        ],
    )
    with pytest.raises(SystemExit) as first_exit:
        runpy.run_path(str(root / "scripts" / "run_full_matrix.py"), run_name="__main__")
    assert first_exit.value.code == 0

    config_path.write_text(
        json.dumps(
            {
                "project": {"name": "tiny-matrix-run", "seed": 77},
                "benchmark": {
                    "source": "data/release/sources/suite_humaneval_plus_release.normalized.jsonl",
                    "prepared_output": "data/release/sources/suite_humaneval_plus_release.normalized.jsonl",
                    "limit": 1,
                    "include_reference_kinds": ["canonical"],
                },
                "provider": {"mode": "offline_mock", "parameters": {}},
                "watermark": {"scheme": "comment", "strength": 0.9},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--cpu-workers",
            "1",
            "--retry-count",
            "0",
            "--resume",
        ],
    )
    with pytest.raises(SystemExit) as second_exit:
        runpy.run_path(str(root / "scripts" / "run_full_matrix.py"), run_name="__main__")
    assert second_exit.value.code == 0

    index_path = output_root / "suite_all_models_methods" / "matrix_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    tiny_run = next(item for item in payload["runs"] if item["run_id"] == "tiny_run")
    assert tiny_run["status"] == "success"
    assert tiny_run["attempts"] == 1
    report_path = output_root / "suite_all_models_methods" / "tiny_run" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["config"]["watermark_strength"] == pytest.approx(0.9)


def test_run_full_matrix_marks_missing_report_output(tmp_path, monkeypatch) -> None:
    module = _load_runner_module()
    config_path = tmp_path / "tiny.json"
    config_path.write_text(
        json.dumps(
            {
                "project": {"name": "tiny-matrix-run", "seed": 77},
                "benchmark": {
                    "source": "data/release/sources/suite_humaneval_plus_release.normalized.jsonl",
                    "prepared_output": "data/release/sources/suite_humaneval_plus_release.normalized.jsonl",
                    "limit": 1,
                    "include_reference_kinds": ["canonical"],
                },
                "provider": {"mode": "offline_mock", "parameters": {}},
                "watermark": {"scheme": "comment", "strength": 0.5},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    run = module.MatrixRun(
        run_id="missing_report",
        config_path=config_path,
        resource="cpu",
        output_dir=tmp_path / "run_out",
        report_path=tmp_path / "run_out" / "report.json",
        log_path=tmp_path / "run_out" / "run.log",
    )
    results: dict[str, dict[str, object]] = {}

    monkeypatch.setattr(module, "_load_run_metadata_for_matrix", lambda _: module._blank_metadata())
    monkeypatch.setattr(module, "_execute_subprocess", lambda *args, **kwargs: 0)

    args = type(
        "Args",
        (),
        {
            "fail_fast": False,
            "retry_count": 0,
            "python_bin": sys.executable,
            "command_timeout_seconds": 10,
        },
    )()
    metadata = module._blank_metadata()
    import threading
    results_lock = threading.Lock()

    record = None
    def exercise_failure_path():
        nonlocal record
        attempts_allowed = 1
        last_exit_code = 0
        last_device = ""
        last_failure_reason = "command_failed"
        last_failure_detail = ""
        started_at = time.time()
        module.ensure_dir(run.output_dir)
        with results_lock:
            results[run.run_id] = {
                **module._matrix_record_base(run, metadata=metadata),
                "status": "running",
                "attempts": 0,
                "retries_used": 0,
                "exit_code": 0,
                "started_at": started_at,
                    "finished_at": None,
                    "duration_seconds": 0.0,
                    "cuda_visible_devices": "",
                }
        for attempt in range(1, attempts_allowed + 1):
            env = {}
            module._cleanup_previous_final_outputs(run)
            device_token = ""
            exit_code = module._execute_subprocess(
                module._command_for(run, args.python_bin),
                env=env,
                log_path=run.log_path,
                timeout_seconds=args.command_timeout_seconds,
            )
            failure_reason = module._failure_reason_for_exit_code(exit_code, stage="run_command")
            failure_detail = f"exit_code={exit_code}" if exit_code != 0 else ""
            if exit_code == 0:
                if not run.report_path.exists():
                    module._append_log_message(run.log_path, "run completed without producing report.json")
                    exit_code = 90
                    failure_reason = "missing_report_output"
                    failure_detail = "report.json"
            last_exit_code = exit_code
            last_device = device_token
            if exit_code != 0:
                last_failure_reason = failure_reason or "run_command_failed"
                last_failure_detail = failure_detail
        finished_at = time.time()
        record = {
            **module._matrix_record_base(run, metadata=metadata),
            "status": "failed",
            "reason": last_failure_reason,
            "reason_detail": last_failure_detail,
            "attempts": attempts_allowed,
            "retries_used": attempts_allowed - 1,
            "exit_code": last_exit_code,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": round(finished_at - started_at, 2),
            "cuda_visible_devices": last_device,
        }

    exercise_failure_path()

    assert record is not None
    assert record["reason"] == "missing_report_output"
    assert record["reason_detail"] == "report.json"


def test_run_full_matrix_baseline_eval_command_uses_explicit_private_payload_path() -> None:
    module = _load_runner_module()
    run = module.MatrixRun(
        run_id="runtime_eval",
        config_path=Path("configs/public_humaneval_plus_stone_runtime.yaml"),
        resource="gpu",
        output_dir=Path("results/runtime_eval"),
        report_path=Path("results/runtime_eval/report.json"),
        log_path=Path("results/runtime_eval/run.log"),
        gpu_pool="runtime",
        baseline_eval=True,
    )

    command = module._baseline_eval_command(
        run,
        "python",
        payload_path=Path("/tmp/runtime_eval_payload.private.jsonl"),
    )

    assert "--ppl-model" in command
    assert "bigcode/starcoder2-7b" in command
    assert "--cache-dir" in command
    assert "model_cache/huggingface" in command
    assert "--local-files-only" in command
    assert "--payloads" in command
    assert str(Path("/tmp/runtime_eval_payload.private.jsonl")) in command


def test_run_full_matrix_baseline_eval_command_uses_run_execution_context() -> None:
    module = _load_runner_module()
    run = module.MatrixRun(
        run_id="runtime_eval",
        config_path=Path("configs/public_humaneval_plus_stone_runtime.yaml"),
        resource="gpu",
        output_dir=Path("results/runtime_eval"),
        report_path=Path("results/runtime_eval/report.json"),
        log_path=Path("results/runtime_eval/run.log"),
        gpu_pool="runtime",
        baseline_eval=True,
        config_overrides={
            "watermark": {
                "cache_dir": "/tmp/runtime_cache",
                "local_files_only": True,
                "token_env": "RUNTIME_TOKEN",
                "trust_remote_code": True,
                "device": "cuda:1",
            }
        },
    )

    command = module._baseline_eval_command(run, "python")

    assert "--cache-dir" in command
    assert "/tmp/runtime_cache" in command
    assert "--token-env" in command
    assert "RUNTIME_TOKEN" in command
    assert "--local-files-only" in command
    assert "--trust-remote-code" in command
    assert "--device" in command
    assert "cuda" in command


def test_run_full_matrix_cleanup_sensitive_baseline_payloads_deletes_temp_file(tmp_path) -> None:
    module = _load_runner_module()
    payload_path = tmp_path / "payload.private.jsonl"
    payload_path.write_text("secret\n", encoding="utf-8")

    module._cleanup_sensitive_baseline_payloads(payload_path)

    assert payload_path.exists() is False


def test_run_full_matrix_lock_rejects_live_owner(tmp_path: Path) -> None:
    module = _load_runner_module()
    lock_path = tmp_path / ".matrix_runner.lock"
    lock_path.write_text(
        json.dumps(
            {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "profile": "suite_all_models_methods",
                "manifest": "configs/matrices/suite_all_models_methods.json",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="matrix lock already held"):
        module._acquire_matrix_lock(
            lock_path,
            manifest_path=Path("configs/matrices/suite_all_models_methods.json"),
            profile="suite_all_models_methods",
        )


def test_run_full_matrix_lock_reclaims_stale_owner(tmp_path: Path) -> None:
    module = _load_runner_module()
    lock_path = tmp_path / ".matrix_runner.lock"
    lock_path.write_text(
        json.dumps(
            {
                "host": socket.gethostname(),
                "pid": 999999,
                "profile": "suite_all_models_methods",
                "manifest": "configs/matrices/suite_all_models_methods.json",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    owner = module._acquire_matrix_lock(
        lock_path,
        manifest_path=Path("configs/matrices/suite_all_models_methods.json"),
        profile="suite_all_models_methods",
    )
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] == owner["pid"] == os.getpid()
    module._release_matrix_lock(lock_path, owner)
    assert not lock_path.exists()


def test_run_full_matrix_lock_force_clears_cross_host_owner(tmp_path: Path) -> None:
    module = _load_runner_module()
    lock_path = tmp_path / ".matrix_runner.lock"
    lock_path.write_text(
        json.dumps(
            {
                "host": "other-host",
                "pid": 424242,
                "profile": "suite_all_models_methods",
                "manifest": "configs/matrices/suite_all_models_methods.json",
                "created_at": time.time() - 3600,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    owner = module._acquire_matrix_lock(
        lock_path,
        manifest_path=Path("configs/matrices/suite_all_models_methods.json"),
        profile="suite_all_models_methods",
        force_clear_stale_lock=True,
    )

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["host"] == socket.gethostname()
    assert payload["pid"] == owner["pid"] == os.getpid()
    module._release_matrix_lock(lock_path, owner)
    assert not lock_path.exists()


def test_terminate_process_kills_descendants_on_posix(monkeypatch) -> None:
    module = _load_runner_module()

    class FakeProcess:
        pid = 4321

        def poll(self):
            return None

        def kill(self):
            raise AssertionError("posix tree termination should not fall back to process.kill() in this test")

    killed: list[object] = []
    monkeypatch.setattr(module.os, "name", "posix", raising=False)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: type("Completed", (), {"stdout": "5000 4321\n6000 5000\n"})(),
    )
    monkeypatch.setattr(module.os, "kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: killed.append(("pg", pid)), raising=False)

    module._terminate_process(FakeProcess())

    assert 6000 in killed
    assert 5000 in killed
    assert ("pg", 4321) in killed

