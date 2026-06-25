from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from tests._stone_test_helpers import create_runtime_checkout


def _prepare_runtime_upstreams(tmp_path, monkeypatch) -> None:
    runtime_env = {
        "stone_runtime": ("POSTEDITBENCH_STONE_UPSTREAM_ROOT", "POSTEDITBENCH_STONE_UPSTREAM_MANIFEST"),
        "sweet_runtime": ("POSTEDITBENCH_SWEET_UPSTREAM_ROOT", "POSTEDITBENCH_SWEET_UPSTREAM_MANIFEST"),
        "ewd_runtime": ("POSTEDITBENCH_EWD_UPSTREAM_ROOT", "POSTEDITBENCH_EWD_UPSTREAM_MANIFEST"),
        "kgw_runtime": ("POSTEDITBENCH_KGW_UPSTREAM_ROOT", "POSTEDITBENCH_KGW_UPSTREAM_MANIFEST"),
    }
    for method, (root_env, manifest_env) in runtime_env.items():
        checkout, manifest, _ = create_runtime_checkout(tmp_path, method)
        monkeypatch.setenv(root_env, str(checkout))
        monkeypatch.setenv(manifest_env, str(manifest))


def _write_valid_safetensor(path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"weight": torch.zeros(1)}, str(path))


def _write_hf_text_assets(snapshot_dir: Path) -> None:
    (snapshot_dir / "config.json").write_text(json.dumps({"architectures": ["MockModel"]}) + "\n", encoding="utf-8")
    (snapshot_dir / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 512}) + "\n", encoding="utf-8")
    (snapshot_dir / "tokenizer.json").write_text(json.dumps({"version": "1.0", "truncation": None}) + "\n", encoding="utf-8")


def _write_hf_cache_entry(cache_root: Path, model_name: str, *, root: bool, hub: bool) -> None:
    entry_name = "models--" + model_name.replace("/", "--")
    for base in [cache_root if root else None, cache_root / "hub" if hub else None]:
        if base is None:
            continue
        entry = base / entry_name
        (entry / "refs").mkdir(parents=True, exist_ok=True)
        snapshot_dir = entry / "snapshots" / "snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (entry / "refs" / "main").write_text("snapshot\n", encoding="utf-8")
        _write_hf_text_assets(snapshot_dir)
        _write_valid_safetensor(snapshot_dir / "model.safetensors")


def test_audit_full_matrix_reports_clean_when_all_methods_are_covered(tmp_path, monkeypatch) -> None:
    _prepare_runtime_upstreams(tmp_path, monkeypatch)

    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "suite_all_models_methods",
                "runs": [
                    {"run_id": "kgw", "profile": "suite_all_models_methods", "config": "tests/fixtures/configs/public_humaneval_plus.yaml", "resource": "cpu"},
                    {"run_id": "comment", "profile": "suite_all_models_methods", "config": "tests/fixtures/configs/public_humaneval_plus_comment.yaml", "resource": "cpu"},
                    {"run_id": "identifier", "profile": "suite_all_models_methods", "config": "tests/fixtures/configs/public_humaneval_plus_identifier.yaml", "resource": "cpu"},
                    {"run_id": "structural_flow", "profile": "suite_all_models_methods", "config": "tests/fixtures/configs/public_humaneval_plus_structural_flow.yaml", "resource": "cpu"},
                    {"run_id": "stone_runtime", "profile": "suite_all_models_methods", "config": "configs/public_humaneval_plus_stone_runtime.yaml", "resource": "gpu"},
                    {"run_id": "sweet_runtime", "profile": "suite_all_models_methods", "config": "configs/public_humaneval_plus_sweet_runtime.yaml", "resource": "gpu"},
                    {"run_id": "ewd_runtime", "profile": "suite_all_models_methods", "config": "configs/public_humaneval_plus_ewd_runtime.yaml", "resource": "gpu"},
                    {"run_id": "kgw_runtime", "profile": "suite_all_models_methods", "config": "configs/public_humaneval_plus_kgw_runtime.yaml", "resource": "gpu"},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "audit.json"
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--skip-hf-access",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(root / "scripts" / "audit_full_matrix.py"), run_name="__main__")
    assert exit_info.value.code == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "clean"
    assert payload["missing_methods"] == []
    assert all(item["status"] in {"ok", "skipped"} for item in payload["method_smoke"])


def test_runtime_hf_access_uses_watermark_cache_settings(tmp_path, monkeypatch) -> None:
    _prepare_runtime_upstreams(tmp_path, monkeypatch)

    cache_root = tmp_path / "hf_cache"
    runtime_cache = cache_root / "hub" / "models--bigcode--starcoder2-7b"
    (runtime_cache / "refs").mkdir(parents=True, exist_ok=True)
    (runtime_cache / "snapshots" / "snapshot").mkdir(parents=True, exist_ok=True)
    (runtime_cache / "refs" / "main").write_text("snapshot\n", encoding="utf-8")

    runtime_templates = {
        "stone_runtime": "configs/public_humaneval_plus_stone_runtime.yaml",
        "sweet_runtime": "configs/public_humaneval_plus_sweet_runtime.yaml",
        "ewd_runtime": "configs/public_humaneval_plus_ewd_runtime.yaml",
        "kgw_runtime": "configs/public_humaneval_plus_kgw_runtime.yaml",
    }
    runtime_configs: dict[str, Path] = {}
    root = Path(__file__).resolve().parents[1]
    for method, template in runtime_templates.items():
        source = json.loads((root / template).read_text(encoding="utf-8"))
        source["watermark"]["cache_dir"] = str(cache_root)
        source["watermark"]["local_files_only"] = True
        source["watermark"]["token_env"] = "HF_ACCESS_TOKEN"
        config_path = tmp_path / f"{method}.json"
        config_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
        runtime_configs[method] = config_path

    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "suite_all_models_methods",
                "runs": [
                    {"run_id": "kgw", "profile": "suite_all_models_methods", "config": "tests/fixtures/configs/public_humaneval_plus.yaml", "resource": "cpu"},
                    {"run_id": "comment", "profile": "suite_all_models_methods", "config": "tests/fixtures/configs/public_humaneval_plus_comment.yaml", "resource": "cpu"},
                    {"run_id": "identifier", "profile": "suite_all_models_methods", "config": "tests/fixtures/configs/public_humaneval_plus_identifier.yaml", "resource": "cpu"},
                    {"run_id": "structural_flow", "profile": "suite_all_models_methods", "config": "tests/fixtures/configs/public_humaneval_plus_structural_flow.yaml", "resource": "cpu"},
                    {"run_id": "stone_runtime", "profile": "suite_all_models_methods", "config": str(runtime_configs["stone_runtime"]), "resource": "gpu"},
                    {"run_id": "sweet_runtime", "profile": "suite_all_models_methods", "config": str(runtime_configs["sweet_runtime"]), "resource": "gpu"},
                    {"run_id": "ewd_runtime", "profile": "suite_all_models_methods", "config": str(runtime_configs["ewd_runtime"]), "resource": "gpu"},
                    {"run_id": "kgw_runtime", "profile": "suite_all_models_methods", "config": str(runtime_configs["kgw_runtime"]), "resource": "gpu"},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    module = runpy.run_path(str(root / "scripts" / "audit_full_matrix.py"))
    main = module["main"]
    output_path = tmp_path / "audit.json"
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--skip-provider-credentials",
        ],
    )

    assert main() == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    runtime_result = next(item for item in payload["hf_access"] if item["model"] == "bigcode/starcoder2-7b")
    assert runtime_result["accessible"] is True
    assert runtime_result["reason"] == "local_cache"
    assert runtime_result["local_files_only"] is True
    assert runtime_result["cache_dir"] == str(cache_root)
    assert runtime_result["requested_revision"] == "bb9afde76d7945da5745592525db122d4d729eb1"


def test_hf_access_probe_uses_pinned_revision_for_canonical_model(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root / "scripts" / "audit_full_matrix.py"))
    probe = module["_probe_hf_model"]
    seen: dict[str, str] = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=20.0):
        seen["url"] = request.full_url
        return _Response()

    monkeypatch.setattr(probe.__globals__["urllib"].request, "urlopen", fake_urlopen)

    payload = probe("Qwen/Qwen2.5-Coder-14B-Instruct", "token")

    assert payload["accessible"] is True
    assert payload["requested_revision"] == "aedcc2d42b622764e023cf882b6652e646b95671"
    assert seen["url"].endswith(
        "/Qwen/Qwen2.5-Coder-14B-Instruct/resolve/aedcc2d42b622764e023cf882b6652e646b95671/config.json"
    )


def test_local_hf_access_accepts_root_cache_even_when_hub_directory_exists(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "hf_cache"
    (cache_root / "hub").mkdir(parents=True, exist_ok=True)
    model_name = "acme/test-root-only-model"
    _write_hf_cache_entry(cache_root, model_name, root=True, hub=False)

    root = Path(__file__).resolve().parents[1]
    config_payload = json.loads((root / "tests" / "fixtures" / "configs" / "public_humaneval_plus_local_hf_qwen25_7b.yaml").read_text(encoding="utf-8"))
    config_payload["provider"]["parameters"]["model"] = model_name
    config_payload["provider"]["parameters"]["cache_dir"] = str(cache_root)
    config_payload["provider"]["parameters"]["local_files_only"] = True
    config_path = tmp_path / "root_only_local_hf.json"
    config_path.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")

    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "suite_canary_heavy",
                "required_watermark_methods": ["kgw"],
                "required_provider_modes": ["local_hf"],
                "required_gpu_pools": ["local_hf"],
                "runs": [
                    {
                        "run_id": "cert_local_model",
                        "profile": "suite_canary_heavy",
                        "config": str(config_path),
                        "resource": "gpu",
                        "gpu_pool": "local_hf",
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "audit.json"
    module = runpy.run_path(str(root / "scripts" / "audit_full_matrix.py"))
    main = module["main"]
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--profile",
            "suite_canary_heavy",
            "--output",
            str(output_path),
            "--skip-provider-credentials",
        ],
    )

    assert main() == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    result = payload["hf_access"][0]
    assert result["model"] == model_name
    assert result["accessible"] is True
    assert result["reason"] == "local_cache"


def test_audit_full_matrix_honors_manifest_required_coverage(tmp_path, monkeypatch) -> None:
    _prepare_runtime_upstreams(tmp_path, monkeypatch)

    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "model_invocation_smoke",
                "required_watermark_methods": ["kgw", "comment", "stone_runtime"],
                "required_provider_modes": ["offline_mock", "local_hf"],
                "required_gpu_pools": ["runtime", "local_hf"],
                "runs": [
                    {"run_id": "quick_cpu", "profile": "model_invocation_smoke", "config": "tests/fixtures/configs/public_humaneval_plus_comment.yaml", "resource": "cpu"},
                    {
                        "run_id": "quick_runtime",
                        "profile": "model_invocation_smoke",
                        "config": "configs/public_humaneval_plus_stone_runtime.yaml",
                        "resource": "gpu",
                        "gpu_pool": "runtime",
                    },
                    {
                        "run_id": "quick_model",
                        "profile": "model_invocation_smoke",
                        "config": "tests/fixtures/configs/public_humaneval_plus_local_hf_qwen25_7b.yaml",
                        "resource": "gpu",
                        "gpu_pool": "local_hf",
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "audit.json"
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--profile",
            "model_invocation_smoke",
            "--output",
            str(output_path),
            "--skip-hf-access",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(root / "scripts" / "audit_full_matrix.py"), run_name="__main__")
    assert exit_info.value.code == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "clean"
    assert payload["required_methods"] == ["kgw", "comment", "stone_runtime"]
    assert payload["missing_methods"] == []
    assert payload["required_provider_modes"] == ["offline_mock", "local_hf"]
    assert payload["missing_provider_modes"] == []
    assert payload["required_gpu_pools"] == ["runtime", "local_hf"]
    assert payload["missing_gpu_pools"] == []


def test_audit_full_matrix_smoke_scope_tracks_only_matrix_methods() -> None:
    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root / "scripts" / "audit_full_matrix.py"))
    smoke_methods = module["_smoke_methods_for_matrix"]

    methods = smoke_methods({"kgw", "comment", "stone_runtime"})

    assert methods == ["kgw", "comment", "stone_runtime"]


def test_merge_hf_model_rejects_conflicting_requirements() -> None:
    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root / "scripts" / "audit_full_matrix.py"))
    merge_hf_model = module["_merge_hf_model"]

    bucket: dict[str, dict[str, object]] = {}
    merge_hf_model(
        bucket,
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        revision="rev-a",
        cache_dir="model_cache/huggingface",
        local_files_only=True,
        token_env="HF_ACCESS_TOKEN",
        trust_remote_code=False,
        device="cuda",
        dtype="float16",
        usage="runtime",
        config_path=Path("configs/runtime.yaml"),
    )

    with pytest.raises(ValueError, match="HF model requirement conflict"):
        merge_hf_model(
            bucket,
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            revision="rev-b",
            cache_dir="model_cache/huggingface",
            local_files_only=True,
            token_env="HF_ACCESS_TOKEN",
            trust_remote_code=False,
            device="cuda",
            dtype="float16",
            usage="baseline_eval",
            config_path=Path("configs/baseline.yaml"),
        )


def test_runtime_smoke_representatives_select_one_run_per_method_model_with_source_preference() -> None:
    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root / "scripts" / "audit_full_matrix.py"))
    representatives = module["_runtime_smoke_representatives"]

    chosen = representatives(
        [
            {
                "run_id": "run-humaneval",
                "effective_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "benchmark_label": "HumanEval+",
                "source_slug": "humaneval_plus",
                "config": "humaneval.json",
            },
            {
                "run_id": "run-crafted",
                "effective_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "benchmark_label": "Crafted Original",
                "source_slug": "crafted_original",
                "config": "crafted.json",
            },
            {
                "run_id": "run-second-model",
                "effective_model": "deepseek-ai/deepseek-coder-6.7b-instruct",
                "benchmark_label": "MBXP-5lang",
                "source_slug": "mbxp_5lang",
                "config": "mbxp.json",
            },
        ]
    )

    assert [item["run_id"] for item in chosen] == ["run-crafted", "run-second-model"]
    assert [item["effective_model"] for item in chosen] == [
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
    ]


def test_runtime_checkout_failure_is_reported_once_per_method_and_skips_runtime_smoke(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "suite_all_models_methods",
                "runs": [
                    {
                        "run_id": "stone_qwen_humaneval",
                        "profile": "suite_all_models_methods",
                        "config": "configs/public_humaneval_plus_stone_runtime.yaml",
                        "config_overrides": {
                            "watermark": {"model_name": "Qwen/Qwen2.5-Coder-7B-Instruct", "max_new_tokens": 64}
                        },
                        "resource": "gpu",
                        "gpu_pool": "runtime",
                        "source_slug": "humaneval_plus",
                    },
                    {
                        "run_id": "stone_qwen_crafted",
                        "profile": "suite_all_models_methods",
                        "config": "configs/public_humaneval_plus_stone_runtime.yaml",
                        "config_overrides": {
                            "watermark": {"model_name": "Qwen/Qwen2.5-Coder-7B-Instruct", "max_new_tokens": 64}
                        },
                        "resource": "gpu",
                        "gpu_pool": "runtime",
                        "source_slug": "crafted_original",
                    },
                    {
                        "run_id": "stone_deepseek_mbxp",
                        "profile": "suite_all_models_methods",
                        "config": "configs/public_humaneval_plus_stone_runtime.yaml",
                        "config_overrides": {
                            "watermark": {"model_name": "deepseek-ai/deepseek-coder-6.7b-instruct", "max_new_tokens": 64}
                        },
                        "resource": "gpu",
                        "gpu_pool": "runtime",
                        "source_slug": "mbxp_5lang",
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    import posteditbench.baselines.stone_family.common as stone_common

    def fake_validate_checkout(method: str) -> list[str]:
        if method == "stone_runtime":
            return ["stone_runtime: checkout invalid"]
        return []

    monkeypatch.setattr(stone_common, "validate_checkout", fake_validate_checkout)

    module = runpy.run_path(str(root / "scripts" / "audit_full_matrix.py"))
    main = module["main"]
    main.__globals__["_runtime_smoke"] = lambda _run_item, *, timeout_seconds: (_ for _ in ()).throw(
        AssertionError("runtime smoke should be skipped")
    )
    output_path = tmp_path / "audit.json"
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--profile",
            "suite_all_models_methods",
            "--output",
            str(output_path),
            "--runtime-smoke",
            "--skip-hf-access",
            "--skip-provider-credentials",
        ],
    )

    assert main() == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["runtime_smoke_scope"] == "method_x_model"
    assert payload["issues"].count("stone_runtime: checkout invalid") == 1
    skipped = [item for item in payload["method_smoke"] if item["method"] == "stone_runtime"]
    assert len(skipped) == 2
    assert all(item["status"] == "skipped" for item in skipped)
    assert all(item["reason"] == "checkout_validation_failed" for item in skipped)


def test_runtime_smoke_reports_representative_effective_model(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root / "scripts" / "audit_full_matrix.py"))
    runtime_smoke = module["_runtime_smoke"]

    class _Completed:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(*args, **kwargs):
        report_path = Path(kwargs["env"]["POSTEDITBENCH_AUDIT_REPORT_PATH"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"rows": [{"task_id": "fixture"}]}), encoding="utf-8")
        return _Completed()

    monkeypatch.chdir(root)
    monkeypatch.setattr(runtime_smoke.__globals__["subprocess"], "run", fake_run)
    monkeypatch.setitem(runtime_smoke.__globals__, "ROOT", tmp_path)
    monkeypatch.setitem(runtime_smoke.__globals__, "load_json", lambda _path: {"rows": [{"task_id": "fixture"}]})

    payload = runtime_smoke(
        {
            "config": "configs/public_humaneval_plus_stone_runtime.yaml",
            "run_id": "suite_qwen25_7b_crafted_original_stone_runtime",
            "effective_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        },
        timeout_seconds=123,
    )

    assert payload["status"] == "ok"
    assert payload["effective_model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"


def test_runtime_smoke_passes_timeout_to_subprocess(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root / "scripts" / "audit_full_matrix.py"))
    runtime_smoke = module["_runtime_smoke"]
    seen: dict[str, object] = {}

    class _Completed:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(*args, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        report_path = Path(kwargs["env"]["POSTEDITBENCH_AUDIT_REPORT_PATH"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"rows": [{"task_id": "fixture"}]}), encoding="utf-8")
        return _Completed()

    monkeypatch.chdir(root)
    monkeypatch.setattr(runtime_smoke.__globals__["subprocess"], "run", fake_run)
    monkeypatch.setitem(runtime_smoke.__globals__, "ROOT", tmp_path)
    monkeypatch.setitem(runtime_smoke.__globals__, "load_json", lambda _path: {"rows": [{"task_id": "fixture"}]})

    payload = runtime_smoke(
        {
            "config": "configs/public_humaneval_plus_stone_runtime.yaml",
            "run_id": "suite_qwen25_7b_crafted_original_stone_runtime",
            "effective_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        },
        timeout_seconds=456,
    )

    assert payload["status"] == "ok"
    assert seen["timeout"] == 456


def test_audit_full_matrix_strict_cache_rejects_hub_only_entries_when_root_entry_is_missing(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "hf_cache"
    model_name = "acme/test-local-model"
    _write_hf_cache_entry(cache_root, model_name, root=False, hub=True)

    root = Path(__file__).resolve().parents[1]
    config_payload = json.loads((root / "tests" / "fixtures" / "configs" / "public_humaneval_plus_local_hf_qwen25_7b.yaml").read_text(encoding="utf-8"))
    config_payload["provider"]["parameters"]["model"] = model_name
    config_payload["provider"]["parameters"]["cache_dir"] = str(cache_root)
    config_payload["provider"]["parameters"]["local_files_only"] = True
    config_path = tmp_path / "strict_local_hf.json"
    config_path.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")

    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "suite_canary_heavy",
                "required_watermark_methods": ["kgw"],
                "required_provider_modes": ["local_hf"],
                "required_gpu_pools": ["local_hf"],
                "runs": [
                    {
                        "run_id": "cert_local_model",
                        "profile": "suite_canary_heavy",
                        "config": str(config_path),
                        "resource": "gpu",
                        "gpu_pool": "local_hf",
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "audit.json"
    module = runpy.run_path(str(root / "scripts" / "audit_full_matrix.py"))
    main = module["main"]
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--profile",
            "suite_canary_heavy",
            "--output",
            str(output_path),
            "--strict-hf-cache",
            "--skip-hf-access",
            "--skip-provider-credentials",
        ],
    )

    assert main() == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "has_issues"
    assert payload["required_hf_models"] == [model_name]
    assert payload["hf_cache_validation"][0]["status"] == "failed"
    assert any("missing official cache entry" in issue for issue in payload["hf_cache_validation"][0]["issues"])


def test_audit_full_matrix_strict_cache_collects_runtime_and_local_hf_models(tmp_path, monkeypatch) -> None:
    _prepare_runtime_upstreams(tmp_path, monkeypatch)

    cache_root = tmp_path / "hf_cache"
    local_model = "acme/test-local-model"
    runtime_model = "acme/test-runtime-model"
    _write_hf_cache_entry(cache_root, local_model, root=True, hub=False)
    _write_hf_cache_entry(cache_root, runtime_model, root=True, hub=False)

    root = Path(__file__).resolve().parents[1]
    local_payload = json.loads((root / "tests" / "fixtures" / "configs" / "public_humaneval_plus_local_hf_qwen25_7b.yaml").read_text(encoding="utf-8"))
    local_payload["provider"]["parameters"]["model"] = local_model
    local_payload["provider"]["parameters"]["cache_dir"] = str(cache_root)
    local_payload["provider"]["parameters"]["local_files_only"] = True
    local_config = tmp_path / "strict_local_hf.json"
    local_config.write_text(json.dumps(local_payload, indent=2) + "\n", encoding="utf-8")

    runtime_payload = json.loads((root / "configs" / "public_humaneval_plus_stone_runtime.yaml").read_text(encoding="utf-8"))
    runtime_payload["watermark"]["model_name"] = runtime_model
    runtime_payload["watermark"]["cache_dir"] = str(cache_root)
    runtime_payload["watermark"]["local_files_only"] = True
    runtime_config = tmp_path / "strict_runtime.json"
    runtime_config.write_text(json.dumps(runtime_payload, indent=2) + "\n", encoding="utf-8")

    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "suite_canary_heavy",
                "required_watermark_methods": ["kgw", "stone_runtime"],
                "required_provider_modes": ["offline_mock", "local_hf"],
                "required_gpu_pools": ["runtime", "local_hf"],
                "runs": [
                    {
                        "run_id": "cert_runtime",
                        "profile": "suite_canary_heavy",
                        "config": str(runtime_config),
                        "resource": "gpu",
                        "gpu_pool": "runtime",
                    },
                    {
                        "run_id": "cert_local_model",
                        "profile": "suite_canary_heavy",
                        "config": str(local_config),
                        "resource": "gpu",
                        "gpu_pool": "local_hf",
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "audit.json"
    module = runpy.run_path(str(root / "scripts" / "audit_full_matrix.py"))
    main = module["main"]
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--profile",
            "suite_canary_heavy",
            "--output",
            str(output_path),
            "--strict-hf-cache",
            "--skip-hf-access",
            "--skip-provider-credentials",
        ],
    )

    assert main() == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "clean"
    assert payload["required_hf_models"] == sorted([local_model, runtime_model])
    assert {item["model"] for item in payload["hf_cache_validation"]} == {local_model, runtime_model}
    assert all(item["status"] == "ok" for item in payload["hf_cache_validation"])


def test_audit_full_matrix_uses_manifest_config_overrides_for_model_collection(tmp_path, monkeypatch) -> None:
    _prepare_runtime_upstreams(tmp_path, monkeypatch)

    cache_root = tmp_path / "hf_cache"
    runtime_model_one = "acme/override-runtime-model-one"
    runtime_model_two = "acme/override-runtime-model-two"
    _write_hf_cache_entry(cache_root, runtime_model_one, root=True, hub=False)
    _write_hf_cache_entry(cache_root, runtime_model_two, root=True, hub=False)

    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "suite_all_models_methods",
                "required_watermark_methods": ["stone_runtime", "sweet_runtime"],
                "required_provider_modes": ["offline_mock"],
                "required_gpu_pools": ["runtime"],
                "runs": [
                    {
                        "run_id": "suite_runtime_stone",
                        "profile": "suite_all_models_methods",
                        "config": "configs/public_humaneval_plus_stone_runtime.yaml",
                        "config_overrides": {
                            "watermark": {
                                "model_name": runtime_model_one,
                                "cache_dir": str(cache_root),
                                "local_files_only": True,
                            },
                        },
                        "resource": "gpu",
                        "gpu_pool": "runtime",
                    },
                    {
                        "run_id": "suite_runtime_sweet",
                        "profile": "suite_all_models_methods",
                        "config": "configs/public_humaneval_plus_sweet_runtime.yaml",
                        "config_overrides": {
                            "watermark": {
                                "model_name": runtime_model_two,
                                "cache_dir": str(cache_root),
                                "local_files_only": True,
                            }
                        },
                        "resource": "gpu",
                        "gpu_pool": "runtime",
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "audit.json"
    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root / "scripts" / "audit_full_matrix.py"))
    main = module["main"]
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--profile",
            "suite_all_models_methods",
            "--output",
            str(output_path),
            "--strict-hf-cache",
            "--skip-hf-access",
            "--skip-provider-credentials",
        ],
    )

    assert main() == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "clean"
    assert payload["required_hf_models"] == sorted([runtime_model_one, runtime_model_two])


def test_audit_full_matrix_model_load_smoke_covers_runtime_models(tmp_path, monkeypatch) -> None:
    _prepare_runtime_upstreams(tmp_path, monkeypatch)

    cache_root = tmp_path / "hf_cache"
    local_model = "acme/test-local-model"
    runtime_model = "acme/test-runtime-model"
    _write_hf_cache_entry(cache_root, local_model, root=True, hub=False)
    _write_hf_cache_entry(cache_root, runtime_model, root=True, hub=False)

    root = Path(__file__).resolve().parents[1]
    local_payload = json.loads((root / "tests" / "fixtures" / "configs" / "public_humaneval_plus_local_hf_qwen25_7b.yaml").read_text(encoding="utf-8"))
    local_payload["provider"]["parameters"]["model"] = local_model
    local_payload["provider"]["parameters"]["cache_dir"] = str(cache_root)
    local_payload["provider"]["parameters"]["local_files_only"] = True
    local_config = tmp_path / "smoke_local_hf.json"
    local_config.write_text(json.dumps(local_payload, indent=2) + "\n", encoding="utf-8")

    runtime_payload = json.loads((root / "configs" / "public_humaneval_plus_stone_runtime.yaml").read_text(encoding="utf-8"))
    runtime_payload["watermark"]["model_name"] = runtime_model
    runtime_payload["watermark"]["cache_dir"] = str(cache_root)
    runtime_payload["watermark"]["local_files_only"] = True
    runtime_config = tmp_path / "smoke_runtime.json"
    runtime_config.write_text(json.dumps(runtime_payload, indent=2) + "\n", encoding="utf-8")

    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "suite_canary_heavy",
                "required_watermark_methods": ["kgw", "stone_runtime"],
                "required_provider_modes": ["offline_mock", "local_hf"],
                "required_gpu_pools": ["runtime", "local_hf"],
                "runs": [
                    {
                        "run_id": "cert_runtime",
                        "profile": "suite_canary_heavy",
                        "config": str(runtime_config),
                        "resource": "gpu",
                        "gpu_pool": "runtime",
                    },
                    {
                        "run_id": "cert_local_model",
                        "profile": "suite_canary_heavy",
                        "config": str(local_config),
                        "resource": "gpu",
                        "gpu_pool": "local_hf",
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "audit.json"
    module = runpy.run_path(str(root / "scripts" / "audit_full_matrix.py"))
    main = module["main"]
    smoke_calls: list[str] = []
    evaluator_calls: list[str] = []

    def fake_model_smoke(requirement):
        smoke_calls.append(requirement.model)
        return {"model": requirement.model, "status": "ok", "issues": []}

    def fake_evaluator_smoke(requirement):
        evaluator_calls.append(requirement.model)
        return {"model": requirement.model, "status": "ok", "issues": []}

    main.__globals__["smoke_load_local_hf_model"] = fake_model_smoke
    main.__globals__["smoke_load_local_hf_evaluator"] = fake_evaluator_smoke

    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--profile",
            "suite_canary_heavy",
            "--output",
            str(output_path),
            "--strict-hf-cache",
            "--model-load-smoke",
            "--skip-hf-access",
            "--skip-provider-credentials",
        ],
    )

    assert main() == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "clean"
    assert set(smoke_calls) == {local_model, runtime_model}
    assert set(evaluator_calls) == {local_model, runtime_model}
    assert {item["model"] for item in payload["hf_model_smoke"]} == {local_model, runtime_model}
    assert {item["model"] for item in payload["hf_evaluator_smoke"]} == {local_model, runtime_model}


def test_audit_full_matrix_includes_baseline_eval_model_in_hf_checks(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "hf_cache"
    local_model = "acme/test-local-model"
    baseline_model = "bigcode/starcoder2-7b"
    _write_hf_cache_entry(cache_root, local_model, root=True, hub=False)
    baseline_revision = "bb9afde76d7945da5745592525db122d4d729eb1"
    baseline_entry = cache_root / "models--bigcode--starcoder2-7b"
    (baseline_entry / "refs").mkdir(parents=True, exist_ok=True)
    baseline_snapshot_dir = baseline_entry / "snapshots" / baseline_revision
    baseline_snapshot_dir.mkdir(parents=True, exist_ok=True)
    (baseline_entry / "refs" / "main").write_text(baseline_revision + "\n", encoding="utf-8")
    _write_hf_text_assets(baseline_snapshot_dir)
    _write_valid_safetensor(baseline_snapshot_dir / "model.safetensors")

    root = Path(__file__).resolve().parents[1]
    config_payload = json.loads(
        (root / "tests" / "fixtures" / "configs" / "public_humaneval_plus_local_hf_qwen25_7b.yaml").read_text(encoding="utf-8")
    )
    config_payload["provider"]["parameters"]["model"] = local_model
    config_payload["provider"]["parameters"]["cache_dir"] = str(cache_root)
    config_payload["provider"]["parameters"]["local_files_only"] = True
    config_path = tmp_path / "baseline_eval_local_hf.json"
    config_path.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")

    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "suite_canary_heavy",
                "required_watermark_methods": ["kgw"],
                "required_provider_modes": ["local_hf"],
                "required_gpu_pools": ["local_hf"],
                "runs": [
                    {
                        "run_id": "cert_local_model",
                        "profile": "suite_canary_heavy",
                        "config": str(config_path),
                        "resource": "gpu",
                        "gpu_pool": "local_hf",
                        "baseline_eval": True,
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "audit.json"
    module = runpy.run_path(str(root / "scripts" / "audit_full_matrix.py"))
    main = module["main"]
    requirement_cls = module["HFModelRequirement"]
    smoke_calls: list[str] = []
    evaluator_calls: list[str] = []

    def fake_model_smoke(requirement):
        smoke_calls.append(requirement.model)
        return {"model": requirement.model, "status": "ok", "issues": []}

    def fake_evaluator_smoke(requirement):
        evaluator_calls.append(requirement.model)
        return {"model": requirement.model, "status": "ok", "issues": []}

    def fake_baseline_requirement(*, metadata=None, device="cuda", usage=("baseline_eval", "evaluator"), **_kwargs):
        return requirement_cls(
            model=baseline_model,
            revision=baseline_revision,
            cache_dir=str(cache_root),
            local_files_only=True,
            trust_remote_code=False,
            device=device,
            dtype="float16",
            token_env="HF_ACCESS_TOKEN",
            usage=usage,
            config_paths=("scripts/evaluate_baseline_family.py",),
        )

    main.__globals__["smoke_load_local_hf_model"] = fake_model_smoke
    main.__globals__["smoke_load_local_hf_evaluator"] = fake_evaluator_smoke
    main.__globals__["derived_baseline_eval_requirement"] = fake_baseline_requirement

    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--profile",
            "suite_canary_heavy",
            "--output",
            str(output_path),
            "--strict-hf-cache",
            "--model-load-smoke",
            "--skip-hf-access",
            "--skip-provider-credentials",
        ],
    )

    assert main() == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "clean"
    assert payload["required_hf_models"] == sorted([baseline_model, local_model])
    assert set(smoke_calls) == {baseline_model, local_model}
    assert set(evaluator_calls) == {baseline_model, local_model}


def test_audit_full_matrix_enforces_all_model_fairness_slices(tmp_path, monkeypatch) -> None:
    _prepare_runtime_upstreams(tmp_path, monkeypatch)

    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "suite_all_models_methods",
                "model_roster": [
                    "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "bigcode/starcoder2-7b",
                ],
                "benchmark_roster": ["HumanEval+"],
                "required_watermark_methods": ["stone_runtime", "sweet_runtime"],
                "required_provider_modes": ["offline_mock"],
                "required_gpu_pools": ["runtime"],
                "runs": [
                    {
                        "run_id": "suite_qwen_stone",
                        "profile": "suite_all_models_methods",
                        "config": "configs/public_humaneval_plus_stone_runtime.yaml",
                        "config_overrides": {
                            "watermark": {"model_name": "Qwen/Qwen2.5-Coder-7B-Instruct"},
                        },
                        "resource": "gpu",
                        "gpu_pool": "runtime",
                    },
                    {
                        "run_id": "suite_qwen_sweet",
                        "profile": "suite_all_models_methods",
                        "config": "configs/public_humaneval_plus_sweet_runtime.yaml",
                        "config_overrides": {
                            "watermark": {"model_name": "Qwen/Qwen2.5-Coder-7B-Instruct"},
                        },
                        "resource": "gpu",
                        "gpu_pool": "runtime",
                    },
                    {
                        "run_id": "suite_starcoder_stone",
                        "profile": "suite_all_models_methods",
                        "config": "configs/public_humaneval_plus_stone_runtime.yaml",
                        "config_overrides": {
                            "watermark": {"model_name": "bigcode/starcoder2-7b"},
                        },
                        "resource": "gpu",
                        "gpu_pool": "runtime",
                    },
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "audit.json"
    root = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(root / "scripts" / "audit_full_matrix.py"))
    main = module["main"]
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_full_matrix.py",
            "--manifest",
            str(manifest_path),
            "--profile",
            "suite_all_models_methods",
            "--output",
            str(output_path),
            "--skip-hf-access",
            "--skip-provider-credentials",
        ],
    )

    assert main() == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "has_issues"
    assert payload["missing_model_roster"] == []
    assert payload["missing_benchmark_roster"] == []
    assert payload["missing_slice_methods"] == [
        {
            "model": "bigcode/starcoder2-7b",
            "benchmark": "HumanEval+",
            "missing_methods": ["sweet_runtime"],
        }
    ]

