from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import validate_setup as article_validate

from tests._stone_test_helpers import (
    DEFAULT_REPO_URL,
    create_runtime_checkout,
    create_stone_checkout,
)


def _write_valid_safetensor(path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"weight": torch.zeros(1)}, str(path))


def _write_hf_cache_entry(
    cache_root: Path,
    model_name: str,
    *,
    root: bool,
    hub: bool,
    revision: str = "snapshot",
) -> None:
    entry_name = "models--" + model_name.replace("/", "--")
    for base in [cache_root if root else None, cache_root / "hub" if hub else None]:
        if base is None:
            continue
        entry = base / entry_name
        (entry / "refs").mkdir(parents=True, exist_ok=True)
        snapshot_dir = entry / "snapshots" / revision
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (entry / "refs" / "main").write_text(f"{revision}\n", encoding="utf-8")
        (snapshot_dir / "config.json").write_text('{"architectures":["MockModel"]}\n', encoding="utf-8")
        (snapshot_dir / "tokenizer_config.json").write_text('{"model_max_length":512}\n', encoding="utf-8")
        (snapshot_dir / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
        _write_valid_safetensor(snapshot_dir / "model.safetensors")


def _write_minimal_validation_fixture(tmp_path: Path) -> tuple[Path, Path]:
    benchmark_path = tmp_path / "benchmark.normalized.jsonl"
    benchmark_path.write_text(
        json.dumps(
            {
                "task_id": "fixture-1",
                "language": "python",
                "prompt": "Write a function that returns 1.",
                "reference_solution": "def solve():\n    return 1\n",
                "reference_kind": "canonical",
                "source_group": "fixture",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = benchmark_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_count": 1,
                "observed_languages": ["python"],
                "reference_kind_counts": {"canonical": 1},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    attack_matrix = tmp_path / "attack_matrix.json"
    attack_matrix.write_text(
        json.dumps({"attacks": [{"name": "comment_strip"}]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return benchmark_path, attack_matrix


def test_benchmark_content_scan_flags_non_public_email(tmp_path):
    benchmark_path = tmp_path / "data" / "fixtures" / "benchmark.jsonl"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "task_id": "fixture-1",
        "language": "python",
        "prompt": "Contact the maintainer at leak@example.invalid for help.",
        "reference_solution": "def solve():\n    return 1\n",
    }
    benchmark_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    findings = article_validate.scan_for_identity_markers_in_benchmark(benchmark_path)

    assert any("matched email" in finding for finding in findings)


def test_validate_setup_defaults_to_canonical_public_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["validate_setup.py"])

    args = article_validate.parse_args()

    assert args.config == Path("configs/default.yaml")


def test_benchmark_content_scan_keeps_public_email_whitelist_but_flags_metadata(tmp_path):
    benchmark_path = tmp_path / "data" / "public" / "snapshot.jsonl"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "task_id": "public-1",
        "language": "python",
        "prompt": "Validate emails like person@example.com in user input.",
        "reference_solution": "def solve():\n    return True\n",
        "source_path": "C:" + r"\Users\alice\artifact\source.jsonl",
    }
    benchmark_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    findings = article_validate.scan_for_identity_markers_in_benchmark(benchmark_path)

    assert any("matched personal_path" in finding for finding in findings)
    assert not any("matched email" in finding for finding in findings)


def test_stone_checkout_metadata_reports_clean_vendored_checkout(monkeypatch, tmp_path):
    checkout, manifest, commit = create_stone_checkout(tmp_path)
    monkeypatch.setenv("POSTEDITBENCH_STONE_UPSTREAM_ROOT", str(checkout))
    monkeypatch.setenv("POSTEDITBENCH_STONE_UPSTREAM_MANIFEST", str(manifest))

    provenance = article_validate.stone_family_checkout_metadata()

    assert provenance["origin"] == "vendored_snapshot"
    assert provenance["bundle_eligible"] is True
    assert provenance["checkout_present"] is True
    assert provenance["checkout_valid"] is True
    assert provenance["checkout_issues"] == []
    assert provenance["manifest_repo_url"] == DEFAULT_REPO_URL
    assert provenance["manifest_pinned_commit"] == commit
    assert provenance["manifest_license_status"] == "redistributable"
    assert provenance["repo_url"] == DEFAULT_REPO_URL
    assert provenance["remote_url"] == DEFAULT_REPO_URL
    assert provenance["upstream_commit"] == commit
    assert provenance["license_status"] == "redistributable"


def test_stone_checkout_validation_reports_remote_mismatch(monkeypatch, tmp_path):
    checkout, manifest, _ = create_stone_checkout(tmp_path)
    monkeypatch.setenv("POSTEDITBENCH_STONE_UPSTREAM_ROOT", str(checkout))
    monkeypatch.setenv("POSTEDITBENCH_STONE_UPSTREAM_MANIFEST", str(manifest))

    subprocess.run(["git", "-C", str(checkout), "remote", "set-url", "origin", "https://example.invalid/STONE.git"], check=True)

    issues = article_validate.validate_stone_checkout()

    assert any("origin remote mismatch" in issue for issue in issues)
    provenance = article_validate.stone_family_checkout_metadata()
    assert provenance["checkout_valid"] is False
    assert any("origin remote mismatch" in issue for issue in provenance["checkout_issues"])


def test_stone_checkout_validation_reports_dirty_worktree(monkeypatch, tmp_path):
    checkout, manifest, _ = create_stone_checkout(tmp_path, dirty=True)
    monkeypatch.setenv("POSTEDITBENCH_STONE_UPSTREAM_ROOT", str(checkout))
    monkeypatch.setenv("POSTEDITBENCH_STONE_UPSTREAM_MANIFEST", str(manifest))

    issues = article_validate.validate_stone_checkout()

    assert any("uncommitted changes" in issue for issue in issues)


@pytest.mark.parametrize(
    ("method", "root_env", "manifest_env"),
    [
        ("sweet_runtime", "POSTEDITBENCH_SWEET_UPSTREAM_ROOT", "POSTEDITBENCH_SWEET_UPSTREAM_MANIFEST"),
        ("ewd_runtime", "POSTEDITBENCH_EWD_UPSTREAM_ROOT", "POSTEDITBENCH_EWD_UPSTREAM_MANIFEST"),
        ("kgw_runtime", "POSTEDITBENCH_KGW_UPSTREAM_ROOT", "POSTEDITBENCH_KGW_UPSTREAM_MANIFEST"),
    ],
)
def test_runtime_checkout_ignores_generated_python_cache_noise(
    monkeypatch,
    tmp_path,
    method: str,
    root_env: str,
    manifest_env: str,
):
    checkout, manifest, _ = create_runtime_checkout(tmp_path, method)
    pycache = checkout / "__pycache__"
    pycache.mkdir(parents=True, exist_ok=True)
    (pycache / "runtime.cpython-312.pyc").write_bytes(b"fixture-pyc")
    monkeypatch.setenv(root_env, str(checkout))
    monkeypatch.setenv(manifest_env, str(manifest))

    provenance = article_validate.stone_family_checkout_metadata(method)
    issues = article_validate.validate_stone_checkout(method)

    assert provenance["checkout_valid"] is True
    assert provenance["dirty"] is False
    assert provenance["checkout_issues"] == []
    assert issues == []


def test_stone_checkout_validation_requires_license_for_vendored_snapshot(monkeypatch, tmp_path):
    checkout, manifest, _ = create_stone_checkout(tmp_path, include_license=False, license_status="unverified")
    monkeypatch.setenv("POSTEDITBENCH_STONE_UPSTREAM_ROOT", str(checkout))
    monkeypatch.setenv("POSTEDITBENCH_STONE_UPSTREAM_MANIFEST", str(manifest))

    issues = article_validate.validate_stone_checkout()

    assert any("missing LICENSE/COPYING" in issue for issue in issues)


def test_stone_checkout_validation_prefers_valid_external_checkout_when_vendored_placeholder_is_invalid(monkeypatch, tmp_path):
    vendored_root = tmp_path / "third_party" / "STONE-watermarking" / "stone_implementation"
    vendored_root.mkdir(parents=True, exist_ok=True)
    (vendored_root / "__init__.py").write_text("", encoding="utf-8")
    external_checkout, manifest, commit = create_stone_checkout(
        tmp_path,
        relative_path=".coordination/external/STONE-watermarking.gitcheckout",
        include_license=False,
        license_status="unverified",
    )
    monkeypatch.delenv("POSTEDITBENCH_STONE_UPSTREAM_ROOT", raising=False)
    monkeypatch.setenv("POSTEDITBENCH_STONE_UPSTREAM_MANIFEST", str(manifest))
    monkeypatch.setattr("posteditbench.baselines.stone_family.common._workspace_root", lambda: tmp_path)

    provenance = article_validate.stone_family_checkout_metadata()
    issues = article_validate.validate_stone_checkout()

    assert provenance["origin"] == "external_checkout"
    assert provenance["checkout_valid"] is True
    assert provenance["upstream_commit"] == commit
    assert issues == []
    assert str(external_checkout) in provenance["repo_root"]
def test_validate_setup_local_hf_accepts_root_cache_layout(tmp_path, monkeypatch):
    checkout, manifest, _ = create_stone_checkout(tmp_path)
    monkeypatch.setenv("POSTEDITBENCH_STONE_UPSTREAM_ROOT", str(checkout))
    monkeypatch.setenv("POSTEDITBENCH_STONE_UPSTREAM_MANIFEST", str(manifest))
    benchmark_path, attack_matrix = _write_minimal_validation_fixture(tmp_path)
    cache_root = tmp_path / "hf_cache"
    model_name = "acme/test-local-model"
    _write_hf_cache_entry(cache_root, model_name, root=True, hub=False)
    _write_hf_cache_entry(
        cache_root,
        "bigcode/starcoder2-7b",
        root=True,
        hub=False,
        revision=article_validate.baseline_eval_requirement(
            model_name="bigcode/starcoder2-7b",
            token_env="HF_ACCESS_TOKEN",
            device="cuda",
            cache_dir=str(cache_root),
            local_files_only=True,
            trust_remote_code=False,
        ).revision,
    )

    config_path = tmp_path / "local_hf.json"
    config_path.write_text(
        json.dumps(
            {
                "project": {"name": "strict-local-hf", "seed": 17},
                "benchmark": {
                    "prepared_output": str(benchmark_path),
                    "source": str(benchmark_path),
                    "limit": 1,
                    "include_reference_kinds": ["canonical"],
                    "languages": ["python"],
                },
                "provider": {
                    "mode": "local_hf",
                    "parameters": {
                        "model": model_name,
                        "cache_dir": str(cache_root),
                        "local_files_only": True,
                        "device": "cuda",
                        "dtype": "float16",
                    },
                },
                "watermark": {
                    "scheme": "stone_runtime",
                    "strength": 0.55,
                    "model_name": model_name,
                    "cache_dir": str(cache_root),
                    "local_files_only": True,
                },
                "attacks": {"include": ["comment_strip"]},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_setup.py",
            "--config",
            str(config_path),
            "--attack-matrix",
            str(attack_matrix),
            "--report",
            str(report_path),
        ],
    )
    monkeypatch.setattr(article_validate, "smoke_load_local_hf_model", lambda requirement: {"model": requirement.model, "status": "ok", "issues": []})
    monkeypatch.setattr(
        article_validate,
        "smoke_load_local_hf_evaluator",
        lambda requirement: {"model": requirement.model, "status": "ok", "issues": []},
    )

    assert article_validate.main() == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["hf_cache_validation"]["status"] == "ok"
    assert payload["hf_cache_validation"]["issues"] == []


def test_validate_setup_runtime_accepts_root_cache_layout(tmp_path, monkeypatch):
    checkout, manifest, _ = create_stone_checkout(tmp_path)
    monkeypatch.setenv("POSTEDITBENCH_STONE_UPSTREAM_ROOT", str(checkout))
    monkeypatch.setenv("POSTEDITBENCH_STONE_UPSTREAM_MANIFEST", str(manifest))

    benchmark_path, attack_matrix = _write_minimal_validation_fixture(tmp_path)
    cache_root = tmp_path / "hf_cache"
    model_name = "acme/test-runtime-model"
    _write_hf_cache_entry(cache_root, model_name, root=True, hub=False)
    _write_hf_cache_entry(
        cache_root,
        "bigcode/starcoder2-7b",
        root=True,
        hub=False,
        revision=article_validate.baseline_eval_requirement(
            model_name="bigcode/starcoder2-7b",
            token_env="HF_ACCESS_TOKEN",
            device="cuda",
            cache_dir=str(cache_root),
            local_files_only=True,
            trust_remote_code=False,
        ).revision,
    )

    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps(
            {
                "project": {"name": "strict-runtime", "seed": 17},
                "benchmark": {
                    "prepared_output": str(benchmark_path),
                    "source": str(benchmark_path),
                    "limit": 1,
                    "include_reference_kinds": ["canonical"],
                    "languages": ["python"],
                },
                "provider": {"mode": "offline_mock", "parameters": {}},
                "watermark": {
                    "scheme": "stone_runtime",
                    "strength": 0.6,
                    "model_name": model_name,
                    "cache_dir": str(cache_root),
                    "local_files_only": True,
                    "token_env": "HF_ACCESS_TOKEN",
                    "trust_remote_code": True,
                    "device": "cuda",
                },
                "attacks": {"include": ["comment_strip"]},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_setup.py",
            "--config",
            str(config_path),
            "--attack-matrix",
            str(attack_matrix),
            "--report",
            str(report_path),
        ],
    )
    monkeypatch.setattr(article_validate, "smoke_load_local_hf_model", lambda requirement: {"model": requirement.model, "status": "ok", "issues": []})
    monkeypatch.setattr(
        article_validate,
        "smoke_load_local_hf_evaluator",
        lambda requirement: {"model": requirement.model, "status": "ok", "issues": []},
    )

    assert article_validate.main() == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["hf_cache_validation"]["status"] == "ok"
    assert payload["hf_cache_validation"]["issues"] == []


def test_validate_setup_fails_when_offline_evaluator_smoke_fails(tmp_path, monkeypatch):
    checkout, manifest, _ = create_stone_checkout(tmp_path)
    monkeypatch.setenv("POSTEDITBENCH_STONE_UPSTREAM_ROOT", str(checkout))
    monkeypatch.setenv("POSTEDITBENCH_STONE_UPSTREAM_MANIFEST", str(manifest))
    benchmark_path, attack_matrix = _write_minimal_validation_fixture(tmp_path)
    cache_root = tmp_path / "hf_cache"
    model_name = "acme/test-runtime-model"
    _write_hf_cache_entry(cache_root, model_name, root=True, hub=False)
    _write_hf_cache_entry(
        cache_root,
        "bigcode/starcoder2-7b",
        root=True,
        hub=False,
        revision=article_validate.canonical_baseline_eval_requirement(
            device="cuda",
            token_env="HF_ACCESS_TOKEN",
            cache_dir=str(cache_root),
            local_files_only=True,
            trust_remote_code=False,
        ).revision,
    )

    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps(
            {
                "project": {"name": "strict-runtime", "seed": 17},
                "benchmark": {
                    "prepared_output": str(benchmark_path),
                    "source": str(benchmark_path),
                    "limit": 1,
                    "include_reference_kinds": ["canonical"],
                    "languages": ["python"],
                },
                "provider": {"mode": "offline_mock", "parameters": {}},
                "watermark": {
                    "scheme": "stone_runtime",
                    "strength": 0.6,
                    "model_name": model_name,
                    "cache_dir": str(cache_root),
                    "local_files_only": True,
                    "token_env": "HF_ACCESS_TOKEN",
                    "trust_remote_code": True,
                    "device": "cuda",
                },
                "attacks": {"include": ["comment_strip"]},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_setup.py",
            "--config",
            str(config_path),
            "--attack-matrix",
            str(attack_matrix),
            "--report",
            str(report_path),
        ],
    )
    monkeypatch.setattr(article_validate, "smoke_load_local_hf_model", lambda requirement: {"model": requirement.model, "status": "ok", "issues": []})
    monkeypatch.setattr(
        article_validate,
        "smoke_load_local_hf_evaluator",
        lambda requirement: {"model": requirement.model, "status": "failed", "issues": ["forced evaluator failure"]},
    )

    assert article_validate.main() == 1
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["hf_cache_validation"]["status"] == "failed"
    assert any("offline evaluator smoke" in issue for issue in payload["hf_cache_validation"]["issues"])


def test_mbxp_only_declared_languages_match_canonical_fixture() -> None:
    config = article_validate.load_config(Path("configs/mbxp_only.yaml"))
    benchmark_path = Path(config["benchmark"]["prepared_output"])
    manifest = article_validate.load_json(benchmark_path.with_suffix(".manifest.json"))

    claimed = [str(language).lower() for language in config["benchmark"]["languages"]]
    observed = [str(language).lower() for language in manifest["observed_languages"]]

    missing_claimed = [language for language in claimed if language not in observed]
    assert missing_claimed == []


def test_release_candidate_files_include_release_sources_and_license() -> None:
    candidates = article_validate.release_candidate_files()

    assert Path("LICENSE") in candidates
    assert article_validate.DATA_DIR / "release" in candidates
