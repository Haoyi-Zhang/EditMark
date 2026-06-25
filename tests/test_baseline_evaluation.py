from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import evaluate_baseline_family as baseline_eval
from posteditbench.baselines.stone_family.evaluation import binary_auroc, calculate_stem
from posteditbench.suite import SUITE_MODEL_REVISIONS


def _write_hf_cache_entry(cache_root: Path, model_name: str, revision: str) -> Path:
    entry_name = "models--" + model_name.replace("/", "--")
    entry = cache_root / entry_name
    (entry / "refs").mkdir(parents=True, exist_ok=True)
    snapshot_dir = entry / "snapshots" / revision
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (entry / "refs" / "main").write_text(revision + "\n", encoding="utf-8")
    (snapshot_dir / "config.json").write_text('{"architectures":["MockModel"]}\n', encoding="utf-8")
    (snapshot_dir / "tokenizer_config.json").write_text('{"model_max_length":512}\n', encoding="utf-8")
    (snapshot_dir / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    return snapshot_dir


def test_binary_auroc_prefers_higher_positive_scores():
    assert binary_auroc([0.1, 0.2], [0.8, 0.9]) == 1.0


def test_calculate_stem_matches_project_formula():
    assert calculate_stem(0.8, 0.9, 10.0, 11.0) == 0.7333


def test_evaluate_records_skip_perplexity_uses_report_and_detection_scores():
    report = {"summary": {"watermarked_functional_metrics": {"pass@1": 1.0}}}
    records = [
        {
            "baseline_family": "runtime_official",
            "baseline_origin": "external_checkout",
            "baseline_upstream_commit": "abc123",
            "watermark_scheme": "stone_runtime",
            "human_detect_score": 0.1,
            "clean_reference_detect_score": 0.2,
            "watermarked_detect_score": 0.9,
            "watermarked_validation": {"available": True, "passed": True},
            "human_reference_solution": "def add(a, b): return a + b",
            "clean_reference_solution": "def add(a, b):\n    return a + b\n",
            "watermarked_source": "def add(a, b):\n    return a + b\n",
        }
    ]

    result = baseline_eval.evaluate_records(
        report,
        records,
        ppl_model="unused",
        device="cpu",
        skip_perplexity=True,
        sample_limit=None,
        token_env="HF_ACCESS_TOKEN",
    )

    assert result["baseline_family"] == "runtime_official"
    assert result["datasets"] == []
    assert result["source_groups"] == []
    assert result["model_labels"] == []
    assert result["evaluation_tracks"] == []
    assert result["human_vs_watermarked_auroc"] == 1.0
    assert result["clean_reference_vs_watermarked_auroc"] == 1.0
    assert result["watermarked_pass_rate"] == 1.0
    assert "stem_human_reference" not in result


def test_evaluate_records_forwards_cache_contract_to_perplexity(monkeypatch):
    report = {"summary": {"watermarked_functional_metrics": {"pass@1": 1.0}}}
    records = [
        {
            "baseline_family": "runtime_official",
            "baseline_origin": "external_checkout",
            "baseline_upstream_commit": "abc123",
            "watermark_scheme": "stone_runtime",
            "human_detect_score": 0.1,
            "clean_reference_detect_score": 0.2,
            "watermarked_detect_score": 0.9,
            "watermarked_validation": {"available": True, "passed": True},
            "human_reference_solution": "def add(a, b): return a + b",
            "clean_reference_solution": "def add(a, b):\n    return a + b\n",
            "watermarked_source": "def add(a, b):\n    return a + b\n",
        }
    ]

    calls = []

    def fake_average_perplexity(texts, **kwargs):
        calls.append({"texts": list(texts), **kwargs})
        return 11.0

    monkeypatch.setattr(baseline_eval, "_average_perplexity", fake_average_perplexity)

    result = baseline_eval.evaluate_records(
        report,
        records,
        ppl_model="bigcode/starcoder2-7b",
        device="cuda:0",
        skip_perplexity=False,
        sample_limit=None,
        token_env="HF_ACCESS_TOKEN",
        cache_dir="model_cache/huggingface",
        local_files_only=True,
        trust_remote_code=True,
    )

    assert len(calls) == 3
    assert all(call["model_name"] == "bigcode/starcoder2-7b" for call in calls)
    assert all(call["cache_dir"] == "model_cache/huggingface" for call in calls)
    assert all(call["local_files_only"] is True for call in calls)
    assert all(call["trust_remote_code"] is True for call in calls)
    assert result["average_perplexity_watermarked"] == 11.0
    assert "stem_human_reference" in result


def test_evaluate_records_emits_context_metadata() -> None:
    report = {"summary": {"watermarked_functional_metrics": {"pass@1": 1.0}}}
    records = [
        {
            "baseline_family": "runtime_official",
            "baseline_origin": "external_checkout",
            "baseline_upstream_commit": "abc123",
            "dataset": "HumanEval+",
            "source_group": "public_humaneval_plus",
            "model_label": "bigcode/starcoder2-7b",
            "evaluation_track": "generation_time",
            "watermark_scheme": "stone_runtime",
            "human_detect_score": 0.1,
            "clean_reference_detect_score": 0.2,
            "watermarked_detect_score": 0.9,
            "watermarked_validation": {"available": True, "passed": True},
            "human_reference_solution": "def add(a, b): return a + b",
            "clean_reference_solution": "def add(a, b):\n    return a + b\n",
            "watermarked_source": "def add(a, b):\n    return a + b\n",
        }
    ]

    result = baseline_eval.evaluate_records(
        report,
        records,
        ppl_model="unused",
        device="cpu",
        skip_perplexity=True,
        sample_limit=None,
        token_env="HF_ACCESS_TOKEN",
    )

    assert result["datasets"] == ["HumanEval+"]
    assert result["source_groups"] == ["public_humaneval_plus"]
    assert result["model_labels"] == ["bigcode/starcoder2-7b"]
    assert result["evaluation_tracks"] == ["generation_time"]


def test_evaluate_records_uses_private_payloads_for_perplexity(monkeypatch) -> None:
    report = {"summary": {"watermarked_functional_metrics": {"pass@1": 1.0}}}
    records = [
        {
            "example_id": "ex-1",
            "baseline_family": "runtime_official",
            "baseline_origin": "external_checkout",
            "baseline_upstream_commit": "abc123",
            "watermark_scheme": "stone_runtime",
            "human_detect_score": 0.1,
            "clean_reference_detect_score": 0.2,
            "watermarked_detect_score": 0.9,
            "watermarked_validation": {"available": True, "passed": True},
        }
    ]
    payloads = [
        {
            "example_id": "ex-1",
            "human_reference_solution": "def add(a, b): return a + b",
            "clean_reference_solution": "def add(a, b):\n    return a + b\n",
            "watermarked_source": "def add(a, b):\n    return a + b\n",
        }
    ]

    calls = []

    def fake_average_perplexity(texts, **kwargs):
        calls.append(list(texts))
        return 9.5

    monkeypatch.setattr(baseline_eval, "_average_perplexity", fake_average_perplexity)

    result = baseline_eval.evaluate_records(
        report,
        records,
        payloads=payloads,
        ppl_model="bigcode/starcoder2-7b",
        device="cuda:0",
        skip_perplexity=False,
        sample_limit=None,
        token_env="HF_ACCESS_TOKEN",
    )

    assert len(calls) == 3
    assert result["perplexity_available"] is True
    assert result["average_perplexity_watermarked"] == 9.5
    assert "stem_clean_reference" in result


def test_resolve_perplexity_load_context_uses_pinned_snapshot_for_offline_mode(tmp_path: Path) -> None:
    cache_root = tmp_path / "model_cache"
    model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
    revision = SUITE_MODEL_REVISIONS[model_name]
    snapshot_dir = _write_hf_cache_entry(cache_root, model_name, revision)

    device, load_target, tokenizer_kwargs, model_kwargs, offline_env = baseline_eval._resolve_perplexity_load_context(
        model_name=model_name,
        token_env="HF_ACCESS_TOKEN",
        device="cpu",
        cache_dir=str(cache_root),
        local_files_only=True,
        trust_remote_code=True,
    )

    assert device == "cpu"
    assert load_target == str(snapshot_dir)
    assert tokenizer_kwargs["local_files_only"] is True
    assert model_kwargs["local_files_only"] is True
    assert "revision" not in tokenizer_kwargs
    assert "revision" not in model_kwargs
    assert offline_env["HF_HUB_OFFLINE"] == "1"
    assert offline_env["TRANSFORMERS_OFFLINE"] == "1"


def test_resolve_perplexity_load_context_pins_revision_for_online_mode() -> None:
    model_name = "bigcode/starcoder2-7b"
    revision = SUITE_MODEL_REVISIONS[model_name]

    device, load_target, tokenizer_kwargs, model_kwargs, offline_env = baseline_eval._resolve_perplexity_load_context(
        model_name=model_name,
        token_env="HF_ACCESS_TOKEN",
        device="cuda:0",
        cache_dir="model_cache/huggingface",
        local_files_only=False,
        trust_remote_code=True,
    )

    assert device == "cuda:0"
    assert load_target == model_name
    assert tokenizer_kwargs["revision"] == revision
    assert model_kwargs["revision"] == revision
    assert tokenizer_kwargs["cache_dir"] == "model_cache/huggingface"
    assert model_kwargs["cache_dir"] == "model_cache/huggingface"
    assert offline_env == {}
