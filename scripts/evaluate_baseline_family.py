from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import dump_json, markdown_table, read_jsonl
from _hf_readiness import HFModelRequirement, _load_context_for_local_hf_model
from posteditbench.baselines.stone_family.evaluation import binary_auroc, calculate_stem
from posteditbench.hf_auth import resolve_token_env_value
from posteditbench.suite import resolve_model_revision

DEFAULT_PPL_MODEL = "bigcode/starcoder2-7b"
DEFAULT_PPL_TOKEN_ENV = "HF_ACCESS_TOKEN"
DEFAULT_PPL_CACHE_DIR = "model_cache/huggingface"
DEFAULT_PPL_LOCAL_FILES_ONLY = True
DEFAULT_PPL_TRUST_REMOTE_CODE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate imported official runtime baseline runs with AUROC, PPL, and STEM.")
    parser.add_argument("--input", type=Path, required=True, help="Run directory or report.json path.")
    parser.add_argument("--records", type=Path, default=None, help="Optional baseline_eval_records.jsonl path.")
    parser.add_argument("--payloads", type=Path, default=None, help="Optional private payload JSONL with raw texts for perplexity.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--ppl-model", type=str, default=DEFAULT_PPL_MODEL, help="Reference model for perplexity.")
    parser.add_argument("--device", type=str, default="", help="Torch device for perplexity evaluation.")
    parser.add_argument("--skip-perplexity", action="store_true", help="Skip perplexity and STEM naturalness terms.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional cap for PPL samples.")
    parser.add_argument("--token-env", type=str, default=DEFAULT_PPL_TOKEN_ENV, help="Credential env for gated HF models.")
    parser.add_argument("--cache-dir", type=str, default="", help="Optional Hugging Face cache root for perplexity evaluation.")
    parser.add_argument("--local-files-only", action="store_true", help="Require local-only HF loading for perplexity evaluation.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code for perplexity evaluation.")
    return parser.parse_args()


def resolve_report_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "report.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"{path} does not contain report.json")
    return path


def resolve_records_path(report_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidate = report_path.with_name("baseline_eval_records.jsonl")
    if not candidate.exists():
        raise FileNotFoundError(f"Missing baseline eval records: {candidate}")
    return candidate


def resolve_payloads_path(report_path: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    candidate = report_path.with_name("baseline_eval_payloads.private.jsonl")
    return candidate if candidate.exists() else None


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _payload_texts(
    records: list[dict[str, Any]],
    payloads: list[dict[str, Any]] | None,
    *,
    field: str,
) -> list[str]:
    by_example: dict[str, dict[str, Any]] = {}
    for payload in payloads or []:
        example_id = str(payload.get("example_id", "")).strip()
        if example_id:
            by_example[example_id] = payload
    values: list[str] = []
    for record in records:
        example_id = str(record.get("example_id", "")).strip()
        payload = by_example.get(example_id, {})
        text = str(payload.get(field, "")).strip()
        if not text:
            text = str(record.get(field, "")).strip()
        if text:
            values.append(text)
    return values


def _device_name(value: str) -> str:
    if value.strip():
        return value.strip()
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def baseline_eval_requirement(
    *,
    model_name: str,
    token_env: str,
    device: str,
    cache_dir: str,
    local_files_only: bool,
    trust_remote_code: bool,
    usage: tuple[str, ...] = ("evaluator",),
) -> HFModelRequirement:
    requested_device = str(device or "cuda").strip() or "cuda"
    return HFModelRequirement(
        model=model_name,
        revision=resolve_model_revision(model_name, None, require_canonical=False),
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        device=requested_device,
        dtype="float16" if requested_device.startswith("cuda") else "float32",
        token_env=token_env,
        usage=usage,
        config_paths=("scripts/evaluate_baseline_family.py",),
    )


def canonical_baseline_eval_requirement(
    *,
    device: str = "cuda",
    token_env: str = DEFAULT_PPL_TOKEN_ENV,
    cache_dir: str = DEFAULT_PPL_CACHE_DIR,
    local_files_only: bool = DEFAULT_PPL_LOCAL_FILES_ONLY,
    trust_remote_code: bool = DEFAULT_PPL_TRUST_REMOTE_CODE,
    usage: tuple[str, ...] = ("baseline_eval", "evaluator"),
) -> HFModelRequirement:
    return baseline_eval_requirement(
        model_name=DEFAULT_PPL_MODEL,
        token_env=token_env,
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        usage=usage,
    )


def derived_baseline_eval_requirement(
    *,
    metadata: Mapping[str, Any] | None = None,
    ppl_model: str = DEFAULT_PPL_MODEL,
    device: str = "cuda",
    token_env: str = "",
    cache_dir: str = "",
    local_files_only: bool | None = None,
    trust_remote_code: bool | None = None,
    usage: tuple[str, ...] = ("baseline_eval", "evaluator"),
) -> HFModelRequirement:
    metadata_payload = dict(metadata or {})
    watermark_metadata = dict(metadata_payload.get("watermark", {}) or {})
    provider_metadata = dict(metadata_payload.get("provider", {}) or {})

    derived_token_env = str(
        token_env
        or watermark_metadata.get("token_env")
        or provider_metadata.get("token_env")
        or DEFAULT_PPL_TOKEN_ENV
    ).strip() or DEFAULT_PPL_TOKEN_ENV
    derived_cache_dir = str(
        cache_dir
        or watermark_metadata.get("cache_dir")
        or provider_metadata.get("cache_dir")
        or DEFAULT_PPL_CACHE_DIR
    ).strip() or DEFAULT_PPL_CACHE_DIR
    derived_device = str(
        device
        or watermark_metadata.get("device")
        or provider_metadata.get("device")
        or "cuda"
    ).strip() or "cuda"
    if local_files_only is None:
        derived_local_files_only = bool(
            watermark_metadata.get("local_files_only")
            or provider_metadata.get("local_files_only")
            or DEFAULT_PPL_LOCAL_FILES_ONLY
        )
    else:
        derived_local_files_only = bool(local_files_only)
    if trust_remote_code is None:
        derived_trust_remote_code = bool(
            watermark_metadata.get("trust_remote_code")
            or provider_metadata.get("trust_remote_code")
            or DEFAULT_PPL_TRUST_REMOTE_CODE
        )
    else:
        derived_trust_remote_code = bool(trust_remote_code)
    return baseline_eval_requirement(
        model_name=ppl_model,
        token_env=derived_token_env,
        device=derived_device,
        cache_dir=derived_cache_dir,
        local_files_only=derived_local_files_only,
        trust_remote_code=derived_trust_remote_code,
        usage=usage,
    )


@contextmanager
def _temporary_env(overrides: dict[str, str]):
    previous: dict[str, str | None] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _resolve_perplexity_load_context(
    *,
    model_name: str,
    token_env: str,
    device: str,
    cache_dir: str,
    local_files_only: bool,
    trust_remote_code: bool,
) -> tuple[str, str, dict[str, Any], dict[str, Any], dict[str, str]]:
    if local_files_only:
        requirement = baseline_eval_requirement(
            model_name=model_name,
            token_env=token_env,
            device=device,
            cache_dir=cache_dir,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            usage=("evaluator",),
        )
        _, resolved_device, load_target, tokenizer_kwargs, model_kwargs, offline_env = _load_context_for_local_hf_model(
            requirement
        )
        return resolved_device, load_target, tokenizer_kwargs, model_kwargs, offline_env

    token = resolve_token_env_value(token_env)
    resolved_revision = resolve_model_revision(model_name, None, require_canonical=False)
    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if token:
        tokenizer_kwargs["token"] = token
        model_kwargs["token"] = token
    if cache_dir:
        tokenizer_kwargs["cache_dir"] = cache_dir
        model_kwargs["cache_dir"] = cache_dir
    if resolved_revision:
        tokenizer_kwargs["revision"] = resolved_revision
        model_kwargs["revision"] = resolved_revision
    return device, model_name, tokenizer_kwargs, model_kwargs, {}


def _average_perplexity(
    texts: list[str],
    *,
    model_name: str,
    token_env: str,
    device: str,
    cache_dir: str,
    local_files_only: bool,
    trust_remote_code: bool,
) -> float:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    resolved_device, load_target, tokenizer_kwargs, model_kwargs, offline_env = _resolve_perplexity_load_context(
        model_name=model_name,
        token_env=token_env,
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = None
    model = None
    try:
        with _temporary_env(offline_env):
            tokenizer = AutoTokenizer.from_pretrained(load_target, **tokenizer_kwargs)
            model = AutoModelForCausalLM.from_pretrained(load_target, **model_kwargs).to(resolved_device)
            values: list[float] = []
            for text in texts:
                encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(resolved_device)
                with torch.no_grad():
                    outputs = model(**encodings, labels=encodings["input_ids"])
                    values.append(float(torch.exp(outputs.loss).item()))
            return round(mean(values), 4) if values else 0.0
    finally:
        del model
        del tokenizer
        gc = getattr(torch, "cuda", None)
        if gc is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_records(
    report: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    payloads: list[dict[str, Any]] | None = None,
    ppl_model: str,
    device: str,
    skip_perplexity: bool,
    sample_limit: int | None,
    token_env: str,
    cache_dir: str = "",
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    if sample_limit is not None and sample_limit > 0:
        records = records[:sample_limit]
    summary = dict(report.get("summary", {}))
    baseline_family = ""
    origins = sorted({str(record.get("baseline_origin", "")).strip() for record in records if str(record.get("baseline_origin", "")).strip()})
    commits = sorted({str(record.get("baseline_upstream_commit", "")).strip() for record in records if str(record.get("baseline_upstream_commit", "")).strip()})
    datasets = sorted({str(record.get("dataset", "")).strip() for record in records if str(record.get("dataset", "")).strip()})
    source_groups = sorted({str(record.get("source_group", "")).strip() for record in records if str(record.get("source_group", "")).strip()})
    model_labels = sorted({str(record.get("model_label", "")).strip() for record in records if str(record.get("model_label", "")).strip()})
    evaluation_tracks = sorted({str(record.get("evaluation_track", "")).strip() for record in records if str(record.get("evaluation_track", "")).strip()})
    if records:
        baseline_family = str(records[0].get("baseline_family", ""))
    human_scores = [float(record.get("human_detect_score", 0.0)) for record in records if record.get("human_detect_score") is not None]
    watermarked_scores = [float(record.get("watermarked_detect_score", 0.0)) for record in records if record.get("watermarked_detect_score") is not None]
    clean_reference_scores = [
        float(record.get("clean_reference_detect_score", 0.0))
        for record in records
        if record.get("clean_reference_detect_score") is not None
    ]
    watermarked_validations = [record.get("watermarked_validation", {}) for record in records if isinstance(record.get("watermarked_validation"), dict)]
    watermarked_passes = [1.0 if validation.get("passed") else 0.0 for validation in watermarked_validations if validation.get("available")]
    watermarked_pass_rate = round(mean(watermarked_passes), 4) if watermarked_passes else 0.0
    human_ppl: float | None = None
    clean_reference_ppl: float | None = None
    watermarked_ppl: float | None = None
    perplexity_available = False
    if not skip_perplexity:
        human_texts = _payload_texts(records, payloads, field="human_reference_solution")
        clean_reference_texts = _payload_texts(records, payloads, field="clean_reference_solution")
        watermarked_texts = _payload_texts(records, payloads, field="watermarked_source")
        if human_texts and clean_reference_texts and watermarked_texts:
            perplexity_available = True
            human_ppl = _average_perplexity(
                human_texts,
                model_name=ppl_model,
                token_env=token_env,
                device=device,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            clean_reference_ppl = _average_perplexity(
                clean_reference_texts,
                model_name=ppl_model,
                token_env=token_env,
                device=device,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            watermarked_ppl = _average_perplexity(
                watermarked_texts,
                model_name=ppl_model,
                token_env=token_env,
                device=device,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
    human_auroc = round(binary_auroc(human_scores, watermarked_scores), 4)
    clean_reference_auroc = round(binary_auroc(clean_reference_scores, watermarked_scores), 4)
    result = {
        "baseline_family": baseline_family,
        "record_count": len(records),
        "baseline_origins": origins,
        "baseline_upstream_commits": commits,
        "datasets": datasets,
        "source_groups": source_groups,
        "model_labels": model_labels,
        "evaluation_tracks": evaluation_tracks,
        "watermark_schemes": sorted({str(record.get("watermark_scheme", "")).strip() for record in records if str(record.get("watermark_scheme", "")).strip()}),
        "human_vs_watermarked_auroc": human_auroc,
        "clean_reference_vs_watermarked_auroc": clean_reference_auroc,
        "watermarked_pass_rate": watermarked_pass_rate,
        "perplexity_available": perplexity_available,
        "summary_watermarked_functional_metrics": summary.get("watermarked_functional_metrics", {}),
        "average_perplexity_human": human_ppl,
        "average_perplexity_clean_reference": clean_reference_ppl,
        "average_perplexity_watermarked": watermarked_ppl,
    }
    if not skip_perplexity and perplexity_available and human_ppl is not None and clean_reference_ppl is not None and watermarked_ppl is not None:
        result["stem_human_reference"] = calculate_stem(watermarked_pass_rate, human_auroc, human_ppl, watermarked_ppl)
        result["stem_clean_reference"] = calculate_stem(watermarked_pass_rate, clean_reference_auroc, clean_reference_ppl, watermarked_ppl)
    elif not skip_perplexity:
        result["perplexity_unavailable_reason"] = "missing_private_payloads"
    return result


def main() -> int:
    args = parse_args()
    try:
        report_path = resolve_report_path(args.input)
        records_path = resolve_records_path(report_path, args.records)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    payloads_path = resolve_payloads_path(report_path, args.payloads)
    report = load_report(report_path)
    records = read_jsonl(records_path)
    payloads = read_jsonl(payloads_path) if payloads_path is not None and payloads_path.exists() else []
    config = dict(report.get("config", {}))
    evaluator_requirement = derived_baseline_eval_requirement(
        metadata=dict(config.get("metadata", {})),
        ppl_model=args.ppl_model,
        device=_device_name(args.device),
        token_env=args.token_env,
        cache_dir=args.cache_dir,
        local_files_only=True if args.local_files_only else None,
        trust_remote_code=True if args.trust_remote_code else None,
        usage=("baseline_eval", "evaluator"),
    )
    evaluation = evaluate_records(
        report,
        records,
        payloads=payloads,
        ppl_model=evaluator_requirement.model,
        device=evaluator_requirement.device,
        skip_perplexity=args.skip_perplexity,
        sample_limit=args.sample_limit,
        token_env=evaluator_requirement.token_env,
        cache_dir=evaluator_requirement.cache_dir,
        local_files_only=evaluator_requirement.local_files_only,
        trust_remote_code=evaluator_requirement.trust_remote_code,
    )
    print(f"Evaluated {report_path}")
    table_rows = [[key, value] for key, value in evaluation.items() if key not in {"summary_watermarked_functional_metrics"}]
    print(markdown_table(["metric", "value"], table_rows))
    if args.output is not None:
        dump_json(args.output, evaluation)
        print(f"Wrote baseline evaluation to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
