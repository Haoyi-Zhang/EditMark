from __future__ import annotations

import gc
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open

try:
    from _shared import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover
    from scripts._shared import PROJECT_ROOT
from posteditbench.hf_auth import resolve_token_env_value
from posteditbench.suite import resolve_model_revision


@dataclass(frozen=True, slots=True)
class HFModelRequirement:
    model: str
    cache_dir: str
    local_files_only: bool
    revision: str = ""
    trust_remote_code: bool = False
    device: str = "cuda"
    dtype: str = "float16"
    token_env: str = "HF_ACCESS_TOKEN"
    usage: tuple[str, ...] = ()
    config_paths: tuple[str, ...] = ()


def repo_cache_dirname(model_name: str) -> str:
    normalized = str(model_name or "").strip().replace("\\", "/").strip("/")
    if not normalized:
        return ""
    return "models--" + normalized.replace("/", "--")


def resolve_cache_roots(cache_dir: str) -> tuple[Path, Path]:
    configured = str(cache_dir or "").strip()
    if configured:
        base = Path(configured)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
    else:
        hf_home = str(os.environ.get("HF_HOME", "")).strip()
        base = Path(hf_home) if hf_home else PROJECT_ROOT / "model_cache" / "huggingface"
    if base.name == "hub":
        return base.parent, base
    return base, base / "hub"


def cache_entry_paths(model_name: str, cache_dir: str) -> tuple[Path, Path]:
    root_cache, hub_cache = resolve_cache_roots(cache_dir)
    entry = repo_cache_dirname(model_name)
    return root_cache / entry, hub_cache / entry


def preferred_cache_entry_path(model_name: str, cache_dir: str) -> tuple[Path, Path, Path]:
    root_entry, hub_entry = cache_entry_paths(model_name, cache_dir)
    configured = str(cache_dir or "").strip()
    if configured:
        base = Path(configured)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
    else:
        hf_home = str(os.environ.get("HF_HOME", "")).strip()
        base = Path(hf_home) if hf_home else PROJECT_ROOT / "model_cache" / "huggingface"
    ordered = [hub_entry, root_entry] if base.name == "hub" else [root_entry, hub_entry]
    selected = next((entry for entry in ordered if entry.exists()), ordered[0])
    return root_entry, hub_entry, selected


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


def _snapshot_candidates(entry_path: Path) -> list[Path]:
    snapshots_dir = entry_path / "snapshots"
    if not snapshots_dir.exists():
        return []
    return sorted(path for path in snapshots_dir.iterdir() if path.is_dir())


def _resolved_requirement_revision(requirement: HFModelRequirement) -> str:
    return resolve_model_revision(requirement.model, requirement.revision, require_canonical=False)


def resolve_snapshot_dir(entry_path: Path, *, revision: str = "") -> tuple[Path | None, list[str]]:
    issues: list[str] = []
    normalized_revision = str(revision or "").strip()
    snapshots = _snapshot_candidates(entry_path)
    if normalized_revision:
        candidate = entry_path / "snapshots" / normalized_revision
        if candidate.exists():
            return candidate, issues
        issues.append(f"{entry_path}: pinned revision {normalized_revision} is missing from snapshots/")
        return None, issues
    refs_main = entry_path / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        if not revision:
            issues.append(f"{entry_path}: refs/main is empty")
        else:
            candidate = entry_path / "snapshots" / revision
            if candidate.exists():
                return candidate, issues
            issues.append(f"{entry_path}: refs/main points to missing snapshot {revision}")
    if len(snapshots) == 1:
        return snapshots[0], issues
    if not snapshots:
        issues.append(f"{entry_path}: missing snapshots directory entries")
    else:
        issues.append(f"{entry_path}: unable to resolve snapshot uniquely")
    return None, issues


def _required_safetensor_names(snapshot_dir: Path) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    index_path = snapshot_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return [], [f"{snapshot_dir}: invalid model.safetensors.index.json: {exc}"]
        weight_map = payload.get("weight_map", {})
        if not isinstance(weight_map, dict):
            return [], [f"{snapshot_dir}: model.safetensors.index.json missing weight_map"]
        names = sorted({str(name) for name in weight_map.values() if str(name).strip()})
        if not names:
            issues.append(f"{snapshot_dir}: model.safetensors.index.json does not reference any shard files")
        return names, issues
    names = sorted(path.name for path in snapshot_dir.glob("*.safetensors"))
    if not names:
        issues.append(f"{snapshot_dir}: no .safetensors shards found")
    return names, issues


def _required_snapshot_files(snapshot_dir: Path) -> list[str]:
    return ["config.json", "tokenizer_config.json"]


def _has_minimal_tokenizer_assets(snapshot_dir: Path) -> bool:
    if (snapshot_dir / "tokenizer.json").exists():
        return True
    if any((snapshot_dir / name).exists() for name in ("tokenizer.model", "spiece.model")):
        return True
    return (snapshot_dir / "vocab.json").exists() and (snapshot_dir / "merges.txt").exists()


def _validate_snapshot_assets(snapshot_dir: Path) -> list[str]:
    issues: list[str] = []
    for required_name in _required_snapshot_files(snapshot_dir):
        if not (snapshot_dir / required_name).exists():
            issues.append(f"{snapshot_dir}: missing required asset {required_name}")
    if not _has_minimal_tokenizer_assets(snapshot_dir):
        issues.append(
            f"{snapshot_dir}: missing tokenizer assets (expected tokenizer.json, tokenizer.model/spiece.model, or vocab.json + merges.txt)"
        )
    return issues


def validate_local_hf_cache(
    requirement: HFModelRequirement,
    *,
    require_root_entry: bool = True,
) -> dict[str, Any]:
    root_entry, hub_entry, entry_path = preferred_cache_entry_path(requirement.model, requirement.cache_dir)
    issues: list[str] = []
    root_exists = root_entry.exists()
    hub_exists = hub_entry.exists()
    if require_root_entry and not root_exists:
        issues.append(f"{requirement.model}: missing official cache entry {root_entry}")
    resolved_revision = _resolved_requirement_revision(requirement)
    if not entry_path.exists():
        return {
            "model": requirement.model,
            "cache_dir": requirement.cache_dir,
            "requested_revision": resolved_revision,
            "root_entry": str(root_entry),
            "hub_entry": str(hub_entry),
            "root_entry_exists": root_exists,
            "hub_entry_exists": hub_exists,
            "resolved_entry": "",
            "resolved_snapshot": "",
            "required_shards": [],
            "missing_shards": [],
            "validated_shards": [],
            "shard_errors": [],
            "issues": issues,
            "status": "failed",
        }

    refs_main = entry_path / "refs" / "main"
    if not refs_main.exists() and not resolved_revision:
        issues.append(f"{requirement.model}: missing refs/main under {entry_path}")

    snapshot_dir, snapshot_issues = resolve_snapshot_dir(entry_path, revision=resolved_revision)
    issues.extend(snapshot_issues)
    required_shards: list[str] = []
    missing_shards: list[str] = []
    validated_shards: list[str] = []
    shard_errors: list[str] = []
    if snapshot_dir is not None:
        issues.extend(_validate_snapshot_assets(snapshot_dir))
        required_shards, shard_requirement_issues = _required_safetensor_names(snapshot_dir)
        issues.extend(shard_requirement_issues)
        for shard_name in required_shards:
            shard_path = snapshot_dir / shard_name
            if not shard_path.exists():
                missing_shards.append(shard_name)
                issues.append(f"{requirement.model}: missing shard {shard_name} in {snapshot_dir}")
                continue
            try:
                with safe_open(str(shard_path), framework="pt") as handle:
                    list(handle.keys())[:1]
                validated_shards.append(shard_name)
            except Exception as exc:  # pragma: no cover - depends on third-party parser wording
                error = f"{shard_name}: {exc}"
                shard_errors.append(error)
                issues.append(f"{requirement.model}: invalid shard {error}")

    return {
        "model": requirement.model,
        "cache_dir": requirement.cache_dir,
        "requested_revision": resolved_revision,
        "root_entry": str(root_entry),
        "hub_entry": str(hub_entry),
        "root_entry_exists": root_exists,
        "hub_entry_exists": hub_exists,
        "resolved_entry": str(entry_path),
        "resolved_snapshot": str(snapshot_dir) if snapshot_dir is not None else "",
        "required_shards": required_shards,
        "missing_shards": missing_shards,
        "validated_shards": validated_shards,
        "shard_errors": shard_errors,
        "issues": issues,
        "status": "ok" if not issues else "failed",
    }


def _load_context_for_local_hf_model(requirement: HFModelRequirement) -> tuple[Any, str, str, dict[str, Any], dict[str, Any], dict[str, str]]:
    import torch

    requested_device = str(requirement.device or "cuda").strip().lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested for {requirement.model} but no CUDA device is available")
    device = requested_device if requested_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = str(requirement.dtype or "float16").strip().lower()
    dtype_map = {
        "float16": getattr(torch, "float16", None),
        "fp16": getattr(torch, "float16", None),
        "bfloat16": getattr(torch, "bfloat16", None),
        "bf16": getattr(torch, "bfloat16", None),
        "float32": getattr(torch, "float32", None),
        "fp32": getattr(torch, "float32", None),
    }
    torch_dtype = dtype_map.get(dtype_name) or getattr(torch, "float16" if device.startswith("cuda") else "float32")
    token = ""
    if requirement.token_env and not requirement.local_files_only:
        token = resolve_token_env_value(requirement.token_env)
    _, _, entry_path = preferred_cache_entry_path(requirement.model, requirement.cache_dir)
    cache_root = entry_path.parent if entry_path.name.startswith("models--") else resolve_cache_roots(requirement.cache_dir)[0]
    resolved_revision = _resolved_requirement_revision(requirement)
    snapshot_dir = None
    if requirement.local_files_only and entry_path.exists():
        snapshot_dir, _ = resolve_snapshot_dir(entry_path, revision=resolved_revision)
    load_target = str(snapshot_dir) if snapshot_dir is not None else requirement.model
    tokenizer_kwargs: dict[str, Any] = {
        "cache_dir": str(cache_root),
        "local_files_only": True,
        "trust_remote_code": bool(requirement.trust_remote_code),
    }
    model_kwargs: dict[str, Any] = {
        "cache_dir": str(cache_root),
        "local_files_only": True,
        "trust_remote_code": bool(requirement.trust_remote_code),
        "low_cpu_mem_usage": True,
        "torch_dtype": torch_dtype,
    }
    if token:
        tokenizer_kwargs["token"] = token
        model_kwargs["token"] = token
    if resolved_revision and load_target == requirement.model:
        tokenizer_kwargs["revision"] = resolved_revision
        model_kwargs["revision"] = resolved_revision
    offline_env = {}
    if requirement.local_files_only:
        offline_env["HF_HUB_OFFLINE"] = "1"
        offline_env["TRANSFORMERS_OFFLINE"] = "1"
    return torch, device, load_target, tokenizer_kwargs, model_kwargs, offline_env


def smoke_load_local_hf_model(
    requirement: HFModelRequirement,
    *,
    prompt: str = "def add(a, b):\n    return",
    max_new_tokens: int = 4,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - dependency-dependent
        return {"model": requirement.model, "status": "failed", "issues": [f"missing dependency: {exc}"]}

    try:
        torch, device, load_target, tokenizer_kwargs, model_kwargs, offline_env = _load_context_for_local_hf_model(requirement)
    except Exception as exc:  # pragma: no cover - depends on local runtime
        return {"model": requirement.model, "status": "failed", "issues": [f"{exc.__class__.__name__}: {exc}"]}
    model = None
    tokenizer = None
    try:
        with _temporary_env(offline_env):
            tokenizer = AutoTokenizer.from_pretrained(load_target, **tokenizer_kwargs)
            if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
                eos_token = getattr(tokenizer, "eos_token", None)
                if eos_token is not None:
                    tokenizer.pad_token = eos_token
            model = AutoModelForCausalLM.from_pretrained(load_target, **model_kwargs)
            if device != "cpu":
                model = model.to(device)
            if hasattr(model, "eval"):
                model.eval()
            batch = tokenizer(prompt, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            generation_kwargs = {
                "max_new_tokens": max(1, int(max_new_tokens)),
                "do_sample": False,
            }
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id
            inference_mode = getattr(torch, "inference_mode", None)
            context = inference_mode() if callable(inference_mode) else None
            if context is None:
                output = model.generate(**batch, **generation_kwargs)
            else:
                with context:
                    output = model.generate(**batch, **generation_kwargs)
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            return {
                "model": requirement.model,
                "status": "ok",
                "device": device,
                "generated_preview": decoded[:120],
                "issues": [],
            }
    except Exception as exc:  # pragma: no cover - depends on local HF runtime
        return {"model": requirement.model, "status": "failed", "device": device, "issues": [f"{exc.__class__.__name__}: {exc}"]}
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def smoke_load_local_hf_evaluator(
    requirement: HFModelRequirement,
    *,
    text: str = "def add(a, b):\n    return a + b\n",
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - dependency-dependent
        return {"model": requirement.model, "status": "failed", "issues": [f"missing dependency: {exc}"]}

    try:
        torch, device, load_target, tokenizer_kwargs, model_kwargs, offline_env = _load_context_for_local_hf_model(requirement)
    except Exception as exc:  # pragma: no cover - depends on local runtime
        return {"model": requirement.model, "status": "failed", "issues": [f"{exc.__class__.__name__}: {exc}"]}
    model = None
    tokenizer = None
    try:
        with _temporary_env(offline_env):
            tokenizer = AutoTokenizer.from_pretrained(load_target, **tokenizer_kwargs)
            model = AutoModelForCausalLM.from_pretrained(load_target, **model_kwargs)
            if device != "cpu":
                model = model.to(device)
            if hasattr(model, "eval"):
                model.eval()
            batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            batch = {key: value.to(device) for key, value in batch.items()}
            no_grad = getattr(torch, "no_grad", None)
            context = no_grad() if callable(no_grad) else None
            if context is None:
                outputs = model(**batch, labels=batch["input_ids"])
            else:
                with context:
                    outputs = model(**batch, labels=batch["input_ids"])
            loss_value = float(getattr(outputs, "loss").item())
            return {
                "model": requirement.model,
                "status": "ok",
                "device": device,
                "loss": round(loss_value, 6),
                "issues": [],
            }
    except Exception as exc:  # pragma: no cover - depends on local HF runtime
        return {"model": requirement.model, "status": "failed", "device": device, "issues": [f"{exc.__class__.__name__}: {exc}"]}
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
