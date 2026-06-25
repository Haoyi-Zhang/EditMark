from __future__ import annotations

import contextlib
import os
import shlex
import subprocess
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from .hf_auth import resolve_token_env_value
from .models import BenchmarkExample
from .suite import resolve_model_revision


class CompletionProvider(Protocol):
    name: str

    def generate(self, example: BenchmarkExample, *, seed: int = 0) -> str:
        raise NotImplementedError


@dataclass(slots=True)
class OfflineMockProvider:
    name: str = "offline_mock"

    def generate(self, example: BenchmarkExample, *, seed: int = 0) -> str:
        # Deterministic and network-free by default.
        return example.reference_solution


def _bool_from_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_dtype_name(value: Any, *, device: str | None = None) -> str:
    text = str(value or "").strip().lower()
    if not text or text == "auto":
        return "float16" if str(device or "").strip().lower().startswith("cuda") else "float32"
    if text in {"fp16", "half"}:
        return "float16"
    if text in {"bf16"}:
        return "bfloat16"
    if text in {"fp32"}:
        return "float32"
    return text


def _parse_torch_dtype(torch_module: Any, dtype_name: str, *, device: str) -> Any:
    normalized = _normalize_dtype_name(dtype_name, device=device)
    mapping = {
        "float16": getattr(torch_module, "float16", None),
        "bfloat16": getattr(torch_module, "bfloat16", None),
        "float32": getattr(torch_module, "float32", None),
    }
    if normalized not in mapping or mapping[normalized] is None:
        raise ValueError(
            f"unsupported local_hf dtype '{dtype_name}' (expected auto, float16, bfloat16, fp16, bf16, float32, or fp32)"
        )
    return mapping[normalized]


def _resolve_local_hf_device(torch_module: Any, device: str | None) -> str:
    requested = str(device or "auto").strip().lower()
    if not requested or requested == "auto":
        has_cuda = bool(getattr(getattr(torch_module, "cuda", None), "is_available", lambda: False)())
        return "cuda" if has_cuda else "cpu"
    if requested.startswith("cuda") and not bool(getattr(getattr(torch_module, "cuda", None), "is_available", lambda: False)()):
        raise RuntimeError(f"CUDA device requested but not available: {device}")
    return requested


def _move_batch_to_device(batch: Any, device: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        converted = [_move_batch_to_device(value, device) for value in batch]
        return type(batch)(converted)
    return batch


def _strip_prompt_prefix(prompt: str, generated: str) -> str:
    text = str(generated)
    if prompt and text.startswith(prompt):
        return text[len(prompt) :].lstrip("\n\r")
    return text


def _repo_cache_dirname(model_name: str) -> str:
    normalized = str(model_name or "").strip().replace("\\", "/").strip("/")
    if not normalized:
        return ""
    return "models--" + normalized.replace("/", "--")


def _resolved_local_hf_revision(model_name: str, requested: str | None = None) -> str:
    return resolve_model_revision(model_name, requested, require_canonical=False)


def _resolve_local_hf_cache_dir(cache_dir: str, model_name: str, *, local_files_only: bool) -> str:
    repo_dirname = _repo_cache_dirname(model_name)

    def _contains_repo(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        if repo_dirname and (path / repo_dirname).exists():
            return True
        return any(child.is_dir() and child.name.startswith(("models--", "datasets--", "spaces--")) for child in path.iterdir())

    # The local loader intentionally accepts either the root Hugging Face cache
    # directory or its nested hub/ layout. Selection follows the configured path
    # first (hub stays hub, root stays root), and the readiness validator uses
    # the same deterministic preference order before the benchmark can run.
    configured = str(cache_dir or "").strip()
    if configured:
        configured_path = Path(configured)
        if _contains_repo(configured_path):
            return str(configured_path)
        if configured_path.exists():
            hub_path = configured_path / "hub"
            if _contains_repo(hub_path):
                return str(hub_path)
        if configured_path.exists() or not local_files_only:
            return configured
    if local_files_only:
        hf_home = str(os.environ.get("HF_HOME", "")).strip()
        if hf_home:
            hf_home_path = Path(hf_home)
            if _contains_repo(hf_home_path):
                return hf_home
            hub_path = hf_home_path / "hub"
            if _contains_repo(hub_path):
                return str(hub_path)
            return hf_home
    return configured


def _resolve_local_hf_snapshot_path(cache_dir: str, model_name: str, *, revision: str = "") -> str:
    repo_dirname = _repo_cache_dirname(model_name)
    selected_cache_dir = _resolve_local_hf_cache_dir(cache_dir, model_name, local_files_only=True)
    if not selected_cache_dir or not repo_dirname:
        return ""

    base = Path(selected_cache_dir)
    entry_path = base / repo_dirname
    if not entry_path.exists():
        return ""
    normalized_revision = str(revision or "").strip()
    if normalized_revision:
        candidate = entry_path / "snapshots" / normalized_revision
        if candidate.exists():
            return str(candidate)
    refs_main = entry_path / "refs" / "main"
    if refs_main.exists():
        refs_revision = refs_main.read_text(encoding="utf-8").strip()
        if refs_revision:
            candidate = entry_path / "snapshots" / refs_revision
            if candidate.exists():
                return str(candidate)
    snapshots_dir = entry_path / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if len(snapshots) == 1:
            return str(snapshots[0])
    return ""


@dataclass(slots=True)
class LocalHFProvider:
    name: str = "local_hf"
    model: str = ""
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    no_repeat_ngram_size: int = 0
    trust_remote_code: bool = False
    token_env: str = "HF_ACCESS_TOKEN"
    revision: str = ""
    cache_dir: str = ""
    local_files_only: bool = False

    def _backend(self) -> "_LocalHFBackend":
        return _load_local_hf_backend(
            self.model,
            self.device,
            self.dtype,
            self.max_new_tokens,
            self.do_sample,
            self.temperature,
            self.top_p,
            self.no_repeat_ngram_size,
            self.trust_remote_code,
            self.token_env,
            self.revision,
            self.cache_dir,
            self.local_files_only,
        )

    def generate(self, example: BenchmarkExample, *, seed: int = 0) -> str:
        backend = self._backend()
        return backend.generate(example.prompt, seed=seed)


@dataclass(slots=True)
class _LocalHFBackend:
    model: Any
    tokenizer: Any
    torch_module: Any
    device: str
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    no_repeat_ngram_size: int

    def generate(self, prompt: str, *, seed: int = 0) -> str:
        batch = self.tokenizer(prompt, return_tensors="pt")
        batch = _move_batch_to_device(batch, self.device)
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            manual_seed = getattr(self.torch_module, "manual_seed", None)
            if callable(manual_seed):
                manual_seed(int(seed))
            cuda_module = getattr(self.torch_module, "cuda", None)
            manual_seed_all = getattr(cuda_module, "manual_seed_all", None)
            if callable(manual_seed_all):
                manual_seed_all(int(seed))
            generation_kwargs["temperature"] = max(float(self.temperature), 1e-5)
            generation_kwargs["top_p"] = max(1e-5, min(float(self.top_p), 1.0))
        else:
            generation_kwargs["temperature"] = 0.0
        if self.no_repeat_ngram_size > 0:
            generation_kwargs["no_repeat_ngram_size"] = int(self.no_repeat_ngram_size)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            generation_kwargs["pad_token_id"] = eos_token_id
        elif pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        inference_mode = getattr(self.torch_module, "inference_mode", None)
        context = inference_mode() if callable(inference_mode) else contextlib.nullcontext()
        with context:
            outputs = self.model.generate(**batch, **generation_kwargs)
        if hasattr(outputs, "tolist"):
            outputs = outputs.tolist()
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], list):
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            decoded = self.tokenizer.decode(outputs, skip_special_tokens=True)
        text = _strip_prompt_prefix(prompt, str(decoded).strip())
        return text if text else ""


@lru_cache(maxsize=4)
def _load_local_hf_backend(
    model_name: str,
    device: str,
    dtype_name: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    no_repeat_ngram_size: int,
    trust_remote_code: bool,
    token_env: str,
    revision: str,
    cache_dir: str,
    local_files_only: bool,
) -> _LocalHFBackend:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - dependency-dependent
        raise RuntimeError("local_hf provider requires torch and transformers to be installed") from exc

    resolved_device = _resolve_local_hf_device(torch, device)
    torch_dtype = _parse_torch_dtype(torch, dtype_name, device=resolved_device)
    token = resolve_token_env_value(token_env) if token_env else ""
    resolved_cache_dir = _resolve_local_hf_cache_dir(cache_dir, model_name, local_files_only=local_files_only)
    resolved_revision = _resolved_local_hf_revision(model_name, revision)
    load_target = model_name
    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if local_files_only and resolved_cache_dir:
        snapshot_path = _resolve_local_hf_snapshot_path(
            resolved_cache_dir,
            model_name,
            revision=resolved_revision,
        )
        if snapshot_path:
            load_target = snapshot_path
    if token:
        tokenizer_kwargs["token"] = token
        model_kwargs["token"] = token
    if resolved_revision and load_target == model_name:
        tokenizer_kwargs["revision"] = resolved_revision
        model_kwargs["revision"] = resolved_revision
    if resolved_cache_dir:
        tokenizer_kwargs["cache_dir"] = resolved_cache_dir
        model_kwargs["cache_dir"] = resolved_cache_dir
    if local_files_only:
        tokenizer_kwargs["local_files_only"] = True
        model_kwargs["local_files_only"] = True

    tokenizer = AutoTokenizer.from_pretrained(load_target, **tokenizer_kwargs)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
    model = AutoModelForCausalLM.from_pretrained(load_target, **model_kwargs)
    if resolved_device != "cpu":
        model = model.to(resolved_device)
    if hasattr(model, "eval"):
        model.eval()
    return _LocalHFBackend(
        model=model,
        tokenizer=tokenizer,
        torch_module=torch,
        device=resolved_device,
        max_new_tokens=max(1, int(max_new_tokens)),
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_p=float(top_p),
        no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
    )


@dataclass(slots=True)
class LocalCommandProvider:
    name: str = "local_command"
    command: str = ""
    shell: bool = False
    timeout: float = 120.0
    encoding: str = "utf-8"
    env: dict[str, str] = field(default_factory=dict)

    def generate(self, example: BenchmarkExample, *, seed: int = 0) -> str:
        if not self.command:
            return ""
        cmd = self.command
        if not self.shell:
            cmd = shlex.split(cmd)
        completed = subprocess.run(
            cmd,
            input=example.prompt,
            capture_output=True,
            text=True,
            encoding=self.encoding,
            timeout=self.timeout,
            env={**os.environ, **self.env},
            shell=self.shell,
            check=False,
        )
        stdout = (completed.stdout or "").strip()
        return stdout


def available_providers() -> tuple[str, ...]:
    return ("offline_mock", "local_hf", "local_command")


_REMOVED_API_PROVIDER_ISSUE = (
    "API support removed: openai_compatible providers are no longer supported; use local_hf, local_command, or offline_mock"
)


def summarize_provider_configuration(mode: str, parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
    params = dict(parameters or {})
    mode = str(mode or "offline_mock").lower()
    if mode == "openai_compatible":
        return {
            "provider_mode": mode,
            "provider_type": "removed",
            "provider_removed": True,
            "provider_note": _REMOVED_API_PROVIDER_ISSUE,
        }
    if mode == "local_hf":
        model_name = str(params.get("model", params.get("name", "")))
        cache_dir = str(params.get("cache_dir", "")).strip()
        revision = _resolved_local_hf_revision(model_name, str(params.get("revision", "")))
        return {
            "provider_mode": mode,
            "provider_type": "local_hf",
            "provider_model": model_name,
            "provider_revision": revision,
            "provider_device": str(params.get("device", "auto")),
            "provider_dtype": _normalize_dtype_name(params.get("dtype", "auto"), device=str(params.get("device", "auto"))),
            "provider_max_new_tokens": int(params.get("max_new_tokens", 256)),
            "provider_do_sample": _bool_from_value(params.get("do_sample"), False),
            "provider_temperature": float(params.get("temperature", 0.0)),
            "provider_top_p": float(params.get("top_p", 1.0)),
            "provider_no_repeat_ngram_size": int(params.get("no_repeat_ngram_size", 0)),
            "provider_trust_remote_code": _bool_from_value(params.get("trust_remote_code"), False),
            "provider_token_env": str(params.get("token_env", "HF_ACCESS_TOKEN")),
            "provider_local_files_only": _bool_from_value(params.get("local_files_only"), False),
            "provider_cache_dir_set": bool(cache_dir),
        }
    return {
        "provider_mode": mode,
        "provider_type": mode,
    }


def validate_provider_configuration(mode: str, parameters: Mapping[str, Any] | None = None, *, env: Mapping[str, str] | None = None) -> list[str]:
    params = dict(parameters or {})
    mode = str(mode or "offline_mock").lower()
    issues: list[str] = []
    if mode == "openai_compatible":
        return [_REMOVED_API_PROVIDER_ISSUE]
    if mode != "local_hf":
        return issues

    env = env or os.environ
    model = str(params.get("model", params.get("name", ""))).strip()
    device = str(params.get("device", "auto")).strip()
    dtype = str(params.get("dtype", "auto")).strip().lower()
    max_new_tokens = params.get("max_new_tokens", 256)
    do_sample = _bool_from_value(params.get("do_sample"), False)
    temperature = params.get("temperature", 0.0)
    top_p = params.get("top_p", 1.0)
    no_repeat_ngram_size = params.get("no_repeat_ngram_size", 0)
    if not model:
        issues.append("local_hf provider requires provider.parameters.model")
    if not device:
        issues.append("local_hf provider requires provider.parameters.device")
    try:
        if int(max_new_tokens) <= 0:
            issues.append("local_hf provider requires provider.parameters.max_new_tokens to be a positive integer")
    except Exception:
        issues.append("local_hf provider requires provider.parameters.max_new_tokens to be a positive integer")
    if dtype and _normalize_dtype_name(dtype, device=device) not in {"float16", "bfloat16", "float32"}:
        issues.append(
            f"unsupported local_hf dtype '{dtype}' (expected auto, float16, bfloat16, fp16, bf16, float32, or fp32)"
        )
    try:
        if int(no_repeat_ngram_size) < 0:
            issues.append("local_hf provider requires provider.parameters.no_repeat_ngram_size to be a non-negative integer")
    except Exception:
        issues.append("local_hf provider requires provider.parameters.no_repeat_ngram_size to be a non-negative integer")
    try:
        temperature_value = float(temperature)
        if temperature_value < 0:
            issues.append("local_hf provider requires provider.parameters.temperature to be non-negative")
    except Exception:
        issues.append("local_hf provider requires provider.parameters.temperature to be numeric")
        temperature_value = 0.0
    try:
        top_p_value = float(top_p)
        if not (0.0 < top_p_value <= 1.0):
            issues.append("local_hf provider requires provider.parameters.top_p to be within (0, 1]")
    except Exception:
        issues.append("local_hf provider requires provider.parameters.top_p to be numeric")
        top_p_value = 1.0
    if do_sample and temperature_value <= 0:
        issues.append("local_hf provider requires provider.parameters.temperature > 0 when do_sample=true")
    return issues


def build_provider(mode: str, parameters: dict[str, Any] | None = None) -> CompletionProvider:
    params = dict(parameters or {})
    mode = (mode or "offline_mock").lower()
    if mode == "offline_mock":
        return OfflineMockProvider()
    if mode == "local_hf":
        return LocalHFProvider(
            model=str(params.get("model", params.get("name", ""))),
            device=str(params.get("device", "auto")),
            dtype=str(params.get("dtype", "auto")),
            max_new_tokens=int(params.get("max_new_tokens", 256)),
            do_sample=_bool_from_value(params.get("do_sample"), False),
            temperature=float(params.get("temperature", 0.0)),
            top_p=float(params.get("top_p", 1.0)),
            no_repeat_ngram_size=int(params.get("no_repeat_ngram_size", 0)),
            trust_remote_code=_bool_from_value(params.get("trust_remote_code"), False),
            token_env=str(params.get("token_env", "HF_ACCESS_TOKEN")),
            revision=str(params.get("revision", "")),
            cache_dir=str(params.get("cache_dir", "")),
            local_files_only=_bool_from_value(params.get("local_files_only"), False),
        )
    if mode == "openai_compatible":
        raise KeyError(_REMOVED_API_PROVIDER_ISSUE)
    if mode == "local_command":
        command = params.get("command", "")
        if isinstance(command, (list, tuple)):
            command = " ".join(shlex.quote(str(part)) for part in command)
        return LocalCommandProvider(
            command=str(command),
            shell=bool(params.get("shell", False)),
            timeout=float(params.get("timeout", 120.0)),
            encoding=str(params.get("encoding", "utf-8")),
            env={str(k): str(v) for k, v in dict(params.get("env", {})).items()},
        )
    raise KeyError(f"unknown provider mode: {mode}")
