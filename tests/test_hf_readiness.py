from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

from posteditbench.suite import suite_model_revision


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "_hf_readiness.py"
    spec = importlib.util.spec_from_file_location("_hf_readiness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_smoke_load_local_hf_model_forces_offline_mode(monkeypatch, tmp_path) -> None:
    module = _load_module()

    fake_torch = ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)

    @contextlib.contextmanager
    def inference_mode():
        yield

    fake_torch.inference_mode = inference_mode

    @dataclass
    class FakeTensor:
        value: str
        device: str | None = None

        def to(self, device: str):
            self.device = device
            return self

    class FakeTokenizer:
        last_prompt = ""

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            assert kwargs["local_files_only"] is True
            assert "token" not in kwargs
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            return cls()

        def __init__(self):
            self.pad_token_id = None
            self.eos_token_id = 0
            self.eos_token = "<eos>"
            self.pad_token = None

        def __call__(self, prompt: str, return_tensors: str = "pt"):
            FakeTokenizer.last_prompt = prompt
            return {"input_ids": FakeTensor("input_ids"), "attention_mask": FakeTensor("attention_mask")}

        def decode(self, token_ids, skip_special_tokens: bool = True):
            return self.last_prompt + " a + b\n"

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            assert kwargs["local_files_only"] is True
            assert "token" not in kwargs
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            return cls()

        def to(self, device: str):
            self.device = device
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            return [[1, 2, 3]]

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=FakeTokenizer.from_pretrained)
    fake_transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=FakeModel.from_pretrained)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setenv("HF_ACCESS_TOKEN", "secret-token")

    requirement = module.HFModelRequirement(
        model="acme/test-model",
        cache_dir=str(tmp_path / "hf_cache"),
        local_files_only=True,
        token_env="HF_ACCESS_TOKEN",
        device="cuda",
        dtype="float16",
    )

    result = module.smoke_load_local_hf_model(requirement)

    assert result["status"] == "ok"
    assert result["device"] == "cuda"
    assert result["generated_preview"].endswith("a + b\n")


def test_validate_local_hf_cache_rejects_hub_only_layout_when_root_entry_is_required(monkeypatch, tmp_path) -> None:
    module = _load_module()
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    revision = suite_model_revision(model_name)
    entry = tmp_path / "hf_cache" / "hub" / "models--Qwen--Qwen2.5-Coder-7B-Instruct"
    snapshot_dir = entry / "snapshots" / revision
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "config.json").write_text('{"architectures":["MockModel"]}\n', encoding="utf-8")
    (snapshot_dir / "tokenizer_config.json").write_text('{"model_max_length":512}\n', encoding="utf-8")
    (snapshot_dir / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    (snapshot_dir / "model.safetensors").write_bytes(b"fixture")

    @contextlib.contextmanager
    def fake_safe_open(path: str, framework: str = "pt"):
        class Handle:
            @staticmethod
            def keys():
                return ["weight"]

        yield Handle()

    monkeypatch.setattr(module, "safe_open", fake_safe_open)

    requirement = module.HFModelRequirement(
        model=model_name,
        cache_dir=str(tmp_path / "hf_cache"),
        local_files_only=True,
    )

    result = module.validate_local_hf_cache(requirement, require_root_entry=True)

    assert result["status"] == "failed"
    assert result["requested_revision"] == revision
    assert result["root_entry_exists"] is False
    assert result["hub_entry_exists"] is True
    assert result["resolved_snapshot"] == str(snapshot_dir)
    assert any("missing official cache entry" in issue for issue in result["issues"])


def test_validate_local_hf_cache_requires_tokenizer_assets(monkeypatch, tmp_path) -> None:
    module = _load_module()
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    revision = suite_model_revision(model_name)
    entry = tmp_path / "hf_cache" / "hub" / "models--Qwen--Qwen2.5-Coder-7B-Instruct"
    snapshot_dir = entry / "snapshots" / revision
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "config.json").write_text('{"architectures":["MockModel"]}\n', encoding="utf-8")
    (snapshot_dir / "model.safetensors").write_bytes(b"fixture")

    @contextlib.contextmanager
    def fake_safe_open(path: str, framework: str = "pt"):
        class Handle:
            @staticmethod
            def keys():
                return ["weight"]

        yield Handle()

    monkeypatch.setattr(module, "safe_open", fake_safe_open)

    requirement = module.HFModelRequirement(
        model=model_name,
        cache_dir=str(tmp_path / "hf_cache"),
        local_files_only=True,
    )

    result = module.validate_local_hf_cache(requirement, require_root_entry=True)

    assert result["status"] == "failed"
    assert any("missing required asset tokenizer_config.json" in issue for issue in result["issues"])
    assert any("missing tokenizer assets" in issue for issue in result["issues"])


def test_smoke_load_local_hf_evaluator_forces_offline_mode(monkeypatch, tmp_path) -> None:
    module = _load_module()

    fake_torch = ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)

    @contextlib.contextmanager
    def no_grad():
        yield

    fake_torch.no_grad = no_grad

    @dataclass
    class FakeTensor:
        value: str
        device: str | None = None

        def to(self, device: str):
            self.device = device
            return self

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            assert kwargs["local_files_only"] is True
            assert "token" not in kwargs
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            return cls()

        def __call__(self, text: str, return_tensors: str = "pt", truncation: bool = True, max_length: int = 128):
            return {"input_ids": FakeTensor("input_ids"), "attention_mask": FakeTensor("attention_mask")}

    class FakeLoss:
        @staticmethod
        def item():
            return 1.25

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            assert kwargs["local_files_only"] is True
            assert "token" not in kwargs
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            return cls()

        def to(self, device: str):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(loss=FakeLoss())

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=FakeTokenizer.from_pretrained)
    fake_transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=FakeModel.from_pretrained)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setenv("HF_ACCESS_TOKEN", "secret-token")

    requirement = module.HFModelRequirement(
        model="acme/test-model",
        cache_dir=str(tmp_path / "hf_cache"),
        local_files_only=True,
        token_env="HF_ACCESS_TOKEN",
        device="cuda",
        dtype="float16",
    )

    result = module.smoke_load_local_hf_evaluator(requirement)

    assert result["status"] == "ok"
    assert result["device"] == "cuda"
    assert result["loss"] == 1.25
