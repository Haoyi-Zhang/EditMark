from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

from posteditbench.config import build_experiment_config, load_config, validate_experiment_config
from posteditbench.models import BenchmarkExample
from posteditbench.providers import _load_local_hf_backend, _resolve_local_hf_snapshot_path, build_provider


LOCAL_HF_CONFIGS = (
    "tests/fixtures/configs/local_hf_starcoder2.yaml",
    "tests/fixtures/configs/local_hf_starcoder2_3b.yaml",
    "tests/fixtures/configs/local_hf_qwen25_7b.yaml",
    "tests/fixtures/configs/local_hf_qwen25_1p5b.yaml",
    "tests/fixtures/configs/local_hf_qwen25_14b.yaml",
    "tests/fixtures/configs/local_hf_deepseek_coder_6p7b.yaml",
    "tests/fixtures/configs/public_humaneval_plus_local_hf_starcoder2.yaml",
    "tests/fixtures/configs/public_humaneval_plus_local_hf_starcoder2_3b.yaml",
    "tests/fixtures/configs/public_humaneval_plus_local_hf_qwen25_7b.yaml",
    "tests/fixtures/configs/public_humaneval_plus_local_hf_qwen25_1p5b.yaml",
    "tests/fixtures/configs/public_humaneval_plus_local_hf_qwen25_14b.yaml",
    "tests/fixtures/configs/public_humaneval_plus_local_hf_deepseek_coder_6p7b.yaml",
    "tests/fixtures/configs/public_mbpp_plus_local_hf_starcoder2.yaml",
    "tests/fixtures/configs/public_mbpp_plus_local_hf_starcoder2_3b.yaml",
    "tests/fixtures/configs/public_mbpp_plus_local_hf_qwen25_7b.yaml",
    "tests/fixtures/configs/public_mbpp_plus_local_hf_qwen25_1p5b.yaml",
    "tests/fixtures/configs/public_mbpp_plus_local_hf_qwen25_14b.yaml",
    "tests/fixtures/configs/public_mbpp_plus_local_hf_deepseek_coder_6p7b.yaml",
)


def test_local_hf_config_validates_cleanly():
    config = build_experiment_config(load_config(Path("tests/fixtures/configs/local_hf_starcoder2.yaml")))
    assert config.provider_mode == "local_hf"
    assert validate_experiment_config(config) == []
    summary = dict(config.metadata.get("provider_summary", {}))
    assert summary["provider_type"] == "local_hf"
    assert summary["provider_model"] == "bigcode/starcoder2-7b"
    assert summary["provider_device"] == "cuda"
    assert summary["provider_dtype"] == "float16"
    assert summary["provider_max_new_tokens"] == 256
    assert summary["provider_do_sample"] is True
    assert summary["provider_temperature"] == 0.2
    assert summary["provider_top_p"] == 0.95
    assert summary["provider_no_repeat_ngram_size"] == 4
    assert summary["provider_trust_remote_code"] is False
    assert summary["provider_cache_dir_set"] is True


def test_all_local_hf_configs_validate_cleanly():
    for relative_path in LOCAL_HF_CONFIGS:
        config = build_experiment_config(load_config(Path(relative_path)))
        assert config.provider_mode == "local_hf"
        assert validate_experiment_config(config) == []


def test_local_hf_provider_uses_transformers_backend(monkeypatch):
    fake_torch = ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.seed = None
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True, manual_seed_all=lambda seed: setattr(fake_torch, "cuda_seed", seed))
    fake_torch.manual_seed = lambda seed: setattr(fake_torch, "seed", seed)

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
        last_kwargs: dict[str, object] = {}

        def __init__(self):
            self.pad_token_id = None
            self.eos_token_id = 0
            self.eos_token = "<eos>"
            self.pad_token = None

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            cls.last_kwargs = {"model_name": model_name, **kwargs}
            return cls()

        def __call__(self, prompt: str, return_tensors: str = "pt"):
            self.last_prompt = prompt
            FakeTokenizer.last_prompt = prompt
            return {"input_ids": FakeTensor("input_ids"), "attention_mask": FakeTensor("attention_mask")}

        def decode(self, token_ids, skip_special_tokens: bool = True):
            return self.last_prompt + "\nreturn 42\n"

    class FakeModel:
        last_kwargs: dict[str, object] = {}
        last_instance = None

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            cls.last_kwargs = {"model_name": model_name, **kwargs}
            instance = cls()
            cls.last_instance = instance
            return instance

        def to(self, device: str):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

        def generate(self, **kwargs):
            self.last_generate_kwargs = kwargs
            return [[1, 2, 3]]

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=FakeTokenizer.from_pretrained)
    fake_transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=FakeModel.from_pretrained)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    _load_local_hf_backend.cache_clear()

    config = build_experiment_config(load_config(Path("tests/fixtures/configs/local_hf_starcoder2.yaml")))
    provider = build_provider(config.provider_mode, dict(config.provider_parameters))
    example = BenchmarkExample(
        example_id="example-1",
        language="python",
        prompt="def add(a, b):",
        reference_solution="def add(a, b):\n    return a + b\n",
    )

    text = provider.generate(example, seed=23)

    assert text == "return 42"
    assert FakeTokenizer.last_kwargs["model_name"] == "bigcode/starcoder2-7b"
    assert FakeTokenizer.last_kwargs["trust_remote_code"] is False
    assert FakeTokenizer.last_kwargs["cache_dir"] in {"model_cache/huggingface", "model_cache/huggingface/hub"}
    assert FakeModel.last_kwargs["torch_dtype"] == "float16"
    assert FakeModel.last_kwargs["low_cpu_mem_usage"] is True
    assert fake_torch.seed == 23
    assert fake_torch.cuda_seed == 23
    assert FakeModel.last_instance.last_generate_kwargs["max_new_tokens"] == 256
    assert FakeModel.last_instance.last_generate_kwargs["do_sample"] is True
    assert FakeModel.last_instance.last_generate_kwargs["temperature"] == 0.2
    assert FakeModel.last_instance.last_generate_kwargs["top_p"] == 0.95
    assert FakeModel.last_instance.last_generate_kwargs["no_repeat_ngram_size"] == 4


def test_local_hf_provider_does_not_fallback_to_prompt_or_reference(monkeypatch):
    fake_torch = ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
    fake_torch.manual_seed = lambda seed: None
    fake_torch.cuda.manual_seed_all = lambda seed: None

    @contextlib.contextmanager
    def inference_mode():
        yield

    fake_torch.inference_mode = inference_mode

    class FakeTokenizer:
        def __init__(self):
            self.pad_token_id = None
            self.eos_token_id = 0
            self.eos_token = "<eos>"
            self.pad_token = None
            self.last_prompt = ""

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            return cls()

        def __call__(self, prompt: str, return_tensors: str = "pt"):
            self.last_prompt = prompt
            return {"input_ids": SimpleNamespace(to=lambda device: SimpleNamespace()), "attention_mask": SimpleNamespace(to=lambda device: SimpleNamespace())}

        def decode(self, token_ids, skip_special_tokens: bool = True):
            return self.last_prompt

    class FakeModel:
        last_instance = None

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            instance = cls()
            cls.last_instance = instance
            return instance

        def to(self, device: str):
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
    _load_local_hf_backend.cache_clear()

    config = build_experiment_config(load_config(Path("tests/fixtures/configs/local_hf_starcoder2.yaml")))
    provider = build_provider(config.provider_mode, dict(config.provider_parameters))
    example = BenchmarkExample(
        example_id="example-2",
        language="python",
        prompt="def add(a, b):",
        reference_solution="def add(a, b):\n    return a + b\n",
    )

    assert provider.generate(example) == ""


def test_local_hf_snapshot_path_prefers_refs_main(tmp_path: Path) -> None:
    entry = tmp_path / "models--Qwen--Qwen2.5-Coder-14B-Instruct"
    snapshot = entry / "snapshots" / "abc123"
    snapshot.mkdir(parents=True, exist_ok=True)
    (entry / "refs").mkdir(parents=True, exist_ok=True)
    (entry / "refs" / "main").write_text("abc123\n", encoding="utf-8")

    resolved = _resolve_local_hf_snapshot_path(str(tmp_path), "Qwen/Qwen2.5-Coder-14B-Instruct")

    assert resolved == str(snapshot)
