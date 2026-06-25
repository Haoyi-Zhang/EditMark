from __future__ import annotations

import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "check_model_access.py"
    spec = importlib.util.spec_from_file_location("check_model_access", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_model_matrix_is_populated():
    module = _load_module()
    assert "Qwen/Qwen2.5-Coder-14B-Instruct" in module.DEFAULT_MODELS
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" in module.DEFAULT_MODELS
    assert "Qwen/Qwen2.5-Coder-1.5B-Instruct" in module.DEFAULT_MODELS
    assert "bigcode/starcoder2-7b" in module.DEFAULT_MODELS
    assert "deepseek-ai/deepseek-coder-6.7b-instruct" in module.DEFAULT_MODELS
    assert len(module.DEFAULT_MODELS) == 5
    assert all("codegemma" not in model.lower() for model in module.DEFAULT_MODELS)


def test_probe_marks_accessible_on_success(monkeypatch):
    module = _load_module()
    seen = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _urlopen(request, timeout=30.0):
        seen["url"] = request.full_url
        return _Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", _urlopen)
    result = module._probe("Qwen/Qwen2.5-Coder-7B-Instruct", "c03e6d358207e414f1eca0bb1891e29f1db0e242", "token", 30.0)
    assert result["accessible"] is True
    assert result["status"] == 200
    assert result["requested_revision"] == "c03e6d358207e414f1eca0bb1891e29f1db0e242"
    assert seen["url"].endswith("/Qwen/Qwen2.5-Coder-7B-Instruct/resolve/c03e6d358207e414f1eca0bb1891e29f1db0e242/config.json")


def test_default_targets_use_pinned_revisions():
    module = _load_module()

    targets = module._resolve_targets(module.DEFAULT_MODELS, None)

    assert ("Qwen/Qwen2.5-Coder-14B-Instruct", "aedcc2d42b622764e023cf882b6652e646b95671") in targets
    assert ("bigcode/starcoder2-7b", "bb9afde76d7945da5745592525db122d4d729eb1") in targets


def test_custom_targets_require_explicit_revision_outside_canonical_roster():
    module = _load_module()

    try:
        module._resolve_targets(["acme/custom-model"], None)
    except ValueError as exc:
        assert "provide an explicit revision" in str(exc)
    else:
        raise AssertionError("expected custom model without explicit revision to fail")


def test_main_allows_public_probe_without_token_by_default(monkeypatch, capsys):
    module = _load_module()

    monkeypatch.delenv("HF_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(module, "DEFAULT_MODELS", ("Qwen/Qwen2.5-Coder-7B-Instruct",))
    monkeypatch.setattr(
        module,
        "_probe",
        lambda model_id, revision, token, timeout: {
            "model": model_id,
            "requested_revision": revision,
            "accessible": True,
            "status": 200,
            "reason": "ok",
            "token_used": bool(token),
        },
    )
    monkeypatch.setattr(module, "parse_args", lambda: module.argparse.Namespace(
        token_env="HF_ACCESS_TOKEN",
        models=None,
        revisions=None,
        timeout=30.0,
        require_all=True,
        require_token=False,
    ))

    assert module.main() == 0
    payload = capsys.readouterr().out
    assert '"token_present": false' in payload


def test_main_can_still_require_token(monkeypatch):
    module = _load_module()
    monkeypatch.delenv("HF_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(module, "parse_args", lambda: module.argparse.Namespace(
        token_env="HF_ACCESS_TOKEN",
        models=None,
        revisions=None,
        timeout=30.0,
        require_all=False,
        require_token=True,
    ))

    try:
        module.main()
    except SystemExit as exc:
        assert str(exc) == "Missing HF_ACCESS_TOKEN"
    else:
        raise AssertionError("expected missing-token failure")
