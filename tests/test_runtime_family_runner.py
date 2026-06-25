from __future__ import annotations

from pathlib import Path

from scripts.run_runtime_family import _default_config_path, _default_config_path_for_watermark, _selected_watermarks


def test_runtime_family_runner_defaults_to_runtime_official():
    assert _selected_watermarks(None, "runtime_official") == (
        "stone_runtime",
        "sweet_runtime",
        "ewd_runtime",
        "kgw_runtime",
    )


def test_runtime_family_runner_preserves_stone_family_alias():
    assert _selected_watermarks(None, "stone_family") == (
        "stone_runtime",
        "sweet_runtime",
        "ewd_runtime",
        "kgw_runtime",
    )


def test_runtime_family_runner_preserves_explicit_selection():
    assert _selected_watermarks(["STONE_RUNTIME", "kgw_runtime"], "runtime_official") == (
        "stone_runtime",
        "kgw_runtime",
    )


def test_runtime_family_runner_defaults_to_runtime_official_template():
    args = type("Args", (), {"config": None, "family": "runtime_official"})()

    assert _default_config_path(args) == Path("configs/public_humaneval_plus_stone_runtime.yaml")


def test_runtime_family_runner_uses_method_specific_templates_by_default():
    args = type("Args", (), {"config": None, "family": "runtime_official"})()

    assert _default_config_path_for_watermark(args, "stone_runtime") == Path("configs/public_humaneval_plus_stone_runtime.yaml")
    assert _default_config_path_for_watermark(args, "sweet_runtime") == Path("configs/public_humaneval_plus_sweet_runtime.yaml")
    assert _default_config_path_for_watermark(args, "ewd_runtime") == Path("configs/public_humaneval_plus_ewd_runtime.yaml")
    assert _default_config_path_for_watermark(args, "kgw_runtime") == Path("configs/public_humaneval_plus_kgw_runtime.yaml")
