from __future__ import annotations

from posteditbench.config import build_experiment_config, validate_experiment_config
from posteditbench.providers import build_provider, summarize_provider_configuration


def test_removed_api_provider_configs_fail_with_explicit_message() -> None:
    config = build_experiment_config(
        {
            "benchmark": {"prepared_output": "data/fixtures/benchmark.normalized.jsonl"},
            "provider": {
                "mode": "openai_compatible",
                "parameters": {
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-5.4-mini",
                },
            },
            "watermark": {"scheme": "kgw", "strength": 0.5},
        }
    )

    issues = validate_experiment_config(config)

    assert any("API support removed" in issue for issue in issues)


def test_removed_provider_summary_is_explicit() -> None:
    summary = summarize_provider_configuration(
        "openai_compatible",
        {"base_url": "https://api.openai.com", "model": "gpt-5.4-mini"},
    )
    assert summary["provider_removed"] is True
    assert "API support removed" in summary["provider_note"]


def test_build_provider_rejects_removed_api_mode() -> None:
    try:
        build_provider("openai_compatible", {"model": "gpt-5.4-mini"})
    except KeyError as exc:
        assert "API support removed" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("openai_compatible provider should not build successfully")
