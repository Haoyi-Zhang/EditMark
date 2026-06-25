from __future__ import annotations

import json
from pathlib import Path

import click

from .config import build_experiment_config, load_config
from .pipeline import generate_corpus, run_experiment
from .utils import scrub_paths


@click.group()
def cli() -> None:
    """Executable benchmark tooling for code watermark evaluation."""


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--benchmark", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--seed", type=int, default=None)
@click.option("--count", type=int, default=None)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--provider-mode", type=str, default=None)
@click.option("--provider-model", type=str, default=None)
@click.option("--provider-command", type=str, default=None)
@click.option("--provider-timeout", type=float, default=None)
@click.option("--provider-temperature", type=float, default=None)
def run(
    config_path: Path | None,
    benchmark: Path | None,
    seed: int | None,
    count: int | None,
    output: Path | None,
    provider_mode: str | None,
    provider_model: str | None,
    provider_command: str | None,
    provider_timeout: float | None,
    provider_temperature: float | None,
) -> None:
    source = load_config(config_path) if config_path else {}
    overrides = {}
    benchmark_overrides = {}
    if seed is not None:
        overrides["seed"] = seed
    if count is not None:
        benchmark_overrides["limit"] = count
    if output is not None:
        overrides["output_path"] = str(output)
    provider_overrides = {}
    provider_parameters = {}
    if provider_mode is not None:
        provider_overrides["mode"] = provider_mode
    if provider_model is not None:
        provider_parameters["model"] = provider_model
    if provider_command is not None:
        provider_parameters["command"] = provider_command
    if provider_timeout is not None:
        provider_parameters["timeout"] = provider_timeout
    if provider_temperature is not None:
        provider_parameters["temperature"] = provider_temperature
    if provider_parameters:
        provider_overrides["parameters"] = provider_parameters
    if provider_overrides:
        overrides["provider"] = provider_overrides
    if benchmark_overrides:
        benchmark_config = dict(source.raw.get("benchmark", {})) if hasattr(source, "raw") else dict(source.get("benchmark", {}))
        benchmark_config.update(benchmark_overrides)
        overrides["benchmark"] = benchmark_config
    if benchmark is not None:
        benchmark_override = dict(
            overrides.get("benchmark", source.raw.get("benchmark", {}) if hasattr(source, "raw") else source.get("benchmark", {}))
        )
        benchmark_override.update({"prepared_output": str(benchmark), "source": str(benchmark)})
        overrides["benchmark"] = benchmark_override
    config = build_experiment_config(source, **overrides)
    result = run_experiment(config)
    click.echo(json.dumps(result.report.as_dict(), indent=2, sort_keys=True))


@cli.command()
@click.option("--count", type=int, default=5)
@click.option("--seed", type=int, default=7)
def synthesize(count: int, seed: int) -> None:
    corpus = generate_corpus(count, seed=seed)
    click.echo(json.dumps([example.as_dict() for example in corpus], indent=2, sort_keys=True))


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def sanitize(path: Path) -> None:
    data = path.read_text(encoding="utf-8")
    click.echo(scrub_paths(data))


if __name__ == "__main__":
    cli()
