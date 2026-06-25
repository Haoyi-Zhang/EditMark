from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from scripts import prepare_data


def test_collection_input_force_rebuilds_existing_public_source(tmp_path, monkeypatch) -> None:
    prepared = tmp_path / "public.normalized.jsonl"
    prepared.write_text("{}", encoding="utf-8")
    prepared.with_suffix(".manifest.json").write_text("{}", encoding="utf-8")
    calls: list[tuple[str, Path, bool, Path]] = []

    def fake_prepare(name: str, *, output_path: Path, fetch: bool, cache_dir: Path):
        calls.append((name, Path(output_path), fetch, Path(cache_dir)))
        return {"benchmark": name}

    monkeypatch.setattr(prepare_data, "prepare_public_benchmark", fake_prepare)

    resolved = prepare_data._collection_input(
        {"type": "public", "name": "human_eval", "path": str(prepared)},
        {"cache_dir": str(tmp_path / "cache")},
        False,
        force=True,
    )

    assert resolved == prepared
    assert calls == [("human_eval", prepared, False, tmp_path / "cache")]


def test_collection_input_force_rebuilds_existing_crafted_source(tmp_path, monkeypatch) -> None:
    prepared = tmp_path / "crafted.normalized.jsonl"
    prepared.write_text("{}", encoding="utf-8")
    prepared.with_suffix(".manifest.json").write_text("{}", encoding="utf-8")
    calls: list[tuple[str, Path]] = []

    def fake_write(kind: str, *, output_path: Path):
        calls.append((kind, Path(output_path)))
        return {"benchmark": kind}

    monkeypatch.setattr(prepare_data, "write_crafted_benchmark", fake_write)

    resolved = prepare_data._collection_input(
        {"type": "crafted", "name": "crafted_original", "path": str(prepared)},
        {},
        False,
        force=True,
    )

    assert resolved == prepared
    assert calls == [("crafted_original", prepared)]


def test_collection_input_uses_cached_public_source_without_force(tmp_path, monkeypatch) -> None:
    prepared = tmp_path / "public.normalized.jsonl"
    prepared.write_text("{}", encoding="utf-8")
    prepared.with_suffix(".manifest.json").write_text("{}", encoding="utf-8")
    called = False

    def fake_prepare(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("prepare_public_benchmark should not run when force is disabled and cache is present")

    monkeypatch.setattr(prepare_data, "prepare_public_benchmark", fake_prepare)

    resolved = prepare_data._collection_input(
        {"type": "public", "name": "human_eval", "path": str(prepared)},
        {},
        False,
        force=False,
    )

    assert resolved == prepared
    assert called is False
