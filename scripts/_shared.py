from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_CACHE_DIR = PROJECT_ROOT / "model_cache"
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_FIXTURE = DATA_DIR / "fixtures" / "benchmarks" / "synthetic_tasks.jsonl"
DEFAULT_NORMALIZED_BENCHMARK = DATA_DIR / "fixtures" / "benchmark.normalized.jsonl"
DEFAULT_ATTACKS = DATA_DIR / "fixtures" / "attacks" / "attack_matrix.json"
DEFAULT_INTERIM_DIR = DATA_DIR / "interim"
DEFAULT_RUNS_DIR = RESULTS_DIR / "runs"

STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "be",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "return",
    "the",
    "to",
    "with",
}

ATTACK_SEVERITY = {
    "noop": 0.0,
    "formatting": 0.08,
    "comment_strip": 0.12,
    "renaming": 0.20,
    "control_flow_flatten": 0.42,
    "token_shuffle": 0.45,
    "paraphrase": 0.32,
    "translation_roundtrip": 0.58,
    "adversarial_edit": 0.78,
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8", newline="\n")


def load_json(path: Path) -> Any:
    return json.loads(read_text(path))


def dump_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=str(path.parent),
        delete=False,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    try:
        for _ in range(5):
            try:
                temp_path.replace(path)
                return
            except PermissionError:
                time.sleep(0.05)
        path.write_text(payload, encoding="utf-8", newline="\n")
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in read_text(path).splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            handle.write("\n")


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    payload = load_json(path)
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    raise ValueError(f"Expected list-like JSON payload in {path}")


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(read_text(path))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} must contain JSON-compatible YAML. "
            "Use strict JSON syntax inside the .yaml file."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a top-level object in {path}")
    return payload


def resolve_prepared_benchmark_config(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    project_root = (root or PROJECT_ROOT).resolve()
    config_path = path if path.is_absolute() else project_root / path
    config = load_config(config_path)
    benchmark_cfg = dict(config.get("benchmark", {}))
    paths_cfg = dict(config.get("paths", {}))
    prepared = Path(
        benchmark_cfg.get("prepared_output")
        or paths_cfg.get("prepared_benchmark")
        or benchmark_cfg.get("source")
        or DEFAULT_NORMALIZED_BENCHMARK
    )
    if not prepared.is_absolute():
        prepared = project_root / prepared
    prepared = prepared.resolve()
    return {
        "config_path": config_path.resolve(),
        "prepared_path": prepared,
        "manifest_path": prepared.with_suffix(".manifest.json"),
        "compose": bool(benchmark_cfg.get("collection_sources")),
        "config": config,
    }


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def stable_number(*parts: object, scale: float = 1.0) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], "big") / 2**64
    return raw * scale


def stable_choice(options: Sequence[str], *parts: object) -> str:
    if not options:
        raise ValueError("options must not be empty")
    index = int(stable_number(*parts, scale=len(options))) % len(options)
    return options[index]


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def list_files(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    yield child


def group_mean(rows: Iterable[dict[str, Any]], key: str, value: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        buckets[str(row[key])].append(float(row[value]))
    return {bucket: sum(values) / len(values) for bucket, values in buckets.items() if values}


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        rendered.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(rendered)


def parse_attack_matrix(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    attacks = payload.get("attacks")
    if not isinstance(attacks, list):
        raise ValueError(f"{path} must define an 'attacks' list")
    normalized: list[dict[str, Any]] = []
    for attack in attacks:
        if not isinstance(attack, dict) or "name" not in attack:
            raise ValueError("Each attack entry must be an object with a name")
        item = dict(attack)
        item["severity"] = float(item.get("severity", ATTACK_SEVERITY.get(str(item["name"]), 0.25)))
        item["description"] = str(item.get("description", ""))
        normalized.append(item)
    return normalized


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-_.") or "artifact"
