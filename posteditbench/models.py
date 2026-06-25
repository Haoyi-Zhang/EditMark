from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


def _freeze_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        try:
            return _json_safe(value.detach().cpu().tolist())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(dict(value.__dict__))
        except Exception:
            pass
    return str(value)


@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    example_id: str
    language: str
    prompt: str
    reference_solution: str
    reference_tests: tuple[str, ...] = ()
    execution_tests: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = _json_safe(asdict(self))
        data["metadata"] = _json_safe(dict(self.metadata))
        return data


@dataclass(frozen=True, slots=True)
class WatermarkSpec:
    name: str
    secret: str
    payload: str
    strength: float = 1.0
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = _json_safe(asdict(self))
        data["parameters"] = _json_safe(dict(self.parameters))
        return data


@dataclass(frozen=True, slots=True)
class WatermarkedSnippet:
    example_id: str
    language: str
    source: str
    watermark: WatermarkSpec
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = _json_safe(asdict(self))
        data["metadata"] = _json_safe(dict(self.metadata))
        data["watermark"] = self.watermark.as_dict()
        return data


@dataclass(frozen=True, slots=True)
class DetectionResult:
    example_id: str
    method: str
    score: float
    detected: bool
    threshold: float
    evidence: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = _json_safe(asdict(self))
        data["metadata"] = _json_safe(dict(self.metadata))
        return data


@dataclass(frozen=True, slots=True)
class AttackOutcome:
    attack_name: str
    source: str
    changed: bool
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = _json_safe(asdict(self))
        data["metadata"] = _json_safe(dict(self.metadata))
        return data


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    seed: int = 7
    corpus_size: int | None = None
    language: str = "python"
    watermark_name: str = "stone_runtime"
    watermark_secret: str = "redacted"
    watermark_payload: str = "wm"
    watermark_strength: float = 1.0
    attacks: tuple[str, ...] = ("comment_strip", "identifier_rename", "whitespace_normalize")
    attack_parameters: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    corpus_parameters: Mapping[str, Any] = field(default_factory=dict)
    provider_mode: str = "offline_mock"
    provider_parameters: Mapping[str, Any] = field(default_factory=dict)
    validation_scope: str = "python_first"
    output_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = _json_safe(asdict(self))
        data["attacks"] = _json_safe(list(self.attacks))
        data["attack_parameters"] = _json_safe({k: dict(v) for k, v in self.attack_parameters.items()})
        data["corpus_parameters"] = _json_safe(dict(self.corpus_parameters))
        data["provider_parameters"] = _json_safe(dict(self.provider_parameters))
        data["metadata"] = _json_safe(dict(self.metadata))
        return data


BenchmarkConfig = ExperimentConfig


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    example_id: str
    attack_name: str
    task_id: str = ""
    dataset: str = ""
    language: str = ""
    task_category: str = ""
    reference_kind: str = ""
    method_origin: str = ""
    evaluation_track: str = ""
    model_label: str = ""
    baseline_family: str = ""
    baseline_origin: str = ""
    baseline_upstream_commit: str = ""
    source_group: str = ""
    origin_type: str = ""
    family_id: str = ""
    difficulty: str = ""
    attack_severity: float = 0.0
    watermark_scheme: str = ""
    watermark_strength: float = 0.0
    prompt_digest: str = ""
    clean_score: float = 0.0
    watermarked_score: float | None = None
    attacked_score: float = 0.0
    clean_detected: bool = False
    watermarked_detected: bool | None = None
    attacked_detected: bool = False
    quality_score: float = 0.0
    stealth_score: float = 0.0
    mutation_distance: float = 0.0
    watermark_retention: float = 0.0
    robustness_score: float = 0.0
    semantic_validation_available: bool = False
    semantic_preserving: bool | None = None
    status: str = "needs-review"
    comment: str = ""
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = _json_safe(asdict(self))
        data["metadata"] = _json_safe(dict(self.metadata))
        return data

    @property
    def positive_score(self) -> float:
        if self.watermarked_score is not None:
            return float(self.watermarked_score)
        return float(self.clean_score)

    @property
    def positive_detected(self) -> bool:
        if self.watermarked_detected is not None:
            return bool(self.watermarked_detected)
        return bool(self.clean_detected)


def attack_row_supported(row: BenchmarkRow) -> bool:
    metadata = row.metadata if isinstance(row.metadata, Mapping) else {}
    attack_metadata = metadata.get("attack_metadata", {})
    if isinstance(attack_metadata, Mapping) and "supported" in attack_metadata:
        return bool(attack_metadata.get("supported"))
    return True


def supported_attack_rows(rows: Iterable[BenchmarkRow]) -> list[BenchmarkRow]:
    return [row for row in rows if attack_row_supported(row)]


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    config: ExperimentConfig
    rows: tuple[BenchmarkRow, ...]
    summary: Mapping[str, Any]
    output_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return _json_safe({
            "config": self.config.as_dict(),
            "rows": [
                {
                    **row.as_dict(),
                    "attack_outcome_score": float(row.robustness_score),
                }
                for row in self.rows
            ],
            "summary": dict(self.summary),
            "output_path": self.output_path,
        })

    def to_json(self, path: str | Path | None = None) -> str:
        import json

        payload = json.dumps(self.as_dict(), indent=2, sort_keys=True)
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload


@dataclass(frozen=True, slots=True)
class BudgetCurvePoint:
    budget: int
    detector_score: float
    quality_score: float
    semantic_preserving: bool | None = None
    step_name: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "detector_score": self.detector_score,
            "quality_score": self.quality_score,
            "semantic_preserving": self.semantic_preserving,
            "step_name": self.step_name,
        }
