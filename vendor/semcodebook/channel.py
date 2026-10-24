from __future__ import annotations

from dataclasses import dataclass

from .protocol import CarrierEvidence, CarrierScheduleEntry


@dataclass(frozen=True, slots=True)
class SymbolObservation:
    family: str
    role: str
    bit_index: int | None
    prob_zero: float
    prob_one: float
    reliability: float
    erased: bool
    applicable: bool


@dataclass(frozen=True, slots=True)
class ChannelSummary:
    data_observations: tuple[SymbolObservation, ...]
    anchor_observation: SymbolObservation | None
    erasure_count: int
    mean_reliability: float
    applicable_ratio: float
    notes: tuple[str, ...] = ()


def observation_from_evidence(
    evidence: CarrierEvidence,
    entry: CarrierScheduleEntry,
    *,
    erasure_threshold: float = 0.12,
) -> SymbolObservation:
    reliability = round(abs(evidence.prob_one - evidence.prob_zero), 4)
    erased = (not evidence.applicable) or evidence.confidence <= 0.0 or reliability < erasure_threshold
    return SymbolObservation(
        family=evidence.family,
        role=entry.role,
        bit_index=entry.bit_index,
        prob_zero=evidence.prob_zero,
        prob_one=evidence.prob_one,
        reliability=reliability,
        erased=erased,
        applicable=evidence.applicable,
    )


def summarize_channel(
    ordered_evidence: tuple[CarrierEvidence, ...],
    schedule: tuple[CarrierScheduleEntry, ...],
) -> ChannelSummary:
    observations = tuple(
        observation_from_evidence(item, entry)
        for item, entry in zip(ordered_evidence, schedule, strict=True)
    )
    data_observations = tuple(item for item in observations if item.role == "data")
    anchor_observation = next((item for item in observations if item.role == "anchor"), None)
    erasure_count = sum(1 for item in data_observations if item.erased)
    applicable_ratio = round(
        sum(1.0 for item in data_observations if item.applicable) / len(data_observations),
        4,
    ) if data_observations else 0.0
    mean_reliability = round(
        sum(item.reliability for item in data_observations) / len(data_observations),
        4,
    ) if data_observations else 0.0
    return ChannelSummary(
        data_observations=data_observations,
        anchor_observation=anchor_observation,
        erasure_count=erasure_count,
        mean_reliability=mean_reliability,
        applicable_ratio=applicable_ratio,
        notes=("explicit_erasure_model", "anchor_channel_slot"),
    )


def soft_observations(summary: ChannelSummary) -> tuple[tuple[float, float], ...]:
    return tuple((item.prob_zero, item.prob_one) for item in summary.data_observations[:7])


def soft_anchor_observation(summary: ChannelSummary) -> tuple[float, float] | None:
    if summary.anchor_observation is None:
        return None
    return (summary.anchor_observation.prob_zero, summary.anchor_observation.prob_one)
