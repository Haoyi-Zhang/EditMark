from __future__ import annotations

from dataclasses import dataclass

from .adaptive_ecc import AdaptiveDecodeResult
from .ecc import anchor_bit_for_nibble
from .protocol import CarrierEvidence, CarrierScheduleEntry


@dataclass(frozen=True, slots=True)
class PositiveSupportCertificate:
    accepted: bool
    score: float
    reasons: tuple[str, ...]
    primary_failure_reason: str | None
    applicable_ratio: float
    structural_level_count: int
    positive_support: int
    schedule_aligned_support: int
    rewrite_backed_family_count: int
    rewrite_backed_level_count: int


@dataclass(frozen=True, slots=True)
class NegativeControlAssessment:
    accepted: bool
    score: float
    reasons: tuple[str, ...]
    primary_failure_reason: str | None
    failed_gates: tuple[str, ...]
    applicable_ratio: float
    structural_level_count: int
    positive_support: int


def _base_fingerprints(entry: CarrierScheduleEntry) -> set[str]:
    fingerprints: set[str] = set()
    for note in entry.notes:
        text = str(note)
        if text.startswith("fingerprint:"):
            fingerprints.add(text.split(":", 1)[1])
    return fingerprints


def _locally_rewrite_backed(entry: CarrierScheduleEntry) -> bool:
    return any(
        note == "discriminative_rewrite_backed_carrier"
        or str(note).startswith("target_alignment_reason:rewrite_aligned")
        for note in entry.notes
    )


def _generation_planned(entry: CarrierScheduleEntry) -> bool:
    return any(note == "discriminative_generation_planned_carrier" for note in entry.notes)


def _base_observed_bit(entry: CarrierScheduleEntry) -> int | None:
    for note in entry.notes:
        text = str(note)
        if not text.startswith("target_alignment_base_bit:"):
            continue
        value = text.split(":", 1)[1]
        if value in {"0", "1"}:
            return int(value)
    return None


def _structural_delta_observed(entry: CarrierScheduleEntry, current_structural_fingerprint: str | None) -> bool:
    if not current_structural_fingerprint:
        return False
    fingerprints = _base_fingerprints(entry)
    return bool(fingerprints) and all(fingerprint != current_structural_fingerprint for fingerprint in fingerprints)


def _generation_planned_bit_delta(entry: CarrierScheduleEntry, observed_bit: int | None) -> bool:
    base_bit = _base_observed_bit(entry)
    return (
        base_bit in {0, 1}
        and observed_bit in {0, 1}
        and entry.target_bit in {0, 1}
        and observed_bit == entry.target_bit
        and observed_bit != base_bit
    )


def _discriminative_support(
    entry: CarrierScheduleEntry,
    current_structural_fingerprint: str | None,
    observed_bit: int | None = None,
) -> bool:
    if _locally_rewrite_backed(entry):
        return True
    return _generation_planned(entry) and (
        _structural_delta_observed(entry, current_structural_fingerprint)
        or _generation_planned_bit_delta(entry, observed_bit)
    )


def _support_facts(
    ordered_evidence: tuple[CarrierEvidence, ...],
    schedule: tuple[CarrierScheduleEntry, ...],
    decode_result: AdaptiveDecodeResult,
    *,
    current_structural_fingerprint: str | None = None,
) -> dict[str, object]:
    data_evidence = tuple(item for item, entry in zip(ordered_evidence, schedule, strict=True) if entry.role == "data")
    applicable_ratio = round(
        sum(1.0 for item in data_evidence if item.applicable and item.confidence > 0.0) / len(data_evidence),
        4,
    ) if data_evidence else 0.0

    schedule_aligned_support = 0
    rewrite_backed_slot_count = 0
    rewrite_backed_families: set[str] = set()
    rewrite_backed_levels: set[str] = set()
    high_confidence_families: set[str] = set()
    structural_levels: set[str] = set()
    anchor_required = False
    anchor_evidence_observed = False
    anchor_gate = False
    generation_planned_slot_count = sum(1 for entry in schedule if entry.role == "data" and _generation_planned(entry))
    generation_planned_delta_count = 0
    expected_anchor = anchor_bit_for_nibble(decode_result.value) if decode_result.value is not None else None

    for item, entry in zip(ordered_evidence, schedule, strict=True):
        if item.confidence > 0.0 and item.structural_level:
            structural_levels.add(item.structural_level)
        if not item.applicable or item.confidence < 0.72 or not item.option.startswith("bit_"):
            continue
        observed_bit = int(item.option[-1])
        discriminative_slot = _discriminative_support(entry, current_structural_fingerprint, observed_bit)
        if entry.role == "data" and discriminative_slot and observed_bit == entry.target_bit:
            rewrite_backed_slot_count += 1
            if _generation_planned(entry):
                generation_planned_delta_count += 1
        high_confidence_families.add(item.family)
        if entry.role == "anchor":
            if _discriminative_support(entry, current_structural_fingerprint, observed_bit):
                anchor_required = True
                anchor_evidence_observed = True
                anchor_gate = expected_anchor is not None and observed_bit == expected_anchor and item.confidence >= 0.82
            continue
        if entry.bit_index is None or entry.bit_index >= len(decode_result.codeword):
            continue
        if observed_bit != decode_result.codeword[entry.bit_index]:
            continue
        schedule_aligned_support += 1
        if _discriminative_support(entry, current_structural_fingerprint, observed_bit):
            rewrite_backed_families.add(item.family)
            if item.structural_level:
                rewrite_backed_levels.add(item.structural_level)

    return {
        "applicable_ratio": applicable_ratio,
        "schedule_aligned_support": schedule_aligned_support,
        "rewrite_backed_slot_count": rewrite_backed_slot_count,
        "rewrite_backed_families": rewrite_backed_families,
        "rewrite_backed_levels": rewrite_backed_levels,
        "high_confidence_families": high_confidence_families,
        "structural_level_count": len(structural_levels),
        "anchor_required": anchor_required,
        "anchor_evidence_observed": anchor_evidence_observed,
        "anchor_gate": anchor_gate,
        "generation_planned_slot_count": generation_planned_slot_count,
        "generation_planned_delta_count": generation_planned_delta_count,
    }


def certify_positive_support(
    ordered_evidence: tuple[CarrierEvidence, ...],
    schedule: tuple[CarrierScheduleEntry, ...],
    decode_result: AdaptiveDecodeResult,
    *,
    detector_threshold: float,
    current_structural_fingerprint: str | None = None,
) -> PositiveSupportCertificate:
    facts = _support_facts(
        ordered_evidence,
        schedule,
        decode_result,
        current_structural_fingerprint=current_structural_fingerprint,
    )
    rewrite_backed_families = facts["rewrite_backed_families"]
    rewrite_backed_levels = facts["rewrite_backed_levels"]
    applicable_gate = float(facts["applicable_ratio"]) >= 0.35
    schedule_alignment_gate = int(facts["schedule_aligned_support"]) >= 2
    high_confidence_family_count = len(facts["high_confidence_families"])
    exact_codeword_support_gate = (
        int(facts["schedule_aligned_support"]) >= len(decode_result.codeword) - 1
        and high_confidence_family_count >= len(decode_result.codeword) - 1
        and decode_result.raw_bit_error_count <= 1
        and decode_result.erasure_count == 0
        and decode_result.decoder_status in {
            "decoded_clean",
            "decoded_corrected",
        }
    )
    distinctive_delta_gate = int(facts["generation_planned_delta_count"]) >= 2 or (
        int(facts["generation_planned_delta_count"]) >= 1
        and high_confidence_family_count >= len(decode_result.codeword) - 1
    )
    exact_codeword_delta_gate = (
        exact_codeword_support_gate
        and distinctive_delta_gate
        and len(rewrite_backed_families) >= 1
        and len(rewrite_backed_levels) >= 1
    )
    positive_gate = len(rewrite_backed_families) >= 2 or exact_codeword_delta_gate
    rewrite_level_gate = len(rewrite_backed_levels) >= 2 or exact_codeword_delta_gate
    anchor_consistency_gate = (not facts["anchor_required"]) or (
        facts["anchor_evidence_observed"] and facts["anchor_gate"]
    )
    decoder_gate = decode_result.decoder_status in {
        "decoded_clean",
        "decoded_with_erasures",
        "decoded_corrected",
        "decoded_corrected_with_erasures",
    }
    correction_gate = decode_result.raw_bit_error_count <= 2
    decode_confidence_gate = decode_result.confidence >= max(detector_threshold, 0.55)
    structural_decode_gate = (
        int(facts["schedule_aligned_support"]) >= 5
        and len(rewrite_backed_families) >= 2
        and len(rewrite_backed_levels) >= 2
        and decode_result.raw_bit_error_count <= 2
        and decode_result.erasure_count <= 1
    ) or exact_codeword_delta_gate
    decode_reliability_gate = decode_confidence_gate or structural_decode_gate
    ordered_gates = (
        ("low_applicable_ratio", applicable_gate),
        ("insufficient_schedule_aligned_support", schedule_alignment_gate),
        ("missing_rewrite_backed_positive_support", positive_gate),
        ("insufficient_rewrite_level_diversity", rewrite_level_gate),
        ("anchor_consistency_gate_failed", anchor_consistency_gate),
        ("uncorrectable_decode", decoder_gate),
        ("correction_gate_failed", correction_gate),
        ("decode_reliability_below_gate", decode_reliability_gate),
    )
    failed_gates = tuple(name for name, passed in ordered_gates if not passed)
    score = round(sum(1.0 for _, passed in ordered_gates if passed) / len(ordered_gates), 4)
    reasons = (
        f"positive_support_gate_fraction:{score:.4f}",
        f"schedule_aligned_support:{facts['schedule_aligned_support']}",
        f"rewrite_backed_family_count:{len(rewrite_backed_families)}",
        f"rewrite_backed_level_count:{len(rewrite_backed_levels)}",
        f"high_confidence_family_count:{high_confidence_family_count}",
        f"decode_confidence_gate:{str(decode_confidence_gate).lower()}",
        f"structural_decode_gate:{str(structural_decode_gate).lower()}",
        f"exact_codeword_delta_gate:{str(exact_codeword_delta_gate).lower()}",
        f"exact_codeword_support_gate:{str(exact_codeword_support_gate).lower()}",
        f"distinctive_delta_gate:{str(distinctive_delta_gate).lower()}",
        f"generation_planned_delta_count:{facts['generation_planned_delta_count']}",
    ) + failed_gates
    return PositiveSupportCertificate(
        accepted=not failed_gates,
        score=score,
        reasons=reasons,
        primary_failure_reason=failed_gates[0] if failed_gates else None,
        applicable_ratio=float(facts["applicable_ratio"]),
        structural_level_count=int(facts["structural_level_count"]),
        positive_support=len(rewrite_backed_families),
        schedule_aligned_support=int(facts["schedule_aligned_support"]),
        rewrite_backed_family_count=len(rewrite_backed_families),
        rewrite_backed_level_count=len(rewrite_backed_levels),
    )


def assess_negative_control(
    ordered_evidence: tuple[CarrierEvidence, ...],
    schedule: tuple[CarrierScheduleEntry, ...],
    decode_result: AdaptiveDecodeResult,
    *,
    detector_threshold: float,
    current_structural_fingerprint: str | None = None,
) -> NegativeControlAssessment:
    facts = _support_facts(
        ordered_evidence,
        schedule,
        decode_result,
        current_structural_fingerprint=current_structural_fingerprint,
    )
    rewrite_backed_families = facts["rewrite_backed_families"]
    rewrite_backed_levels = facts["rewrite_backed_levels"]
    high_confidence_families = facts["high_confidence_families"]
    applicable_gate = float(facts["applicable_ratio"]) >= 0.55
    discriminative_schedule_gate = int(facts["rewrite_backed_slot_count"]) >= 2
    structural_gate = int(facts["structural_level_count"]) >= 2 and len(high_confidence_families) >= 4
    rewrite_level_gate = len(rewrite_backed_levels) >= 2
    positive_gate = len(rewrite_backed_families) >= 2
    anchor_consistency_gate = (not facts["anchor_required"]) or (
        facts["anchor_evidence_observed"] and facts["anchor_gate"]
    )
    decoder_gate = decode_result.decoder_status in {
        "decoded_clean",
        "decoded_with_erasures",
        "decoded_corrected",
        "decoded_corrected_with_erasures",
    }
    erasure_gate = decode_result.erasure_count <= 3
    correction_gate = decode_result.raw_bit_error_count <= 2
    decode_confidence_gate = decode_result.confidence >= max(detector_threshold, 0.60)
    ordered_gates = (
        ("low_applicable_ratio", applicable_gate),
        ("schedule_not_discriminative_enough", discriminative_schedule_gate),
        ("insufficient_structural_diversity", structural_gate),
        ("insufficient_rewrite_level_diversity", rewrite_level_gate),
        ("missing_rewrite_backed_positive_support", positive_gate),
        ("anchor_consistency_gate_failed", anchor_consistency_gate),
        ("uncorrectable_decode", decoder_gate),
        ("erasure_gate_failed", erasure_gate),
        ("correction_gate_failed", correction_gate),
        ("decode_confidence_below_gate", decode_confidence_gate),
    )
    failed_gates = tuple(name for name, passed in ordered_gates if not passed)
    score = round(sum(1.0 for _, passed in ordered_gates if passed) / len(ordered_gates), 4)
    reasons = (
        f"decoder_status:{decode_result.decoder_status}",
        f"gate_fraction:{score:.4f}",
        f"schedule_aligned_support:{facts['schedule_aligned_support']}",
        f"rewrite_backed_slot_count:{facts['rewrite_backed_slot_count']}",
        f"rewrite_backed_family_count:{len(rewrite_backed_families)}",
        f"rewrite_backed_level_count:{len(rewrite_backed_levels)}",
        f"generation_planned_delta_count:{facts['generation_planned_delta_count']}",
        f"high_confidence_family_count:{len(high_confidence_families)}",
    ) + failed_gates
    return NegativeControlAssessment(
        accepted=not failed_gates,
        score=score,
        reasons=reasons,
        primary_failure_reason=failed_gates[0] if failed_gates else None,
        failed_gates=failed_gates,
        applicable_ratio=float(facts["applicable_ratio"]),
        structural_level_count=int(facts["structural_level_count"]),
        positive_support=len(rewrite_backed_families),
    )
