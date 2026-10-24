from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from .protocol import CarrierScheduleEntry
from .typed_ast import stable_ast_fingerprint, summarize_typed_ast


@dataclass(frozen=True, slots=True)
class SlotCommitment:
    family: str
    role: str
    bit_index: int | None
    digest: str
    structural_fingerprint: str


@dataclass(frozen=True, slots=True)
class ScheduleCommitment:
    schedule_context_hash: str
    slot_commitments: tuple[SlotCommitment, ...]
    commitment_root: str
    notes: tuple[str, ...] = ()


def _derive_key(carrier_key: str, label: str) -> bytes:
    return hmac.new(carrier_key.encode("utf-8"), label.encode("utf-8"), hashlib.sha256).digest()


def stable_structural_fingerprint(code: str, language: str) -> str:
    return stable_ast_fingerprint(summarize_typed_ast(code, language))


def build_slot_commitment(
    carrier_key: str,
    language: str,
    code: str,
    entry: CarrierScheduleEntry,
) -> SlotCommitment:
    fingerprint = stable_structural_fingerprint(code, language)
    material = "|".join(
        (
            language,
            fingerprint,
            entry.family,
            entry.role,
            str(entry.bit_index) if entry.bit_index is not None else "anchor",
            f"{entry.applicability_score:.4f}",
            f"{entry.schedule_priority:.4f}",
        )
    )
    digest = hmac.new(_derive_key(carrier_key, "semcodebook-slot-v1"), material.encode("utf-8"), hashlib.sha256).hexdigest()[:24]
    return SlotCommitment(
        family=entry.family,
        role=entry.role,
        bit_index=entry.bit_index,
        digest=digest,
        structural_fingerprint=fingerprint,
    )


def build_schedule_commitment(
    carrier_key: str,
    language: str,
    code: str,
    entries: tuple[CarrierScheduleEntry, ...],
) -> ScheduleCommitment:
    context_hash = hmac.new(
        _derive_key(carrier_key, "semcodebook-context-v1"),
        f"{language}|{stable_structural_fingerprint(code, language)}|{len(entries)}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:24]
    slot_commitments = tuple(build_slot_commitment(carrier_key, language, code, entry) for entry in entries)
    root_material = "|".join(item.digest for item in slot_commitments)
    commitment_root = hashlib.sha256(root_material.encode("utf-8")).hexdigest()[:24] if slot_commitments else "empty_schedule"
    return ScheduleCommitment(
        schedule_context_hash=context_hash,
        slot_commitments=slot_commitments,
        commitment_root=commitment_root,
        notes=("hmac_slot_commitments", "structural_context_hash"),
    )
