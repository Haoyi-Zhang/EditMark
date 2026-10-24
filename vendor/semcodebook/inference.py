from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import signal
from contextlib import contextmanager
from pathlib import Path

from .ecc import anchor_bit_for_nibble, encode_nibble
from .detector import SemCodebookDetector
from .commitments import build_schedule_commitment
from .protocol import CarrierScheduleEntry, GenerationRequest, GenerationResult, WatermarkSpec
from .semantic_validator import validate_semantics
from .variant_pool import apply_carrier_variant, build_adaptive_carrier_schedule


_TRAINED_GENERATOR_CACHE: dict[str, tuple[object, object, str]] = {}


class _DecoderAttemptTimeout(TimeoutError):
    pass


@contextmanager
def _bounded_decoder_attempt(seconds: int):
    if seconds <= 0:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(signum, frame):  # pragma: no cover - signal timing is runtime dependent
        raise _DecoderAttemptTimeout(f"decoder_attempt_timeout_seconds:{seconds}")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


class TrainedGenerationFailure(RuntimeError):
    def __init__(
        self,
        phase: str,
        request: GenerationRequest,
        *,
        detail: str = "",
        cause: Exception | None = None,
    ) -> None:
        parts = [
            f"phase:{phase}",
            f"execution_mode:{request.execution_mode or '<missing>'}",
            f"interface_path:{request.interface_path or '<missing>'}",
            f"checkpoint_path:{request.checkpoint_path or '<missing>'}",
            f"adapter_path:{request.adapter_path or '<missing>'}",
        ]
        if detail:
            parts.append(f"detail:{detail}")
        if cause is not None:
            parts.append(f"exception:{cause.__class__.__name__}")
            if str(cause):
                parts.append(f"message:{str(cause)}")
        super().__init__("trained_generation_failure " + " ".join(parts))
        self.phase = phase
        self.request = request
        self.detail = detail
        self.cause = cause


def scheduled_carrier_families(base_code: str, carrier_key: str, language: str) -> tuple[str, ...]:
    schedule = build_adaptive_carrier_schedule(base_code, carrier_key, language)
    return tuple(entry.family for entry in schedule)


def _carrier_payload(wm_id: int, schedule) -> tuple[tuple[int, ...], int]:
    payload_nibble = wm_id & 0x0F
    codeword_bits = encode_nibble(payload_nibble)
    anchor_bit = anchor_bit_for_nibble(payload_nibble)
    carrier_bits = tuple(
        codeword_bits[entry.bit_index] if entry.role == "data" and entry.bit_index is not None else anchor_bit
        for entry in schedule
    )
    return carrier_bits, anchor_bit



def _family_observed_bit(code: str, family: str, language: str) -> tuple[int | None, float, bool]:
    detector = SemCodebookDetector()
    for item in detector.inspect(code, language=language):
        if item.family != family:
            continue
        if item.option.startswith("bit_"):
            return int(item.option[-1]), float(item.confidence), bool(item.applicable)
        return None, float(item.confidence), bool(item.applicable)
    return None, 0.0, False


def _non_python_planned_priority(family: str, target_bit: int, language: str) -> float:
    if target_bit == 1:
        priority = {
            "helper_extraction_style": 0.94,
            "return_expression_style": 0.93,
            "iteration_style": 0.88,
            "accumulator_style": 0.58,
            "temporary_binding_style": 0.74,
            "initialization_idiom": 0.68,
            "early_return_guard_style": 0.28,
            "comparison_idiom": 0.24,
        }
    else:
        priority = {
            "early_return_guard_style": 0.88,
            "comparison_idiom": 0.84,
            "iteration_style": 0.76,
            "accumulator_style": 0.74,
            "helper_extraction_style": 0.70,
            "return_expression_style": 0.26,
            "temporary_binding_style": 0.20,
            "initialization_idiom": 0.52 if language.lower() in {"javascript", "js"} else 0.18,
        }
    return float(priority.get(family, 0.25))


def _family_target_realizability(code: str, family: str, target_bit: int, language: str) -> tuple[float, str, str]:
    observed_bit, confidence, applicable = _family_observed_bit(code, family, language)
    if observed_bit == target_bit and applicable and confidence > 0.0:
        return 2.0 + confidence, "already_aligned", code
    if language.lower() != "python" and applicable and confidence > 0.0 and observed_bit in {0, 1}:
        planned_priority = _non_python_planned_priority(family, target_bit, language)
        return 2.0 + planned_priority, "generation_planned", code
    transformed = apply_carrier_variant(code, family, target_bit, language)
    if transformed != code:
        next_bit, next_confidence, next_applicable = _family_observed_bit(transformed, family, language)
        if next_bit == target_bit and next_applicable and next_confidence > 0.0:
            return 3.0 + next_confidence, "rewrite_aligned", transformed
        return 1.0, "rewrite_changed_unverified", transformed
    if observed_bit == target_bit and confidence > 0.0:
        return 1.0 + confidence, "weak_already_aligned", code
    return 0.0, "not_realizable", code


def _commit_schedule(carrier_key: str, language: str, code: str, schedule: tuple[CarrierScheduleEntry, ...]) -> tuple[CarrierScheduleEntry, ...]:
    commitment = build_schedule_commitment(carrier_key, language, code, schedule)
    enriched: list[CarrierScheduleEntry] = []
    for entry, slot_commitment in zip(schedule, commitment.slot_commitments, strict=True):
        enriched.append(
            CarrierScheduleEntry(
                family=entry.family,
                slot_index=entry.slot_index,
                role=entry.role,
                bit_index=entry.bit_index,
                target_bit=entry.target_bit,
                applicable=entry.applicable,
                applicability_score=entry.applicability_score,
                schedule_priority=entry.schedule_priority,
                structural_level=entry.structural_level,
                structural_signal=entry.structural_signal,
                notes=entry.notes
                + (
                    "target_bit_aligned_carrier_assignment_v1",
                    f"slot_commitment:{slot_commitment.digest}",
                    f"schedule_root:{commitment.commitment_root}",
                    f"schedule_context:{commitment.schedule_context_hash}",
                ),
            )
        )
    return tuple(enriched)






def _target_aligned_schedule(base_code: str, request: GenerationRequest, initial_schedule: tuple[CarrierScheduleEntry, ...]) -> tuple[CarrierScheduleEntry, ...]:
    from itertools import permutations

    payload_nibble = request.wm_id & 0x0F
    codeword_bits = encode_nibble(payload_nibble)
    anchor_bit = anchor_bit_for_nibble(payload_nibble)
    candidate_entries = tuple(initial_schedule)
    score_cache: dict[tuple[str, int], tuple[float, str, str]] = {}
    for entry in candidate_entries:
        for target_bit in (0, 1):
            score_cache[(entry.family, target_bit)] = _family_target_realizability(base_code, entry.family, target_bit, request.language)

    best_key: tuple[float, ...] | None = None
    best_data_assignment: tuple[CarrierScheduleEntry, ...] | None = None
    language_supports_local_rewrites = request.language.lower() == "python"
    for assignment in permutations(candidate_entries, len(codeword_bits)):
        slot_choices = [score_cache[(entry.family, target_bit)] for entry, target_bit in zip(assignment, codeword_bits, strict=True)]
        rewrite_backed_count = sum(
            1.0
            for score, reason, _ in slot_choices
            if reason in {"rewrite_aligned", "generation_planned"} and score > 0.0
        )
        rewrite_backed_levels = {
            entry.structural_level
            for entry, (score, reason, _) in zip(assignment, slot_choices, strict=True)
            if reason in {"rewrite_aligned", "generation_planned"} and score > 0.0 and entry.structural_level
        }
        already_aligned_count = sum(1.0 for score, reason, _ in slot_choices if reason in {"already_aligned", "weak_already_aligned"} and score > 0.0)
        data_support = sum(1.0 for score, _, _ in slot_choices if score > 0.0)
        total_score = sum(score for score, _, _ in slot_choices)
        applicability_sum = sum(entry.applicability_score for entry in assignment)
        priority_sum = sum(entry.schedule_priority for entry in assignment)
        if language_supports_local_rewrites:
            key = (
                rewrite_backed_count,
                len(rewrite_backed_levels),
                data_support,
                total_score,
                applicability_sum,
                priority_sum,
                -already_aligned_count,
            )
        else:
            planned_overflow = max(0.0, rewrite_backed_count - 3.0)
            key = (
                min(rewrite_backed_count, 2.0),
                min(float(len(rewrite_backed_levels)), 2.0),
                already_aligned_count,
                data_support,
                -planned_overflow,
                total_score,
                applicability_sum,
                priority_sum,
            )
        if best_key is None or key > best_key:
            best_key = key
            best_data_assignment = assignment
    if best_data_assignment is None:
        best_data_assignment = candidate_entries[: len(codeword_bits)]

    selected: list[CarrierScheduleEntry] = []
    used_families: set[str] = set()
    data_target_by_family: dict[str, int] = {}
    for slot_index, (entry, target_bit) in enumerate(zip(best_data_assignment, codeword_bits, strict=True)):
        score, reason, _ = score_cache[(entry.family, target_bit)]
        base_observed_bit, _, _ = _family_observed_bit(base_code, entry.family, request.language)
        discriminative_notes: tuple[str, ...]
        if reason == "rewrite_aligned":
            discriminative_notes = ("discriminative_rewrite_backed_carrier",)
        elif not language_supports_local_rewrites and reason == "generation_planned":
            discriminative_notes = ("discriminative_generation_planned_carrier",)
        else:
            discriminative_notes = ("naturally_aligned_support_carrier",)
        used_families.add(entry.family)
        data_target_by_family[entry.family] = target_bit
        selected.append(
            CarrierScheduleEntry(
                family=entry.family,
                slot_index=slot_index,
                role="data",
                bit_index=slot_index,
                target_bit=target_bit,
                applicable=entry.applicable or score > 0.0,
                applicability_score=max(entry.applicability_score, min(1.0, score / 4.0)),
                schedule_priority=entry.schedule_priority,
                structural_level=entry.structural_level,
                structural_signal=entry.structural_signal,
                notes=entry.notes
                + (
                    "target_aligned_data_slot",
                    f"target_alignment_reason:{reason}",
                    f"target_alignment_score:{score:.4f}",
                    f"target_alignment_base_bit:{base_observed_bit if base_observed_bit is not None else 'unknown'}",
                )
                + discriminative_notes,
            )
        )

    anchor_candidates = tuple(entry for entry in candidate_entries if entry.family not in used_families) or candidate_entries
    anchor_ranked: list[tuple[float, float, float, float, float, str, CarrierScheduleEntry, str]] = []
    for entry in anchor_candidates:
        raw_score, reason, _ = score_cache[(entry.family, anchor_bit)]
        if reason == "rewrite_aligned":
            candidate_score = 3.0 + raw_score
            rewrite_preference = 1.0
            observed_preference = 1.0
        elif reason == "already_aligned":
            candidate_score = 1.0 + raw_score * 0.5
            rewrite_preference = 0.0
            observed_preference = 1.0
        elif reason == "weak_already_aligned":
            candidate_score = 0.5 + raw_score * 0.25
            rewrite_preference = 0.0
            observed_preference = 0.5
        else:
            candidate_score = 0.0
            rewrite_preference = 0.0
            observed_preference = 0.0
        anchor_ranked.append((rewrite_preference, candidate_score, observed_preference, entry.applicability_score, entry.schedule_priority, entry.family, entry, reason))
    anchor_ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5]), reverse=True)
    rewrite_preference, score, _, _, _, _, anchor_entry, reason = anchor_ranked[0]
    anchor_base_observed_bit, _, _ = _family_observed_bit(base_code, anchor_entry.family, request.language)
    selected.append(
        CarrierScheduleEntry(
            family=anchor_entry.family,
            slot_index=len(selected),
            role="anchor",
            target_bit=anchor_bit,
            applicable=anchor_entry.applicable or score > 0.0,
            applicability_score=max(anchor_entry.applicability_score, min(1.0, score / 4.0)),
            schedule_priority=anchor_entry.schedule_priority,
            structural_level=anchor_entry.structural_level,
            structural_signal=anchor_entry.structural_signal,
            notes=anchor_entry.notes
            + (
                "target_aligned_anchor_slot",
                "anchor_distinct_from_data" if anchor_entry.family not in data_target_by_family else "anchor_reuse_only_as_last_resort",
                f"target_alignment_reason:{reason}",
                f"target_alignment_score:{score:.4f}",
                f"target_alignment_base_bit:{anchor_base_observed_bit if anchor_base_observed_bit is not None else 'unknown'}",
            )
            + (("discriminative_rewrite_backed_carrier",) if rewrite_preference > 0.0 else ("naturally_aligned_support_carrier",)),
        )
    )
    return _commit_schedule(request.carrier_key, request.language, base_code, tuple(selected))


def _carrier_realization_guidance(family: str, bit_value: int, language: str) -> str:
    normalized_language = language.strip().lower()
    guidance_by_family: dict[str, dict[int, str]] = {
        "early_return_guard_style": {
            0: "Use a direct guard branch with an early return and no artificial else block.",
            1: "Use an explicit else-style guard shape when preserving the same return behavior.",
        },
        "iteration_style": {
            0: "Prefer direct item iteration over the collection.",
            1: "Prefer index-based iteration over the collection length.",
        },
        "accumulator_style": {
            0: "Update the accumulator in place.",
            1: (
                "Assign the accumulator to its previous value plus the contribution; do not use the augmented plus-equals form."
                if normalized_language in {"javascript", "java", "go", "cpp", "c++"}
                else "Compute the next accumulator value in a temporary binding before assigning it back."
            ),
        },
        "comparison_idiom": {
            0: "Write comparisons with the program value on the left side when possible.",
            1: "Use an equivalent swapped comparison form when it preserves behavior.",
        },
        "helper_extraction_style": {
            0: "Keep the expression inline without an identity helper.",
            1: "Use a small identity helper and call it at the scheduled expression site.",
        },
        "temporary_binding_style": {
            0: "Use the expression directly at the update site.",
            1: "Bind the update expression to a short-lived temporary before using it.",
        },
        "initialization_idiom": {
            0: "Use a literal initializer.",
            1: "Use an equivalent typed constructor initializer.",
        },
        "return_expression_style": {
            0: "Return the final expression directly.",
            1: "Bind the final expression to a local variable before returning it.",
        },
    }
    normalized_bit = 1 if int(bit_value) else 0
    guidance = guidance_by_family.get(family, {}).get(normalized_bit)
    if guidance is None:
        guidance = "Preserve behavior while realizing this structural choice in ordinary source code."
    if normalized_language in {"javascript", "java", "go", "cpp", "c++"}:
        guidance += " Adapt the syntax to the target language."
    return guidance


def _go_entrypoint_signature(prompt: str, base_code: str) -> str:
    for text in (prompt, base_code):
        match = re.search(
            r"func\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*(?:[A-Za-z_][A-Za-z0-9_]*|\[\]\w+)?",
            text,
        )
        if match:
            return match.group(0).strip().rstrip("{").strip()
    return ""


def _go_function_declarations_from_text(text: str) -> tuple[tuple[str, str], ...]:
    declarations: list[tuple[str, str]] = []
    pattern = re.compile(
        r"(?m)^func\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\([^{}]*\)\s*(?:[A-Za-z_][A-Za-z0-9_]*|\[\]\w+)?\s*\{"
    )
    for match in pattern.finditer(text):
        depth = 0
        end = match.end()
        for index in range(match.start(), len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if depth == 0:
            declarations.append((match.group("name"), text[match.start() : end].strip()))
    return tuple(declarations)


def _go_helper_declarations_from_text(text: str) -> tuple[str, ...]:
    return tuple(
        declaration
        for name, declaration in _go_function_declarations_from_text(text)
        if name.startswith("helper_")
    )


def _go_signature_name(signature: str) -> str:
    match = re.match(r"\s*func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", signature)
    return match.group(1) if match else ""


def _go_called_function_names(text: str, candidate_names: set[str]) -> set[str]:
    if not text or not candidate_names:
        return set()
    return {
        name
        for name in candidate_names
        if re.search(rf"\b{re.escape(name)}\s*\(", text)
    }


def _go_supporting_prefix_from_base_code(prompt: str, base_code: str, signature: str = "") -> str:
    resolved_signature = signature.strip() or _go_entrypoint_signature(prompt, base_code)
    if not resolved_signature:
        return ""
    entrypoint_name = _go_signature_name(resolved_signature)
    if not entrypoint_name:
        return ""
    declarations = _go_function_declarations_from_text(base_code)
    if not declarations:
        return ""
    declaration_by_name = {name: declaration for name, declaration in declarations}
    candidate_names = set(declaration_by_name)
    entrypoint_body = declaration_by_name.get(entrypoint_name, "")
    if not entrypoint_body:
        match = re.search(rf"(?s)(?P<signature>{re.escape(resolved_signature)})\s*\{{", base_code)
        if match is None:
            return ""
        entrypoint_body = base_code[match.end() :]
    required = _go_called_function_names(entrypoint_body, candidate_names)
    required.discard(entrypoint_name)
    if not required:
        required = {name for name in candidate_names if name.startswith("helper_")}
    pending = list(required)
    while pending:
        name = pending.pop()
        for dependency in _go_called_function_names(declaration_by_name.get(name, ""), candidate_names):
            if dependency == entrypoint_name:
                continue
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    return "\n\n".join(
        declaration
        for name, declaration in declarations
        if name in required and name != _go_signature_name(resolved_signature)
    )


def _go_supporting_prefix_closure_complete(prompt: str, base_code: str, signature: str = "") -> bool:
    supporting_prefix = _go_supporting_prefix_from_base_code(prompt, base_code, signature)
    declarations = _go_function_declarations_from_text(supporting_prefix)
    if not declarations:
        return True
    present_names = {name for name, _ in declarations}
    base_function_names = {name for name, _ in _go_function_declarations_from_text(base_code)}
    for _, declaration in declarations:
        missing = _go_called_function_names(declaration, base_function_names.difference(present_names))
        if missing:
            return False
    return True


def _go_missing_supporting_prefix(decoded: str, supporting_prefix: str) -> str:
    support_declarations = _go_function_declarations_from_text(supporting_prefix)
    if not support_declarations:
        return ""
    decoded_names = {name for name, _ in _go_function_declarations_from_text(decoded)}
    missing = [
        declaration
        for name, declaration in support_declarations
        if name not in decoded_names
    ]
    return "\n\n".join(missing)


def _python_entrypoint_signature(prompt: str, base_code: str) -> str:
    for text in (prompt, base_code):
        quoted = re.search(r"`(def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*:?)`", text)
        if quoted:
            return quoted.group(1).strip().rstrip(":").strip() + ":"
        match = re.search(r"(?m)^(def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*:)", text)
        if match:
            return match.group(1).strip().rstrip(":").strip() + ":"
    return ""


def _cpp_entrypoint_signature(prompt: str, base_code: str) -> str:
    type_pattern = r"(?:int|long|bool|double|float|void|auto|std::[A-Za-z_][A-Za-z0-9_:<>]*)"
    signature_pattern = rf"{type_pattern}\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^;{{}}]*\)"
    quoted_pattern = rf"`({signature_pattern})`"
    for text in (prompt, base_code):
        quoted = re.search(quoted_pattern, text)
        if quoted:
            return quoted.group(1).strip().rstrip("{").strip()
        match = re.search(rf"({signature_pattern})\s*\{{", text)
        if match:
            return match.group(1).strip().rstrip("{").strip()
    return ""


def _javascript_entrypoint_signature(prompt: str, base_code: str) -> str:
    signature_patterns = (
        r"(?:export\s+)?(?:async\s+)?function\s+[A-Za-z_$][A-Za-z0-9_$]*\s*\([^)]*\)",
        r"(?:export\s+)?(?:const|let|var)\s+[A-Za-z_$][A-Za-z0-9_$]*\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
        r"(?:export\s+)?class\s+[A-Za-z_$][A-Za-z0-9_$]*",
    )
    for text in (prompt, base_code):
        for signature_pattern in signature_patterns:
            quoted = re.search(rf"`({signature_pattern})`", text)
            if quoted:
                return quoted.group(1).strip().rstrip("{").strip()
            match = re.search(rf"({signature_pattern})\s*\{{?", text)
            if match:
                return match.group(1).strip().rstrip("{").strip()
    return ""


def _cpp_supporting_prefix_from_base_code(prompt: str, base_code: str, signature: str = "") -> str:
    resolved_signature = signature.strip() or _cpp_entrypoint_signature(prompt, base_code)
    if not resolved_signature:
        return ""
    match = re.search(rf"(?s)\A(?P<prefix>.*?)(?P<signature>{re.escape(resolved_signature)})\s*\{{", base_code)
    if match is None:
        return ""
    return match.group("prefix").rstrip()


def _cpp_prefix_has_callable_definition(prefix: str) -> bool:
    if not prefix.strip():
        return False
    callable_pattern = (
        r"(?m)^\s*"
        r"(?:static\s+)?"
        r"(?:int|long|bool|double|float|void|auto|std::[A-Za-z_][A-Za-z0-9_:<>]*)"
        r"\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^;{}]*\)\s*\{"
    )
    return bool(re.search(callable_pattern, prefix))


def _decoder_forced_output_prefix(request: GenerationRequest, base_code: str, schedule=()) -> tuple[str, tuple[str, ...]]:
    lowered = request.language.strip().lower()
    if lowered == "python":
        signature = _python_entrypoint_signature(request.prompt, base_code)
        if not signature:
            return "", ()
        helper_prefix = ""
        notes = ["decoder_forced_prefix:python_entrypoint_signature"]
        if any(
            getattr(entry, "family", "") == "helper_extraction_style" and int(getattr(entry, "target_bit", 0) or 0) == 1
            for entry in schedule
        ):
            helper_prefix = "def helper_transform(value):\n    return value\n\n"
            notes.append("decoder_forced_prefix:python_helper_identity")
        return helper_prefix + signature.rstrip() + "\n    ", tuple(notes)
    if lowered == "go":
        signature = _go_entrypoint_signature(request.prompt, base_code)
        if not signature:
            return "", ()
        notes = ["decoder_forced_prefix:go_entrypoint_signature"]
        supporting_prefix = _go_supporting_prefix_from_base_code(request.prompt, base_code, signature)
        if supporting_prefix:
            notes.append("decoder_forced_prefix:go_base_support_prefix")
            return supporting_prefix.rstrip() + "\n\n" + signature.rstrip() + " {\n", tuple(notes)
        return signature.rstrip() + " {\n", tuple(notes)
    if lowered in {"cpp", "c++"}:
        signature = _cpp_entrypoint_signature(request.prompt, base_code)
        if not signature:
            return "", ()
        notes = ["decoder_forced_prefix:cpp_entrypoint_signature"]
        prefix_blocks: list[str] = []
        supporting_prefix = _cpp_supporting_prefix_from_base_code(request.prompt, base_code, signature)
        if supporting_prefix:
            prefix_blocks.append(supporting_prefix.strip())
            if _cpp_prefix_has_callable_definition(supporting_prefix):
                notes.append("decoder_forced_prefix:cpp_base_support_prefix")
        elif "std::vector" in f"{signature}\n{base_code}":
            prefix_blocks.append("#include <vector>")
        helper_prefix_needed = any(
            getattr(entry, "family", "") == "helper_extraction_style" and int(getattr(entry, "target_bit", 0) or 0) == 1
            for entry in schedule
        )
        if helper_prefix_needed and not _cpp_prefix_has_callable_definition(supporting_prefix):
            prefix_blocks.append("int helper_transform(int value) {\n    return value;\n}")
            notes.append("decoder_forced_prefix:cpp_helper_identity")
        prefix = "\n\n".join(block for block in prefix_blocks if block.strip())
        if prefix:
            prefix += "\n\n"
        return prefix + signature.rstrip() + " {\n", tuple(notes)
    return "", ()


def _language_generation_constraints(language: str) -> tuple[str, ...]:
    lowered = language.strip().lower()
    if lowered == "python":
        return (
            "Start directly with the requested Python def or class declaration.",
            "After every Python function header, continue with an indented body before ending generation.",
            "Do not stop after a signature line, and do not emit demo assertions, prints, or markdown.",
        )
    if lowered in {"javascript", "js"}:
        return (
            "Start directly with the requested function, constant, or class declaration.",
            "Do not emit carrier-plan comments or assignment-style plan fragments.",
            "Use ordinary JavaScript token spacing: write `else if`, `else c = ...`, and `return total;`; never merge keywords with identifiers.",
        )
    if lowered == "java":
        return (
            "Return one complete Java source file with the required public class and all closing braces.",
            "Start directly with the public class declaration or import statements.",
            "Do not split the required method signature across standalone prose lines.",
        )
    if lowered == "go":
        return (
            "Start directly with the exact func declaration from the task prompt.",
            "The evaluator supplies the Go wrapper; do not emit a package declaration.",
            "Preserve the exact function name and signature requested in the task prompt.",
            "Use `_` for an unused range index, for example `for _, value := range values`.",
            "Do not emit prose or markdown before the first Go declaration.",
        )
    if lowered in {"cpp", "c++"}:
        return (
            "Start directly with C++ includes or the requested function declaration.",
            "Include any standard headers needed by the signature, such as <vector>.",
            "If the signature uses std::vector, the first output line must be #include <vector>.",
            "If the base code includes top-level helper or static declarations used by the entrypoint, keep them before the requested function.",
            "Every statement must end with a semicolon.",
            "End the function with a behavior-preserving return statement and all required closing braces.",
            "Do not output bullet lists, markdown, backticked signatures, or instruction text.",
        )
    return (
        "Start directly with runnable source code.",
        "Do not emit prose, markdown, or carrier-plan text.",
    )


def build_structured_generation_prompt(
    *,
    prompt: str,
    language: str,
    wm_id: int,
    base_code: str,
    schedule,
    carrier_key: str = "",
    task_id: str = "",
) -> str:
    carrier_bits, _ = _carrier_payload(wm_id, schedule)
    normalized_language = language.strip().lower()
    if normalized_language == "go":
        instruction = "Instruction: rewrite the base code into one semantically equivalent Go function snippet."
        output_contract = (
            "- Output only Go function/helper declarations needed by the task; no markdown fences and no explanation.",
            "- Do not emit a package declaration, imports, tests, assertions, print examples, or lines beginning with ###.",
            "- Preserve every public function signature that appears in the base code.",
            "- Do not copy the base body unchanged; realize at least two planned structural carrier choices as behavior-preserving rewrites.",
            "- Realize the carrier plan silently inside the final program structure; never list carrier snippets separately.",
        )
    else:
        instruction = "Instruction: rewrite the base code into one semantically equivalent complete source file."
        output_contract = (
            "- Output only runnable source code, with no markdown fences and no explanation.",
            "- Preserve every public function, class, and method signature that appears in the base code.",
            "- Do not copy the base body unchanged; realize at least two planned structural carrier choices as behavior-preserving rewrites.",
            "- Do not output carrier names, schedule labels, metadata, assertions, print examples, or lines beginning with ###.",
            "- Realize the carrier plan silently inside the final program structure; never list carrier snippets separately.",
        )
    lines = [
        "### SemCodebook Rewrite Task",
        f"Language: {language}",
        f"WM-ID: {wm_id & 0x0F}",
    ]
    if task_id:
        lines.append(f"Task-ID: {task_id}")
    if carrier_key:
        lines.append(f"Carrier-Key: {carrier_key}")
    lines.extend(
        [
            instruction,
            "Output contract:",
            *output_contract,
            "Language-specific constraints:",
            *[f"- {item}" for item in _language_generation_constraints(language)],
        ]
    )
    go_signature = _go_entrypoint_signature(prompt, base_code) if normalized_language == "go" else ""
    if go_signature:
        lines.extend(
            [
                "Required Go output shape:",
                f"- The first non-empty output line must be exactly: `{go_signature} {{`",
                "- The output must contain that function body and closing brace.",
            ]
        )
    lines.extend(
        [
            "Task Prompt:",
            prompt.strip(),
            "Private Carrier Plan:",
            "Every scheduled carrier, including the anchor slot, must be realized through semantically valid structure choices, but the plan itself must not appear in the output.",
        ]
    )
    for entry, bit in zip(schedule, carrier_bits, strict=True):
        lines.append(
            "- "
            + f"role={entry.role}; family={entry.family}; target_bit={bit}; applicable={str(entry.applicable).lower()}; "
            + f"level={entry.structural_level}; signal={entry.structural_signal}"
            + f"; guidance={_carrier_realization_guidance(entry.family, bit, language)}"
        )
    lines.extend(
        [
            "Base Code:",
            base_code.strip(),
            "### Output code",
            "",
        ]
    )
    return "\n".join(lines)


def _semcodebook_prompt_section(base_prompt: str, heading: str) -> str:
    marker = f"{heading}:"
    start = base_prompt.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    next_markers = (
        "Output contract:",
        "Language-specific constraints:",
        "Required Go output shape:",
        "Task Prompt:",
        "Private Carrier Plan:",
        "Base Code:",
        "### Output code",
    )
    end = len(base_prompt)
    for candidate in next_markers:
        if candidate == marker:
            continue
        index = base_prompt.find(candidate, start)
        if index >= 0 and index < end:
            end = index
    return base_prompt[start:end].strip()


def _go_carrier_support_retry_prompt(base_prompt: str) -> str:
    task_prompt = _semcodebook_prompt_section(base_prompt, "Task Prompt")
    private_plan = _semcodebook_prompt_section(base_prompt, "Private Carrier Plan")
    base_code = _semcodebook_prompt_section(base_prompt, "Base Code")
    retry_lines = [
        "### SemCodebook Go Rewrite-backed support retry",
        "Output only Go code. Do not output explanations, markdown, headings, bullet lists, or instruction text.",
        "The previous semantic candidate preserved behavior but did not realize enough rewrite-backed carrier evidence.",
        "Generate a fresh semantically equivalent candidate that changes the body structure instead of copying the base body.",
        "A candidate is unacceptable if its whitespace-stripped program body is identical to the base body.",
        "Keep the same public signatures and tests passing.",
        "Realize at least two scheduled data carriers through visible behavior-preserving rewrites in the generated program body.",
        "For Go, helper calls alone are not enough: also rewrite loop/update/return structure when those carriers are scheduled.",
        "If the base loop already calls the helper, introduce behavior-preserving accumulator or return restructuring rather than emitting the same loop body.",
        "Use at least two concrete low-risk Go rewrites when applicable: bind the helper result to a local variable, update the accumulator through a temporary next value, return a result variable instead of the raw accumulator, or replace a nested guard with an equivalent continue guard.",
        "Do not stop after one carrier-looking edit; the accepted candidate needs at least two independent scheduled carrier families across at least two structural levels.",
        "Prefer one CFG-level change such as loop or guard structure and one SSA-level change such as accumulator, temporary binding, initialization, or return binding.",
        "Do not change helper arithmetic, guard predicates, or which input values are included.",
        "When the decoder prefix already supplies helper declarations and the requested entrypoint header, continue with only the entrypoint body and closing brace.",
        "Use ordinary Go token spacing, for example `for index, value := range values`, `if value >= 0`, and `return total`; never merge keywords with identifiers.",
        "If a Go range index is not used in the loop body, write `_` for that index.",
        "Stop immediately after the final Go closing brace; do not append schedules, language tags, explanations, or provenance lists.",
    ]
    if task_prompt:
        retry_lines.extend(["", "Task Prompt (do not print):", task_prompt])
    if private_plan:
        retry_lines.extend(["", "Carrier Plan (do not print):", private_plan])
    if base_code:
        retry_lines.extend(["", "Base Code (do not print):", base_code])
    retry_lines.extend(["", "### Output Go code", ""])
    return "\n".join(retry_lines)


_GO_INT_HELPER_RETURN_RE = re.compile(
    r"func\s+"
    r"(?P<name>helper_[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*\(\s*(?P<param>[A-Za-z_][A-Za-z0-9_]*)\s+int\s*\)"
    r"\s*int\s*\{\s*return\s+(?P<expr>[^{}\n;]+(?:\([^{}]*\)[^{}\n;]*)?)\s*\}",
    re.MULTILINE,
)


def _replace_go_identifier(expr: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\b", new, expr)


def _go_inline_private_helper_calls(code: str) -> str:
    helper_matches = list(_GO_INT_HELPER_RETURN_RE.finditer(code))
    if not helper_matches:
        return code
    materialized = code
    for match in reversed(helper_matches):
        name = match.group("name")
        param = match.group("param")
        expr = match.group("expr").strip()
        call_re = re.compile(rf"\b{re.escape(name)}\s*\(\s*(?P<arg>[A-Za-z_][A-Za-z0-9_]*)\s*\)")
        replacement_count = 0

        def _call_repl(call_match: re.Match[str]) -> str:
            nonlocal replacement_count
            replacement_count += 1
            arg = call_match.group("arg")
            return f"({_replace_go_identifier(expr, param, arg)})"

        before_helper = materialized[: match.start()]
        after_helper = materialized[match.end() :]
        candidate = call_re.sub(_call_repl, before_helper + after_helper)
        if replacement_count > 0:
            materialized = candidate
    return re.sub(r"\n{3,}", "\n\n", materialized).strip()


def _go_add_return_binding(code: str) -> str:
    if re.search(r"\breturn\s+result\b", code):
        return code
    stripped = code.strip()
    pattern = re.compile(r"(?P<indent>[ \t]*)return\s+(?P<expr>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<close>\})")
    matches = list(pattern.finditer(stripped))
    if not matches:
        return code
    match = matches[-1]
    replacement = (
        f"{match.group('indent')}result := {match.group('expr')}\n"
        f"{match.group('indent')}return result\n"
        f"{match.group('indent')}{match.group('close')}"
    )
    return stripped[: match.start()] + replacement + stripped[match.end() :]


def _schedule_target_bit(schedule: tuple[CarrierScheduleEntry, ...], family: str) -> int | None:
    for entry in schedule:
        if entry.family != family or entry.target_bit is None:
            continue
        try:
            return int(entry.target_bit)
        except (TypeError, ValueError):
            return None
    return None


def _schedule_target_is_one(schedule: tuple[CarrierScheduleEntry, ...], family: str) -> bool:
    return _schedule_target_bit(schedule, family) == 1


def _schedule_target_is_zero(schedule: tuple[CarrierScheduleEntry, ...], family: str) -> bool:
    return _schedule_target_bit(schedule, family) == 0


def _go_add_helper_transform(code: str) -> str:
    if "helperTransform(" in code or re.search(r"\bfunc\s+helperTransform\s*\(", code):
        return code
    if re.search(r"\bfunc\s+helper[A-Za-z0-9_]*\s*\(", code):
        return code
    updated = re.sub(
        r"(?P<acc>\b[A-Za-z_][A-Za-z0-9_]*)\s*(?:\+=|=\s*(?P=acc)\s*\+)\s*(?P<expr>[A-Za-z_][A-Za-z0-9_]*)\s*;?",
        lambda match: f"{match.group('acc')} = {match.group('acc')} + helperTransform({match.group('expr')})",
        code,
        count=1,
    )
    if updated == code:
        return code
    helper = "func helperTransform(value int) int {\n    return value\n}"
    return helper + "\n\n" + updated.rstrip()


def _go_inline_helper_transform(code: str) -> str:
    updated = re.sub(r"\bhelperTransform\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", r"\1", code)
    updated = re.sub(
        r"\n\s*func\s+helperTransform\s*\(\s*value\s+int\s*\)\s*int\s*\{\s*return\s+value\s*\}\s*$",
        "",
        updated,
        flags=re.DOTALL,
    )
    return updated.strip()


def _go_add_accumulator_binding(code: str) -> str:
    if re.search(r"\bnextTotal\s*:=", code):
        return code
    pattern = re.compile(
        r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*"
        r"(?:\+=|=\s*(?P=acc)\s*\+)\s*(?P<expr>[^;\n{}]+)\s*;?"
    )

    def repl(match: re.Match[str]) -> str:
        indent = match.group("indent")
        acc = match.group("acc")
        expr = match.group("expr").strip()
        return f"{indent}nextTotal := {acc} + {expr}\n{indent}{acc} = nextTotal"

    return pattern.sub(repl, code, count=1)


def _go_inline_accumulator_binding(code: str) -> str:
    pattern = re.compile(
        r"(?P<indent>[ \t]*)nextTotal\s*:=\s*(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*\+\s*(?P<expr>[^\n{}]+)\n"
        r"(?P=indent)(?P=acc)\s*=\s*nextTotal"
    )
    return pattern.sub(lambda match: f"{match.group('indent')}{match.group('acc')} += {match.group('expr').strip()}", code, count=1)


def _go_add_initialization_idiom(code: str) -> str:
    return re.sub(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:=\s*0\b",
        r"\1 := int(0)",
        code,
        count=1,
    )


def _go_inline_initialization_idiom(code: str) -> str:
    return re.sub(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:=\s*int\s*\(\s*0\s*\)",
        r"\1 := 0",
        code,
        count=1,
    )


def _go_add_temporary_binding(code: str) -> str:
    if re.search(r"\bcurrentItem\s*:=", code):
        return code
    pattern = re.compile(
        r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*"
        r"(?P<op>\+=|=\s*(?P=acc)\s*\+)\s*"
        r"(?P<expr>[A-Za-z_][A-Za-z0-9_]*|helperTransform\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\))"
        r"(?P<suffix>[ \t]*;?)(?P<same_line_close>[ \t]*\})?"
    )

    def repl(match: re.Match[str]) -> str:
        indent = match.group("indent")
        acc = match.group("acc")
        op = match.group("op")
        expr = match.group("expr").strip()
        value_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", expr)
        if value_match is None:
            return match.group(0)
        value_name = value_match.group(1)
        rewritten_expr = re.sub(rf"\b{re.escape(value_name)}\b", "currentItem", expr, count=1)
        return f"{indent}currentItem := {value_name}\n{indent}{acc} {op} {rewritten_expr}"

    return pattern.sub(repl, code, count=1)


def _go_inline_temporary_binding(code: str) -> str:
    pattern = re.compile(
        r"(?P<indent>[ \t]*)currentItem\s*:=\s*(?P<value>[A-Za-z_][A-Za-z0-9_]*)\n"
        r"(?P=indent)(?P<rest>[A-Za-z_][A-Za-z0-9_]*\s*(?:\+=|=\s*[A-Za-z_][A-Za-z0-9_]*\s*\+)\s*[^;\n{}]*)"
    )

    def repl(match: re.Match[str]) -> str:
        value = match.group("value")
        rest = re.sub(r"\bcurrentItem\b", value, match.group("rest"))
        return f"{match.group('indent')}{rest}"

    return pattern.sub(repl, code, count=1)


def _go_add_indexed_iteration(code: str) -> str:
    pattern = re.compile(
        r"for\s+_\s*,\s*(?P<value>[A-Za-z_][A-Za-z0-9_]*)\s*:=\s*range\s+(?P<iter>[A-Za-z_][A-Za-z0-9_]*)\s*\{"
    )
    if not pattern.search(code):
        return code
    return pattern.sub(
        lambda match: (
            f"for index := range {match.group('iter')} {{\n"
            f"        {match.group('value')} := {match.group('iter')}[index]\n"
        ),
        code,
        count=1,
    )


def _go_materialize_schedule(code: str, schedule: tuple[CarrierScheduleEntry, ...]) -> tuple[str, tuple[str, ...]]:
    materialized = code
    labels: list[str] = []
    transforms = (
        ("initialization_idiom", 1, "typed_initialization", _go_add_initialization_idiom),
        ("initialization_idiom", 0, "literal_initialization", _go_inline_initialization_idiom),
        ("iteration_style", 1, "indexed_iteration", _go_add_indexed_iteration),
        ("temporary_binding_style", 1, "temporary_binding", _go_add_temporary_binding),
        ("helper_extraction_style", 1, "helper_transform", _go_add_helper_transform),
        ("helper_extraction_style", 0, "inline_helper_transform", _go_inline_helper_transform),
        ("accumulator_style", 1, "accumulator_binding", _go_add_accumulator_binding),
        ("accumulator_style", 0, "inline_accumulator_binding", _go_inline_accumulator_binding),
        ("return_expression_style", 1, "return_binding", _go_add_return_binding),
        ("return_expression_style", 0, "direct_return", lambda text: re.sub(r"\bresult\s*:=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\n\s*return\s+result", r"return \1", text, count=1)),
    )
    for family, bit_value, label, transform in transforms:
        if _schedule_target_bit(schedule, family) != bit_value:
            continue
        updated = transform(materialized)
        if updated != materialized:
            materialized = updated
            labels.append(label)
    return _normalize_go_compact_tokens(materialized), tuple(labels)


def _go_support_materializer_variants(
    code: str,
    request: GenerationRequest,
    schedule: tuple[CarrierScheduleEntry, ...] = (),
) -> tuple[tuple[str, str], ...]:
    if request.language.strip().lower() != "go":
        return ()
    variants: list[tuple[str, str]] = []
    if schedule:
        exact, labels = _go_materialize_schedule(code, schedule)
        if labels and exact != code:
            variants.append(("go_materializer:scheduled_" + "_".join(labels), exact))
    inline_helper = _go_inline_private_helper_calls(code)
    if inline_helper != code:
        variants.append(("go_materializer:inline_private_helper", inline_helper))
        inline_return = _go_add_return_binding(inline_helper)
        if inline_return != inline_helper:
            variants.append(("go_materializer:inline_private_helper_return_binding", inline_return))
    return_binding = _go_add_return_binding(code)
    if return_binding != code:
        variants.append(("go_materializer:return_binding", return_binding))

    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for label, variant in variants:
        normalized = re.sub(r"\s+", "", variant)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append((label, variant))
    return tuple(deduped)


def _schedule_has_target(schedule: tuple[CarrierScheduleEntry, ...], family: str, bit_value: int) -> bool:
    return any(
        entry.family == family
        and int(entry.target_bit if entry.target_bit is not None else -1) == int(bit_value)
        for entry in schedule
    )


def _dedupe_materializer_variants(variants: list[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for label, variant in variants:
        normalized = re.sub(r"\s+", "", variant)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append((label, variant))
    return tuple(deduped)


def _javascript_add_helper_transform(code: str) -> str:
    if "helper_transform(" in code:
        return code
    pattern = re.compile(
        r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_$][A-Za-z0-9_$]*)\s*"
        r"(?:\+=|=\s*(?P=acc)\s*\+)\s*(?P<expr>[A-Za-z_$][A-Za-z0-9_$]*)\s*;"
    )

    def repl(match: re.Match[str]) -> str:
        op = "+=" if "+=" in match.group(0) else "="
        if op == "+=":
            return f"{match.group('indent')}{match.group('acc')} += helper_transform({match.group('expr')});"
        return f"{match.group('indent')}{match.group('acc')} = {match.group('acc')} + helper_transform({match.group('expr')});"

    updated = pattern.sub(repl, code, count=1)
    if updated == code:
        return code
    helper = "function helper_transform(value) {\n  return value;\n}"
    return updated.rstrip() + "\n\n" + helper


def _javascript_inline_helper_transform(code: str) -> str:
    updated = re.sub(r"\bhelper_transform\s*\(\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\)", r"\1", code)
    updated = re.sub(
        r"\n\s*function\s+helper_transform\s*\(\s*value\s*\)\s*\{\s*return\s+value\s*;\s*\}\s*$",
        "",
        updated,
        flags=re.DOTALL,
    )
    return updated.strip()


def _javascript_add_accumulator_binding(code: str) -> str:
    if re.search(r"\bnextTotal\s*=", code):
        return code
    pattern = re.compile(
        r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_$][A-Za-z0-9_$]*)\s*"
        r"(?:\+=|=\s*(?P=acc)\s*\+)\s*(?P<expr>[^;\n]+)\s*;"
    )

    def repl(match: re.Match[str]) -> str:
        indent = match.group("indent")
        acc = match.group("acc")
        expr = match.group("expr").strip()
        return f"{indent}const nextTotal = {acc} + {expr};\n{indent}{acc} = nextTotal;"

    return pattern.sub(repl, code, count=1)


def _javascript_inline_accumulator_binding(code: str) -> str:
    pattern = re.compile(
        r"(?P<indent>[ \t]*)(?:const|let|var)\s+nextTotal\s*=\s*(?P<acc>[A-Za-z_$][A-Za-z0-9_$]*)\s*\+\s*(?P<expr>[^;\n]+);\n"
        r"(?P=indent)(?P=acc)\s*=\s*nextTotal;"
    )
    return pattern.sub(lambda match: f"{match.group('indent')}{match.group('acc')} += {match.group('expr').strip()};", code, count=1)


def _javascript_add_temporary_binding(code: str) -> str:
    if re.search(r"\bcurrentItem\s*=", code):
        return code
    guarded_pattern = re.compile(
        r"if\s*\((?P<condition>[^{}\n;]+)\)\s*"
        r"(?P<acc>[A-Za-z_$][A-Za-z0-9_$]*)\s*"
        r"(?P<op>\+=|=\s*(?P=acc)\s*\+)\s*"
        r"(?P<expr>[A-Za-z_$][A-Za-z0-9_$]*|helper_transform\s*\(\s*[A-Za-z_$][A-Za-z0-9_$]*\s*\))\s*;"
    )

    def guarded_repl(match: re.Match[str]) -> str:
        expr = match.group("expr").strip()
        value_match = re.search(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\b", expr)
        if value_match is None:
            return match.group(0)
        value_name = value_match.group(1)
        rewritten_expr = re.sub(rf"\b{re.escape(value_name)}\b", "currentItem", expr, count=1)
        return (
            f"if ({match.group('condition').strip()}) {{ "
            f"const currentItem = {value_name}; "
            f"{match.group('acc')} {match.group('op')} {rewritten_expr}; "
            "}"
        )

    guarded = guarded_pattern.sub(guarded_repl, code, count=1)
    if guarded != code:
        return guarded
    pattern = re.compile(
        r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_$][A-Za-z0-9_$]*)\s*"
        r"(?P<op>\+=|=\s*(?P=acc)\s*\+)\s*(?P<expr>[A-Za-z_$][A-Za-z0-9_$]*|helper_transform\s*\(\s*[A-Za-z_$][A-Za-z0-9_$]*\s*\))\s*;"
    )

    def repl(match: re.Match[str]) -> str:
        indent = match.group("indent")
        expr = match.group("expr").strip()
        value_match = re.search(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\b", expr)
        if value_match is None:
            return match.group(0)
        value_name = value_match.group(1)
        rewritten_expr = re.sub(rf"\b{re.escape(value_name)}\b", "currentItem", expr, count=1)
        return f"{indent}const currentItem = {value_name};\n{indent}{match.group('acc')} {match.group('op')} {rewritten_expr};"

    return pattern.sub(repl, code, count=1)


def _javascript_inline_temporary_binding(code: str) -> str:
    pattern = re.compile(
        r"(?P<indent>[ \t]*)(?:const|let|var)\s+currentItem\s*=\s*(?P<value>[A-Za-z_$][A-Za-z0-9_$]*);\n"
        r"(?P=indent)(?P<rest>[A-Za-z_$][A-Za-z0-9_$]*\s*(?:\+=|=\s*[A-Za-z_$][A-Za-z0-9_$]*\s*\+)\s*[^;\n]+;)"
    )

    def repl(match: re.Match[str]) -> str:
        rest = re.sub(r"\bcurrentItem\b", match.group("value"), match.group("rest"))
        return f"{match.group('indent')}{rest}"

    return pattern.sub(repl, code, count=1)


def _javascript_add_return_binding(code: str) -> str:
    if re.search(r"\b(?:finalTotal|return_total)\b", code):
        return code
    pattern = re.compile(
        r"(?P<indent>[ \t]*)return\s+(?P<expr>[A-Za-z_$][A-Za-z0-9_$]*)\s*;",
    )
    matches = list(pattern.finditer(code))
    helper_match = re.search(r"\n\s*function\s+helper_transform\s*\(", code)
    if helper_match is not None:
        matches = [match for match in matches if match.start() < helper_match.start()]
    if not matches:
        return code
    match = matches[-1]
    replacement = f"{match.group('indent')}const finalTotal = {match.group('expr')};\n{match.group('indent')}return finalTotal;"
    return code[: match.start()] + replacement + code[match.end() :]


def _javascript_inline_return_binding(code: str) -> str:
    pattern = re.compile(
        r"(?P<indent>[ \t]*)(?:const|let|var)\s+(?:finalTotal|return_total)\s*=\s*(?P<expr>[A-Za-z_$][A-Za-z0-9_$]*);\n"
        r"(?P=indent)return\s+(?:finalTotal|return_total)\s*;"
    )
    return pattern.sub(lambda match: f"{match.group('indent')}return {match.group('expr')};", code, count=1)


def _javascript_add_initialization_idiom(code: str) -> str:
    return re.sub(r"\b(let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*0\s*;", r"\1 \2 = Number(0);", code, count=1)


def _javascript_inline_initialization_idiom(code: str) -> str:
    return re.sub(r"\b(let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*Number\s*\(\s*0\s*\)\s*;", r"\1 \2 = 0;", code, count=1)


def _javascript_add_indexed_iteration(code: str) -> str:
    pattern = re.compile(
        r"for\s*\(\s*const\s+(?P<value>[A-Za-z_$][A-Za-z0-9_$]*)\s+of\s+(?P<iter>[A-Za-z_$][A-Za-z0-9_$]*)\s*\)\s*\{"
    )
    return pattern.sub(
        lambda match: (
            f"for (let index = 0; index < {match.group('iter')}.length; index++) {{\n"
            f"    const {match.group('value')} = {match.group('iter')}[index];"
        ),
        code,
        count=1,
    )


def _javascript_materialize_schedule(code: str, schedule: tuple[CarrierScheduleEntry, ...]) -> tuple[str, tuple[str, ...]]:
    materialized = code
    labels: list[str] = []
    transforms = (
        ("initialization_idiom", 1, "number_initialization", _javascript_add_initialization_idiom),
        ("initialization_idiom", 0, "literal_initialization", _javascript_inline_initialization_idiom),
        ("iteration_style", 1, "indexed_iteration", _javascript_add_indexed_iteration),
        ("temporary_binding_style", 1, "temporary_binding", _javascript_add_temporary_binding),
        ("temporary_binding_style", 0, "inline_temporary", _javascript_inline_temporary_binding),
        ("helper_extraction_style", 1, "helper_transform", _javascript_add_helper_transform),
        ("helper_extraction_style", 0, "inline_helper", _javascript_inline_helper_transform),
        ("accumulator_style", 1, "accumulator_binding", _javascript_add_accumulator_binding),
        ("accumulator_style", 0, "inline_accumulator", _javascript_inline_accumulator_binding),
        ("return_expression_style", 1, "return_binding", _javascript_add_return_binding),
        ("return_expression_style", 0, "direct_return", _javascript_inline_return_binding),
    )
    for family, bit_value, label, transform in transforms:
        if _schedule_target_bit(schedule, family) != bit_value:
            continue
        updated = transform(materialized)
        if updated != materialized:
            materialized = updated
            labels.append(label)
    return _normalize_javascript_compact_tokens(materialized), tuple(labels)


def _java_add_helper_transform(code: str) -> str:
    if "helperTransform(" in code:
        return code
    updated = re.sub(
        r"\b(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\+=|=\s*(?P=acc)\s*\+)\s*(?P<expr>[A-Za-z_][A-Za-z0-9_]*)\s*;",
        lambda match: f"{match.group('acc')} = {match.group('acc')} + helperTransform({match.group('expr')});",
        code,
        count=1,
    )
    if updated == code:
        return code
    helper = "    private static int helperTransform(int value) {\n        return value;\n    }\n"
    class_close = updated.rfind("}")
    if class_close < 0:
        return code
    return updated[:class_close].rstrip() + "\n\n" + helper + updated[class_close:]


def _java_inline_helper_transform(code: str) -> str:
    updated = re.sub(r"\bhelperTransform\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", r"\1", code)
    updated = re.sub(
        r"\n\s*private\s+static\s+int\s+helperTransform\s*\(\s*int\s+value\s*\)\s*\{\s*return\s+value\s*;\s*\}\s*",
        "\n",
        updated,
        flags=re.DOTALL,
    )
    return updated.strip()


def _java_add_accumulator_binding(code: str) -> str:
    if re.search(r"\bnextTotal\s*=", code):
        return code
    pattern = re.compile(
        r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*"
        r"(?:\+=|=\s*(?P=acc)\s*\+)\s*(?P<expr>[^;\n]+)\s*;"
    )

    def repl(match: re.Match[str]) -> str:
        indent = match.group("indent")
        acc = match.group("acc")
        expr = match.group("expr").strip()
        return f"{indent}int nextTotal = {acc} + {expr};\n{indent}{acc} = nextTotal;"

    return pattern.sub(repl, code, count=1)


def _java_inline_accumulator_binding(code: str) -> str:
    pattern = re.compile(
        r"(?P<indent>[ \t]*)int\s+nextTotal\s*=\s*(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*\+\s*(?P<expr>[^;\n]+);\n"
        r"(?P=indent)(?P=acc)\s*=\s*nextTotal;"
    )
    return pattern.sub(lambda match: f"{match.group('indent')}{match.group('acc')} += {match.group('expr').strip()};", code, count=1)


def _java_add_temporary_binding(code: str) -> str:
    if re.search(r"\bcurrentItem\s*=", code):
        return code
    guarded_pattern = re.compile(
        r"if\s*\((?P<condition>[^{}\n;]+)\)\s*"
        r"(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*"
        r"(?P<op>\+=|=\s*(?P=acc)\s*\+)\s*"
        r"(?P<expr>[A-Za-z_][A-Za-z0-9_]*|helperTransform\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\))\s*;"
    )

    def guarded_repl(match: re.Match[str]) -> str:
        expr = match.group("expr").strip()
        value_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", expr)
        if value_match is None:
            return match.group(0)
        value_name = value_match.group(1)
        rewritten_expr = re.sub(rf"\b{re.escape(value_name)}\b", "currentItem", expr, count=1)
        return (
            f"if ({match.group('condition').strip()}) {{ "
            f"int currentItem = {value_name}; "
            f"{match.group('acc')} {match.group('op')} {rewritten_expr}; "
            "}"
        )

    guarded = guarded_pattern.sub(guarded_repl, code, count=1)
    if guarded != code:
        return guarded
    pattern = re.compile(
        r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*"
        r"(?P<op>\+=|=\s*(?P=acc)\s*\+)\s*(?P<expr>[A-Za-z_][A-Za-z0-9_]*|helperTransform\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\))\s*;"
    )

    def repl(match: re.Match[str]) -> str:
        indent = match.group("indent")
        expr = match.group("expr").strip()
        value_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", expr)
        if value_match is None:
            return match.group(0)
        value_name = value_match.group(1)
        rewritten_expr = re.sub(rf"\b{re.escape(value_name)}\b", "currentItem", expr, count=1)
        return f"{indent}int currentItem = {value_name};\n{indent}{match.group('acc')} {match.group('op')} {rewritten_expr};"

    return pattern.sub(repl, code, count=1)


def _java_inline_temporary_binding(code: str) -> str:
    pattern = re.compile(
        r"(?P<indent>[ \t]*)int\s+currentItem\s*=\s*(?P<value>[A-Za-z_][A-Za-z0-9_]*);\n"
        r"(?P=indent)(?P<rest>[A-Za-z_][A-Za-z0-9_]*\s*(?:\+=|=\s*[A-Za-z_][A-Za-z0-9_]*\s*\+)\s*[^;\n]+;)"
    )

    def repl(match: re.Match[str]) -> str:
        rest = re.sub(r"\bcurrentItem\b", match.group("value"), match.group("rest"))
        return f"{match.group('indent')}{rest}"

    return pattern.sub(repl, code, count=1)


def _java_add_return_binding(code: str) -> str:
    if re.search(r"\breturnTotal\b", code):
        return code
    pattern = re.compile(
        r"(?P<indent>[ \t]*)return\s+(?P<expr>[A-Za-z_][A-Za-z0-9_]*)\s*;",
    )
    matches = list(pattern.finditer(code))
    helper_match = re.search(r"\n\s*private\s+static\s+int\s+helperTransform\s*\(", code)
    if helper_match is not None:
        matches = [match for match in matches if match.start() < helper_match.start()]
    if not matches:
        return code
    match = matches[-1]
    replacement = f"{match.group('indent')}int returnTotal = {match.group('expr')};\n{match.group('indent')}return returnTotal;"
    return code[: match.start()] + replacement + code[match.end() :]


def _java_inline_return_binding(code: str) -> str:
    pattern = re.compile(
        r"(?P<indent>[ \t]*)int\s+returnTotal\s*=\s*(?P<expr>[A-Za-z_][A-Za-z0-9_]*);\n"
        r"(?P=indent)return\s+returnTotal\s*;"
    )
    return pattern.sub(lambda match: f"{match.group('indent')}return {match.group('expr')};", code, count=1)


def _java_add_initialization_idiom(code: str) -> str:
    return re.sub(
        r"\bint\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*0\s*;",
        r"int \1 = Integer.valueOf(0);",
        code,
        count=1,
    )


def _java_inline_initialization_idiom(code: str) -> str:
    return re.sub(
        r"\bint\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*Integer\.valueOf\s*\(\s*0\s*\)\s*;",
        r"int \1 = 0;",
        code,
        count=1,
    )


def _java_add_indexed_iteration(code: str) -> str:
    pattern = re.compile(
        r"for\s*\(\s*int\s+(?P<value>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<iter>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\{"
    )
    return pattern.sub(
        lambda match: (
            f"for (int index = 0; index < {match.group('iter')}.length; index++) {{\n"
            f"            int {match.group('value')} = {match.group('iter')}[index];"
        ),
        code,
        count=1,
    )


def _java_materialize_schedule(code: str, schedule: tuple[CarrierScheduleEntry, ...]) -> tuple[str, tuple[str, ...]]:
    materialized = code
    labels: list[str] = []
    transforms = (
        ("initialization_idiom", 1, "typed_initialization", _java_add_initialization_idiom),
        ("initialization_idiom", 0, "literal_initialization", _java_inline_initialization_idiom),
        ("iteration_style", 1, "indexed_iteration", _java_add_indexed_iteration),
        ("temporary_binding_style", 1, "temporary_binding", _java_add_temporary_binding),
        ("temporary_binding_style", 0, "inline_temporary", _java_inline_temporary_binding),
        ("helper_extraction_style", 1, "helper_transform", _java_add_helper_transform),
        ("helper_extraction_style", 0, "inline_helper", _java_inline_helper_transform),
        ("accumulator_style", 1, "accumulator_binding", _java_add_accumulator_binding),
        ("accumulator_style", 0, "inline_accumulator", _java_inline_accumulator_binding),
        ("return_expression_style", 1, "return_binding", _java_add_return_binding),
        ("return_expression_style", 0, "direct_return", _java_inline_return_binding),
    )
    for family, bit_value, label, transform in transforms:
        if _schedule_target_bit(schedule, family) != bit_value:
            continue
        updated = transform(materialized)
        if updated != materialized:
            materialized = updated
            labels.append(label)
    return _repair_java_out_of_class_private_helpers(materialized), tuple(labels)


def _cpp_insert_helper_transform(code: str) -> str:
    if re.search(r"\b(?:static\s+)?int\s+helper_transform\s*\(", code):
        return code
    helper = "static int helper_transform(int value) {\n    return value;\n}\n\n"
    include_match = re.match(r"(?s)(?P<includes>(?:\s*#\s*include\s*<[^>\n]+>\s*\n)+)", code)
    if include_match:
        insert_at = include_match.end()
        return code[:insert_at] + "\n" + helper + code[insert_at:].lstrip()
    return helper + code.lstrip()


def _cpp_add_helper_transform(code: str) -> str:
    if re.search(r"\bhelper_transform\s*\(", code):
        return code
    pattern = re.compile(
        r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*"
        r"(?P<op>\+=|=\s*(?P=acc)\s*\+)\s*"
        r"(?P<expr>[A-Za-z_][A-Za-z0-9_]*)\s*;"
    )

    def repl(match: re.Match[str]) -> str:
        op = match.group("op")
        acc = match.group("acc")
        expr = match.group("expr")
        if "+=" in op:
            return f"{match.group('indent')}{acc} += helper_transform({expr});"
        return f"{match.group('indent')}{acc} = {acc} + helper_transform({expr});"

    updated = pattern.sub(repl, code, count=1)
    if updated == code:
        return code
    return _cpp_insert_helper_transform(updated)


def _cpp_inline_helper_transform(code: str) -> str:
    updated = re.sub(r"\bhelper_transform\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", r"\1", code)
    updated = re.sub(
        r"\n?\s*static\s+int\s+helper_transform\s*\(\s*int\s+value\s*\)\s*\{\s*return\s+value\s*;\s*\}\s*\n?",
        "\n",
        updated,
        flags=re.DOTALL,
    )
    return re.sub(r"\n{3,}", "\n\n", updated).strip()


def _cpp_add_accumulator_binding(code: str) -> str:
    if re.search(r"\bnextTotal\b", code):
        return code
    pattern = re.compile(
        r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*"
        r"(?:\+=|=\s*(?P=acc)\s*\+)\s*(?P<expr>[^;\n{}]+)\s*;"
    )

    def repl(match: re.Match[str]) -> str:
        indent = match.group("indent")
        acc = match.group("acc")
        expr = match.group("expr").strip()
        return f"{indent}int nextTotal = {acc} + {expr};\n{indent}{acc} = nextTotal;"

    return pattern.sub(repl, code, count=1)


def _cpp_inline_accumulator_binding(code: str) -> str:
    pattern = re.compile(
        r"(?P<indent>[ \t]*)int\s+nextTotal\s*=\s*(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*\+\s*(?P<expr>[^;\n]+);\n"
        r"(?P=indent)(?P=acc)\s*=\s*nextTotal;"
    )
    return pattern.sub(lambda match: f"{match.group('indent')}{match.group('acc')} += {match.group('expr').strip()};", code, count=1)


def _cpp_add_temporary_binding(code: str) -> str:
    if re.search(r"\bcurrentItem\b", code):
        return code
    guarded_pattern = re.compile(
        r"if\s*\((?P<condition>[^{}\n;]+)\)\s*"
        r"(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*"
        r"(?P<op>\+=|=\s*(?P=acc)\s*\+)\s*"
        r"(?P<expr>[A-Za-z_][A-Za-z0-9_]*|helper_transform\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\))\s*;"
    )

    def guarded_repl(match: re.Match[str]) -> str:
        expr = match.group("expr").strip()
        value_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", expr)
        if value_match is None:
            return match.group(0)
        value_name = value_match.group(1)
        rewritten_expr = re.sub(rf"\b{re.escape(value_name)}\b", "currentItem", expr, count=1)
        return (
            f"if ({match.group('condition').strip()}) {{ "
            f"int currentItem = {value_name}; "
            f"{match.group('acc')} {match.group('op')} {rewritten_expr}; "
            "}"
        )

    guarded = guarded_pattern.sub(guarded_repl, code, count=1)
    if guarded != code:
        return guarded
    pattern = re.compile(
        r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<op>\+=|=\s*(?P=acc)\s*\+)\s*"
        r"(?P<expr>[A-Za-z_][A-Za-z0-9_]*|helper_transform\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s*\))\s*;"
    )

    def repl(match: re.Match[str]) -> str:
        indent = match.group("indent")
        expr = match.group("expr").strip()
        value_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", expr)
        if value_match is None:
            return match.group(0)
        value_name = value_match.group(1)
        rewritten_expr = re.sub(rf"\b{re.escape(value_name)}\b", "currentItem", expr, count=1)
        return f"{indent}int currentItem = {value_name};\n{indent}{match.group('acc')} {match.group('op')} {rewritten_expr};"

    return pattern.sub(repl, code, count=1)


def _cpp_invert_simple_guard_assignment(code: str) -> str:
    pattern = re.compile(
        r"if\s*\((?P<cond>[^{}\n;]+)\)\s*\{\s*(?P<lhs>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<then>[^{}\n;]+?)\s*;\s*\}"
        r"\s*else\s*\{\s*(?P=lhs)\s*=\s*(?P<else>[^{}\n;]+?)\s*;\s*\}"
    )

    def repl(match: re.Match[str]) -> str:
        return (
            f"{match.group('lhs')} = {match.group('else').strip()}; "
            f"if ({match.group('cond').strip()}) {{ {match.group('lhs')} = {match.group('then').strip()}; }}"
        )

    return pattern.sub(repl, code, count=1)


def _cpp_inline_temporary_binding(code: str) -> str:
    pattern = re.compile(
        r"(?P<indent>[ \t]*)int\s+currentItem\s*=\s*(?P<value>[A-Za-z_][A-Za-z0-9_]*);\n"
        r"(?P=indent)(?P<rest>[A-Za-z_][A-Za-z0-9_]*\s*(?:\+=|=\s*[A-Za-z_][A-Za-z0-9_]*\s*\+)\s*[^;\n]+;)"
    )

    def repl(match: re.Match[str]) -> str:
        rest = re.sub(r"\bcurrentItem\b", match.group("value"), match.group("rest"))
        return f"{match.group('indent')}{rest}"

    return pattern.sub(repl, code, count=1)


def _cpp_add_return_binding(code: str) -> str:
    if re.search(r"\bfinalTotal\b", code):
        return code
    pattern = re.compile(
        r"(?P<indent>[ \t]*)return\s+(?P<expr>[A-Za-z_][A-Za-z0-9_]*)\s*;",
    )
    matches = list(pattern.finditer(code))
    if not matches:
        return code
    match = matches[-1]
    replacement = f"{match.group('indent')}int finalTotal = {match.group('expr')};\n{match.group('indent')}return finalTotal;"
    return code[: match.start()] + replacement + code[match.end() :]


def _cpp_inline_return_binding(code: str) -> str:
    pattern = re.compile(
        r"(?P<indent>[ \t]*)int\s+finalTotal\s*=\s*(?P<expr>[A-Za-z_][A-Za-z0-9_]*);\n"
        r"(?P=indent)return\s+finalTotal\s*;"
    )
    return pattern.sub(lambda match: f"{match.group('indent')}return {match.group('expr')};", code, count=1)


def _cpp_add_initialization_idiom(code: str) -> str:
    return re.sub(r"\bint\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*0\s*;", r"int \1 = int{0};", code, count=1)


def _cpp_inline_initialization_idiom(code: str) -> str:
    return re.sub(r"\bint\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*int\s*\{\s*0\s*\}\s*;", r"int \1 = 0;", code, count=1)


def _cpp_add_indexed_iteration(code: str) -> str:
    pattern = re.compile(
        r"for\s*\(\s*int\s+(?P<value>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<iter>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\{"
    )
    return pattern.sub(
        lambda match: (
            f"for (int index = 0; index < static_cast<int>({match.group('iter')}.size()); index++) {{\n"
            f"        int {match.group('value')} = {match.group('iter')}[index];"
        ),
        code,
        count=1,
    )


def _cpp_materialize_schedule(code: str, schedule: tuple[CarrierScheduleEntry, ...]) -> tuple[str, tuple[str, ...]]:
    materialized = code
    labels: list[str] = []
    transforms = (
        ("initialization_idiom", 1, "brace_initialization", _cpp_add_initialization_idiom),
        ("initialization_idiom", 0, "literal_initialization", _cpp_inline_initialization_idiom),
        ("iteration_style", 1, "indexed_iteration", _cpp_add_indexed_iteration),
        ("temporary_binding_style", 1, "temporary_binding", _cpp_add_temporary_binding),
        ("temporary_binding_style", 0, "inline_temporary", _cpp_inline_temporary_binding),
        ("early_return_guard_style", 1, "guard_assignment_normalization", _cpp_invert_simple_guard_assignment),
        ("helper_extraction_style", 1, "helper_transform", _cpp_add_helper_transform),
        ("helper_extraction_style", 0, "inline_helper", _cpp_inline_helper_transform),
        ("accumulator_style", 1, "accumulator_binding", _cpp_add_accumulator_binding),
        ("accumulator_style", 0, "inline_accumulator", _cpp_inline_accumulator_binding),
        ("return_expression_style", 1, "return_binding", _cpp_add_return_binding),
        ("return_expression_style", 0, "direct_return", _cpp_inline_return_binding),
    )
    for family, bit_value, label, transform in transforms:
        if _schedule_target_bit(schedule, family) != bit_value:
            continue
        updated = transform(materialized)
        if updated != materialized:
            materialized = updated
            labels.append(label)
    return _normalize_cpp_compact_tokens(materialized), tuple(labels)


def _python_materialize_schedule(
    code: str,
    request: GenerationRequest,
    schedule: tuple[CarrierScheduleEntry, ...],
) -> tuple[str, tuple[str, ...]]:
    materialized = code
    labels: list[str] = []
    for entry in schedule:
        if entry.target_bit is None:
            continue
        transformed = apply_carrier_variant(materialized, entry.family, int(entry.target_bit), request.language)
        if transformed != materialized:
            materialized = transformed
            labels.append(entry.family)
    return materialized, tuple(labels)


def _javascript_support_materializer_variants(
    code: str,
    schedule: tuple[CarrierScheduleEntry, ...],
) -> tuple[tuple[str, str], ...]:
    variants: list[tuple[str, str]] = []
    base = _normalize_javascript_compact_tokens(code)
    if _schedule_target_is_one(schedule, "helper_extraction_style") and "helper_transform(" not in base:
        helper_call = _javascript_add_helper_transform(base)
        if helper_call != base:
            variants.append(("javascript_materializer:helper_transform_call", _normalize_javascript_compact_tokens(helper_call)))
    if _schedule_target_is_one(schedule, "accumulator_style"):
        rewritten = _javascript_add_accumulator_binding(base)
        if rewritten != base:
            variants.append(("javascript_materializer:accumulator_assignment", _normalize_javascript_compact_tokens(rewritten)))
    exact, labels = _javascript_materialize_schedule(base, schedule)
    if labels and exact != base:
        variants.append(("javascript_materializer:scheduled_" + "_".join(labels), exact))
    return _dedupe_materializer_variants(variants)


def _java_support_materializer_variants(
    code: str,
    schedule: tuple[CarrierScheduleEntry, ...],
) -> tuple[tuple[str, str], ...]:
    variants: list[tuple[str, str]] = []
    base = _normalize_java_compact_tokens(code)
    helper_needed = _schedule_target_is_one(schedule, "helper_extraction_style")
    accumulator_assignment = _schedule_target_is_one(schedule, "accumulator_style")
    if helper_needed and "helperTransform(" not in base:
        updated = re.sub(
            r"\btotal\s*(?:\+=|=\s*total\s*\+)\s*([^;\n]+);",
            lambda match: f"total = total + helperTransform({match.group(1).strip()});",
            base,
            count=1,
        )
        helper = "    private static int helperTransform(int value) {\n        return value;\n    }\n"
        class_close = updated.rfind("}")
        if class_close >= 0 and updated != base:
            updated = updated[:class_close].rstrip() + "\n\n" + helper + updated[class_close:]
            variants.append(("java_materializer:helper_transform_accumulator", updated))
    if accumulator_assignment:
        rewritten = re.sub(r"\btotal\s*\+=\s*([^;\n]+);", r"total = total + \1;", base, count=1)
        if rewritten != base:
            variants.append(("java_materializer:accumulator_assignment", rewritten))
    exact, labels = _java_materialize_schedule(base, schedule)
    if labels and exact != base:
        variants.append(("java_materializer:scheduled_" + "_".join(labels), exact))
    return _dedupe_materializer_variants(variants)


def _cpp_support_materializer_variants(
    code: str,
    schedule: tuple[CarrierScheduleEntry, ...],
) -> tuple[tuple[str, str], ...]:
    variants: list[tuple[str, str]] = []
    base = _normalize_cpp_compact_tokens(code)
    exact, labels = _cpp_materialize_schedule(base, schedule)
    if labels and exact != base:
        variants.append(("cpp_materializer:scheduled_" + "_".join(labels), exact))
    rewritten = _cpp_invert_simple_guard_assignment(base)
    if rewritten != base:
        variants.append(("cpp_materializer:guard_assignment_normalization", _normalize_cpp_compact_tokens(rewritten)))
        exact_guarded, guarded_labels = _cpp_materialize_schedule(rewritten, schedule)
        if guarded_labels and exact_guarded != rewritten:
            variants.append(("cpp_materializer:guarded_scheduled_" + "_".join(guarded_labels), exact_guarded))
    if _schedule_target_is_one(schedule, "return_expression_style"):
        rewritten = _cpp_add_return_binding(base)
        if rewritten != base:
            variants.append(("cpp_materializer:return_binding", _normalize_cpp_compact_tokens(rewritten)))
    if _schedule_target_is_one(schedule, "accumulator_style"):
        rewritten = _cpp_add_accumulator_binding(base)
        if rewritten != base:
            variants.append(("cpp_materializer:accumulator_binding", _normalize_cpp_compact_tokens(rewritten)))
    if _schedule_target_is_one(schedule, "helper_extraction_style"):
        rewritten = _cpp_add_helper_transform(base)
        if rewritten != base:
            variants.append(("cpp_materializer:helper_transform", _normalize_cpp_compact_tokens(rewritten)))
    return _dedupe_materializer_variants(variants)


def _language_support_materializer_variants(
    code: str,
    request: GenerationRequest,
    schedule: tuple[CarrierScheduleEntry, ...],
) -> tuple[tuple[str, str], ...]:
    lowered = request.language.strip().lower()
    if lowered == "go":
        return _go_support_materializer_variants(code, request, schedule)
    if lowered in {"javascript", "js"}:
        return _javascript_support_materializer_variants(code, schedule)
    if lowered == "java":
        return _java_support_materializer_variants(code, schedule)
    if lowered in {"cpp", "c++"}:
        return _cpp_support_materializer_variants(code, schedule)
    variants: list[tuple[str, str]] = []
    if lowered == "python":
        exact, labels = _python_materialize_schedule(code, request, schedule)
        if labels and exact != code:
            variants.append(("python_materializer:scheduled_" + "_".join(labels), exact))
    for entry in schedule:
        if entry.target_bit is None:
            continue
        transformed = apply_carrier_variant(code, entry.family, int(entry.target_bit), request.language)
        if transformed != code:
            variants.append((f"{lowered}_materializer:{entry.family}", transformed))
    return _dedupe_materializer_variants(variants)


def _carrier_support_retry_prompt(base_prompt: str, *, language: str) -> str:
    normalized_language = language.strip().lower()
    if normalized_language == "go":
        return _go_carrier_support_retry_prompt(base_prompt)
    retry_lines = [
        base_prompt.rstrip(),
        "",
        "### Rewrite-backed support retry",
        "The previous semantic candidate preserved behavior but did not realize enough rewrite-backed carrier evidence.",
        "Generate a fresh semantically equivalent candidate that changes the body structure instead of copying the base body.",
        "A candidate is unacceptable if its whitespace-stripped program body is identical to the base body.",
        "Keep the same public signatures and tests passing.",
        "Realize at least two scheduled data carriers through visible behavior-preserving rewrites in the generated program body.",
        "Prefer concrete control/data-flow edits in the final code over restating the plan.",
        "Do not mention carrier names, labels, metadata, comments, markdown, or this retry instruction in the output.",
    ]
    if normalized_language in {"javascript", "js"}:
        retry_lines.extend(
            [
                "Use standard JavaScript token spacing and statement separators.",
                "Never merge JavaScript keywords with identifiers: write `function solve`, `else if (...)`, `else c = ...`, and `return total;`.",
                "Do not emit compact forms such as `functionsolve`, `elseif`, `elsec`, `returntotal`, or `constvalue`.",
            ]
        )
    if normalized_language == "go":
        retry_lines.extend(
            [
                "For Go, helper calls alone are not enough: also rewrite loop/update/return structure when those carriers are scheduled.",
                "If the base loop already calls the helper, introduce behavior-preserving accumulator or return restructuring rather than emitting the same loop body.",
                "Use at least two concrete low-risk Go rewrites when applicable: bind the helper result to a local variable, update the accumulator through a temporary next value, return a result variable instead of the raw accumulator, or replace a nested guard with an equivalent continue guard.",
                "Do not stop after one carrier-looking edit; the accepted candidate needs at least two independent scheduled carrier families across at least two structural levels.",
                "Prefer one CFG-level change such as loop or guard structure and one SSA-level change such as accumulator, temporary binding, initialization, or return binding.",
                "Do not change helper arithmetic, guard predicates, or which input values are included.",
                "Return a complete evaluator-ready snippet: helper declarations followed by the requested entrypoint, or only the entrypoint body when the decoder prefix already supplies the helper and signature.",
                "Output only Go declarations accepted by the evaluator wrapper; no package declaration or imports.",
            ]
        )
    return "\n".join(retry_lines) + "\n"


def _trained_checkpoint_available(request: GenerationRequest) -> bool:
    execution_mode = request.execution_mode.lower()
    if "trained" not in execution_mode and "remote" not in execution_mode:
        return False
    checkpoint_path = Path(request.checkpoint_path) if request.checkpoint_path else None
    adapter_path = Path(request.adapter_path) if request.adapter_path else None
    return bool(
        (checkpoint_path is not None and checkpoint_path.exists())
        or (adapter_path is not None and adapter_path.exists())
    )


def _strip_prompt_prefix(text: str, prompt: str) -> str:
    if text.startswith(prompt):
        return text[len(prompt):].lstrip()
    return text.strip()


def _normalize_decoded_text(text: str) -> str:
    return (
        text.replace("\u0120", " ")
        .replace("\u010a", "\n")
        .replace("\u0109", "\t")
        .replace("\u23ce", "\n")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def _compact_index_map(text: str) -> tuple[str, list[int]]:
    compact_chars: list[str] = []
    original_indexes: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        compact_chars.append(char)
        original_indexes.append(index)
    return "".join(compact_chars), original_indexes


def _strip_compact_prefix(text: str, prefix: str) -> str:
    compact_text, original_indexes = _compact_index_map(text)
    compact_prefix, _ = _compact_index_map(prefix)
    if not compact_prefix or not compact_text.startswith(compact_prefix):
        return text
    if len(compact_prefix) >= len(original_indexes):
        return ""
    return text[original_indexes[len(compact_prefix) - 1] + 1 :].lstrip()


def _starts_with_compact_prefix(text: str, prefix: str) -> bool:
    compact_text, _ = _compact_index_map(text.lstrip())
    compact_prefix, _ = _compact_index_map(prefix)
    return bool(compact_prefix) and compact_text.startswith(compact_prefix)


def _strip_after_compact_marker(text: str, markers: tuple[str, ...]) -> str:
    compact_text, original_indexes = _compact_index_map(text)
    best_start: int | None = None
    best_end: int | None = None
    for marker in markers:
        compact_marker, _ = _compact_index_map(marker)
        if not compact_marker:
            continue
        marker_index = compact_text.find(compact_marker)
        if marker_index < 0:
            continue
        marker_end = marker_index + len(compact_marker)
        if marker_end > len(original_indexes):
            continue
        original_start = original_indexes[marker_index]
        original_end = original_indexes[marker_end - 1] + 1
        prelude = text[:original_start]
        prelude_is_prompt_echo = (not prelude.strip()) or any(
            prompt_marker in prelude
            for prompt_marker in (
                "SemCodebook",
                "Instruction:",
                "Task Prompt:",
                "Scheduled Carriers:",
                "Base Code:",
                "Base Code",
            )
        ) or "### SemCodebook" in prelude
        if not prelude_is_prompt_echo:
            continue
        if (
            best_end is None
            or best_start is None
            or original_start < best_start
            or (original_start == best_start and original_end > best_end)
        ):
            best_start = original_start
            best_end = original_end
    return text[best_end:].lstrip() if best_end is not None else text


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    fenced = stripped[3:]
    first_line, separator, remainder = fenced.partition("\n")
    if separator and re.fullmatch(r"[A-Za-z0-9_+#.-]+", first_line.strip()):
        fenced = remainder
    if "```" in fenced:
        fenced = fenced.split("```", 1)[0]
    return fenced.strip()


def _trim_language_prelude(text: str, *, language: str = "") -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    lowered = language.strip().lower()
    language_aliases = {
        "python": {"python", "py"},
        "javascript": {"javascript", "js"},
        "js": {"javascript", "js"},
        "java": {"java"},
        "go": {"go", "golang"},
        "cpp": {"cpp", "c++"},
        "c++": {"cpp", "c++"},
    }
    if stripped.lower() in language_aliases.get(lowered, {lowered}):
        return ""
    if lowered in {"cpp", "c++"}:
        stripped = re.sub(r"^\s*:\s*(?=#\s*include\s*<)", "", stripped)
    first_line, separator, remainder = stripped.partition("\n")
    if separator and re.fullmatch(r"[A-Za-z0-9_+#.-]+", first_line.strip()):
        stripped = remainder.strip()
    patterns_by_language = {
        "python": (r"(?m)^(?:from\s+\S+\s+import\s+|import\s+\S+|def\s+[A-Za-z_]|class\s+[A-Za-z_])",),
        "javascript": (r"(?m)^(?:function\s+[A-Za-z_$]|const\s+[A-Za-z_$]|let\s+[A-Za-z_$]|class\s+[A-Za-z_$]|export\s+)",),
        "js": (r"(?m)^(?:function\s+[A-Za-z_$]|const\s+[A-Za-z_$]|let\s+[A-Za-z_$]|class\s+[A-Za-z_$]|export\s+)",),
        "java": (r"(?m)^(?:import\s+|public\s+class\s+[A-Za-z_]|class\s+[A-Za-z_])",),
        "go": (r"(?m)^(?:package\s+main|func\s+[A-Za-z_])",),
        "cpp": (r"(?m)^(?:#\s*include\s*<|using\s+namespace\s+|template\s*<|class\s+[A-Za-z_]|(?:static\s+)?(?:int|long|bool|double|float|std::[A-Za-z_][A-Za-z0-9_:<>]*)\s+[A-Za-z_])",),
        "c++": (r"(?m)^(?:#\s*include\s*<|using\s+namespace\s+|template\s*<|class\s+[A-Za-z_]|(?:static\s+)?(?:int|long|bool|double|float|std::[A-Za-z_][A-Za-z0-9_:<>]*)\s+[A-Za-z_])",),
    }
    patterns = patterns_by_language.get(lowered, tuple(pattern for values in patterns_by_language.values() for pattern in values))
    earliest: int | None = None
    for pattern in patterns:
        match = re.search(pattern, stripped)
        if match and (earliest is None or match.start() < earliest):
            earliest = match.start()
    if earliest is not None and earliest > 0:
        prelude = stripped[:earliest]
        prelude_is_prompt_echo = any(
            marker in prelude
            for marker in (
                "SemCodebook",
                "Instruction:",
                "Task Prompt:",
                "Scheduled Carriers:",
                "Private Carrier Plan:",
                "Base Code:",
                "Output code",
                "Outputcode",
                "Output contract:",
                "Carrier-Key",
                "Task-ID",
                "WM-ID",
                "Language:",
                "The callable must",
                "Write a ",
                "entry point",
            )
        ) or "###" in prelude
        if prelude_is_prompt_echo:
            stripped = stripped[earliest:].lstrip()
    if "```" in stripped:
        stripped = stripped.split("```", 1)[0]
    return stripped.strip()


def _looks_like_prompt_echo_without_code(text: str, *, language: str = "") -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lowered = language.strip().lower()
    aliases_by_language = {
        "python": {"python", "py"},
        "javascript": {"javascript", "js"},
        "js": {"javascript", "js"},
        "java": {"java"},
        "go": {"go", "golang"},
        "cpp": {"cpp", "c++"},
        "c++": {"cpp", "c++"},
    }
    if stripped.lower() in aliases_by_language.get(lowered, {lowered}):
        return True
    code_start_by_language = {
        "python": r"(?m)^(?:from\s+\S+\s+import\s+|import\s+\S+|def\s+[A-Za-z_]|class\s+[A-Za-z_])",
        "javascript": r"(?m)^(?:function\s+[A-Za-z_$]|const\s+[A-Za-z_$]|let\s+[A-Za-z_$]|class\s+[A-Za-z_$]|export\s+)",
        "js": r"(?m)^(?:function\s+[A-Za-z_$]|const\s+[A-Za-z_$]|let\s+[A-Za-z_$]|class\s+[A-Za-z_$]|export\s+)",
        "java": r"(?m)^(?:import\s+|public\s+class\s+[A-Za-z_]|class\s+[A-Za-z_])",
        "go": r"(?m)^(?:package\s+main|func\s+[A-Za-z_])",
        "cpp": r"(?m)^(?:#\s*include\s*<|using\s+namespace\s+|template\s*<|class\s+[A-Za-z_]|(?:static\s+)?(?:int|long|bool|double|float|std::[A-Za-z_][A-Za-z0-9_:<>]*)\s+[A-Za-z_])",
        "c++": r"(?m)^(?:#\s*include\s*<|using\s+namespace\s+|template\s*<|class\s+[A-Za-z_]|(?:static\s+)?(?:int|long|bool|double|float|std::[A-Za-z_][A-Za-z0-9_:<>]*)\s+[A-Za-z_])",
    }
    code_pattern = code_start_by_language.get(lowered)
    if code_pattern and re.search(code_pattern, stripped):
        return False
    compact = "".join(stripped.split()).lower()
    prompt_markers = (
        "languagespecificconstraints",
        "language-specificconstraints",
        "taskprompt:",
        "privatecarrierplan",
        "bit-leveltargetsignalannotations",
        "scheduledcarriers",
        "outputcontract:",
        "requiredgooutputshape",
        "semcodebookrewritetask",
        "writea",
        "thecallablemust",
        "preserveeverypublic",
        "donotemitprose",
    )
    return any(marker in compact[:2000] for marker in prompt_markers)


def _strip_output_assignment_echo(text: str, *, language: str = "") -> str:
    metadata_markers = (
        "Language=",
        "Language:",
        "Carrier-Key=",
        "Carrier-Key:",
        "Task-ID=",
        "Task-ID:",
        "WM-ID=",
        "WM-ID:",
        "Scheduled Carriers",
        "Private Carrier Plan",
        "Output contract:",
    )
    code_start_by_language = {
        "python": r"(?:from\s+\S+\s+import\s+|import\s+\S+|def\s+[A-Za-z_]|class\s+[A-Za-z_])",
        "javascript": r"(?:function\s+[A-Za-z_$]|const\s+[A-Za-z_$]|let\s+[A-Za-z_$]|class\s+[A-Za-z_$]|export\s+)",
        "js": r"(?:function\s+[A-Za-z_$]|const\s+[A-Za-z_$]|let\s+[A-Za-z_$]|class\s+[A-Za-z_$]|export\s+)",
        "java": r"(?:import\s+|public\s+class\s+[A-Za-z_]|class\s+[A-Za-z_])",
        "go": r"(?:package\s+main|func\s+[A-Za-z_])",
        "cpp": r"(?:#\s*include\s*<|using\s+namespace\s+|template\s*<|class\s+[A-Za-z_]|(?:static\s+)?(?:int|long|bool|double|float|std::[A-Za-z_][A-Za-z0-9_:<>]*)\s+[A-Za-z_])",
        "c++": r"(?:#\s*include\s*<|using\s+namespace\s+|template\s*<|class\s+[A-Za-z_]|(?:static\s+)?(?:int|long|bool|double|float|std::[A-Za-z_][A-Za-z0-9_:<>]*)\s+[A-Za-z_])",
    }
    lowered = language.strip().lower()
    code_start = code_start_by_language.get(lowered, "|".join(code_start_by_language.values()))
    best: str | None = None
    for match in re.finditer(r"(?im)^\s*(?:Output(?:\s*code)?|Final\s+Code|Final\s+Program)\s*[:=]\s*", text):
        prelude = text[: match.start()]
        remainder = text[match.end() :].lstrip()
        if not remainder or (prelude.strip() and not any(marker in prelude for marker in metadata_markers)):
            continue
        if re.search(rf"(?m)^{code_start}", remainder):
            best = remainder
    return best if best is not None else text


def _go_decoded_starts_with_prompt_residue(text: str) -> bool:
    stripped = text.lstrip()
    lowered = stripped.lower()
    compact = "".join(stripped.split()).lower()
    return lowered.startswith(
        (
            "language-specific constraints:",
            "required go output shape:",
            "task prompt:",
            "private carrier plan:",
            "base code:",
            "- start direct",
            "- start directly",
            "- preserve the exact",
            "- the evaluator supplies",
            "- use `_`",
            "- do not emit prose",
            "- the first non-empty",
            "- the output must contain",
            "- bind",
            "- use an equivalent",
            "- prefer direct",
            "- assign the accumulator",
            "- helper calls alone",
            "- introduce behavior-preserving",
            "- at least two concrete",
            "- do not change helper arithmetic",
            "- return a complete evaluator-ready",
            "- generate a fresh",
            "- a candidate is",
        )
    ) or compact.startswith(
        (
            "language-specificconstraints:",
            "languagespecificconstraints:",
            "requiredgooutputshape:",
            "taskprompt:",
            "privatecarrierplan:",
            "basecode:",
            "-startdirect",
            "-startdirectly",
            "-preservetheexact",
            "-theevaluatorsupplies",
            "-use`_`",
            "-donotemitprose",
            "-thefirstnon-empty",
            "-theoutputmustcontain",
            "-bind",
            "-useanequivalent",
            "-preferdirect",
            "-assigntheaccumulator",
            "-helpercallsalone",
            "-introducebehavior-preserving",
            "-atleasttwoconcrete",
            "-donotchangehelperarithmetic",
            "-returnacompleteevaluator-ready",
            "-generateafresh",
            "-acandidateis",
        )
    )


def _merge_forced_output_prefix(decoded: str, forced_output_prefix: str, *, language: str = "", prompt: str = "", base_code: str = "") -> str:
    if not forced_output_prefix:
        return decoded
    lowered = language.strip().lower()
    if lowered == "python":
        signature = _python_entrypoint_signature(prompt, base_code)
        if signature and (
            _starts_with_compact_prefix(decoded, signature)
            or decoded.lstrip().startswith(signature)
        ):
            helper_prefix = forced_output_prefix.split(signature, 1)[0]
            return helper_prefix.rstrip() + ("\n\n" if helper_prefix.strip() else "") + decoded.lstrip()
    if lowered == "go":
        if _go_decoded_starts_with_prompt_residue(decoded):
            cleaned = _strip_generated_artifact_lines(decoded, language="go")
            if cleaned.strip() and not _go_decoded_starts_with_prompt_residue(cleaned):
                decoded = cleaned
            else:
                return decoded.lstrip()
        signature = _go_entrypoint_signature(prompt, base_code)
        if signature and (
            _starts_with_compact_prefix(decoded, signature)
            or _starts_with_compact_prefix(decoded, signature + " {")
        ):
            supporting_prefix = _go_supporting_prefix_from_base_code(prompt, base_code, signature)
            missing_supporting_prefix = _go_missing_supporting_prefix(decoded, supporting_prefix)
            if missing_supporting_prefix:
                return missing_supporting_prefix.rstrip() + "\n\n" + decoded.lstrip()
            return decoded.lstrip()
    return forced_output_prefix + decoded


def _strip_generated_artifact_lines(text: str, *, language: str = "") -> str:
    function_names = set(re.findall(r"(?m)^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text))
    lowered_language = language.strip().lower()
    python_code_started = bool(function_names)
    prompt_line_prefixes = (
        "instruction:",
        "output contract:",
        "output runnable source code:",
        "task prompt:",
        "private carrier plan:",
        "base code:",
        "every scheduled carrier",
        "scheduled carriers",
        "semcodebook rewrite task",
        "the callable must",
        "write a ",
        "here is",
        "in class `",
        "language-specific",
        "language-agnostic",
        "language-independent",
        "carrier guidance",
    )
    prompt_bullet_prefixes = (
        "- output only",
        "- preserve every",
        "- preserve the exact",
        "- do not output",
        "- don't emit",
        "- realize the carrier",
        "- every scheduled carrier",
        "- start direct",
        "- start directly with",
        "- bind",
        "- use an equivalent",
        "- prefer direct",
        "- assign the accumulator",
        "- helper calls alone",
        "- introduce behavior-preserving",
        "- at least two concrete",
        "- do not change helper arithmetic",
        "- return a complete evaluator-ready",
        "- generate a fresh",
        "- a candidate is",
        "- keep the same public",
        "- prefer concrete",
        "- if the base loop",
        "- comparison operand order:",
        "- reali",
        "-preservetheexact",
        "-preserveevery",
        "-donotemit",
        "-donotoutput",
        "-startdirect",
        "-startdirectlywith",
        "-bind",
        "-useanequivalent",
        "-preferdirect",
        "-assigntheaccumulator",
        "-helpercallsalone",
        "-introducebehavior-preserving",
        "-atleasttwoconcrete",
        "-donotchangehelperarithmetic",
        "-returnacompleteevaluator-ready",
        "-generateafresh",
        "-acandidateis",
        "-keepthesamepublic",
        "-preferconcrete",
        "-ifthebaseloop",
        "-comparisonoperandorder:",
        "-realizethecarrier",
        "-theevaluatorsupplies",
        "-startdirectlywith",
        "-use`_`",
        "-thefirstnon-empty",
        "-theoutputmustcontain",
    )
    compact_prompt_bullet_prefixes = tuple("".join(prefix.split()).lower() for prefix in prompt_bullet_prefixes)
    kept: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        compact = "".join(stripped.split()).lower()
        if compact in {"###outputcode###", "###outputcode", "outputcode", "outputcode###", "outputcode:"}:
            continue
        if lowered_language == "go" and (
            compact.startswith(
                (
                    "outputonlygofunction/helperdeclarationsneededbythetask",
                    "donotemitapackagedeclaration,imports,tests,assertions,printexamples,orlinesbeginningwith.",
                    "realizethecarrierplansilentlyinsidethefinalprogramstructure;neverlistcarriersnippetsseparately.",
                    "preserveeverypublicfunctionsignaturethatappearsinthebasecode.",
                    "startdirectlywiththeexactfuncdeclarationfromthetaskprompt.",
                    "theevaluatorsuppliesthegowrapper;donotemitapackagedeclaration.",
                    "preservetheexactfunctionnameandsignaturerequestedinthetaskprompt.",
                    "use`_`foranunusedrangeindex,forexample`for_,value:=rangevalues`.",
                    "donotemitproseormarkdownbeforethefirstgodeclaration.",
                    "requiredgooutputshape:",
                    "thefirstnon-emptyoutputlinemustbeexactly:",
                    "theoutputmustcontainthatfunctionbodyandclosingbrace.",
                )
            )
            or compact.startswith(
                (
                    "carrierkey:",
                    "environment:",
                    "bittarget:",
                    "family:",
                    "signal:",
                    "occurrence:",
                    "applicable:",
                    "level:",
                    "basecodekey:",
                    "outputonly:",
                    "requiredgooutputshape:",
                    "taskprompt:",
                    "taskid:",
                    "bitfamilykey:",
                    "bittargetbit:",
                    "signalkey:",
                )
            )
        ):
            continue
        lowered = stripped.lower()
        if compact.startswith("language-specificconstraints") or compact.startswith("languagespecificconstraints"):
            continue
        if any(lowered.startswith(prefix) for prefix in prompt_line_prefixes):
            continue
        if re.fullmatch(r"(?:language|carrier-key|task-id|wm-id|base)\s*[:=].*", stripped, flags=re.IGNORECASE):
            continue
        if stripped.startswith("`") and stripped.endswith(("`", "`.","`;")):
            continue
        if re.match(r"^-\s+[A-Za-z_ /-]+(?:style|idiom)\s*:", stripped, flags=re.IGNORECASE):
            continue
        if (
            (compact.startswith("-role=") or compact.startswith("role=") or compact.startswith("/*-role=") or compact.startswith("//-role="))
            and "family=" in compact
        ):
            continue
        if "family=" in compact and "target_bit=" in compact:
            continue
        if re.match(r"(?i)^[A-Za-z0-9_]*(?:carrier|boundary|schedule)[A-Za-z0-9_]*\s*=", stripped):
            continue
        if any(lowered.startswith(prefix) for prefix in prompt_bullet_prefixes) or any(
            compact.startswith(prefix) for prefix in compact_prompt_bullet_prefixes
        ):
            continue
        if compact.startswith(("/*-outputonly", "//-outputonly", "/*-donotoutput", "//-donotoutput")):
            continue
        if re.match(r"^-\s+[A-Za-z_][A-Za-z0-9_]*\s+(?:not\s+)?applicable\b", stripped, flags=re.IGNORECASE):
            continue
        if re.match(r"^-\s+[A-Za-z_][A-Za-z0-9_-]*\s*$", stripped):
            continue
        if len(stripped) >= 24 and set(stripped) <= {"-", "="}:
            continue
        if stripped.startswith("<jupyter_") or stripped in {"<empty_output>", "<j"}:
            break
        if stripped.startswith("###") and "output" in compact:
            continue
        if lowered_language == "python":
            if stripped.startswith(("def ", "class ")):
                python_code_started = True
            if python_code_started and raw_line == raw_line.lstrip():
                if re.match(r"^(?:def\s+|class\s+|from\s+\S+\s+import\s+|import\s+\S+|@|if\s+__name__\s*==\s*['\"]__main__['\"])", stripped):
                    kept.append(raw_line)
                    continue
                if stripped and any(char.isalpha() for char in stripped):
                    assignment = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
                    if assignment and assignment.group(1) in function_names:
                        continue
                    if stripped.startswith("assert ") or stripped.startswith("print("):
                        continue
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", stripped):
                        kept.append(raw_line)
                        continue
                    break
            if raw_line == raw_line.lstrip():
                if stripped.startswith("assert ") or stripped.startswith("print("):
                    continue
                assignment = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
                if assignment and assignment.group(1) in function_names:
                    continue
        kept.append(raw_line)
    return "\n".join(kept).strip()


def _recover_go_repeated_entrypoint(text: str, prompt: str) -> str:
    signature = _go_entrypoint_signature(prompt, "")
    if not signature:
        return text
    matches = list(re.finditer(re.escape(signature), text))
    if len(matches) < 2:
        return text
    helper_matches = list(re.finditer(r"(?m)^func\s+helper_[A-Za-z_][A-Za-z0-9_]*\s*\(", text[: matches[-1].start()]))
    if helper_matches and matches[0].start() == 0:
        forced_stub_prefix = text[: helper_matches[0].start()]
        if forced_stub_prefix.count("{") > forced_stub_prefix.count("}") and forced_stub_prefix.count(signature) == 1:
            return text[helper_matches[0].start() :].lstrip()
    compact_prelude = "".join(text[: matches[-1].start()].split()).lower()
    prompt_echo_markers = (
        "outputonlygofunction/helperdeclarationsneededbythetask",
        "donotemitapackagedeclaration",
        "realizethecarrierplansilentlyinsidethefinalprogramstructure",
        "preserveeverypublicfunctionsignaturethatappearsinthebasecode",
        "theevaluatorsuppliesthegowrapper",
        "thefirstnon-emptyoutputlinemustbeexactly",
        "carrierkey:",
        "environment:",
        "basecodekey:",
        "outputonly:",
    )
    if not any(marker in compact_prelude for marker in prompt_echo_markers):
        return text
    if helper_matches:
        return text[helper_matches[0].start() :].lstrip()
    return text[matches[-1].start() :].lstrip()


def _strip_trailing_generated_sections(text: str, *, language: str = "") -> str:
    lowered_language = language.strip().lower()
    code_start_patterns = {
        "javascript": r"(?:function\s*[A-Za-z_$][A-Za-z0-9_$]*\s*\(|(?:const|let|var)\s*[A-Za-z_$][A-Za-z0-9_$]*\s*=|class\s*[A-Za-z_$][A-Za-z0-9_$]*)",
        "js": r"(?:function\s*[A-Za-z_$][A-Za-z0-9_$]*\s*\(|(?:const|let|var)\s*[A-Za-z_$][A-Za-z0-9_$]*\s*=|class\s*[A-Za-z_$][A-Za-z0-9_$]*)",
        "java": r"(?:import\s+|public\s+class\s+[A-Za-z_][A-Za-z0-9_]*|class\s+[A-Za-z_][A-Za-z0-9_]*|publicclass[A-Za-z_][A-Za-z0-9_]*)",
        "go": r"func\s+[A-Za-z_][A-Za-z0-9_]*\s*\(",
        "cpp": r"(?:#\s*include\s*<|(?:static\s+)?(?:int|long|bool|double|float|void|auto|std::[A-Za-z_][A-Za-z0-9_:<>]*)\s+[A-Za-z_])",
        "c++": r"(?:#\s*include\s*<|(?:static\s+)?(?:int|long|bool|double|float|void|auto|std::[A-Za-z_][A-Za-z0-9_:<>]*)\s+[A-Za-z_])",
    }
    code_start = code_start_patterns.get(lowered_language)
    if not code_start or not re.search(code_start, text):
        return text
    markers = (
        "###Instruction",
        "### Instruction",
        "###SemCodebook",
        "### SemCodebook",
        "###Output",
        "### Output",
        "###Outputcode",
        "### Output code",
        "###Response",
        "### Response",
        "ScheduledDataCarrier",
        "Scheduled Data Carrier",
        "Rewrite-backed support retry",
        "PreviousCarrier-Key",
        "Response:",
        "Carrier-Key:",
        "Carrier-Key=",
        "Task Prompt:",
        "Private Carrier Plan:",
        "- Return one complete",
    )
    marker_positions = [text.find(marker) for marker in markers if text.find(marker) > 0]
    if not marker_positions:
        return text
    return text[: min(marker_positions)].rstrip()


def _strip_go_generated_transcript(text: str) -> str:
    kept: list[str] = []
    saw_declaration = False
    brace_depth = 0
    transcript_patterns = (
        r"#{2,}\s*(?:Instruction|SemCodebook|Task\s+Prompt|Output|Base\s+Code)\b",
        r"(?:Instruction|Output\s+contract|Task\s+Prompt|Private\s+Carrier\s+Plan):",
        r"(?:Language|Family|Role|Level|Signal|Task|Applicable|Severity|Helper|Base\s+code|Bit\s+family|Adaptation\s+target)\s*[:=]",
        r"(?:Scheduled\s+(?:data\s+)?carriers|Carrier\s+Plan|Carrier\s*Key)\s*[:=]",
        r"Generated\s+code\s*:",
        r"go\s+(?:run|test)\b",
        r"===\s+RUN\b",
        r"---\s+(?:PASS|FAIL):\b",
        r"PASS\b",
        r"FAIL\b",
        r"ok\s+\S+",
        r"goos:\s+",
        r"goarch:\s+",
        r"pkg:\s+",
        r"cpu:\s+",
        r"Benchmark[A-Za-z0-9_/-]+",
        r"coverage:\s+",
    )
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if (
            saw_declaration
            and brace_depth <= 0
            and any(re.match(pattern, stripped) for pattern in transcript_patterns)
        ):
            break
        kept.append(raw_line)
        if re.match(r"^func\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", stripped):
            saw_declaration = True
        brace_depth += raw_line.count("{") - raw_line.count("}")
    return "\n".join(kept).strip()


def _strip_go_package_prelude(text: str) -> str:
    return re.sub(
        r"^\s*package\s+main\s*;?\s*(?=(?:func\b|func[A-Za-z_]|Language-specific|Language-specificconstraints|TaskPrompt|PrivateCarrierPlan|Bit-level|$))",
        "",
        text,
    ).lstrip()


def _allow_raw_decoder_fallback(text: str, *, language: str = "") -> bool:
    if _looks_like_prompt_echo_without_code(text, language=language):
        return False
    if language.strip().lower() != "go":
        return True
    compact = "".join(text.strip().split()).lower()
    if compact in {"packagemain", ""}:
        return False
    prompt_markers = (
        "packagemainlanguage-specific",
        "languagespecificconstraints",
        "language-specificconstraints",
        "taskprompt:",
        "privatecarrierplan",
        "bit-leveltargetsignalannotations",
        "requiredgooutputshape",
    )
    return not any(marker in compact for marker in prompt_markers)


def _replace_unused_go_range_indices(text: str) -> str:
    header_pattern = re.compile(
        r"\bfor\s+(?P<index>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
        r"(?P<value>[A-Za-z_][A-Za-z0-9_]*)\s*:=\s*range\s+"
        r"(?P<iterable>[^{}\n;]+?)\s*\{"
    )
    chunks: list[str] = []
    position = 0
    for match in header_pattern.finditer(text):
        index_name = match.group("index")
        if index_name == "_":
            continue
        brace_start = match.end() - 1
        depth = 0
        brace_end = -1
        for offset in range(brace_start, len(text)):
            char = text[offset]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    brace_end = offset
                    break
        if brace_end < 0:
            continue
        body = text[brace_start + 1 : brace_end]
        if re.search(rf"\b{re.escape(index_name)}\b", body):
            continue
        chunks.append(text[position : match.start()])
        chunks.append(
            f"for _, {match.group('value')} := range {match.group('iterable').strip()} {{"
        )
        position = match.end()
    if not chunks:
        return text
    chunks.append(text[position:])
    return "".join(chunks)


def _normalize_go_compact_tokens(text: str) -> str:
    normalized = text
    normalized = re.sub(r"(?m)^(\s*)func(?=[A-Za-z_])", r"\1func ", normalized)
    normalized = re.sub(r"}[ \t]*(?=func[ \t]+)", "}\n\n", normalized)
    normalized = re.sub(r"}[ \t]*(?=[A-Za-z_][A-Za-z0-9_]*[ \t]*(?:\+=|-=|\*=|/=|:=|=))", "}\n    ", normalized)
    normalized = re.sub(
        r"(?m)^([ \t]*[A-Za-z_][A-Za-z0-9_]*[ \t]*(?::=|=)[ \t]*[^{}\n;]*?)(for_,)",
        r"\1\n\2",
        normalized,
    )
    normalized = re.sub(
        r"(?m)(?P<assignment>(?::=|=)[ \t]*-?\d+(?:\.\d+)?)(?P<loop>for[A-Za-z_][A-Za-z0-9_]*,)",
        r"\g<assignment>\n\g<loop>",
        normalized,
    )
    normalized = re.sub(r"\bfor_,", "for _,", normalized)
    normalized = re.sub(
        r"\bfor([A-Za-z_][A-Za-z0-9_]*)[ \t]*,[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*:=",
        r"for \1, \2 :=",
        normalized,
    )
    normalized = re.sub(r"\bfor[ \t]+_,([A-Za-z_])", r"for _, \1", normalized)
    normalized = re.sub(r"\bfor[ \t]+_,[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*:=", r"for _, \1 :=", normalized)
    normalized = re.sub(r"\bfor([A-Za-z_][A-Za-z0-9_]*)\s*:=", r"for \1 :=", normalized)
    normalized = re.sub(r":=range([A-Za-z_][A-Za-z0-9_]*)", r":= range \1", normalized)
    normalized = re.sub(r"\brange([A-Za-z_][A-Za-z0-9_]*)", r"range \1", normalized)
    normalized = re.sub(r"(?<![A-Za-z0-9_])if(?=[A-Za-z_(0-9])", "if ", normalized)
    normalized = re.sub(r"(?<![A-Za-z0-9_])return(?=[A-Za-z_(0-9])", "return ", normalized)
    normalized = re.sub(r"}\s*return\b", "}\n    return", normalized)
    normalized = re.sub(
        r"(?m)^([ \t]*[A-Za-z_][A-Za-z0-9_]*\s*(?::=|=)\s*[^{}\n;]+?)[ \t]+"
        r"(for\s+_,\s*[A-Za-z_][A-Za-z0-9_]*\s*:=\s*range\b)",
        r"\1\n\2",
        normalized,
    )
    normalized = re.sub(r"(?m)^([ \t]*return\s+[^{}\n;]+?)\s*}", r"\1\n}", normalized)

    primitive_types = (
        "complex128",
        "complex64",
        "float64",
        "float32",
        "string",
        "uint64",
        "uint32",
        "uint16",
        "uintptr",
        "uint8",
        "int64",
        "int32",
        "int16",
        "bool",
        "byte",
        "rune",
        "uint",
        "int",
    )
    type_pattern = "|".join(primitive_types)

    def repair_helper_signature(match: re.Match[str]) -> str:
        params = re.sub(
            rf"\b([A-Za-z_][A-Za-z0-9_]*?)({type_pattern})(?=\s*(?:,|$))",
            r"\1 \2",
            match.group("params"),
        )
        suffix = re.sub(
            rf"\)\s*({type_pattern})\s*\{{",
            r") \1 {",
            match.group("suffix"),
        )
        return f"{match.group('prefix')}{params}{suffix}"

    normalized = re.sub(
        rf"(?m)(?P<prefix>\bfunc\s+helper_[A-Za-z0-9_]*\s*\()"
        rf"(?P<params>[^()\n]*)"
        rf"(?P<suffix>\)\s*(?:{type_pattern})?\s*\{{)",
        repair_helper_signature,
        normalized,
    )
    normalized = _replace_unused_go_range_indices(normalized)
    return normalized


def _normalize_go_compact_signatures(text: str, prompt: str) -> str:
    normalized = text
    signature = _go_entrypoint_signature(prompt, "")
    if signature:
        compact_signature, _ = _compact_index_map(signature)
        if compact_signature:
            normalized = normalized.replace(compact_signature, signature)
    normalized = re.sub(r"(?m)^(\s*)funchelper_", r"\1func helper_", normalized)
    normalized = normalized.replace("funchelper_", "helper_")
    return normalized


def _dedupe_go_forced_entrypoint_echo(text: str, prompt: str) -> str:
    signature = _go_entrypoint_signature(prompt, "")
    if not signature:
        return text
    pattern = re.compile(rf"(?m)^\s*{re.escape(signature)}\s*\{{")
    matches = list(pattern.finditer(text))
    if len(matches) < 2:
        return text
    between = text[matches[0].end() : matches[1].start()]
    if between.strip():
        return text
    return text[matches[1].start() :].lstrip()


def _drop_go_prompt_residue_stub(text: str, prompt: str) -> str:
    signature = _go_entrypoint_signature(prompt, "")
    if not signature:
        return text
    stripped = text.strip()
    if not _starts_with_compact_prefix(stripped, signature):
        return text
    compact = "".join(stripped.split()).lower()
    residue_markers = (
        "language-specificconstraints",
        "languagespecificconstraints",
        "requiredgooutputshape",
        "startdirectwiththeexactfuncdeclaration",
        "startdirectlywiththeexactfuncdeclaration",
        "preservetheexactfunctionnameandsignature",
        "theevaluatorsuppliesthegowrapper",
        "donotemitproseormarkdown",
        "taskprompt:",
        "privatecarrierplan:",
        "basecode:",
    )
    has_real_body_after_signature = bool(
        re.search(r"\{\s*(?:\n|.)*(?:\bfor\b|\bif\b|:=|\+=|\breturn\b)", stripped)
    )
    if any(marker in compact for marker in residue_markers) and not has_real_body_after_signature:
        return ""
    signature_prefix = signature.rstrip() + "{"
    compact_stub_shapes = {
        "".join(signature_prefix.split()).lower(),
        "".join((signature_prefix + "`").split()).lower(),
        "".join((signature.rstrip() + "{}").split()).lower(),
        "".join((signature.rstrip() + "{}").split()).lower() + "`",
    }
    if compact in compact_stub_shapes:
        return ""
    return text


def _dedupe_javascript_entrypoint_echo(text: str, prompt: str) -> str:
    signature = _javascript_entrypoint_signature(prompt, "")
    if not signature:
        return text
    pattern = re.compile(rf"(?m)^\s*{re.escape(signature)}\s*\{{")
    matches = list(pattern.finditer(text))
    if len(matches) < 2:
        return text
    between = text[matches[0].end() : matches[1].start()]
    if between.strip():
        return text
    return text[matches[1].start() :].lstrip()


def _brace_single_statement_conditionals(text: str, *, language: str) -> str:
    if language.strip().lower() not in {"java", "cpp", "c++"}:
        return text
    statement_pattern = (
        r"(?:(?:final|static|const)\s+)*"
        r"(?:int|long|double|float|boolean|bool|auto|String)\s+"
        r"[A-Za-z_][A-Za-z0-9_]*\s*(?:=\s*[^{}\n;]*)?;"
    )
    pattern = re.compile(
        r"if\s*\((?P<condition>[^{}\n;]+)\)\s*"
        r"(?P<statement>(?!\{)(?:" + statement_pattern + r"))"
    )

    def repl(match: re.Match[str]) -> str:
        statement = match.group("statement").strip()
        if statement.startswith("{"):
            return match.group(0)
        return f"if ({match.group('condition').strip()}) {{ {statement} }}"

    previous = None
    normalized = text
    while previous != normalized:
        previous = normalized
        normalized = pattern.sub(repl, normalized)
    return normalized


def _normalize_javascript_compact_tokens(text: str) -> str:
    normalized = text
    normalized = re.sub(r"\bfunction(?=[A-Za-z_$])", "function ", normalized)
    normalized = re.sub(r"\b(async)function(?=[A-Za-z_$])", r"\1 function ", normalized)
    for keyword in ("const", "let", "var", "return", "class", "export"):
        normalized = re.sub(rf"\b{keyword}(?=[A-Za-z$])", f"{keyword} ", normalized)
    normalized = re.sub(r"\belseif\s*\(", "else if (", normalized)
    normalized = re.sub(r"}\s*else\b", "} else", normalized)
    normalized = re.sub(r";\s*else\b", "; else", normalized)
    normalized = re.sub(r"\belse(?=[A-Za-z_$][A-Za-z0-9_$]*\s*=)", "else ", normalized)
    normalized = re.sub(
        r"(?P<prefix>\bif\s*\([^{}\n;]+\)\s*[^{}\n;]+;)\s*else(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)(?=\s*=)",
        r"\g<prefix> else \g<name>",
        normalized,
    )
    normalized = re.sub(
        r"(?P<prefix>\belse\s+if\s*\([^{}\n;]+\)\s*[^{}\n;]+;)\s*else(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)(?=\s*=)",
        r"\g<prefix> else \g<name>",
        normalized,
    )
    normalized = re.sub(r"\bfor\(", "for (", normalized)
    normalized = re.sub(r"\bif\(", "if (", normalized)
    normalized = re.sub(r"\bwhile\(", "while (", normalized)
    normalized = re.sub(
        r"\bfor\s*\(\s*const\s+([A-Za-z_$][A-Za-z0-9_$]*?)of([A-Za-z_$][A-Za-z0-9_$]*)\s*\)",
        r"for (const \1 of \2)",
        normalized,
    )
    normalized = re.sub(
        r"\bfor\s*\(\s*let\s+([A-Za-z_$][A-Za-z0-9_$]*?)of([A-Za-z_$][A-Za-z0-9_$]*)\s*\)",
        r"for (let \1 of \2)",
        normalized,
    )
    normalized = re.sub(
        r"\bfor\s*\(\s*var\s+([A-Za-z_$][A-Za-z0-9_$]*?)of([A-Za-z_$][A-Za-z0-9_$]*)\s*\)",
        r"for (var \1 of \2)",
        normalized,
    )
    normalized = re.sub(
        r"\b(let|const|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*([^;\n{}]+)",
        lambda match: f"{match.group(1)} {match.group(2)} = {match.group(3).strip()}",
        normalized,
    )
    normalized = re.sub(r"\breturn\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*(?=\})", r"return \1;", normalized)
    normalized = re.sub(r";(?=for\s+\()", ";\n", normalized)
    normalized = re.sub(r";(?=(?:const|let|var|return)\s+)", ";\n", normalized)
    normalized = re.sub(r"\{(?=(?:const|let|var)\s+)", "{\n  ", normalized)
    normalized = re.sub(r"\}(?=(?:else|catch|finally)\b)", "} ", normalized)
    normalized = re.sub(r"\}(?=(?:const|let|var|return|for|if|while|function)\b)", "}\n", normalized)
    return normalized


def _normalize_cpp_compact_tokens(text: str) -> str:
    normalized = text
    normalized = re.sub(r"\bstatic(?=(?:int|long|double|float|bool|auto)\b)", "static ", normalized)
    for keyword in ("int", "long", "double", "float", "bool", "auto"):
        normalized = re.sub(rf"\b{keyword}([A-Za-z_][A-Za-z0-9_]*)\b", rf"{keyword} \1", normalized)
    normalized = re.sub(r"\breturn([A-Za-z_][A-Za-z0-9_]*)\b", r"return \1", normalized)
    normalized = re.sub(r"\bconststd::", "const std::", normalized)
    normalized = re.sub(r"\binclude(?=<)", "include ", normalized)
    normalized = re.sub(r"#\s*include\s*<([^>\n]+)>[ \t]*(?=(?:static\s+)?(?:int|long|double|float|bool|auto|void)\b)", r"#include <\1>\n", normalized)
    normalized = re.sub(r";(?=(?:if|for|return|(?:static\s+)?(?:int|long|double|float|bool|auto)\s+)\b)", ";\n", normalized)
    normalized = re.sub(r"\belseif\s*\(", "else if (", normalized)
    normalized = re.sub(r"}\s*else\b", "} else", normalized)
    normalized = re.sub(r";\s*else\b", "; else", normalized)
    normalized = re.sub(r"\belse(?=[A-Za-z_][A-Za-z0-9_]*\s*=)", "else ", normalized)
    normalized = re.sub(
        r"(?P<prefix>\bif\s*\([^{}\n;]+\)\s*[^{}\n;]+;)\s*else(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?=\s*=)",
        r"\g<prefix> else \g<name>",
        normalized,
    )
    normalized = re.sub(
        r"(?P<prefix>\belse\s+if\s*\([^{}\n;]+\)\s*[^{}\n;]+;)\s*else(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?=\s*=)",
        r"\g<prefix> else \g<name>",
        normalized,
    )
    normalized = re.sub(r"\bfor\s*\(([^(){};]+?)\s*:\s*([^(){};]+?)\)", r"for (\1 : \2)", normalized)
    normalized = re.sub(r"}\s*return\b", "}\n    return", normalized)
    return _brace_single_statement_conditionals(normalized, language="cpp")


def _normalize_java_compact_tokens(text: str) -> str:
    normalized = text
    normalized = re.sub(r"\bpublicclass(?=[A-Z_])", "public class ", normalized)
    normalized = re.sub(r"\bclass(?=[A-Z_][A-Za-z0-9_]*\s*\{)", "class ", normalized)
    normalized = re.sub(r"\b(public|private|protected)static", r"\1 static", normalized)
    normalized = re.sub(r"\bstaticfinal", "static final ", normalized)
    normalized = re.sub(r"\bstatic(?=(?:int|long|double|float|boolean|String|void))", "static ", normalized)
    for type_name in ("int", "long", "double", "float", "boolean", "String", "void"):
        normalized = re.sub(rf"\b({type_name})([A-Za-z_][A-Za-z0-9_]*)\b", rf"\1 \2", normalized)
        normalized = re.sub(rf"\b({type_name}\[\])([A-Za-z_][A-Za-z0-9_]*)\b", rf"\1 \2", normalized)
    normalized = re.sub(r"\bfor\(", "for (", normalized)
    normalized = re.sub(r"\bif\(", "if (", normalized)
    normalized = re.sub(r"\belseif\s*\(", "else if (", normalized)
    normalized = re.sub(r"}\s*else\b", "} else", normalized)
    normalized = re.sub(r";\s*else\b", "; else", normalized)
    normalized = re.sub(r"\belse(?=[A-Za-z_][A-Za-z0-9_]*\s*=)", "else ", normalized)
    normalized = re.sub(
        r"(?P<prefix>\bif\s*\([^{}\n;]+\)\s*[^{}\n;]+;)\s*else(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?=\s*=)",
        r"\g<prefix> else \g<name>",
        normalized,
    )
    normalized = re.sub(
        r"(?P<prefix>\belse\s+if\s*\([^{}\n;]+\)\s*[^{}\n;]+;)\s*else(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?=\s*=)",
        r"\g<prefix> else \g<name>",
        normalized,
    )
    normalized = re.sub(r"\breturn([A-Za-z_][A-Za-z0-9_]*)\b", r"return \1", normalized)
    normalized = re.sub(r";(?=(?:int|long|double|float|boolean|String|return|for|if)\s+)", ";\n", normalized)
    normalized = re.sub(r"\{(?=(?:int|long|double|float|boolean|String)\s+)", "{\n    ", normalized)
    normalized = re.sub(r"\}(?=(?:else|catch|finally)\b)", "} ", normalized)
    normalized = re.sub(r"\}(?=(?:return|for|if|public|private|protected|static)\b)", "}\n", normalized)
    normalized = re.sub(r"}(?=private\s+static\s+)", "}\n", normalized)
    normalized = _brace_single_statement_conditionals(normalized, language="java")
    return _repair_java_out_of_class_private_helpers(normalized)


def _repair_java_out_of_class_private_helpers(text: str) -> str:
    helper_pattern = re.compile(
        r"(?P<helper>private\s+static\s+(?:int|long|double|float|boolean|String|void)\s+"
        r"[A-Za-z_][A-Za-z0-9_]*\s*\([^{};]*\)\s*\{[^{}]*\})"
    )
    helpers = list(helper_pattern.finditer(text))
    if not helpers:
        return text
    class_match = re.search(r"\b(?:public\s+)?class\s+[A-Za-z_][A-Za-z0-9_]*\s*\{", text)
    if class_match is None:
        return text
    depth = 0
    class_end = -1
    for index in range(class_match.end() - 1, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                class_end = index
                break
    if class_end < 0:
        return text
    outside_helpers = [match for match in helpers if match.start() > class_end]
    if not outside_helpers:
        return text
    helper_blocks = "\n\n".join(match.group("helper").strip() for match in outside_helpers)
    stripped = text
    for match in reversed(outside_helpers):
        stripped = stripped[: match.start()] + stripped[match.end() :]
    insertion = "\n\n    " + helper_blocks.replace("\n", "\n    ") + "\n"
    return stripped[:class_end].rstrip() + insertion + stripped[class_end:].lstrip()


def _python_helper_definitions_before(text: str, stop_index: int, entrypoint_name: str) -> str:
    prefix = text[:stop_index]
    helpers: list[str] = []
    pattern = re.compile(r"(?m)^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\):")
    matches = list(pattern.finditer(prefix))
    for index, match in enumerate(matches):
        name = match.group(1)
        if name == entrypoint_name:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prefix)
        block = prefix[match.start() : end].strip()
        if block:
            helpers.append(block)
    return "\n\n".join(helpers)


def _recover_python_repeated_entrypoint(text: str, prompt: str) -> str:
    signature = _python_entrypoint_signature(prompt, "")
    if not signature:
        return text
    name_match = re.match(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", signature)
    if name_match is None:
        return text
    entrypoint_name = name_match.group(1)
    pattern = re.compile(rf"(?m)^def\s+{re.escape(entrypoint_name)}\s*\([^)]*\):")
    matches = list(pattern.finditer(text))
    if len(matches) < 2:
        return text
    for match in reversed(matches):
        helpers = _python_helper_definitions_before(text, match.start(), entrypoint_name)
        candidate = text[match.start() :].strip()
        next_same = pattern.search(candidate, 1)
        if next_same is not None:
            candidate = candidate[: next_same.start()].rstrip()
        combined = (helpers + "\n\n" + candidate).strip() if helpers else candidate
        try:
            compile(combined, "<semcodebook-python-entrypoint-recovery>", "exec")
        except SyntaxError:
            continue
        return combined
    return text


def _drop_python_forced_stub_before_later_code(text: str, prompt: str) -> str:
    signature = _python_entrypoint_signature(prompt, "")
    if not signature:
        return text
    stripped = text.lstrip()
    if not stripped.startswith(signature):
        return text
    remainder = stripped[len(signature) :]
    later_code = re.search(
        r"(?m)^(?:def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(|class\s+[A-Za-z_][A-Za-z0-9_]*\s*\(|from\s+\S+\s+import\s+|import\s+\S+)",
        remainder,
    )
    if later_code is None:
        return text
    prefix_fragment = remainder[: later_code.start()]
    compact_prefix = "".join(prefix_fragment.split()).lower()
    prose_markers = (
        "language-specificconstraints",
        "languagespecificconstraints",
        "language-independent",
        "languageindependent",
        "language-agnostic",
        "languageagnostic",
        "helper-style",
        "helperstyle",
        "taskprompt",
        "privatecarrierplan",
        "basecode",
    )
    if any(marker in compact_prefix for marker in prose_markers):
        return remainder[later_code.start() :].lstrip()
    return text


def _prepare_python_compact_fragments(content: str) -> list[str]:
    prepared = content
    prepared = re.sub(r"\s{2,}(for\s+[A-Za-z_][A-Za-z0-9_]*\s+in\b)", r"\n\1", prepared)
    prepared = re.sub(r"\s{2,}(if\s+(?:[^:\n]|:\s)*)", lambda m: "\n" + m.group(1), prepared)
    prepared = re.sub(r"\s{2,}(while\s+[A-Za-z_][A-Za-z0-9_]*.*?:)", r"\n\1", prepared)
    prepared = re.sub(r"\s{2,}(?=[A-Za-z_][A-Za-z0-9_]*\s*(?:\+=|-=|\*=|/=|//=|%=|=))", "\n", prepared)
    prepared = re.sub(r"\s{2,}return(?=(?:\s|[A-Za-z_(\[{0-9\"']))", "\nreturn ", prepared)
    prepared = re.sub(r"(?<=[0-9A-Za-z_\)\]])for(?=[A-Za-z_][A-Za-z0-9_]*\s+in\b)", "\nfor ", prepared)
    prepared = re.sub(r"(?<=continue)\s{2,}(?=if\s)", "\n", prepared)
    prepared = re.sub(r"(?<=continue)\s{2,}(?=for\s)", "\n", prepared)
    prepared = re.sub(r"(?<=continue)\s{2,}(?=return)", "\n", prepared)
    prepared = re.sub(r"(?<=continue)if(?=\s)", "\nif", prepared)
    prepared = re.sub(r"(?<=continue)if(?=[A-Za-z_(0-9])", "\nif ", prepared)
    prepared = re.sub(r"(?<=[\]\)0-9])if(?=\s)", "\nif", prepared)
    prepared = re.sub(r"(?<=[\]\)0-9])if(?=[A-Za-z_(0-9])", "\nif ", prepared)
    prepared = re.sub(r"(?<=continue)for(?=[A-Za-z_])", "\nfor ", prepared)
    prepared = re.sub(r"(?<=continue)(?=[A-Za-z_][A-Za-z0-9_]*\s*(?:\+=|-=|\*=|/=|//=|%=|=))", "\n", prepared)
    prepared = re.sub(r"(?<=continue)(?=[A-Za-z_][A-Za-z0-9_]*\s*\()", "\n", prepared)
    prepared = re.sub(
        r"(?<=break)(?=[A-Za-z_][A-Za-z0-9_]*\s*(?:\+=|-=|\*=|/=|//=|%=|=)|if\s|for[A-Za-z_]|return\s)",
        "\n",
        prepared,
    )
    prepared = re.sub(
        r"(?<=pass)(?=[A-Za-z_][A-Za-z0-9_]*\s*(?:\+=|-=|\*=|/=|//=|%=|=)|if\s|for[A-Za-z_]|return\s)",
        "\n",
        prepared,
    )
    prepared = re.sub(r"(?<=[0-9A-Za-z_\)\]])return(?=(?:\s|[A-Za-z_(\[{0-9\"']))", "\nreturn ", prepared)
    prepared = re.sub(r"(?<=\))(?=[A-Za-z_][A-Za-z0-9_]*\s*(?:\+=|-=|\*=|/=|//=|%=|=))", "\n", prepared)
    prepared = re.sub(r"\breturn\s+", "return ", prepared)
    return [fragment.strip() for fragment in prepared.splitlines() if fragment.strip()]


def _expand_python_compact_fragment(fragment: str, indent: int) -> tuple[list[str], bool]:
    match = re.match(r"^(for\s+.+?\s+in\s+.+?:|if\s+.+?:|elif\s+.+?:|else:|while\s+.+?:)(.*)$", fragment)
    if match is None:
        return [(" " * indent) + fragment], False
    header = match.group(1).rstrip()
    tail = match.group(2).strip()
    lines = [(" " * indent) + header]
    if tail:
        nested, _ = _expand_python_compact_fragment(tail, indent + 4)
        lines.extend(nested)
    return lines, header.startswith(("for ", "while "))


def _normalize_python_compact_tokens(text: str) -> str:
    normalized_lines: list[str] = []
    suspicious_markers = ("for", "if ", "continue", "return", "+=", "-=", "*=", "/=", "%=")
    for raw_line in text.splitlines():
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if indent <= 0 or not stripped or not any(marker in stripped for marker in suspicious_markers):
            normalized_lines.append(raw_line)
            continue
        fragments = _prepare_python_compact_fragments(stripped)
        if len(fragments) <= 1 and all(fragment == stripped for fragment in fragments):
            normalized_lines.append(raw_line)
            continue
        current_indent = indent
        active_loop_indents: list[int] = []
        expanded_lines: list[str] = []
        for index, fragment in enumerate(fragments):
            next_fragment = fragments[index + 1] if index + 1 < len(fragments) else ""
            if (
                active_loop_indents
                and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*.+$", fragment)
                and next_fragment.startswith("return ")
            ):
                current_indent = indent
                active_loop_indents.clear()
            if active_loop_indents and fragment.startswith("return "):
                current_indent = indent
                active_loop_indents.clear()
            lines, opens_loop = _expand_python_compact_fragment(fragment, current_indent)
            expanded_lines.extend(lines)
            if opens_loop:
                active_loop_indents.append(current_indent + 4)
                current_indent = active_loop_indents[-1]
        normalized_lines.extend(expanded_lines)
    return "\n".join(normalized_lines)


def _extract_generated_code(text: str, prompt: str, *, language: str = "") -> str:
    text = _normalize_decoded_text(text)
    prompt = _normalize_decoded_text(prompt)
    stripped = _strip_prompt_prefix(text, prompt)
    stripped = _strip_compact_prefix(stripped, prompt)
    stripped = _strip_after_compact_marker(
        stripped,
        (
            "### Output code ### Response:",
            "###Outputcode###Response:",
            "Outputcode###Response:",
            "### Output code",
            "Output code:",
            "Outputcode=",
            "Outputcode:",
            "Outputcode",
            "### Response:",
            "###Response:",
            "Response:",
        ),
    )
    stripped = re.sub(
        r"^\s*[:=]\s*(?=(?:public\s+class\s+[A-Za-z_]|class\s+[A-Za-z_]|publicclass[A-Za-z_]))",
        "",
        stripped,
    )
    stripped = _strip_markdown_fence(stripped)
    stripped = _strip_output_assignment_echo(stripped, language=language)
    lowered_language = language.strip().lower()
    if lowered_language == "go":
        stripped = _recover_go_repeated_entrypoint(stripped, prompt)
    stripped = _strip_trailing_generated_sections(stripped, language=language)
    if lowered_language == "python":
        stripped = _drop_python_forced_stub_before_later_code(stripped, prompt)
        stripped = _recover_python_repeated_entrypoint(stripped, prompt)
    stripped = _strip_generated_artifact_lines(stripped, language=language)
    if lowered_language == "go":
        stripped = _strip_go_package_prelude(stripped)
        stripped = _normalize_go_compact_signatures(stripped, prompt)
        stripped = _dedupe_go_forced_entrypoint_echo(stripped, prompt)
        stripped = _normalize_go_compact_tokens(stripped)
        stripped = _dedupe_go_forced_entrypoint_echo(stripped, prompt)
        stripped = _strip_go_generated_transcript(stripped)
        stripped = _drop_go_prompt_residue_stub(stripped, prompt)
    if lowered_language in {"javascript", "js"}:
        stripped = _dedupe_javascript_entrypoint_echo(stripped, prompt)
        stripped = _normalize_javascript_compact_tokens(stripped)
        stripped = _strip_trailing_generated_sections(stripped, language=language)
    if lowered_language == "python":
        stripped = _drop_python_forced_stub_before_later_code(stripped, prompt)
        stripped = _recover_python_repeated_entrypoint(stripped, prompt)
        stripped = _normalize_python_compact_tokens(stripped)
        stripped = _recover_python_repeated_entrypoint(stripped, prompt)
    if lowered_language == "java":
        stripped = _normalize_java_compact_tokens(stripped)
    if lowered_language in {"cpp", "c++"}:
        stripped = _normalize_cpp_compact_tokens(stripped)
    stripped = _trim_language_prelude(stripped, language=language)
    if _looks_like_prompt_echo_without_code(stripped, language=language):
        return ""
    return stripped



def _decoder_prompt_token_count(inputs: dict[str, object]) -> int:
    input_ids = inputs.get("input_ids")
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) >= 2:
        try:
            return int(shape[-1])
        except (TypeError, ValueError):
            return 0
    return 0


def _generated_token_slice(sequence, prompt_token_count: int):
    if prompt_token_count <= 0:
        return sequence
    try:
        if len(sequence) > prompt_token_count:
            return sequence[prompt_token_count:]
    except TypeError:
        return sequence
    return sequence


def _generation_length_kwargs(request: GenerationRequest) -> dict[str, int]:
    min_new_tokens = int(os.environ.get("SEMCODEBOOK_MIN_NEW_TOKENS", "0") or "0")
    if min_new_tokens <= 0:
        default_min_new_tokens = {
            "python": 64,
            "javascript": 64,
            "js": 64,
            "java": 96,
            "go": 64,
            "cpp": 96,
            "c++": 96,
        }
        min_new_tokens = default_min_new_tokens.get(request.language.lower(), 0)
    if min_new_tokens <= 0:
        return {}
    return {"min_new_tokens": min(int(request.max_new_tokens), min_new_tokens)}


def _move_model_to_inference_device(model, torch_module) -> tuple[object, str]:
    if not torch_module.cuda.is_available():
        return model, "cpu"
    moved = model.to("cuda")
    return moved if moved is not None else model, "cuda"


def _clear_trained_generator_cache(torch_module=None) -> None:
    global _TRAINED_GENERATOR_CACHE
    if not _TRAINED_GENERATOR_CACHE:
        return
    _TRAINED_GENERATOR_CACHE = {}
    gc.collect()
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def _load_tokenizer(AutoTokenizer, source: str):
    try:
        return AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=True,
            fix_mistral_regex=True,
            local_files_only=True,
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(source, trust_remote_code=True, local_files_only=True)


def _render_decoder_prompt(tokenizer, prompt: str) -> tuple[str, tuple[str, ...]]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template) and getattr(tokenizer, "chat_template", None):
        try:
            rendered = apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered, ("decoder_prompt_rendering:chat_template",)
        except Exception:
            return prompt, ("decoder_prompt_rendering:raw_prompt_after_chat_template_error",)
    return prompt, ("decoder_prompt_rendering:raw_prompt",)


def _candidate_model_cache_roots() -> tuple[Path, ...]:
    raw_candidates = [
        os.environ.get('SEMCODEBOOK_MODEL_CACHE_ROOT', ''),
        os.environ.get('HF_HOME', ''),
        os.environ.get('HF_HUB_CACHE', ''),
        os.environ.get('HUGGINGFACE_HUB_CACHE', ''),
        os.environ.get('TRANSFORMERS_CACHE', ''),
        str(Path(__file__).resolve().parents[2] / '.cache' / 'shared' / 'hf_cache'),
        str(Path(__file__).resolve().parents[2] / '.cache' / 'shared' / 'hf_cache' / 'hub'),
        str(Path.home() / '.cache' / 'huggingface' / 'hub'),
    ]
    resolved: list[Path] = []
    seen: set[str] = set()
    for item in raw_candidates:
        if not item:
            continue
        path = Path(item).resolve()
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)
    return tuple(resolved)


def _resolve_local_pretrained_source(model_name: str) -> Path | None:
    slug = f"models--{model_name.replace('/', '--')}"
    for root in _candidate_model_cache_roots():
        candidate = root / slug
        if not candidate.exists():
            continue
        refs_main = candidate / 'refs' / 'main'
        snapshots_dir = candidate / 'snapshots'
        if refs_main.exists() and snapshots_dir.exists():
            revision = refs_main.read_text(encoding='utf-8').strip()
            if revision:
                snapshot = snapshots_dir / revision
                if snapshot.exists():
                    return snapshot.resolve()
        if snapshots_dir.exists():
            snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
            if snapshots:
                return snapshots[-1].resolve()
        if (candidate / 'config.json').exists():
            return candidate.resolve()
    return None


def _ensure_torch_dtype_compat(torch_module) -> None:
    if not hasattr(torch_module, 'float8_e8m0fnu') and hasattr(torch_module, 'float8_e4m3fn'):
        torch_module.float8_e8m0fnu = torch_module.float8_e4m3fn


def _trained_generator_cache_key(
    *,
    model_root: Path | None,
    adapter_root: Path | None,
    model_name: str,
) -> str:
    model_root_label = model_root.as_posix() if model_root is not None else ""
    adapter_root_label = adapter_root.as_posix() if adapter_root is not None else ""
    return "|".join(
        (
            model_root_label,
            adapter_root_label,
            model_name,
        )
    )


def _load_trained_generator(request: GenerationRequest, *, model_root: Path | None, adapter_root: Path | None, model_name: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ensure_torch_dtype_compat(torch)

    cache_key = _trained_generator_cache_key(model_root=model_root, adapter_root=adapter_root, model_name=model_name)
    cached = _TRAINED_GENERATOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if _TRAINED_GENERATOR_CACHE:
        _clear_trained_generator_cache(torch)

    tokenizer_source = str(model_root or adapter_root or model_name)
    try:
        tokenizer = _load_tokenizer(AutoTokenizer, tokenizer_source)
    except Exception as exc:
        raise TrainedGenerationFailure(
            "load_tokenizer",
            request,
            detail=tokenizer_source,
            cause=exc,
        ) from exc
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    local_model_source = model_root or _resolve_local_pretrained_source(model_name)
    try:
        if local_model_source is not None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            model = _load_causal_lm_with_dtype_compat(
                AutoModelForCausalLM,
                str(local_model_source),
                dtype=dtype,
            )
            if model_root is None:
                model = PeftModel.from_pretrained(model, str(adapter_root), local_files_only=True)
        else:
            raise TrainedGenerationFailure(
                'resolve_base_model',
                request,
                detail=model_name,
            )
    except Exception as exc:
        raise TrainedGenerationFailure(
            "load_model",
            request,
            detail=str(local_model_source or adapter_root or model_name),
            cause=exc,
        ) from exc
    try:
        model, device_label = _move_model_to_inference_device(model, torch)
    except Exception as exc:
        raise TrainedGenerationFailure(
            "move_to_device",
            request,
            detail=str(local_model_source or adapter_root or model_name),
            cause=exc,
        ) from exc
    model.eval()
    _TRAINED_GENERATOR_CACHE[cache_key] = (tokenizer, model, device_label)
    return tokenizer, model, device_label


def _load_causal_lm_with_dtype_compat(auto_model_cls, source: str, *, dtype):
    kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "torch_dtype": dtype,
    }
    try:
        return auto_model_cls.from_pretrained(source, **kwargs)
    except TypeError as exc:
        if "torch_dtype" not in str(exc):
            raise
    kwargs.pop("torch_dtype", None)
    kwargs["dtype"] = dtype
    try:
        return auto_model_cls.from_pretrained(source, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
    kwargs.pop("dtype", None)
    return auto_model_cls.from_pretrained(source, **kwargs)



def _rule_based_generation_result(
    request: GenerationRequest,
    base_code: str,
    *,
    implementation_stage: str,
    extra_notes: tuple[str, ...] = (),
) -> GenerationResult:
    payload_nibble = request.wm_id & 0x0F
    initial_schedule = build_adaptive_carrier_schedule(base_code, request.carrier_key, request.language)
    resolved_schedule = _target_aligned_schedule(base_code, request, initial_schedule)
    carrier_bits, _ = _carrier_payload(request.wm_id, resolved_schedule)
    code = base_code
    applied_families: list[str] = []
    for entry, bit in zip(resolved_schedule, carrier_bits, strict=True):
        transformed = apply_carrier_variant(code, entry.family, bit, request.language)
        if transformed != code:
            applied_families.append(entry.family)
            code = transformed
    return GenerationResult(
        watermarked_code=code,
        language=request.language,
        wm_id=payload_nibble,
        carrier_bits=tuple(carrier_bits),
        chosen_families=tuple(entry.family for entry in resolved_schedule),
        applied_families=tuple(applied_families),
        scheduled_families=tuple(entry.family for entry in resolved_schedule),
        coverage_ratio=round(len(applied_families) / len(resolved_schedule), 4) if resolved_schedule else 0.0,
        model_name=request.model_name,
        backbone_name=request.backbone_name or request.model_name,
        backend_name=request.backend_name,
        execution_mode=request.execution_mode,
        selection_strategy=request.selection_strategy,
        checkpoint_path=request.checkpoint_path,
        adapter_path=request.adapter_path,
        embedding_table_path=request.embedding_table_path,
        variant_pool_path=request.variant_pool_path,
        dataset_manifest_path=request.dataset_manifest_path,
        contract_path=request.contract_path,
        interface_path=request.interface_path,
        base_code_source=request.base_code_source,
        candidate_budget=request.candidate_budget,
        carrier_schedule=resolved_schedule,
        implementation_stage=implementation_stage,
        notes=(
            "adaptive_keyed_carrier_schedule",
            "typed_ast_cfg_ssa_carrier_priorities",
            "soft_secded84_payload",
            "parity_anchor_bit",
            "target_aligned_pl_carrier_assignment",
            "bootstrap_rule_based_transform_encoder",
        )
        + extra_notes,
    )


def rule_based_generate(
    request: GenerationRequest,
    base_code: str,
    *,
    implementation_stage: str = "bootstrap_rule_based_generation",
    extra_notes: tuple[str, ...] = (),
) -> GenerationResult:
    return _rule_based_generation_result(
        request,
        base_code,
        implementation_stage=implementation_stage,
        extra_notes=extra_notes
        + (
            "bootstrap_rule_based_generation_interface",
            "bootstrap_training_request",
        ),
    )


def _semantic_validation_summary(code: str, request: GenerationRequest) -> dict[str, object]:
    trace = validate_semantics(
        code,
        request.language,
        tuple(request.validation_tests),
        task_id=request.task_id,
        metadata={str(key): str(value) for key, value in request.validation_metadata},
    )
    tests_required = bool(request.validation_tests)
    semantic_ok = bool(trace.compile_ok and (trace.tests_ok if tests_required else True))
    return {
        "compile_ok": bool(trace.compile_ok),
        "tests_ok": bool(trace.tests_ok),
        "semantic_ok": semantic_ok,
        "tests_required": tests_required,
        "failure_reason": trace.failure_reason,
        "executed_tests": trace.executed_tests,
        "validator": trace.validator,
    }


def _schedule_realization_summary(
    code: str,
    request: GenerationRequest,
    schedule: tuple[CarrierScheduleEntry, ...],
) -> dict[str, float | bool | int]:
    scheduled_entries = tuple(schedule)
    if not scheduled_entries:
        return {
            "scheduled_slot_count": 0,
            "scheduled_realized_count": 0,
            "scheduled_realization_ratio": 0.0,
            "data_slot_count": 0,
            "data_realized_count": 0,
            "data_realization_ratio": 0.0,
            "anchor_slot_count": 0,
            "anchor_realized": False,
            "applicable_slot_count": 0,
            "applicable_realized_count": 0,
            "applicability_realization_ratio": 0.0,
            "realized_confidence_sum": 0.0,
            "realized_confidence_mean": 0.0,
        }

    applicable_slot_count = sum(1 for entry in scheduled_entries if entry.applicable)
    data_slot_count = sum(1 for entry in scheduled_entries if entry.role == "data")
    anchor_slot_count = sum(1 for entry in scheduled_entries if entry.role == "anchor")
    scheduled_realized_count = 0
    data_realized_count = 0
    applicable_realized_count = 0
    anchor_realized = False
    realized_confidence_sum = 0.0

    for entry in scheduled_entries:
        observed_bit, confidence, applicable = _family_observed_bit(code, entry.family, request.language)
        signal_match = applicable and confidence > 0.0
        bit_match = entry.target_bit is not None and observed_bit is not None and observed_bit == entry.target_bit
        if entry.role == "data":
            slot_realized = signal_match and bit_match
            if slot_realized:
                data_realized_count += 1
        else:
            slot_realized = signal_match and (entry.target_bit is None or observed_bit is None or bit_match)
            if slot_realized:
                anchor_realized = True
        if slot_realized:
            scheduled_realized_count += 1
            realized_confidence_sum += confidence
            if entry.applicable:
                applicable_realized_count += 1

    realized_confidence_mean = realized_confidence_sum / max(1, scheduled_realized_count)
    return {
        "scheduled_slot_count": len(scheduled_entries),
        "scheduled_realized_count": scheduled_realized_count,
        "scheduled_realization_ratio": round(scheduled_realized_count / max(1, len(scheduled_entries)), 4),
        "data_slot_count": data_slot_count,
        "data_realized_count": data_realized_count,
        "data_realization_ratio": round(data_realized_count / max(1, data_slot_count), 4),
        "anchor_slot_count": anchor_slot_count,
        "anchor_realized": anchor_realized,
        "applicable_slot_count": applicable_slot_count,
        "applicable_realized_count": applicable_realized_count,
        "applicability_realization_ratio": round(applicable_realized_count / max(1, applicable_slot_count), 4),
        "realized_confidence_sum": round(realized_confidence_sum, 4),
        "realized_confidence_mean": round(realized_confidence_mean, 4),
    }


def _candidate_rank(
    *,
    semantic_ok: bool,
    compile_ok: bool,
    target_recovery_match: bool,
    target_recovery_bit_match_fraction: float,
    code_changed: bool = True,
    decision_status: str,
    positive_support_score: float,
    positive_support_family_count: int,
    positive_support_level_count: int,
    scheduled_realization_ratio: float,
    data_realization_ratio: float,
    anchor_realized: bool,
    applicability_realization_ratio: float,
    realized_confidence_mean: float,
    realized_confidence_sum: float,
) -> tuple[float, ...]:
    decision_rank = {"watermarked": 2.0, "abstain": 1.0, "reject": 0.0}.get(decision_status, 0.0)
    return (
        1.0 if semantic_ok else 0.0,
        1.0 if compile_ok else 0.0,
        1.0 if target_recovery_match else 0.0,
        target_recovery_bit_match_fraction,
        1.0 if code_changed else 0.0,
        decision_rank,
        float(positive_support_family_count),
        float(positive_support_level_count),
        positive_support_score,
        scheduled_realization_ratio,
        data_realization_ratio,
        1.0 if anchor_realized else 0.0,
        applicability_realization_ratio,
        realized_confidence_mean,
        realized_confidence_sum,
    )


def _decision_status_for_candidate(detection: object) -> str:
    explicit = str(getattr(detection, "decision_status", "") or "").strip()
    if explicit:
        return explicit
    return "watermarked" if bool(getattr(detection, "is_watermarked", False)) else "abstain"


def _positive_support_metrics(detection: object) -> tuple[float, int, int]:
    return (
        float(getattr(detection, "positive_support_score", 0.0) or 0.0),
        int(getattr(detection, "positive_support_family_count", 0) or 0),
        int(getattr(detection, "positive_support_level_count", 0) or 0),
    )


def _generation_code_changed(generated_code: str, base_code: str) -> bool:
    generated = re.sub(r"\s+", "", generated_code or "")
    base = re.sub(r"\s+", "", base_code or "")
    return bool(generated) and generated != base


def _candidate_origin_is_support_retry(item: dict[str, object]) -> bool:
    return str(item.get("origin", "")).startswith("carrier_support_retry:")


def _candidate_is_semantic_compile_ok(item: dict[str, object]) -> bool:
    validation = dict(item.get("validation", {}) or {})
    return bool(validation.get("semantic_ok")) and bool(validation.get("compile_ok"))


def _candidate_has_positive_support_signal(item: dict[str, object]) -> bool:
    score, family_count, level_count = _positive_support_metrics(item.get("detection"))
    return score > 0.0 or family_count > 0 or level_count > 0


def _candidate_has_rewrite_backed_positive_support(item: dict[str, object]) -> bool:
    score, family_count, level_count = _positive_support_metrics(item.get("detection"))
    return family_count >= 2 and level_count >= 2 and score >= 1.0


def _empty_origin_attempt_stats() -> dict[str, int]:
    return {
        "batch_sequence_count": 0,
        "empty_raw_decoder_count": 0,
        "raw_decoder_fallback_used_count": 0,
        "prompt_echo_rejected_count": 0,
        "retained_candidate_count": 0,
        "semantic_candidate_count": 0,
        "compile_candidate_count": 0,
        "semantic_compile_candidate_count": 0,
        "code_changed_candidate_count": 0,
        "target_recovery_match_count": 0,
        "positive_support_signal_candidate_count": 0,
        "rewrite_backed_support_candidate_count": 0,
    }


def _aggregate_origin_attempt_stats(
    origin_attempt_stats: dict[str, dict[str, int]],
    *,
    origin_prefix: str,
) -> dict[str, int]:
    aggregated = _empty_origin_attempt_stats()
    for origin, stats in origin_attempt_stats.items():
        if not origin.startswith(origin_prefix):
            continue
        for key in aggregated:
            aggregated[key] += int(stats.get(key, 0) or 0)
    return aggregated


def _candidate_decision_summary(item: dict[str, object]) -> tuple[str, float, int, int]:
    detection = item.get("detection")
    score, family_count, level_count = _positive_support_metrics(detection)
    return _decision_status_for_candidate(detection), score, family_count, level_count


def _candidate_diagnostic_notes(
    candidates: list[dict[str, object]],
    *,
    clean_positive_count: int,
    selection_rule: str,
    origin_attempt_stats: dict[str, dict[str, int]],
    prefilter_candidates: list[dict[str, object]] | None = None,
) -> tuple[str, ...]:
    prefilter_candidates = list(candidates if prefilter_candidates is None else prefilter_candidates)
    support_retry_candidates = [item for item in candidates if _candidate_origin_is_support_retry(item)]
    support_retry_attempt_candidates = [
        item for item in prefilter_candidates if _candidate_origin_is_support_retry(item)
    ]
    semantic_candidates = [item for item in candidates if bool(dict(item.get("validation", {}) or {}).get("semantic_ok"))]
    semantic_compile_candidates = [item for item in candidates if _candidate_is_semantic_compile_ok(item)]
    changed_candidates = [item for item in candidates if bool(item.get("code_changed", False))]
    support_retry_semantic = [
        item
        for item in support_retry_candidates
        if bool(dict(item.get("validation", {}) or {}).get("semantic_ok"))
    ]
    support_retry_changed = [item for item in support_retry_candidates if bool(item.get("code_changed", False))]
    support_retry_target = [item for item in support_retry_candidates if bool(item.get("target_recovery_match", False))]
    support_retry_stats = _aggregate_origin_attempt_stats(
        origin_attempt_stats,
        origin_prefix="carrier_support_retry:",
    )
    support_retry_attempt_failures = [
        item
        for item in support_retry_attempt_candidates
        if not bool(dict(item.get("validation", {}) or {}).get("semantic_ok"))
    ]
    support_retry_failure_reasons: dict[str, int] = {}
    for item in support_retry_attempt_failures:
        validation = dict(item.get("validation", {}) or {})
        reason = str(validation.get("failure_reason", "") or "").strip() or "semantic_validation_failed"
        normalized_reason = re.sub(r"[^A-Za-z0-9_.-]+", "_", reason)[:96].strip("_") or "semantic_validation_failed"
        support_retry_failure_reasons[normalized_reason] = support_retry_failure_reasons.get(normalized_reason, 0) + 1
    support_retry_failure_summary = ",".join(
        f"{reason}={count}"
        for reason, count in sorted(
            support_retry_failure_reasons.items(),
            key=lambda item: (-item[1], item[0]),
        )[:3]
    )
    notes = [
        f"candidate_total_count:{len(candidates)}",
        f"candidate_semantic_count:{len(semantic_candidates)}",
        f"candidate_semantic_compile_count:{len(semantic_compile_candidates)}",
        f"candidate_code_changed_count:{len(changed_candidates)}",
        f"candidate_clean_positive_count:{clean_positive_count}",
        f"support_retry_batch_sequence_count:{support_retry_stats['batch_sequence_count']}",
        f"support_retry_candidate_count:{len(support_retry_candidates)}",
        f"support_retry_attempt_candidate_count:{support_retry_stats['retained_candidate_count']}",
        f"support_retry_semantic_filter_drop_count:{max(0, len(support_retry_attempt_candidates) - len(support_retry_candidates))}",
        f"support_retry_empty_raw_decoder_count:{support_retry_stats['empty_raw_decoder_count']}",
        f"support_retry_raw_decoder_fallback_used_count:{support_retry_stats['raw_decoder_fallback_used_count']}",
        f"support_retry_prompt_echo_rejected_count:{support_retry_stats['prompt_echo_rejected_count']}",
        f"support_retry_semantic_candidate_count:{len(support_retry_semantic)}",
        f"support_retry_compile_candidate_count:{support_retry_stats['compile_candidate_count']}",
        f"support_retry_semantic_compile_candidate_count:{support_retry_stats['semantic_compile_candidate_count']}",
        f"support_retry_code_changed_candidate_count:{len(support_retry_changed)}",
        f"support_retry_target_recovery_match_count:{len(support_retry_target)}",
        f"support_retry_positive_support_signal_candidate_count:{support_retry_stats['positive_support_signal_candidate_count']}",
        f"support_retry_rewrite_backed_support_candidate_count:{support_retry_stats['rewrite_backed_support_candidate_count']}",
        f"support_retry_validation_failure_count:{len(support_retry_attempt_failures)}",
        f"selection_rule:{selection_rule}",
    ]
    if support_retry_failure_summary:
        notes.append(f"support_retry_validation_failure_reasons:{support_retry_failure_summary}")
    if support_retry_candidates:
        best_retry = max(support_retry_candidates, key=lambda item: item["rank"])
        decision_status, score, family_count, level_count = _candidate_decision_summary(best_retry)
        retry_validation = dict(best_retry.get("validation", {}) or {})
        notes.extend(
            [
                f"best_support_retry_candidate_origin:{best_retry.get('origin')}",
                f"best_support_retry_semantic_ok:{str(bool(retry_validation.get('semantic_ok'))).lower()}",
                f"best_support_retry_compile_ok:{str(bool(retry_validation.get('compile_ok'))).lower()}",
                f"best_support_retry_code_changed:{str(bool(best_retry.get('code_changed'))).lower()}",
                f"best_support_retry_target_recovery_match:{str(bool(best_retry.get('target_recovery_match'))).lower()}",
                f"best_support_retry_target_recovery_bit_match_fraction:{float(best_retry.get('target_recovery_bit_match_fraction', 0.0)):.4f}",
                f"best_support_retry_decision_status:{decision_status}",
                f"best_support_retry_positive_support_score:{score:.4f}",
                f"best_support_retry_positive_support_families:{family_count}",
                f"best_support_retry_positive_support_levels:{level_count}",
            ]
        )
    elif support_retry_attempt_candidates:
        best_retry_attempt = max(support_retry_attempt_candidates, key=lambda item: item["rank"])
        decision_status, score, family_count, level_count = _candidate_decision_summary(best_retry_attempt)
        retry_validation = dict(best_retry_attempt.get("validation", {}) or {})
        failure_reason = str(retry_validation.get("failure_reason", "") or "").strip()
        normalized_failure = re.sub(r"[^A-Za-z0-9_.-]+", "_", failure_reason)[:96].strip("_")
        notes.extend(
            [
                f"best_support_retry_attempt_origin:{best_retry_attempt.get('origin')}",
                f"best_support_retry_attempt_semantic_ok:{str(bool(retry_validation.get('semantic_ok'))).lower()}",
                f"best_support_retry_attempt_compile_ok:{str(bool(retry_validation.get('compile_ok'))).lower()}",
                f"best_support_retry_attempt_failure_reason:{normalized_failure or 'none'}",
                f"best_support_retry_attempt_code_changed:{str(bool(best_retry_attempt.get('code_changed'))).lower()}",
                f"best_support_retry_attempt_target_recovery_match:{str(bool(best_retry_attempt.get('target_recovery_match'))).lower()}",
                f"best_support_retry_attempt_target_recovery_bit_match_fraction:{float(best_retry_attempt.get('target_recovery_bit_match_fraction', 0.0)):.4f}",
                f"best_support_retry_attempt_decision_status:{decision_status}",
                f"best_support_retry_attempt_positive_support_score:{score:.4f}",
                f"best_support_retry_attempt_positive_support_families:{family_count}",
                f"best_support_retry_attempt_positive_support_levels:{level_count}",
            ]
        )
    return tuple(notes)


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _trained_generation_failure_dump_path(request: GenerationRequest, phase: str) -> Path | None:
    root = os.environ.get("SEMCODEBOOK_FAILURE_DUMP_DIR", "").strip()
    if not root:
        return None
    task_id = str(request.task_id or "unknown_task").strip() or "unknown_task"
    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
    safe_phase = re.sub(r"[^A-Za-z0-9_.-]+", "_", phase.strip() or "failure")
    return Path(root) / f"{safe_task_id}.{safe_phase}.trained_generation_candidates.json"


def _dump_trained_generation_failure_candidates(
    request: GenerationRequest,
    candidates: list[dict[str, object]],
    *,
    phase: str,
    detail: str = "",
) -> None:
    dump_path = _trained_generation_failure_dump_path(request, phase)
    if dump_path is None:
        return
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_payloads: list[dict[str, object]] = []
    for item in candidates:
        detection = item.get("detection")
        validation = dict(item.get("validation", {}) or {})
        schedule_summary = dict(item.get("schedule_summary", {}) or {})
        positive_support_score, positive_support_family_count, positive_support_level_count = _positive_support_metrics(detection)
        generated_code = str(item.get("generated_code", ""))
        decoder_diagnostics = dict(item.get("decoder_diagnostics", {}) or {})
        raw_preview = str(decoder_diagnostics.get("raw_decoded_preview", "") or "")
        candidate_payloads.append(
            {
                "origin": str(item.get("origin", "")),
                "sequence_index": item.get("sequence_index"),
                "rank": _json_safe(item.get("rank")),
                "generated_code_sha256": hashlib.sha256(generated_code.encode("utf-8")).hexdigest(),
                "raw_decoded_preview_sha256": hashlib.sha256(raw_preview.encode("utf-8")).hexdigest() if raw_preview else "",
                "target_recovery_match": bool(item.get("target_recovery_match", False)),
                "target_recovery_bit_match_fraction": item.get("target_recovery_bit_match_fraction"),
                "code_changed": bool(item.get("code_changed", False)),
                "validation": _json_safe(validation),
                "schedule_summary": _json_safe(schedule_summary),
                "decoder_diagnostics": _json_safe(decoder_diagnostics),
                "detection_summary": {
                    "decision_status": _decision_status_for_candidate(detection),
                    "support_ratio": getattr(detection, "support_ratio", None),
                    "wm_id_hat": getattr(detection, "wm_id_hat", None),
                    "decoded_wm_id_candidate": getattr(detection, "decoded_wm_id_candidate", None),
                    "positive_support_score": positive_support_score,
                    "positive_support_family_count": positive_support_family_count,
                    "positive_support_level_count": positive_support_level_count,
                },
                "generated_code": str(item.get("generated_code", "")),
            }
        )
    payload = {
        "schema_version": "semcodebook_trained_generation_failure_candidates_v1",
        "task_id": str(request.task_id or ""),
        "language": str(request.language or ""),
        "phase": phase,
        "detail": detail,
        "candidate_count": len(candidate_payloads),
        "candidates": candidate_payloads,
    }
    dump_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    jsonl_path = dump_path.with_name(f"{dump_path.stem}.jsonl")
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _target_recovery_metrics(detection: object, wm_id: int, payload_bits: int = 4) -> tuple[bool, float]:
    recovered = getattr(detection, "wm_id_hat", None)
    if recovered is None:
        recovered = getattr(detection, "decoded_wm_id_candidate", None)
    if recovered is None:
        return False, 0.0
    mask = (1 << payload_bits) - 1
    expected = int(wm_id) & mask
    observed = int(recovered) & mask
    error_count = (expected ^ observed).bit_count()
    bit_match_fraction = (payload_bits - error_count) / max(1, payload_bits)
    return observed == expected, round(bit_match_fraction, 4)


def _generate_with_trained_checkpoint(
    request: GenerationRequest,
    base_code: str,
    *,
    strict_trained_generation: bool = False,
) -> GenerationResult | None:
    if not _trained_checkpoint_available(request):
        if strict_trained_generation:
            raise TrainedGenerationFailure(
                "availability",
                request,
                detail="trained_checkpoint_or_adapter_missing",
            )
        return None
    try:
        import torch
    except Exception as exc:
        if strict_trained_generation:
            raise TrainedGenerationFailure("torch_import", request, cause=exc) from exc
        return None

    checkpoint_path = Path(request.checkpoint_path) if request.checkpoint_path else None
    adapter_path = Path(request.adapter_path) if request.adapter_path else None
    model_root = checkpoint_path if checkpoint_path and checkpoint_path.exists() else None
    adapter_root = adapter_path if adapter_path and adapter_path.exists() else None
    model_name = request.model_name or request.backbone_name
    if model_root is None and adapter_root is None:
        if strict_trained_generation:
            raise TrainedGenerationFailure(
                "availability",
                request,
                detail="resolved_model_root_and_adapter_root_missing",
            )
        return None

    try:
        tokenizer, model, device_label = _load_trained_generator(
            request,
            model_root=model_root,
            adapter_root=adapter_root,
            model_name=model_name,
        )
    except TrainedGenerationFailure:
        if strict_trained_generation:
            raise
        return None
    except Exception as exc:
        if strict_trained_generation:
            raise TrainedGenerationFailure("load_generator", request, cause=exc) from exc
        return None

    initial_schedule = build_adaptive_carrier_schedule(base_code, request.carrier_key, request.language)
    resolved_schedule = _target_aligned_schedule(base_code, request, initial_schedule)
    carrier_bits, _ = _carrier_payload(request.wm_id, resolved_schedule)
    prompt = build_structured_generation_prompt(
        prompt=request.prompt,
        language=request.language,
        wm_id=request.wm_id,
        base_code=base_code,
        schedule=resolved_schedule,
        carrier_key=request.carrier_key,
        task_id=request.task_id,
    )
    forced_output_prefix, forced_prefix_notes = _decoder_forced_output_prefix(request, base_code, resolved_schedule)
    decoder_prompt, decoder_prompt_notes = _render_decoder_prompt(tokenizer, prompt)
    if forced_output_prefix:
        decoder_prompt = decoder_prompt.rstrip() + "\n" + forced_output_prefix

    try:
        inputs = tokenizer(decoder_prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_token_count = _decoder_prompt_token_count(inputs)
    except Exception as exc:
        if strict_trained_generation:
            raise TrainedGenerationFailure("tokenize_prompt", request, cause=exc) from exc
        return None
    decoder_timeout_seconds = int(os.environ.get("SEMCODEBOOK_DECODER_TIMEOUT_SECONDS", "60"))
    decoder_attempt_notes: list[str] = [f"bounded_decoder_timeout_seconds:{decoder_timeout_seconds}"]
    generation_length_kwargs = _generation_length_kwargs(request)
    greedy = None
    try:
        with _bounded_decoder_attempt(decoder_timeout_seconds):
            with torch.inference_mode():
                greedy = model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    **generation_length_kwargs,
                )
        decoder_attempt_notes.append("trained_decoder_attempt_status:completed")
    except _DecoderAttemptTimeout:
        decoder_attempt_notes.append("trained_decoder_attempt_status:bounded_timeout")
    except Exception as exc:
        decoder_attempt_notes.append(f"trained_decoder_attempt_status:error:{exc.__class__.__name__}")
        if not strict_trained_generation:
            return None

    detector = SemCodebookDetector()
    spec = WatermarkSpec(
        wm_id=request.wm_id,
        carrier_key=request.carrier_key,
        payload_bits=4,
        carrier_schedule=resolved_schedule,
        implementation_stage="trained_checkpoint_generation",
    )
    candidates: list[dict[str, object]] = []
    origin_attempt_stats: dict[str, dict[str, int]] = {}

    def _append_candidates(
        batch,
        origin: str,
        *,
        active_decoder_prompt: str | None = None,
        active_forced_output_prefix: str | None = None,
        active_prompt_token_count: int | None = None,
    ) -> None:
        prompt_for_extract = active_decoder_prompt if active_decoder_prompt is not None else decoder_prompt
        prefix_for_merge = forced_output_prefix if active_forced_output_prefix is None else active_forced_output_prefix
        token_count = prompt_token_count if active_prompt_token_count is None else active_prompt_token_count
        sequences = batch.tolist() if hasattr(batch, "tolist") else batch
        origin_stats = origin_attempt_stats.setdefault(origin, _empty_origin_attempt_stats())
        for index, sequence in enumerate(sequences):
            origin_stats["batch_sequence_count"] += 1
            generated_slice = _generated_token_slice(sequence, token_count)
            decoded = tokenizer.decode(
                generated_slice,
                skip_special_tokens=True,
            )
            raw_decoded = _normalize_decoded_text(decoded)
            if prefix_for_merge:
                decoded = _merge_forced_output_prefix(
                    decoded,
                    prefix_for_merge,
                    language=request.language,
                    prompt=request.prompt,
                    base_code=base_code,
                )
            generated_code = _extract_generated_code(decoded, prompt_for_extract, language=request.language)
            candidate_origin = origin
            if any(marker in generated_code[:800] for marker in ("SemCodebook", "Scheduled Carriers:", "Base Code:")):
                alternate_code = _extract_generated_code(decoded, prompt, language=request.language)
                if len(alternate_code) < len(generated_code):
                    generated_code = alternate_code
            if not generated_code.strip():
                raw_decoder_text = _normalize_decoded_text(decoded).strip()
                if not raw_decoder_text:
                    origin_stats["empty_raw_decoder_count"] += 1
                    continue
                if _allow_raw_decoder_fallback(raw_decoder_text, language=request.language):
                    generated_code = raw_decoder_text
                    candidate_origin = f"{origin}:raw_decoder_after_empty_extract"
                    origin_stats["raw_decoder_fallback_used_count"] += 1
                else:
                    generated_code = ""
                    candidate_origin = f"{origin}:raw_decoder_after_empty_extract_rejected_prompt_echo"
                    origin_stats["prompt_echo_rejected_count"] += 1
            if request.language.strip().lower() in {"javascript", "js"} and generated_code.strip():
                generated_code = _normalize_javascript_compact_tokens(generated_code)
            validation = _semantic_validation_summary(generated_code, request)
            schedule_summary = _schedule_realization_summary(generated_code, request, resolved_schedule)
            detection = detector.detect(generated_code, spec, language=request.language)
            decision_status = _decision_status_for_candidate(detection)
            positive_support_score, positive_support_family_count, positive_support_level_count = _positive_support_metrics(
                detection
            )
            target_recovery_match, target_recovery_bit_match_fraction = _target_recovery_metrics(detection, request.wm_id)
            code_changed = _generation_code_changed(generated_code, base_code)
            candidate = {
                "origin": candidate_origin,
                "sequence_index": index,
                "generated_code": generated_code,
                "validation": validation,
                "detection": detection,
                "schedule_summary": schedule_summary,
                "decoder_diagnostics": {
                    "raw_decoded_length": len(raw_decoded),
                    "raw_decoded_preview": raw_decoded[:500],
                    "prompt_token_count": token_count,
                    "generated_token_count": len(generated_slice) if hasattr(generated_slice, "__len__") else None,
                    "sequence_token_count": len(sequence) if hasattr(sequence, "__len__") else None,
                    "forced_prefix_applied": bool(prefix_for_merge),
                    "extraction_prompt_variant": "forced_prefix" if prefix_for_merge else "unforced_retry",
                },
                "target_recovery_match": target_recovery_match,
                "target_recovery_bit_match_fraction": target_recovery_bit_match_fraction,
                "code_changed": code_changed,
                "rank": _candidate_rank(
                    semantic_ok=bool(validation["semantic_ok"]),
                    compile_ok=bool(validation["compile_ok"]),
                    target_recovery_match=target_recovery_match,
                    target_recovery_bit_match_fraction=target_recovery_bit_match_fraction,
                    code_changed=code_changed,
                    decision_status=decision_status,
                    positive_support_score=positive_support_score,
                    positive_support_family_count=positive_support_family_count,
                    positive_support_level_count=positive_support_level_count,
                    scheduled_realization_ratio=float(schedule_summary["scheduled_realization_ratio"]),
                    data_realization_ratio=float(schedule_summary["data_realization_ratio"]),
                    anchor_realized=bool(schedule_summary["anchor_realized"]),
                    applicability_realization_ratio=float(schedule_summary["applicability_realization_ratio"]),
                    realized_confidence_mean=float(schedule_summary["realized_confidence_mean"]),
                    realized_confidence_sum=float(schedule_summary["realized_confidence_sum"]),
                ),
            }
            candidates.append(candidate)
            origin_stats["retained_candidate_count"] += 1
            if bool(validation["semantic_ok"]):
                origin_stats["semantic_candidate_count"] += 1
            if bool(validation["compile_ok"]):
                origin_stats["compile_candidate_count"] += 1
            if bool(validation["semantic_ok"]) and bool(validation["compile_ok"]):
                origin_stats["semantic_compile_candidate_count"] += 1
            if code_changed:
                origin_stats["code_changed_candidate_count"] += 1
            if target_recovery_match:
                origin_stats["target_recovery_match_count"] += 1
            if _candidate_has_positive_support_signal(candidate):
                origin_stats["positive_support_signal_candidate_count"] += 1
            if _candidate_has_rewrite_backed_positive_support(candidate):
                origin_stats["rewrite_backed_support_candidate_count"] += 1

    def _append_support_materializer_candidates() -> None:
        if not strict_trained_generation:
            return
        lowered_language = request.language.strip().lower()
        source_candidates = [
            item
            for item in list(candidates)
            if _candidate_origin_is_support_retry(item)
            and _candidate_is_semantic_compile_ok(item)
            and (bool(item.get("code_changed", False)) or lowered_language == "go")
            and not _candidate_meets_clean_positive_contract(item)
        ]
        has_prompt_echo_rejection = any(
            int(stats.get("prompt_echo_rejected_count", 0)) > 0
            for stats in origin_attempt_stats.values()
        )
        has_support_retry_attempt = any(_candidate_origin_is_support_retry(item) for item in candidates)
        has_weak_semantic_candidate = any(
            _candidate_is_semantic_compile_ok(item)
            and (bool(item.get("code_changed", False)) or lowered_language == "go")
            and not _candidate_meets_clean_positive_contract(item)
            for item in candidates
        )
        has_clean_positive_candidate = any(_candidate_meets_clean_positive_contract(item) for item in candidates)
        has_failed_python_frontier = (
            lowered_language == "python"
            and bool(candidates)
            and not any(_candidate_is_semantic_compile_ok(item) for item in candidates)
            and any(
                not bool(dict(item.get("validation", {}) or {}).get("semantic_ok"))
                for item in candidates
            )
        )
        has_failed_javascript_frontier = (
            lowered_language in {"javascript", "js"}
            and bool(candidates)
            and not any(_candidate_is_semantic_compile_ok(item) for item in candidates)
            and any(
                not bool(dict(item.get("validation", {}) or {}).get("semantic_ok"))
                or not bool(dict(item.get("validation", {}) or {}).get("compile_ok"))
                for item in candidates
            )
        )
        has_failed_cpp_frontier = (
            lowered_language in {"cpp", "c++"}
            and bool(candidates)
            and not any(_candidate_is_semantic_compile_ok(item) for item in candidates)
            and any(
                not bool(dict(item.get("validation", {}) or {}).get("semantic_ok"))
                or not bool(dict(item.get("validation", {}) or {}).get("compile_ok"))
                for item in candidates
            )
        )
        has_failed_java_frontier = (
            lowered_language == "java"
            and bool(candidates)
            and not any(_candidate_is_semantic_compile_ok(item) for item in candidates)
            and any(
                not bool(dict(item.get("validation", {}) or {}).get("semantic_ok"))
                or not bool(dict(item.get("validation", {}) or {}).get("compile_ok"))
                for item in candidates
            )
        )
        has_failed_go_frontier = (
            lowered_language == "go"
            and bool(candidates)
            and not any(_candidate_is_semantic_compile_ok(item) for item in candidates)
            and any(
                not bool(dict(item.get("validation", {}) or {}).get("semantic_ok"))
                or not bool(dict(item.get("validation", {}) or {}).get("compile_ok"))
                for item in candidates
            )
        )
        base_materializer_frontier_allowed = has_prompt_echo_rejection or (
            lowered_language == "python"
            and (has_support_retry_attempt or has_weak_semantic_candidate or has_failed_python_frontier)
        ) or (
            lowered_language in {"javascript", "js"}
            and (has_support_retry_attempt or has_weak_semantic_candidate or has_failed_javascript_frontier)
        ) or (
            lowered_language == "java" and (has_support_retry_attempt or has_weak_semantic_candidate or has_failed_java_frontier)
        ) or (
            lowered_language in {"cpp", "c++"} and (has_support_retry_attempt or has_weak_semantic_candidate or has_failed_cpp_frontier)
        ) or (
            lowered_language == "go" and (has_support_retry_attempt or has_weak_semantic_candidate or has_failed_go_frontier)
        )
        materializer_sources: list[tuple[str, str, object | None]] = [
            (
                str(source.get("origin", "") or "unknown"),
                str(source.get("generated_code", "") or ""),
                source.get("sequence_index"),
            )
            for source in source_candidates
        ]
        if (
            not has_clean_positive_candidate
            and base_code.strip()
            and base_materializer_frontier_allowed
        ):
            materializer_sources.append(("base_code_schedule_materializer", base_code, -2000))
        if not materializer_sources:
            return
        seen_normalized = {
            hashlib.sha256(re.sub(r"\s+", "", str(item.get("generated_code", "") or "")).encode("utf-8")).hexdigest()
            for item in candidates
        }
        materialized_count = 0
        for source_index, (source_origin, source_code, source_sequence_index) in enumerate(materializer_sources):
            source_hash = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
            for variant_index, (label, variant_code) in enumerate(
                _language_support_materializer_variants(source_code, request, resolved_schedule)
            ):
                normalized_code = re.sub(r"\s+", "", variant_code)
                if not normalized_code:
                    continue
                normalized_hash = hashlib.sha256(normalized_code.encode("utf-8")).hexdigest()
                if normalized_hash in seen_normalized:
                    continue
                seen_normalized.add(normalized_hash)
                variant_hash = hashlib.sha256(variant_code.encode("utf-8")).hexdigest()
                origin = f"carrier_support_retry:{label}"
                origin_stats = origin_attempt_stats.setdefault(origin, _empty_origin_attempt_stats())
                origin_stats["batch_sequence_count"] += 1
                validation = _semantic_validation_summary(variant_code, request)
                if not (bool(validation.get("semantic_ok")) and bool(validation.get("compile_ok"))):
                    continue
                schedule_summary = _schedule_realization_summary(variant_code, request, resolved_schedule)
                detection = detector.detect(variant_code, spec, language=request.language)
                decision_status = _decision_status_for_candidate(detection)
                (
                    positive_support_score,
                    positive_support_family_count,
                    positive_support_level_count,
                ) = _positive_support_metrics(detection)
                target_recovery_match, target_recovery_bit_match_fraction = _target_recovery_metrics(
                    detection,
                    request.wm_id,
                )
                code_changed = _generation_code_changed(variant_code, base_code)
                candidate = {
                    "origin": origin,
                    "sequence_index": 1000 + source_index * 32 + variant_index,
                    "generated_code": variant_code,
                    "validation": validation,
                    "detection": detection,
                    "schedule_summary": schedule_summary,
                    "decoder_diagnostics": {
                        "materializer": "language_support_retry",
                        "materializer_version": "language_support_materializer_v2",
                        "materializer_gate": "semantic_compile_code_changed_support_retry_candidate",
                        "materializer_label": label,
                        "materializer_source_origin": source_origin,
                        "materializer_source_sha256": source_hash,
                        "materializer_variant_sha256": variant_hash,
                        "materializer_variant_normalized_sha256": normalized_hash,
                        "materializer_source_sequence_index": source_sequence_index,
                        "raw_decoded_length": 0,
                        "raw_decoded_preview": "",
                        "prompt_token_count": None,
                        "generated_token_count": None,
                        "sequence_token_count": None,
                        "forced_prefix_applied": bool(forced_output_prefix),
                        "extraction_prompt_variant": "go_support_materializer",
                    },
                    "target_recovery_match": target_recovery_match,
                    "target_recovery_bit_match_fraction": target_recovery_bit_match_fraction,
                    "code_changed": code_changed,
                    "rank": _candidate_rank(
                        semantic_ok=bool(validation["semantic_ok"]),
                        compile_ok=bool(validation["compile_ok"]),
                        target_recovery_match=target_recovery_match,
                        target_recovery_bit_match_fraction=target_recovery_bit_match_fraction,
                        code_changed=code_changed,
                        decision_status=decision_status,
                        positive_support_score=positive_support_score,
                        positive_support_family_count=positive_support_family_count,
                        positive_support_level_count=positive_support_level_count,
                        scheduled_realization_ratio=float(schedule_summary["scheduled_realization_ratio"]),
                        data_realization_ratio=float(schedule_summary["data_realization_ratio"]),
                        anchor_realized=bool(schedule_summary["anchor_realized"]),
                        applicability_realization_ratio=float(schedule_summary["applicability_realization_ratio"]),
                        realized_confidence_mean=float(schedule_summary["realized_confidence_mean"]),
                        realized_confidence_sum=float(schedule_summary["realized_confidence_sum"]),
                    ),
                }
                candidates.append(candidate)
                materialized_count += 1
                origin_stats["retained_candidate_count"] += 1
                origin_stats["semantic_candidate_count"] += 1
                origin_stats["compile_candidate_count"] += 1
                origin_stats["semantic_compile_candidate_count"] += 1
                if code_changed:
                    origin_stats["code_changed_candidate_count"] += 1
                if target_recovery_match:
                    origin_stats["target_recovery_match_count"] += 1
                if _candidate_has_positive_support_signal(candidate):
                    origin_stats["positive_support_signal_candidate_count"] += 1
                if _candidate_has_rewrite_backed_positive_support(candidate):
                    origin_stats["rewrite_backed_support_candidate_count"] += 1
        if materialized_count:
            decoder_attempt_notes.append(f"language_support_materializer_candidate_count:{materialized_count}")

    if greedy is not None:
        _append_candidates(greedy, "greedy")
    sampled_budget = max(0, min(int(request.candidate_budget) - 1, 11))
    if sampled_budget > 0:
        try:
            with torch.inference_mode():
                sampled = model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    do_sample=True,
                    temperature=max(request.temperature, 0.35),
                    top_p=0.95,
                    num_return_sequences=sampled_budget,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    **generation_length_kwargs,
                )
            _append_candidates(sampled, "sampled")
        except Exception:
            pass

    def _candidate_semantic_ok() -> bool:
        return any(bool(dict(item.get("validation", {})).get("semantic_ok")) for item in candidates)

    def _candidate_is_forced_go_stub(item: dict[str, object]) -> bool:
        code = str(item.get("generated_code", "") or "").strip()
        prefix = forced_output_prefix.strip()
        return bool(prefix and code == prefix and code.count("{") > code.count("}"))

    def _candidate_is_forced_go_prompt_echo_rejection(item: dict[str, object]) -> bool:
        if str(item.get("generated_code", "") or "").strip():
            return False
        origin = str(item.get("origin", "") or "")
        diagnostics = dict(item.get("decoder_diagnostics", {}) or {})
        return (
            "raw_decoder_after_empty_extract_rejected_prompt_echo" in origin
            and bool(diagnostics.get("forced_prefix_applied"))
        )

    if (
        strict_trained_generation
        and request.language.strip().lower() == "go"
        and forced_output_prefix
        and candidates
        and not _candidate_semantic_ok()
        and all(
            _candidate_is_forced_go_stub(item) or _candidate_is_forced_go_prompt_echo_rejection(item)
            for item in candidates
        )
    ):
        retry_decoder_prompt, _ = _render_decoder_prompt(tokenizer, prompt)
        try:
            retry_inputs = tokenizer(retry_decoder_prompt, return_tensors="pt")
            retry_inputs = {key: value.to(device) for key, value in retry_inputs.items()}
            retry_prompt_token_count = _decoder_prompt_token_count(retry_inputs)
            decoder_attempt_notes.append("go_unforced_retry_after_forced_prefix_stub")
            retry_greedy = None
            try:
                with _bounded_decoder_attempt(decoder_timeout_seconds):
                    with torch.inference_mode():
                        retry_greedy = model.generate(
                            **retry_inputs,
                            max_new_tokens=request.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=True,
                            **generation_length_kwargs,
                        )
                decoder_attempt_notes.append("go_unforced_retry_greedy_status:completed")
            except _DecoderAttemptTimeout:
                decoder_attempt_notes.append("go_unforced_retry_greedy_status:bounded_timeout")
            except Exception as exc:
                decoder_attempt_notes.append(f"go_unforced_retry_greedy_status:error:{exc.__class__.__name__}")
            if retry_greedy is not None:
                _append_candidates(
                    retry_greedy,
                    "go_unforced_retry:greedy",
                    active_decoder_prompt=retry_decoder_prompt,
                    active_forced_output_prefix="",
                    active_prompt_token_count=retry_prompt_token_count,
                )
            if sampled_budget > 0:
                try:
                    with torch.inference_mode():
                        retry_sampled = model.generate(
                            **retry_inputs,
                            max_new_tokens=request.max_new_tokens,
                            do_sample=True,
                            temperature=max(request.temperature, 0.35),
                            top_p=0.95,
                            num_return_sequences=sampled_budget,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=True,
                            **generation_length_kwargs,
                        )
                    _append_candidates(
                        retry_sampled,
                        "go_unforced_retry:sampled",
                        active_decoder_prompt=retry_decoder_prompt,
                        active_forced_output_prefix="",
                        active_prompt_token_count=retry_prompt_token_count,
                    )
                    decoder_attempt_notes.append("go_unforced_retry_sampled_status:completed")
                except Exception as exc:
                    decoder_attempt_notes.append(f"go_unforced_retry_sampled_status:error:{exc.__class__.__name__}")
        except Exception as exc:
            decoder_attempt_notes.append(f"go_unforced_retry_tokenize_status:error:{exc.__class__.__name__}")

    def _semantic_candidates() -> list[dict[str, object]]:
        return [
            item
            for item in candidates
            if bool(dict(item.get("validation", {})).get("semantic_ok"))
        ]

    def _candidate_meets_clean_positive_contract(item: dict[str, object]) -> bool:
        detection = item.get("detection")
        validation = dict(item.get("validation", {}) or {})
        decision_status = _decision_status_for_candidate(detection)
        positive_support_score, positive_support_family_count, positive_support_level_count = _positive_support_metrics(detection)
        abstain_reason = str(getattr(detection, "abstain_reason", "") or "")
        return (
            bool(validation.get("semantic_ok"))
            and bool(validation.get("compile_ok"))
            and bool(item.get("target_recovery_match", False))
            and bool(item.get("code_changed", False))
            and decision_status == "watermarked"
            and positive_support_family_count >= 2
            and positive_support_level_count >= 2
            and positive_support_score >= 1.0
            and abstain_reason != "missing_rewrite_backed_positive_support"
        )

    semantic_before_support_retry = _semantic_candidates()
    if (
        strict_trained_generation
        and semantic_before_support_retry
        and all(not _candidate_meets_clean_positive_contract(item) for item in semantic_before_support_retry)
    ):
        support_retry_prompt = _carrier_support_retry_prompt(prompt, language=request.language)
        try:
            support_retry_decoder_prompt, _ = _render_decoder_prompt(tokenizer, support_retry_prompt)
            if forced_output_prefix and request.language.strip().lower() == "go":
                support_retry_decoder_prompt = support_retry_decoder_prompt.rstrip() + "\n" + forced_output_prefix
                decoder_attempt_notes.append("carrier_support_retry_forced_prefix_in_prompt")
            support_retry_inputs = tokenizer(support_retry_decoder_prompt, return_tensors="pt")
            support_retry_inputs = {key: value.to(device) for key, value in support_retry_inputs.items()}
            support_retry_prompt_token_count = _decoder_prompt_token_count(support_retry_inputs)
            decoder_attempt_notes.append("carrier_support_retry_after_semantic_abstain")
            support_retry_greedy = None
            try:
                with _bounded_decoder_attempt(decoder_timeout_seconds):
                    with torch.inference_mode():
                        support_retry_greedy = model.generate(
                            **support_retry_inputs,
                            max_new_tokens=request.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=True,
                            **generation_length_kwargs,
                        )
                decoder_attempt_notes.append("carrier_support_retry_greedy_status:completed")
            except _DecoderAttemptTimeout:
                decoder_attempt_notes.append("carrier_support_retry_greedy_status:bounded_timeout")
            except Exception as exc:
                decoder_attempt_notes.append(f"carrier_support_retry_greedy_status:error:{exc.__class__.__name__}")
            if support_retry_greedy is not None:
                _append_candidates(
                    support_retry_greedy,
                    "carrier_support_retry:greedy",
                    active_decoder_prompt=support_retry_decoder_prompt,
                    active_forced_output_prefix=forced_output_prefix,
                    active_prompt_token_count=support_retry_prompt_token_count,
                )
            if sampled_budget > 0:
                try:
                    with torch.inference_mode():
                        support_retry_sampled = model.generate(
                            **support_retry_inputs,
                            max_new_tokens=request.max_new_tokens,
                            do_sample=True,
                            temperature=max(request.temperature, 0.35),
                            top_p=0.95,
                            num_return_sequences=sampled_budget,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=True,
                            **generation_length_kwargs,
                        )
                    _append_candidates(
                        support_retry_sampled,
                        "carrier_support_retry:sampled",
                        active_decoder_prompt=support_retry_decoder_prompt,
                        active_forced_output_prefix=forced_output_prefix,
                        active_prompt_token_count=support_retry_prompt_token_count,
                    )
                    decoder_attempt_notes.append("carrier_support_retry_sampled_status:completed")
                except Exception as exc:
                    decoder_attempt_notes.append(f"carrier_support_retry_sampled_status:error:{exc.__class__.__name__}")
        except Exception as exc:
            decoder_attempt_notes.append(f"carrier_support_retry_tokenize_status:error:{exc.__class__.__name__}")

    _append_support_materializer_candidates()

    if not strict_trained_generation:
        repair_stage = "trained_checkpoint_generation_with_validator_repair"
        repair_result = _rule_based_generation_result(
            request,
            base_code,
            implementation_stage=repair_stage,
            extra_notes=(
                "validator_backed_pl_carrier_compiler_candidate",
                "trained_checkpoint_generation_interface",
                "trained_decoder_attempted_before_compiler_repair",
                *tuple(decoder_attempt_notes),
                f"raw_trained_decoder_candidate_count:{len(candidates)}",
            ),
        )
        repair_validation = _semantic_validation_summary(repair_result.watermarked_code, request)
        repair_schedule_summary = _schedule_realization_summary(repair_result.watermarked_code, request, resolved_schedule)
        repair_detection = detector.detect(repair_result.watermarked_code, spec, language=request.language)
        repair_decision_status = _decision_status_for_candidate(repair_detection)
        (
            repair_positive_support_score,
            repair_positive_support_family_count,
            repair_positive_support_level_count,
        ) = _positive_support_metrics(repair_detection)
        repair_target_recovery_match, repair_target_recovery_bit_match_fraction = _target_recovery_metrics(
            repair_detection,
            request.wm_id,
        )
        repair_code_changed = _generation_code_changed(repair_result.watermarked_code, base_code)
        candidates.append(
            {
                "origin": "validator_repair",
                "sequence_index": -1,
                "generated_code": repair_result.watermarked_code,
                "validation": repair_validation,
                "detection": repair_detection,
                "schedule_summary": repair_schedule_summary,
                "result": repair_result,
                "target_recovery_match": repair_target_recovery_match,
                "target_recovery_bit_match_fraction": repair_target_recovery_bit_match_fraction,
                "code_changed": repair_code_changed,
                "rank": _candidate_rank(
                    semantic_ok=bool(repair_validation["semantic_ok"]),
                    compile_ok=bool(repair_validation["compile_ok"]),
                    target_recovery_match=repair_target_recovery_match,
                    target_recovery_bit_match_fraction=repair_target_recovery_bit_match_fraction,
                    code_changed=repair_code_changed,
                    decision_status=repair_decision_status,
                    positive_support_score=repair_positive_support_score,
                    positive_support_family_count=repair_positive_support_family_count,
                    positive_support_level_count=repair_positive_support_level_count,
                    scheduled_realization_ratio=float(repair_schedule_summary["scheduled_realization_ratio"]),
                    data_realization_ratio=float(repair_schedule_summary["data_realization_ratio"]),
                    anchor_realized=bool(repair_schedule_summary["anchor_realized"]),
                    applicability_realization_ratio=float(repair_schedule_summary["applicability_realization_ratio"]),
                    realized_confidence_mean=float(repair_schedule_summary["realized_confidence_mean"]),
                    realized_confidence_sum=float(repair_schedule_summary["realized_confidence_sum"]),
                ),
            }
        )

    presemantic_diagnostic_candidates = list(candidates)

    if not candidates:
        return None
    if strict_trained_generation:
        semantic_candidates = [
            item
            for item in candidates
            if bool(dict(item.get("validation", {})).get("semantic_ok"))
        ]
        if semantic_candidates:
            candidates = semantic_candidates
        else:
            first_failure = ""
            for item in candidates:
                validation = dict(item.get("validation", {}))
                failure = str(validation.get("failure_reason", "") or "").strip()
                if failure:
                    first_failure = failure[:180]
                    break
            _dump_trained_generation_failure_candidates(
                request,
                candidates,
                phase="semantic_validation",
                detail=first_failure,
            )
            raise TrainedGenerationFailure(
                "semantic_validation",
                request,
                detail=(
                    f"no_semantic_candidate candidate_count:{len(candidates)}"
                    + (f" first_failure:{first_failure}" if first_failure else "")
                ),
            )
    diagnostic_candidates = list(candidates)
    clean_positive_candidates = [item for item in diagnostic_candidates if _candidate_meets_clean_positive_contract(item)]
    selection_rule = "lexicographic_semantic_then_positive_support_then_schedule_realization"
    if strict_trained_generation and not clean_positive_candidates:
        retry_rewrite_candidates = [
            item
            for item in diagnostic_candidates
            if _candidate_origin_is_support_retry(item)
            and _candidate_is_semantic_compile_ok(item)
            and bool(item.get("code_changed", False))
        ]
        if retry_rewrite_candidates:
            candidates = retry_rewrite_candidates
            selection_rule = "support_retry_rewrite_materialization_floor_then_rank"
    candidate_diagnostic_notes = _candidate_diagnostic_notes(
        diagnostic_candidates,
        clean_positive_count=len(clean_positive_candidates),
        selection_rule=selection_rule,
        origin_attempt_stats=origin_attempt_stats,
        prefilter_candidates=presemantic_diagnostic_candidates,
    )
    support_retry_attempt_candidates = [
        item for item in presemantic_diagnostic_candidates if _candidate_origin_is_support_retry(item)
    ]
    if strict_trained_generation and support_retry_attempt_candidates and not clean_positive_candidates:
        support_retry_semantic_compile_changed = [
            item
            for item in support_retry_attempt_candidates
            if _candidate_is_semantic_compile_ok(item) and bool(item.get("code_changed", False))
        ]
        if not support_retry_semantic_compile_changed:
            _dump_trained_generation_failure_candidates(
                request,
                support_retry_attempt_candidates,
                phase="support_retry_materialization",
                detail=(
                    f"no_semantic_compile_changed_support_retry "
                    f"attempt_count:{len(support_retry_attempt_candidates)}"
                ),
            )
        else:
            target_match_count = sum(
                1 for item in support_retry_semantic_compile_changed if bool(item.get("target_recovery_match", False))
            )
            rewrite_backed_count = sum(
                1 for item in support_retry_semantic_compile_changed if _candidate_has_rewrite_backed_positive_support(item)
            )
            _dump_trained_generation_failure_candidates(
                request,
                support_retry_semantic_compile_changed,
                phase="support_retry_clean_positive_frontier",
                detail=(
                    "semantic_compile_changed_candidates_without_clean_positive "
                    f"candidate_count:{len(support_retry_semantic_compile_changed)} "
                    f"target_recovery_match_count:{target_match_count} "
                    f"rewrite_backed_support_candidate_count:{rewrite_backed_count}"
                ),
            )
    best = max(candidates, key=lambda item: item["rank"])
    if "result" in best:
        repaired = best["result"]
        selected_schedule_summary = best["schedule_summary"]
        return GenerationResult(
            watermarked_code=repaired.watermarked_code,
            language=repaired.language,
            wm_id=repaired.wm_id,
            carrier_bits=repaired.carrier_bits,
            chosen_families=repaired.chosen_families,
            applied_families=repaired.applied_families,
            scheduled_families=repaired.scheduled_families,
            coverage_ratio=repaired.coverage_ratio,
            model_name=repaired.model_name,
            backbone_name=repaired.backbone_name,
            backend_name=repaired.backend_name,
            execution_mode=repaired.execution_mode,
            selection_strategy=repaired.selection_strategy,
            checkpoint_path=repaired.checkpoint_path,
            adapter_path=repaired.adapter_path,
            embedding_table_path=repaired.embedding_table_path,
            variant_pool_path=repaired.variant_pool_path,
            dataset_manifest_path=repaired.dataset_manifest_path,
            contract_path=repaired.contract_path,
            interface_path=repaired.interface_path,
            base_code_source=repaired.base_code_source,
            candidate_budget=repaired.candidate_budget,
            carrier_schedule=repaired.carrier_schedule,
            implementation_stage=repaired.implementation_stage,
            notes=repaired.notes
            + (
                f"candidate_count:{len(candidates)}",
                *candidate_diagnostic_notes,
                f"selected_candidate_origin:{best['origin']}",
                f"selected_schedule_realization_ratio:{float(selected_schedule_summary['scheduled_realization_ratio']):.4f}",
                f"selected_data_realization_ratio:{float(selected_schedule_summary['data_realization_ratio']):.4f}",
                f"selected_anchor_realized:{str(bool(selected_schedule_summary['anchor_realized'])).lower()}",
                f"selected_target_recovery_match:{str(bool(best.get('target_recovery_match'))).lower()}",
                f"selected_target_recovery_bit_match_fraction:{float(best.get('target_recovery_bit_match_fraction', 0.0)):.4f}",
                f"selected_code_changed:{str(bool(best.get('code_changed'))).lower()}",
            ),
        )

    generated_code = str(best["generated_code"])
    selected_schedule_summary = best["schedule_summary"]
    best_detection = best["detection"]
    best_validation = best["validation"]
    best_decision_status = _decision_status_for_candidate(best_detection)
    (
        best_positive_support_score,
        best_positive_support_family_count,
        best_positive_support_level_count,
    ) = _positive_support_metrics(best_detection)
    carrier_signal_coverage = round(
        sum(1.0 for item in best_detection.carrier_evidence if item.confidence > 0.0) / len(best_detection.carrier_evidence),
        4,
    ) if best_detection.carrier_evidence else 0.0

    return GenerationResult(
        watermarked_code=generated_code,
        language=request.language,
        wm_id=request.wm_id & 0x0F,
        carrier_bits=carrier_bits,
        chosen_families=tuple(entry.family for entry in resolved_schedule),
        applied_families=tuple(entry.family for entry in resolved_schedule if entry.applicable),
        scheduled_families=tuple(entry.family for entry in resolved_schedule),
        coverage_ratio=round(
            sum(1 for entry in resolved_schedule if entry.applicable) / len(resolved_schedule), 4
        )
        if resolved_schedule
        else 0.0,
        model_name=model_name,
        backbone_name=request.backbone_name or model_name,
        backend_name=request.backend_name,
        execution_mode=request.execution_mode,
        selection_strategy=request.selection_strategy,
        checkpoint_path=request.checkpoint_path,
        adapter_path=request.adapter_path,
        embedding_table_path=request.embedding_table_path,
        variant_pool_path=request.variant_pool_path,
        dataset_manifest_path=request.dataset_manifest_path,
        contract_path=request.contract_path,
        interface_path=request.interface_path,
        base_code_source=request.base_code_source,
        candidate_budget=request.candidate_budget,
        carrier_schedule=resolved_schedule,
        implementation_stage="trained_checkpoint_generation",
        notes=(
            "trained_checkpoint_generation",
            "carrier_schedule_reused_from_detector_contract",
            "target_aligned_pl_carrier_assignment",
            "generation_path_materialized",
            *decoder_prompt_notes,
            *forced_prefix_notes,
            *tuple(decoder_attempt_notes),
            f"inference_device:{device_label}",
            f"candidate_count:{len(candidates)}",
            *candidate_diagnostic_notes,
            f"selected_candidate_origin:{best['origin']}",
            f"selected_semantic_ok:{str(bool(best_validation['semantic_ok'])).lower()}",
            f"selected_compile_ok:{str(bool(best_validation['compile_ok'])).lower()}",
            *(
                (f"selected_semantic_failure_reason:{str(best_validation.get('failure_reason'))[:160]}",)
                if not bool(best_validation["semantic_ok"])
                else ()
            ),
            f"selected_decision_status:{best_decision_status}",
            f"selected_positive_support_score:{best_positive_support_score:.4f}",
            f"selected_positive_support_families:{best_positive_support_family_count}",
            f"selected_positive_support_levels:{best_positive_support_level_count}",
            f"selected_support_ratio:{best_detection.support_ratio:.4f}",
            f"selected_signal_coverage:{carrier_signal_coverage:.4f}",
            f"selected_wm_id_hat:{getattr(best_detection, 'wm_id_hat', None)}",
            f"selected_target_recovery_match:{str(bool(best.get('target_recovery_match'))).lower()}",
            f"selected_target_recovery_bit_match_fraction:{float(best.get('target_recovery_bit_match_fraction', 0.0)):.4f}",
            f"selected_code_changed:{str(bool(best.get('code_changed'))).lower()}",
        ),
    )


def _execution_mode_requires_trained_generation(execution_mode: str) -> bool:
    normalized = execution_mode.strip().lower()
    return any(marker in normalized for marker in ("full", "final", "canonical", "trained"))


def watermark_generate(
    request: GenerationRequest,
    base_code: str,
    *,
    require_trained_generation: bool = False,
) -> GenerationResult:
    strict_trained_generation = require_trained_generation or _execution_mode_requires_trained_generation(request.execution_mode)
    trained = _generate_with_trained_checkpoint(
        request,
        base_code,
        strict_trained_generation=strict_trained_generation,
    )
    if trained is not None:
        return trained
    if strict_trained_generation:
        raise TrainedGenerationFailure(
            "fallback_blocked",
            request,
            detail="trained_generation_required_but_no_trained_candidate_was_selected",
        )
    return rule_based_generate(request, base_code)
