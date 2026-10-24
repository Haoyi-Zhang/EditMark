from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .applicability import assess_family_applicability
from .ast_rewrites import rewrite_comparison_idiom, rewrite_initialization_idiom, rewrite_return_expression_style
from .carriers import DEFAULT_CARRIER_FAMILIES
from .cfg_rewrites import rewrite_guard_style, rewrite_helper_extraction_style, rewrite_iteration_style
from .commitments import build_schedule_commitment, stable_structural_fingerprint
from .protocol import CarrierScheduleEntry, VariantExample, VariantPool
from .pl_analysis import evidence_from_profile
from .rewrite_certificates import certificate_is_valid, certify_rewrite_candidate
from .semantic_validator import validate_semantics
from .ssa_rewrites import rewrite_accumulator_style, rewrite_temporary_binding_style


_FAMILY_METADATA: dict[str, dict[str, object]] = {
    "early_return_guard_style": {
        "structural_level": "cfg",
        "structural_signal": "guard_exit_shape",
        "schedule_bias": 1.0,
        "notes": ("cfg_backed", "branch_guard"),
    },
    "iteration_style": {
        "structural_level": "cfg",
        "structural_signal": "loop_topology",
        "schedule_bias": 0.98,
        "notes": ("cfg_backed", "loop_iteration"),
    },
    "accumulator_style": {
        "structural_level": "ssa",
        "structural_signal": "def_use_accumulator",
        "schedule_bias": 0.96,
        "notes": ("ssa_backed", "accumulator_update"),
    },
    "comparison_idiom": {
        "structural_level": "ast",
        "structural_signal": "comparison_operand_order",
        "schedule_bias": 0.88,
        "notes": ("ast_backed", "comparison"),
    },
    "helper_extraction_style": {
        "structural_level": "cfg",
        "structural_signal": "helper_call_boundary",
        "schedule_bias": 0.94,
        "notes": ("cfg_backed", "helper_boundary"),
    },
    "temporary_binding_style": {
        "structural_level": "ssa",
        "structural_signal": "temporary_binding",
        "schedule_bias": 0.92,
        "notes": ("ssa_backed", "temporary_binding"),
    },
    "initialization_idiom": {
        "structural_level": "ssa",
        "structural_signal": "typed_initializer",
        "schedule_bias": 0.84,
        "notes": ("ssa_backed", "initializer"),
    },
    "return_expression_style": {
        "structural_level": "ssa",
        "structural_signal": "return_binding",
        "schedule_bias": 0.9,
        "notes": ("ssa_backed", "return_binding"),
    },
}


def describe_carrier_family(family: str) -> dict[str, object]:
    metadata = _FAMILY_METADATA.get(family, {})
    return {
        "structural_level": str(metadata.get("structural_level", "ast")),
        "structural_signal": str(metadata.get("structural_signal", family)),
        "schedule_bias": float(metadata.get("schedule_bias", 0.75)),
        "notes": tuple(str(item) for item in metadata.get("notes", ())),
    }


def carrier_applicability_profile(code: str, family: str, language: str) -> tuple[bool, float, tuple[str, ...]]:
    applicability = assess_family_applicability(code, family, language)
    metadata = describe_carrier_family(family)
    return applicability.applicable, applicability.applicability_score, tuple(metadata["notes"]) + applicability.notes


def build_adaptive_carrier_schedule(
    code: str,
    carrier_key: str,
    language: str,
) -> tuple[CarrierScheduleEntry, ...]:
    ranked: list[tuple[float, str, CarrierScheduleEntry]] = []
    structural_fingerprint = stable_structural_fingerprint(code, language)
    lexical_fingerprint = hashlib.sha256(" ".join(code.split()).encode("utf-8")).hexdigest()[:24]
    for family in DEFAULT_CARRIER_FAMILIES:
        metadata = describe_carrier_family(family.name)
        applicable, applicability_score, notes = carrier_applicability_profile(code, family.name, language)
        digest = hmac.new(
            carrier_key.encode("utf-8"),
            f"{language}|{family.name}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        digest_priority = int(digest[:8], 16) / 0xFFFFFFFF
        schedule_priority = round(0.65 * applicability_score + 0.2 * float(metadata["schedule_bias"]) + 0.15 * digest_priority, 4)
        miss_aware_notes: tuple[str, ...] = ()
        # Current admitted-cell audit concentrates positive misses in Go and
        # helper_extract_inline. Future schedules should not use the most
        # brittle helper carrier as a primary data bit when other carriers are
        # available; it remains eligible as anchor/support evidence.
        if language.lower() == "go" and family.name == "helper_extraction_style":
            schedule_priority = round(max(0.0, schedule_priority - 0.28), 4)
            miss_aware_notes = ("miss_aware_go_helper_extract_inline_penalty_v1",)
        ranked.append(
            (
                schedule_priority,
                digest,
                CarrierScheduleEntry(
                    family=family.name,
                    slot_index=0,
                    role="data",
                    applicable=applicable,
                    applicability_score=applicability_score,
                    schedule_priority=schedule_priority,
                    structural_level=str(metadata["structural_level"]),
                    structural_signal=str(metadata["structural_signal"]),
                    notes=notes + miss_aware_notes + (f"fingerprint:{structural_fingerprint}",),
                ),
            )
        )
    # The PRF still breaks ties, but it must not outrank semantic applicability:
    # otherwise the ECC data block can contain families the compiler cannot encode.
    ranked.sort(key=lambda item: (0 if item[2].applicable else 1, -item[0], item[1], item[2].family))
    if not ranked:
        return ()

    data_ranked = ranked[:7]
    anchor_priority, anchor_digest, anchor_entry = ranked[7] if len(ranked) > 7 else ranked[-1]
    scheduled: list[CarrierScheduleEntry] = []
    for slot_index, (_, _, entry) in enumerate(data_ranked):
        scheduled.append(
            CarrierScheduleEntry(
                family=entry.family,
                slot_index=slot_index,
                role="data",
                bit_index=slot_index,
                applicable=entry.applicable,
                applicability_score=entry.applicability_score,
                schedule_priority=entry.schedule_priority,
                structural_level=entry.structural_level,
                structural_signal=entry.structural_signal,
                notes=entry.notes + ("adaptive_data_slot",),
            )
        )
    scheduled.append(
        CarrierScheduleEntry(
            family=anchor_entry.family,
            slot_index=len(scheduled),
            role="anchor",
            applicable=anchor_entry.applicable,
            applicability_score=anchor_entry.applicability_score,
            schedule_priority=anchor_entry.schedule_priority,
            structural_level=anchor_entry.structural_level,
            structural_signal=anchor_entry.structural_signal,
            notes=anchor_entry.notes + ("adaptive_anchor_slot", f"anchor_digest:{anchor_digest[:8]}", f"anchor_priority:{anchor_priority:.4f}"),
        )
    )
    schedule = tuple(scheduled)
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
                notes=entry.notes + (
                    f"slot_commitment:{slot_commitment.digest}",
                    f"schedule_root:{commitment.commitment_root}",
                    f"schedule_context:{commitment.schedule_context_hash}",
                ),
            )
        )
    return tuple(enriched)


def apply_carrier_variant(code: str, family: str, bit_value: int, language: str) -> str:
    if bit_value not in {0, 1}:
        return code
    lowered = language.lower()
    if lowered in {"javascript", "js", "java", "go", "cpp", "c++"}:
        return _apply_non_python_carrier_variant(code, family, bit_value, lowered)
    if family == "early_return_guard_style":
        return rewrite_guard_style(code, bit_value=bit_value, language=language)
    if family == "iteration_style":
        return rewrite_iteration_style(code, bit_value=bit_value, language=language)
    if family == "accumulator_style":
        return rewrite_accumulator_style(code, bit_value=bit_value, language=language)
    if family == "comparison_idiom":
        return rewrite_comparison_idiom(code, bit_value=bit_value, language=language)
    if family == "helper_extraction_style":
        return rewrite_helper_extraction_style(code, bit_value=bit_value, language=language)
    if family == "temporary_binding_style":
        return rewrite_temporary_binding_style(code, bit_value=bit_value, language=language)
    if family == "initialization_idiom":
        return rewrite_initialization_idiom(code, bit_value=bit_value, language=language)
    if family == "return_expression_style":
        return rewrite_return_expression_style(code, bit_value=bit_value, language=language)
    return code


def _rewrite_first(pattern: str, repl: str, code: str, *, flags: int = 0) -> str:
    return re.sub(pattern, repl, code, count=1, flags=flags)


def _rewrite_non_python_accumulator(code: str, language: str, bit_value: int) -> str:
    if bit_value != 1:
        return code
    if language in {"go"}:
        return _rewrite_first(
            r"\b(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*\+=\s*(?P<expr>[^\n}]+)",
            r"\g<acc> = \g<acc> + \g<expr>",
            code,
        )
    return _rewrite_first(
        r"\b(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*\+=\s*(?P<expr>[^;\n}]+);",
        r"\g<acc> = \g<acc> + \g<expr>;",
        code,
    )


def _rewrite_non_python_initialization(code: str, language: str, bit_value: int) -> str:
    if bit_value != 1:
        return code
    if language in {"javascript", "js"}:
        return _rewrite_first(r"\blet\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*0;", r"let \g<name> = Number(0);", code)
    if language == "java":
        return _rewrite_first(
            r"\bint\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*0;",
            r"int \g<name> = Integer.valueOf(0);",
            code,
        )
    if language == "go":
        return _rewrite_first(r"\b(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:=\s*0\b", r"\g<name> := int(0)", code)
    if language in {"cpp", "c++"}:
        return _rewrite_first(r"\bint\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*0;", r"int \g<name>{0};", code)
    return code


def _rewrite_non_python_return_binding(code: str, language: str, bit_value: int) -> str:
    if bit_value != 1:
        return code
    if language in {"javascript", "js"}:
        return _rewrite_first(r"(?P<indent>[ \t]*)return\s+(?P<expr>[^;\n]+);", r"\g<indent>const semcodebookReturnValue = \g<expr>;\n\g<indent>return semcodebookReturnValue;", code)
    if language == "java":
        return _rewrite_first(r"(?P<indent>[ \t]*)return\s+(?P<expr>[^;\n]+);", r"\g<indent>int semcodebookReturnValue = \g<expr>;\n\g<indent>return semcodebookReturnValue;", code)
    if language == "go":
        return _rewrite_first(r"(?P<indent>[ \t]*)return\s+(?P<expr>[^\n}]+)", r"\g<indent>semcodebookReturnValue := \g<expr>\n\g<indent>return semcodebookReturnValue", code)
    if language in {"cpp", "c++"}:
        return _rewrite_first(r"(?P<indent>[ \t]*)return\s+(?P<expr>[^;\n]+);", r"\g<indent>int semcodebook_return_value = \g<expr>;\n\g<indent>return semcodebook_return_value;", code)
    return code


def _rewrite_non_python_comparison(code: str, bit_value: int) -> str:
    if bit_value != 1:
        return code
    substitutions = (
        (r"\b(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*>=\s*(?P<num>-?\d+)\b", r"\g<num> <= \g<var>"),
        (r"\b(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*>\s*(?P<num>-?\d+)\b", r"\g<num> < \g<var>"),
        (r"\b(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*<=\s*(?P<num>-?\d+)\b", r"\g<num> >= \g<var>"),
        (r"\b(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*<\s*(?P<num>-?\d+)\b", r"\g<num> > \g<var>"),
    )
    for pattern, repl in substitutions:
        rewritten = _rewrite_first(pattern, repl, code)
        if rewritten != code:
            return rewritten
    return code


def _rewrite_non_python_temporary_binding(code: str, language: str, bit_value: int) -> str:
    if bit_value != 1:
        return code
    if language in {"javascript", "js"}:
        rewritten = _rewrite_first(
            r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*\+=\s*(?P<expr>[^;\n}]+);",
            r"\g<indent>const currentItem = \g<expr>;\n\g<indent>\g<acc> += currentItem;",
            code,
        )
        if rewritten != code:
            return rewritten
    if language == "go":
        return _rewrite_first(
            r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*\+=\s*(?P<expr>[^\n}]+)",
            r"\g<indent>currentItem := \g<expr>\n\g<indent>\g<acc> += currentItem",
            code,
        )
    if language == "java":
        return _rewrite_first(
            r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*\+=\s*(?P<expr>[^;\n}]+);",
            r"\g<indent>int currentItem = \g<expr>;\n\g<indent>\g<acc> += currentItem;",
            code,
        )
    if language in {"cpp", "c++"}:
        return _rewrite_first(
            r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*\+=\s*(?P<expr>[^;\n}]+);",
            r"\g<indent>int currentItem = \g<expr>;\n\g<indent>\g<acc> += currentItem;",
            code,
        )
    return code


def _rewrite_non_python_iteration(code: str, language: str, bit_value: int) -> str:
    if bit_value != 1:
        return code
    if language in {"javascript", "js"}:
        patterns = (
            (
                r"for\s*\(\s*const\s+(?P<value>[A-Za-z_][A-Za-z0-9_]*)\s+of\s+(?P<seq>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\{",
                r"for (let index = 0; index < \g<seq>.length; index++) {\n    const \g<value> = \g<seq>[index];",
            ),
            (
                r"for\s*\(\s*let\s+(?P<value>[A-Za-z_][A-Za-z0-9_]*)\s+of\s+(?P<seq>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\{",
                r"for (let index = 0; index < \g<seq>.length; index++) {\n    const \g<value> = \g<seq>[index];",
            ),
        )
        for pattern, repl in patterns:
            rewritten = _rewrite_first(pattern, repl, code)
            if rewritten != code:
                return rewritten
    if language == "go":
        return _rewrite_first(
            r"for\s+(?P<value>[A-Za-z_][A-Za-z0-9_]*)\s*:=\s*range\s+(?P<seq>[A-Za-z_][A-Za-z0-9_]*)\s*\{",
            r"for index := 0; index < len(\g<seq>); index++ {\n        \g<value> := \g<seq>[index]",
            code,
        )
    if language == "java":
        return _rewrite_first(
            r"for\s*\(\s*int\s+(?P<value>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<seq>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\{",
            r"for (int index = 0; index < \g<seq>.length; index++) {\n            int \g<value> = \g<seq>[index];",
            code,
        )
    if language in {"cpp", "c++"}:
        return _rewrite_first(
            r"for\s*\(\s*int\s+(?P<value>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<seq>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\{",
            r"for (int index = 0; index < static_cast<int>(\g<seq>.size()); index++) {\n        int \g<value> = \g<seq>[index];",
            code,
        )
    return code


def _helper_name(language: str) -> str:
    return "helper_transform" if language in {"javascript", "js", "go", "cpp", "c++"} else "helperTransform"


def _rewrite_non_python_helper_extraction(code: str, language: str, bit_value: int) -> str:
    if bit_value != 1:
        return code
    helper_name = _helper_name(language)
    if f"{helper_name}(" in code:
        return code
    if language in {"javascript", "js"}:
        updated = re.sub(
            r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\+=|=\s*(?P=acc)\s*\+)\s*(?P<expr>[^;\n}]+);",
            r"\g<indent>\g<acc> = \g<acc> + helper_transform(\g<expr>);",
            code,
            count=1,
        )
        if updated != code:
            return "function helper_transform(value) {\n  return value;\n}\n\n" + updated
    if language == "go":
        updated = re.sub(
            r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\+=|=\s*(?P=acc)\s*\+)\s*(?P<expr>[^\n}]+)",
            r"\g<indent>\g<acc> = \g<acc> + helper_transform(\g<expr>)",
            code,
            count=1,
        )
        if updated != code:
            return "func helper_transform(value int) int {\n    return value\n}\n\n" + updated
    if language == "java":
        updated = re.sub(
            r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\+=|=\s*(?P=acc)\s*\+)\s*(?P<expr>[^;\n}]+);",
            r"\g<indent>\g<acc> = \g<acc> + helperTransform(\g<expr>);",
            code,
            count=1,
        )
        class_close = updated.rfind("}")
        if updated != code and class_close >= 0:
            helper = "    private static int helperTransform(int value) {\n        return value;\n    }\n"
            return updated[:class_close].rstrip() + "\n\n" + helper + updated[class_close:]
    if language in {"cpp", "c++"}:
        updated = re.sub(
            r"(?P<indent>[ \t]*)(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\+=|=\s*(?P=acc)\s*\+)\s*(?P<expr>[^;\n}]+);",
            r"\g<indent>\g<acc> = \g<acc> + helper_transform(\g<expr>);",
            code,
            count=1,
        )
        if updated != code:
            return "static int helper_transform(int value) {\n    return value;\n}\n\n" + updated
    return code


def _rewrite_non_python_early_return_guard(code: str, language: str, bit_value: int) -> str:
    if bit_value != 1:
        return code
    if language in {"javascript", "js", "java", "cpp", "c++"}:
        rewritten = re.sub(
            r"if\s*\((?P<cond>[^)]*)\)\s*\{\s*return\s+(?P<expr>[^;\n}]+);\s*\}",
            r"if (!(\g<cond>)) {\n    ;\n} else {\n    return \g<expr>;\n}",
            code,
            count=1,
            flags=re.MULTILINE,
        )
        if rewritten != code:
            return rewritten
    if language == "go":
        return re.sub(
            r"if\s+(?P<cond>[^\n{]+)\s*\{\s*return\s+(?P<expr>[^\n}]+)\s*\}",
            r"if !(\g<cond>) {\n    // keep scanning\n} else {\n    return \g<expr>\n}",
            code,
            count=1,
            flags=re.MULTILINE,
        )
    return code


def _apply_non_python_carrier_variant(code: str, family: str, bit_value: int, language: str) -> str:
    if family == "early_return_guard_style":
        return _rewrite_non_python_early_return_guard(code, language, bit_value)
    if family == "iteration_style":
        return _rewrite_non_python_iteration(code, language, bit_value)
    if family == "accumulator_style":
        return _rewrite_non_python_accumulator(code, language, bit_value)
    if family == "initialization_idiom":
        return _rewrite_non_python_initialization(code, language, bit_value)
    if family == "return_expression_style":
        return _rewrite_non_python_return_binding(code, language, bit_value)
    if family == "comparison_idiom":
        return _rewrite_non_python_comparison(code, bit_value)
    if family == "helper_extraction_style":
        return _rewrite_non_python_helper_extraction(code, language, bit_value)
    if family == "temporary_binding_style":
        return _rewrite_non_python_temporary_binding(code, language, bit_value)
    return code


def _validate_variant(language: str, code: str, tests: tuple[str, ...]) -> tuple[bool, tuple[str, ...]]:
    trace = validate_semantics(code, language, tests)
    if not trace.compile_ok or not trace.tests_ok:
        notes = tuple(filter(None, (trace.validator, trace.failure_reason)))
        return False, notes
    return True, ("compile_test_validated",) + trace.executed_tests




def _reference_variant_signal(code: str, family: str, language: str) -> tuple[int, str, float] | None:
    witness = evidence_from_profile(code, family, language)
    if witness is None or not witness.option.startswith("bit_") or witness.confidence <= 0.0:
        return None
    return int(witness.option[-1]), witness.evidence_source or "profile", float(witness.confidence)


def build_variant_pool(records: list[dict[str, str]]) -> VariantPool:
    created: list[VariantExample] = []
    lang_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    attempted_count = 0
    applicable_count = 0
    emitted_keys: set[tuple[str, str, int, str]] = set()
    validation_cache: dict[tuple[str, str, tuple[str, ...]], tuple[bool, tuple[str, ...]]] = {}

    def _cached_validate(language: str, code: str, tests: tuple[str, ...]) -> tuple[bool, tuple[str, ...]]:
        key = (language, hashlib.sha256(code.encode("utf-8")).hexdigest(), tests)
        cached = validation_cache.get(key)
        if cached is not None:
            return cached
        result = _validate_variant(language, code, tests)
        validation_cache[key] = result
        return result

    def _emit_variant(
        *,
        record: dict[str, str],
        family_name: str,
        bit_value: int,
        transformed_code: str,
        applicability_score: float,
        metadata: dict[str, object],
        applicability_notes: tuple[str, ...],
        validation_notes: tuple[str, ...],
        variant_notes: tuple[str, ...],
    ) -> None:
        transformed_hash = hashlib.sha256(transformed_code.encode("utf-8")).hexdigest()[:16]
        dedupe_key = (str(record["task_id"]), family_name, int(bit_value), transformed_hash)
        if dedupe_key in emitted_keys:
            return
        emitted_keys.add(dedupe_key)
        created.append(
            VariantExample(
                task_id=record["task_id"],
                language=record["language"],
                prompt=record["prompt"],
                reference_code=record["reference_code"],
                family=family_name,
                bit_value=bit_value,
                transformed_code=transformed_code,
                applicable=True,
                validation_passed=True,
                applicability_score=applicability_score,
                schedule_priority=round(0.7 * applicability_score + 0.3 * float(metadata["schedule_bias"]), 4),
                structural_level=str(metadata["structural_level"]),
                structural_signal=str(metadata["structural_signal"]),
                validation_notes=validation_notes,
                split=str(record.get("split", "smoke")),
                validation_mode="compile_test_validated",
                training_eligible=True,
                transformed_code_hash=transformed_hash,
                notes=variant_notes + applicability_notes,
            )
        )
        lang_counter.update([record["language"]])
        family_counter.update([family_name])

    for record in records:
        for family in DEFAULT_CARRIER_FAMILIES:
            attempted_count += 1
            metadata = describe_carrier_family(family.name)
            applicable, applicability_score, applicability_notes = carrier_applicability_profile(
                record["reference_code"],
                family.name,
                record["language"],
            )
            if not applicable:
                continue
            applicable_count += 1

            reference_signal = _reference_variant_signal(record["reference_code"], family.name, record["language"])
            if reference_signal is not None:
                reference_bit, reference_source, reference_confidence = reference_signal
                validation_passed, validation_notes = _cached_validate(
                    record["language"],
                    record["reference_code"],
                    tuple(str(test) for test in record.get("tests", ())),
                )
                if validation_passed:
                    _emit_variant(
                        record=record,
                        family_name=family.name,
                        bit_value=reference_bit,
                        transformed_code=record["reference_code"],
                        applicability_score=applicability_score,
                        metadata=metadata,
                        applicability_notes=applicability_notes,
                        validation_notes=("reference_semantics_validated",) + validation_notes,
                        variant_notes=(
                            "reference_profile_variant",
                            f"reference_evidence_source:{reference_source}",
                            f"reference_evidence_confidence:{reference_confidence:.2f}",
                            "no_rewrite_required_reference_variant",
                        ),
                    )

            for option in family.options:
                transformed = apply_carrier_variant(
                    code=record["reference_code"],
                    family=family.name,
                    bit_value=option.bit_value,
                    language=record["language"],
                )
                if transformed == record["reference_code"]:
                    continue
                validation_passed, validation_notes = _cached_validate(
                    record["language"],
                    transformed,
                    tuple(str(test) for test in record.get("tests", ())),
                )
                certificate = certify_rewrite_candidate(
                    record["reference_code"],
                    transformed,
                    family=family.name,
                    bit_value=option.bit_value,
                    language=record["language"],
                    tests=tuple(str(test) for test in record.get("tests", ())),
                )
                if not validation_passed or not certificate_is_valid(certificate):
                    continue
                _emit_variant(
                    record=record,
                    family_name=family.name,
                    bit_value=option.bit_value,
                    transformed_code=transformed,
                    applicability_score=applicability_score,
                    metadata=metadata,
                    applicability_notes=applicability_notes,
                    validation_notes=validation_notes + certificate.notes,
                    variant_notes=(
                        option.name,
                        f"original_digest:{certificate.original_digest}",
                        f"transformed_digest:{certificate.transformed_digest}",
                        f"original_fingerprint:{certificate.original_fingerprint}",
                        f"transformed_fingerprint:{certificate.transformed_fingerprint}",
                        "rewrite_certificate_valid",
                    ),
                )

    return VariantPool(
        records=tuple(created),
        created_count=len(created),
        language_distribution=dict(lang_counter),
        family_distribution=dict(family_counter),
        attempted_count=attempted_count,
        applicable_count=applicable_count,
        validated_record_count=len(created),
        training_eligible_count=len(created),
        split_distribution=dict(Counter(item.split or "smoke" for item in created)),
        validation_distribution={"compile_test_validated": len(created)},
        notes=(
            "carrier_rewrite_pool",
            "reference_profile_variants_included",
            "two_sided_binary_channel_targeted",
            "rewrite_certificate_guarded",
        ),
    )
