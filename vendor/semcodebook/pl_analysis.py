from __future__ import annotations

from dataclasses import dataclass

from .applicability import assess_family_applicability, build_structural_context, structural_family_witness
from .protocol import CarrierEvidence
from .typed_ast import parse_typed_tree


_INITIALIZATION_CARRIER_KINDS = {"typed_int_zero", "typed_helper_zero", "typed_dict"}


@dataclass(frozen=True, slots=True)
class StructuralProfile:
    language: str
    parses: bool
    function_count: int
    loop_count: int
    if_count: int
    helper_count: int
    has_augassign: bool
    has_index_loop: bool
    has_direct_loop: bool
    has_temporary_binding: bool
    has_named_return: bool
    has_typed_initialization: bool


def parse_python_tree(code: str):
    return parse_typed_tree(code, "python") if code else None


def structural_profile(code: str, language: str) -> StructuralProfile:
    context = build_structural_context(code, language)
    typed = context.typed_ast
    cfg = context.cfg
    ssa = context.ssa
    return StructuralProfile(
        language=language,
        parses=typed.parses,
        function_count=len(typed.function_names),
        loop_count=len(cfg.loop_shapes),
        if_count=len(cfg.guard_regions),
        helper_count=len(typed.helper_names) + len(cfg.helper_boundaries),
        has_augassign=bool(ssa.accumulator_targets),
        has_index_loop=any(item == "indexed_loop" for item in cfg.loop_shapes),
        has_direct_loop=any(item == "direct_loop" for item in cfg.loop_shapes),
        has_temporary_binding=bool(ssa.temporary_targets),
        has_named_return=bool(ssa.return_bindings),
        has_typed_initialization=any(item in _INITIALIZATION_CARRIER_KINDS for item in ssa.typed_initializers),
    )


def carrier_is_applicable(code: str, family: str, language: str) -> bool:
    return assess_family_applicability(code, family, language).applicable


def evidence_from_profile(code: str, family: str, language: str) -> CarrierEvidence | None:
    witness = structural_family_witness(code, family, language)
    if witness is None:
        return None
    option, confidence, matches, level = witness
    applicability = assess_family_applicability(code, family, language)
    return CarrierEvidence(
        family=family,
        option=option,
        confidence=confidence,
        evidence_source=f"{level}_witness",
        structural_level=applicability.structural_level,
        structural_signal=level,
        applicable=applicability.applicable,
        applicability_score=applicability.applicability_score,
        matches=matches,
    )
