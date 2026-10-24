from __future__ import annotations

from dataclasses import dataclass

from .cfg_analysis import CFGWitness, cfg_applicability_score, cfg_family_witness, summarize_cfg
from .ssa_analysis import SSAWitness, ssa_applicability_score, ssa_family_witness, summarize_ssa
from .typed_ast import TypedAstSummary, summarize_typed_ast


@dataclass(frozen=True, slots=True)
class StructuralContext:
    typed_ast: TypedAstSummary
    cfg: CFGWitness
    ssa: SSAWitness


@dataclass(frozen=True, slots=True)
class FamilyApplicability:
    family: str
    applicable: bool
    applicability_score: float
    structural_level: str
    notes: tuple[str, ...]


def build_structural_context(code: str, language: str) -> StructuralContext:
    typed_ast = summarize_typed_ast(code, language)
    return StructuralContext(
        typed_ast=typed_ast,
        cfg=summarize_cfg(code, language),
        ssa=summarize_ssa(code, language),
    )


def assess_family_applicability(code: str, family: str, language: str) -> FamilyApplicability:
    context = build_structural_context(code, language)
    if family in {"early_return_guard_style", "iteration_style", "helper_extraction_style", "comparison_idiom"}:
        score = cfg_applicability_score(context.cfg, family)
        notes = ("cfg_backed",) + context.cfg.notes
        structural_level = "cfg"
    elif family in {"accumulator_style", "temporary_binding_style", "initialization_idiom", "return_expression_style"}:
        score = ssa_applicability_score(context.ssa, family)
        notes = ("ssa_backed",) + context.ssa.notes
        structural_level = "ssa"
    else:
        score = 0.0
        notes = ("unknown_family",)
        structural_level = "ast"
    return FamilyApplicability(
        family=family,
        applicable=score >= 0.5,
        applicability_score=round(score, 4),
        structural_level=structural_level,
        notes=notes + (("applicable",) if score >= 0.5 else ("unsupported_shape",)),
    )


def structural_family_witness(code: str, family: str, language: str) -> tuple[str, float, tuple[str, ...], str] | None:
    context = build_structural_context(code, language)
    cfg_witness = cfg_family_witness(context.cfg, family)
    if cfg_witness is not None:
        option, confidence, matches = cfg_witness
        return option, confidence, matches, "cfg"
    ssa_witness = ssa_family_witness(context.ssa, family)
    if ssa_witness is not None:
        option, confidence, matches = ssa_witness
        return option, confidence, matches, "ssa"
    return None
