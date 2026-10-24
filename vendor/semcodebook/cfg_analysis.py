from __future__ import annotations

from dataclasses import dataclass

from .typed_ast import TypedAstSummary, summarize_typed_ast


@dataclass(frozen=True, slots=True)
class GuardWitness:
    condition_text: str
    has_else: bool
    returns_from_guard: bool
    line: int


@dataclass(frozen=True, slots=True)
class CFGWitness:
    language: str
    parses: bool
    guard_regions: tuple[GuardWitness, ...]
    loop_shapes: tuple[str, ...]
    helper_boundaries: tuple[str, ...]
    branch_count: int
    early_return_count: int
    nested_guard_count: int
    notes: tuple[str, ...] = ()


def summarize_cfg(code: str, language: str) -> CFGWitness:
    typed = summarize_typed_ast(code, language)
    if not typed.parses:
        return CFGWitness(
            language=language,
            parses=False,
            guard_regions=(),
            loop_shapes=(),
            helper_boundaries=(),
            branch_count=0,
            early_return_count=0,
            nested_guard_count=0,
            notes=("cfg_unavailable",),
        )

    guard_regions = tuple(
        GuardWitness(
            condition_text=item.value,
            has_else=item.kind == "guard_with_else",
            returns_from_guard=True,
            line=item.line,
        )
        for item in typed.guard_conditions
    )
    loop_shapes = tuple(item.kind for item in typed.loop_headers)
    helper_boundaries = tuple(item.value for item in typed.helper_calls if item.kind == "helper_call")
    early_return_count = sum(1 for item in typed.guard_conditions if "guard" in item.kind)
    nested_guard_count = sum(1 for item in typed.guard_conditions if item.kind == "guard_with_else")
    return CFGWitness(
        language=language,
        parses=True,
        guard_regions=guard_regions,
        loop_shapes=loop_shapes,
        helper_boundaries=helper_boundaries,
        branch_count=len(typed.guard_conditions),
        early_return_count=early_return_count,
        nested_guard_count=nested_guard_count,
        notes=("cfg_proxy_from_typed_ast",),
    )


def cfg_family_witness(summary: CFGWitness, family: str) -> tuple[str, float, tuple[str, ...]] | None:
    if not summary.parses:
        return None
    if family == "early_return_guard_style":
        if any(not item.has_else for item in summary.guard_regions):
            return "bit_0", 0.86, tuple(item.condition_text for item in summary.guard_regions[:2])
        if any(item.has_else for item in summary.guard_regions):
            return "bit_1", 0.86, tuple(item.condition_text for item in summary.guard_regions[:2])
    if family == "iteration_style":
        if any(item == "indexed_loop" for item in summary.loop_shapes):
            return "bit_1", 0.9, summary.loop_shapes[:2]
        if any(item == "direct_loop" for item in summary.loop_shapes):
            return "bit_0", 0.84, summary.loop_shapes[:2]
    if family == "helper_extraction_style":
        if summary.helper_boundaries:
            return "bit_1", 0.86, summary.helper_boundaries[:3]
        if summary.language.lower() == "go" and (summary.loop_shapes or summary.guard_regions):
            return "bit_0", 0.74, ("inline_logic_with_local_control_flow",)
        return "bit_0", 0.65, ("inline_logic",)
    if family == "comparison_idiom":
        reordered = tuple(
            item.condition_text
            for item in summary.guard_regions
            if item.condition_text.startswith("0 > ") or item.condition_text.startswith("0 < ")
        )
        if reordered:
            return "bit_1", 0.84, reordered[:2]
        if summary.guard_regions:
            return "bit_0", 0.74, tuple(item.condition_text for item in summary.guard_regions[:2])
    return None


def cfg_applicability_score(summary: CFGWitness, family: str) -> float:
    if not summary.parses:
        return 0.0
    if family == "early_return_guard_style":
        return 1.0 if summary.guard_regions else 0.0
    if family == "iteration_style":
        return 1.0 if summary.loop_shapes else 0.0
    if family == "helper_extraction_style":
        return 0.8 if (summary.helper_boundaries or summary.branch_count > 0 or summary.loop_shapes) else 0.0
    if family == "comparison_idiom":
        return 0.9 if summary.guard_regions else 0.0
    return 0.0
