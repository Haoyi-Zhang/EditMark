from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CarrierOption:
    name: str
    description: str
    bit_value: int


@dataclass(frozen=True, slots=True)
class CarrierFamilySpec:
    name: str
    description: str
    rationale: str
    options: tuple[CarrierOption, ...]


DEFAULT_CARRIER_FAMILIES: tuple[CarrierFamilySpec, ...] = (
    CarrierFamilySpec(
        name="early_return_guard_style",
        description="Choose between guard clauses and nested validation.",
        rationale="Early-exit structure is semantically meaningful but often refactor-stable.",
        options=(
            CarrierOption("guard_clause", "Return early on invalid or terminal states.", 0),
            CarrierOption("nested_guard", "Keep a nested control-flow structure.", 1),
        ),
    ),
    CarrierFamilySpec(
        name="iteration_style",
        description="Choose between direct iteration and index-driven iteration.",
        rationale="Loop style survives many formatting attacks and preserves behavior.",
        options=(
            CarrierOption("direct_iter", "Use direct iteration over elements.", 0),
            CarrierOption("indexed_iter", "Use index-based traversal.", 1),
        ),
    ),
    CarrierFamilySpec(
        name="accumulator_style",
        description="Choose between inline accumulation and explicit temporaries.",
        rationale="Accumulation strategy is visible in code shape without changing semantics.",
        options=(
            CarrierOption("inline_update", "Update aggregate directly in the loop.", 0),
            CarrierOption("temp_then_commit", "Use an explicit temporary variable before commit.", 1),
        ),
    ),
    CarrierFamilySpec(
        name="comparison_idiom",
        description="Choose between direct and reordered comparisons.",
        rationale="Equivalent comparison phrasing provides a stable control-flow carrier.",
        options=(
            CarrierOption("direct_comparison", "Use the natural operand ordering.", 0),
            CarrierOption("reordered_comparison", "Swap operands while preserving meaning.", 1),
        ),
    ),
    CarrierFamilySpec(
        name="helper_extraction_style",
        description="Choose between inlined logic and helper decomposition.",
        rationale="Helper extraction offers a higher-level carrier than token choice.",
        options=(
            CarrierOption("inline_logic", "Keep the logic in the main function body.", 0),
            CarrierOption("helper_function", "Move a stable subroutine into a helper.", 1),
        ),
    ),
    CarrierFamilySpec(
        name="temporary_binding_style",
        description="Choose between direct operand use and explicit temporary bindings.",
        rationale="Short-lived temporaries add a stable structural cue without changing semantics.",
        options=(
            CarrierOption("direct_operand", "Use the loop variable directly.", 0),
            CarrierOption("temporary_operand", "Bind the loop value to a named temporary.", 1),
        ),
    ),
    CarrierFamilySpec(
        name="initialization_idiom",
        description="Choose between bare literals and typed constructors for initialization.",
        rationale="Equivalent initializers create light but visible structural cues.",
        options=(
            CarrierOption("literal_init", "Use the literal value directly.", 0),
            CarrierOption("typed_init", "Use an explicit typed constructor.", 1),
        ),
    ),
    CarrierFamilySpec(
        name="return_expression_style",
        description="Choose between direct returns and named temporaries.",
        rationale="Named-return structure survives formatting while preserving semantics.",
        options=(
            CarrierOption("direct_return", "Return the value expression directly.", 0),
            CarrierOption("temporary_return", "Assign to a named variable before returning.", 1),
        ),
    ),
)
