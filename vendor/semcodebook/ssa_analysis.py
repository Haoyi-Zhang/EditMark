from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from .typed_ast import parse_typed_tree, summarize_typed_ast


_INITIALIZATION_CARRIER_KINDS = {"typed_int_zero", "typed_helper_zero", "typed_dict"}


@dataclass(frozen=True, slots=True)
class DefUseWitness:
    name: str
    value_text: str
    kind: str
    line: int


@dataclass(frozen=True, slots=True)
class SSAWitness:
    language: str
    parses: bool
    assignments: tuple[DefUseWitness, ...]
    accumulator_targets: tuple[str, ...]
    temporary_targets: tuple[str, ...]
    typed_initializers: tuple[str, ...]
    return_bindings: tuple[str, ...]
    notes: tuple[str, ...] = ()



def _line_number(code: str, offset: int) -> int:
    return code.count("\n", 0, offset) + 1


def _summarize_non_python_ssa(code: str, language: str, typed: TypedAstSummary) -> SSAWitness:
    assignments: list[DefUseWitness] = []
    seen: set[tuple[str, int]] = set()
    for match in re.finditer(r"\b(?:let|const|var|int|long|double|float|bool|auto)?\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::=|=(?!=))\s*([^;\n]+)", code):
        name = match.group(1)
        if name in {"if", "for", "return"}:
            continue
        line = _line_number(code, match.start())
        key = (name, line)
        if key in seen:
            continue
        seen.add(key)
        assignments.append(DefUseWitness(name=name, value_text=match.group(2).strip(), kind="assign", line=line))
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\1\s*\+\s*([^;\n}]+)", code):
        name = match.group(1)
        line = _line_number(code, match.start())
        key = (f"{name}:accumulator_rewrite", line)
        if key in seen:
            continue
        seen.add(key)
        assignments.append(
            DefUseWitness(
                name=name,
                value_text=f"{name} + {match.group(2).strip()}",
                kind="assign",
                line=line,
            )
        )
    paired_accumulator_pattern = re.compile(
        r"\b(?:let|const|var|int|long|double|float|bool|auto)?\s*"
        r"(?P<temp>[A-Za-z_][A-Za-z0-9_]*)\s*(?::=|=(?!=))\s*"
        r"(?P<acc>[A-Za-z_][A-Za-z0-9_]*)\s*\+\s*(?P<expr>[^;\n}]+)\s*;?"
        r"\s*(?P=acc)\s*=\s*(?P=temp)\b",
        re.MULTILINE,
    )
    for match in paired_accumulator_pattern.finditer(code):
        acc = match.group("acc")
        expr = match.group("expr").strip()
        line = _line_number(code, match.start())
        key = (f"{acc}:paired_accumulator_rewrite", line)
        if key in seen:
            continue
        seen.add(key)
        assignments.append(
            DefUseWitness(
                name=acc,
                value_text=f"{acc} + {expr}",
                kind="assign",
                line=line,
            )
        )
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\+=\s*([^;\n]+)", code):
        name = match.group(1)
        line = _line_number(code, match.start())
        key = (name, line)
        if key in seen:
            continue
        seen.add(key)
        assignments.append(DefUseWitness(name=name, value_text=match.group(2).strip(), kind="augassign", line=line))
    accumulator_targets = tuple(sorted(set(match.group(1) for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\+=", code))))
    temporary_targets = tuple(sorted(set(item.value for item in typed.temporary_bindings if item.value)))
    return_bindings = tuple(sorted(set(item.value for item in typed.return_forms if item.kind == "named_return" and item.value)))
    return SSAWitness(
        language=language,
        parses=True,
        assignments=tuple(assignments),
        accumulator_targets=accumulator_targets,
        temporary_targets=temporary_targets,
        typed_initializers=tuple(item.kind for item in typed.typed_initializers),
        return_bindings=return_bindings,
        notes=("ssa_regex_proxy",),
    )


def summarize_ssa(code: str, language: str) -> SSAWitness:
    typed = summarize_typed_ast(code, language)
    tree = parse_typed_tree(code, language)
    if not typed.parses:
        return SSAWitness(
            language=language,
            parses=False,
            assignments=(),
            accumulator_targets=(),
            temporary_targets=(),
            typed_initializers=(),
            return_bindings=(),
            notes=("ssa_unavailable",),
        )
    if tree is None:
        return _summarize_non_python_ssa(code, language, typed)

    assignments: list[DefUseWitness] = []
    accumulator_targets: list[str] = []
    temporary_targets: list[str] = []
    return_bindings: list[str] = []

    def scan_statements(body: list[ast.stmt]) -> None:
        for index, statement in enumerate(body):
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                target_name = statement.targets[0].id
                value_text = ast.unparse(statement.value) if hasattr(ast, "unparse") else statement.value.__class__.__name__
                assignments.append(
                    DefUseWitness(
                        name=target_name,
                        value_text=value_text,
                        kind="assign",
                        line=getattr(statement, "lineno", 0),
                    )
                )
                next_statement = body[index + 1] if index + 1 < len(body) else None
                if (
                    isinstance(next_statement, ast.Assign)
                    and len(next_statement.targets) == 1
                    and isinstance(next_statement.targets[0], ast.Name)
                    and isinstance(next_statement.value, ast.Name)
                    and next_statement.value.id == target_name
                ):
                    accumulator_targets.append(next_statement.targets[0].id)
                elif (
                    isinstance(next_statement, ast.AugAssign)
                    and isinstance(next_statement.value, ast.Name)
                    and next_statement.value.id == target_name
                ):
                    temporary_targets.append(target_name)
                elif (
                    isinstance(next_statement, ast.Return)
                    and isinstance(next_statement.value, ast.Name)
                    and next_statement.value.id == target_name
                ):
                    return_bindings.append(target_name)
            elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
                accumulator_targets.append(statement.target.id)
                assignments.append(
                    DefUseWitness(
                        name=statement.target.id,
                        value_text=ast.unparse(statement.value) if hasattr(ast, "unparse") else statement.value.__class__.__name__,
                        kind="augassign",
                        line=getattr(statement, "lineno", 0),
                    )
                )
            if isinstance(statement, (ast.For, ast.While, ast.If)):
                scan_statements(statement.body)
                scan_statements(statement.orelse)

    if tree is not None:
        for function in [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]:
            scan_statements(function.body)

    for item in typed.typed_initializers:
        assignments.append(DefUseWitness(name=item.kind, value_text=item.value, kind="initializer", line=item.line))
    for item in typed.return_forms:
        if item.kind == "named_return" and item.value:
            return_bindings.append(item.value)

    return SSAWitness(
        language=language,
        parses=True,
        assignments=tuple(assignments),
        accumulator_targets=tuple(sorted(set(accumulator_targets))),
        temporary_targets=tuple(sorted(set(temporary_targets))),
        typed_initializers=tuple(item.kind for item in typed.typed_initializers),
        return_bindings=tuple(sorted(set(return_bindings))),
        notes=("ssa_proxy_from_typed_ast",),
    )


def ssa_family_witness(summary: SSAWitness, family: str) -> tuple[str, float, tuple[str, ...]] | None:
    if not summary.parses:
        return None
    if family == "accumulator_style":
        rewrite_like = tuple(
            item.name
            for item in summary.assignments
            if item.kind == "assign" and re.search(rf"\b{re.escape(item.name)}\b\s*\+", item.value_text)
        )
        if rewrite_like:
            return "bit_1", 0.88, rewrite_like[:2]
        if summary.accumulator_targets:
            return "bit_0", 0.66, summary.accumulator_targets[:2]
    if family == "temporary_binding_style":
        if summary.temporary_targets:
            return "bit_1", 0.9, summary.temporary_targets[:2]
        if summary.accumulator_targets:
            return "bit_0", 0.58, summary.accumulator_targets[:2]
    if family == "initialization_idiom":
        typed_carriers = tuple(item for item in summary.typed_initializers if item in _INITIALIZATION_CARRIER_KINDS)
        if typed_carriers:
            return "bit_1", 0.88, typed_carriers[:2]
        return "bit_0", 0.72, ("literal_initializer",)
    if family == "return_expression_style":
        if summary.return_bindings:
            return "bit_1", 0.86, summary.return_bindings[:2]
        if summary.assignments:
            return "bit_0", 0.68, tuple(item.name for item in summary.assignments[:2])
    return None


def ssa_applicability_score(summary: SSAWitness, family: str) -> float:
    if not summary.parses:
        return 0.0
    if family == "accumulator_style":
        return 0.9 if summary.accumulator_targets or summary.temporary_targets else 0.0
    if family == "temporary_binding_style":
        return 0.95 if summary.temporary_targets or summary.accumulator_targets else 0.0
    if family == "initialization_idiom":
        return 0.8
    if family == "return_expression_style":
        return 0.85 if summary.return_bindings or summary.assignments else 0.0
    return 0.0
