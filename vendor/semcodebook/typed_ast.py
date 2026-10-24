from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TypedNodeSignature:
    kind: str
    value: str
    line: int


@dataclass(frozen=True, slots=True)
class TypedAstSummary:
    language: str
    parses: bool
    function_names: tuple[str, ...]
    function_arity: tuple[int, ...]
    helper_names: tuple[str, ...]
    guard_conditions: tuple[TypedNodeSignature, ...]
    loop_headers: tuple[TypedNodeSignature, ...]
    helper_calls: tuple[TypedNodeSignature, ...]
    typed_initializers: tuple[TypedNodeSignature, ...]
    return_forms: tuple[TypedNodeSignature, ...]
    comparison_forms: tuple[TypedNodeSignature, ...]
    temporary_bindings: tuple[TypedNodeSignature, ...]
    notes: tuple[str, ...] = ()


def parse_typed_tree(code: str, language: str) -> ast.AST | None:
    if language.lower() != "python":
        return None
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _expr_text(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if hasattr(ast, "unparse"):
        try:
            return ast.unparse(node)
        except Exception:
            return node.__class__.__name__
    return node.__class__.__name__


def _typed_initializer_kind(node: ast.Assign) -> str:
    value_text = _expr_text(node.value)
    if value_text == "int(0)":
        return "typed_int_zero"
    if (
        isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "helper_transform"
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == 0
    ):
        return "typed_helper_zero"
    if value_text == "0":
        return "literal_zero"
    if value_text == "dict()":
        return "typed_dict"
    if value_text == "{}":
        return "literal_dict"
    return "other_initializer"


def _loop_kind(node: ast.For) -> str:
    iterator = _expr_text(node.iter)
    if "range(" in iterator:
        return "indexed_loop"
    return "direct_loop"


def _comparison_kind(node: ast.Compare) -> str:
    left = _expr_text(node.left)
    comparator = _expr_text(node.comparators[0]) if node.comparators else ""
    op = node.ops[0].__class__.__name__ if node.ops else "Unknown"
    if left == "0" and comparator:
        return f"reordered:{op}"
    return f"direct:{op}"


_FUNCTION_PATTERNS: dict[str, str] = {
    "javascript": r"function\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<params>[^)]*)\)",
    "java": r"(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:int|long|double|boolean|String|void)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<params>[^)]*)\)",
    "go": r"func\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<params>[^)]*)\)",
    "cpp": r"(?:int|long|double|bool|void|auto)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<params>[^)]*)\)\s*\{",
}


def _line_number(code: str, offset: int) -> int:
    return code.count("\n", 0, offset) + 1


def _parameter_count(raw: str) -> int:
    return len([item for item in raw.split(",") if item.strip()])


def _non_python_function_signatures(code: str, language: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    pattern = _FUNCTION_PATTERNS.get(language)
    if not pattern:
        return (), ()
    names: list[str] = []
    arities: list[int] = []
    for match in re.finditer(pattern, code, re.MULTILINE):
        names.append(match.group("name"))
        arities.append(_parameter_count(match.group("params")))
    return tuple(names), tuple(arities)


def _non_python_guard_conditions(code: str, language: str) -> tuple[TypedNodeSignature, ...]:
    pattern = r"if\s*\((?P<cond>[^)]*)\)" if language in {"javascript", "java", "cpp"} else r"if\s+(?P<cond>[^\n{]+)\s*\{"
    items: list[TypedNodeSignature] = []
    for match in re.finditer(pattern, code):
        tail = code[match.end() : match.end() + 48]
        items.append(TypedNodeSignature(kind="guard_with_else" if "else" in tail else "guard_without_else", value=match.group("cond").strip(), line=_line_number(code, match.start())))
    return tuple(items)


def _non_python_loop_headers(code: str, language: str) -> tuple[TypedNodeSignature, ...]:
    pattern = r"for\s*\((?P<head>[^)]*)\)" if language in {"javascript", "java", "cpp"} else r"for\s+(?P<head>[^\n{]+)\s*\{"
    items: list[TypedNodeSignature] = []
    for match in re.finditer(pattern, code):
        head = match.group("head").strip()
        indexed = any(token in head for token in ("range(", "len(", ";", "++", "--", "[i]", "[idx]", "index", "idx"))
        items.append(TypedNodeSignature(kind="indexed_loop" if indexed else "direct_loop", value=head, line=_line_number(code, match.start())))
    return tuple(items)


def _non_python_helper_names(function_names: tuple[str, ...]) -> tuple[str, ...]:
    helper_like = {
        name
        for name in function_names
        if name.lower().startswith(("helper", "normalize", "keep", "abs"))
    }
    return tuple(sorted(set(function_names[1:]) | helper_like))


def _looks_like_non_python_declaration_prefix(prefix: str) -> bool:
    line_prefix = prefix.rsplit("\n", 1)[-1]
    return bool(
        re.search(
            r"(?:\bfunc|\bfunction|\bclass|\bpublic|\bprivate|\bprotected|\bstatic|\bfinal|"
            r"\bint|\blong|\bdouble|\bfloat|\bboolean|\bbool|\bvoid|\bauto)\s*$",
            line_prefix,
        )
    )


def _non_python_helper_calls(code: str, helper_names: tuple[str, ...]) -> tuple[TypedNodeSignature, ...]:
    helper_name_set = set(helper_names)
    items: list[TypedNodeSignature] = []
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", code):
        name = match.group(1)
        if _looks_like_non_python_declaration_prefix(code[: match.start()]):
            continue
        if name in helper_name_set or name.startswith(("helper", "keep", "normalize", "norm_", "absInt")):
            items.append(TypedNodeSignature(kind="helper_call", value=name, line=_line_number(code, match.start())))
    return tuple(items)


def _non_python_typed_initializers(code: str, language: str) -> tuple[TypedNodeSignature, ...]:
    items: list[TypedNodeSignature] = []
    patterns = [
        (r"\b(?:let|const|var)\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*Number\s*\(\s*0\s*\)", "typed_int_zero"),
        (r"\b(?:int|long|double|float)\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?:Integer|Long|Double|Float)\.valueOf\s*\(\s*0\s*\)", "typed_int_zero"),
        (r"\b(?:int|long|double|float|bool|auto)\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:=\s*)?(?:int\s*)?\{\s*0\s*\}", "typed_int_zero"),
        (r"\b[A-Za-z_][A-Za-z0-9_]*\s*:=\s*int\s*\(\s*0\s*\)", "typed_int_zero"),
        (r"\b(?:let|const|var|int|long|double|float|bool|auto)\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*0\b", "literal_zero"),
        (r"\b[A-Za-z_][A-Za-z0-9_]*\s*:=\s*0\b", "literal_zero"),
        (r"\b(?:ArrayList|std::vector|make\s*\(\s*\[)", "typed_container"),
    ]
    for pattern, kind in patterns:
        for match in re.finditer(pattern, code):
            items.append(TypedNodeSignature(kind=kind, value=match.group(0).strip(), line=_line_number(code, match.start())))
    return tuple(items)


def _non_python_return_forms(code: str) -> tuple[TypedNodeSignature, ...]:
    items: list[TypedNodeSignature] = []
    materialized_return_names = {
        "result",
        "return_total",
        "returnTotal",
        "final_total",
        "finalTotal",
        "semcodebookReturnValue",
        "semcodebook_return_value",
    }
    for match in re.finditer(r"return\s+([^;\n]+)", code):
        value = match.group(1).strip()
        kind = (
            "named_return"
            if value in materialized_return_names
            else "direct_return"
        )
        items.append(TypedNodeSignature(kind=kind, value=value, line=_line_number(code, match.start())))
    return tuple(items)


def _non_python_comparison_forms(guards: tuple[TypedNodeSignature, ...]) -> tuple[TypedNodeSignature, ...]:
    items: list[TypedNodeSignature] = []
    for guard in guards:
        text = guard.value.strip()
        kind = "reordered:Compare" if text.startswith(("0 <", "0 >", "0 <=", "0 >=")) else "direct:Compare"
        items.append(TypedNodeSignature(kind=kind, value=text, line=guard.line))
    return tuple(items)


def _non_python_temporary_bindings(code: str) -> tuple[TypedNodeSignature, ...]:
    items: list[TypedNodeSignature] = []
    materialized_temporary_names = {
        "currentItem",
        "current_item",
        "currentValue",
        "current_value",
    }
    for match in re.finditer(r"\b(?:let|const|var|int|long|double|float|bool|auto)?\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::=|=(?!=))\s*([^;\n]+)", code):
        name = match.group(1)
        if name in materialized_temporary_names:
            items.append(TypedNodeSignature(kind="temporary_binding", value=name, line=_line_number(code, match.start())))
    return tuple(items)


def _summarize_non_python(code: str, language: str) -> TypedAstSummary:
    function_names, function_arity = _non_python_function_signatures(code, language)
    helper_names = _non_python_helper_names(function_names)
    guard_conditions = _non_python_guard_conditions(code, language)
    loop_headers = _non_python_loop_headers(code, language)
    helper_calls = _non_python_helper_calls(code, helper_names)
    typed_initializers = _non_python_typed_initializers(code, language)
    return_forms = _non_python_return_forms(code)
    temporary_bindings = _non_python_temporary_bindings(code)
    comparison_forms = _non_python_comparison_forms(guard_conditions)
    parses = bool(function_names or guard_conditions or loop_headers or temporary_bindings or return_forms)
    return TypedAstSummary(
        language=language,
        parses=parses,
        function_names=function_names,
        function_arity=function_arity,
        helper_names=helper_names,
        guard_conditions=guard_conditions,
        loop_headers=loop_headers,
        helper_calls=helper_calls,
        typed_initializers=typed_initializers,
        return_forms=return_forms,
        comparison_forms=comparison_forms,
        temporary_bindings=temporary_bindings,
        notes=(f"heuristic_{language}_summary",),
    )


def summarize_typed_ast(code: str, language: str) -> TypedAstSummary:
    tree = parse_typed_tree(code, language)
    if tree is None:
        if language.lower() != "python":
            return _summarize_non_python(code, language)
        return TypedAstSummary(
            language=language,
            parses=False,
            function_names=(),
            function_arity=(),
            helper_names=(),
            guard_conditions=(),
            loop_headers=(),
            helper_calls=(),
            typed_initializers=(),
            return_forms=(),
            comparison_forms=(),
            temporary_bindings=(),
            notes=("parse_failed_or_unsupported_language",),
        )

    function_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    helper_names = tuple(node.name for node in function_nodes[1:])
    guard_conditions: list[TypedNodeSignature] = []
    loop_headers: list[TypedNodeSignature] = []
    helper_calls: list[TypedNodeSignature] = []
    typed_initializers: list[TypedNodeSignature] = []
    return_forms: list[TypedNodeSignature] = []
    comparison_forms: list[TypedNodeSignature] = []
    temporary_bindings: list[TypedNodeSignature] = []

    named_return_bindings: set[str] = set()
    for function in function_nodes:
        for index, statement in enumerate(function.body[:-1]):
            next_statement = function.body[index + 1]
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(next_statement, ast.Return)
                and isinstance(next_statement.value, ast.Name)
                and next_statement.value.id == statement.targets[0].id
            ):
                named_return_bindings.add(statement.targets[0].id)

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            guard_conditions.append(
                TypedNodeSignature(
                    kind="guard_with_else" if node.orelse else "guard_without_else",
                    value=_expr_text(node.test),
                    line=getattr(node, "lineno", 0),
                )
            )
        elif isinstance(node, ast.For):
            loop_headers.append(
                TypedNodeSignature(
                    kind=_loop_kind(node),
                    value=_expr_text(node.iter),
                    line=getattr(node, "lineno", 0),
                )
            )
        elif isinstance(node, ast.Call):
            helper_name = _expr_text(node.func)
            if helper_name not in {"len", "int", "dict", "sum", "range"}:
                helper_calls.append(
                    TypedNodeSignature(
                        kind="helper_call",
                        value=helper_name,
                        line=getattr(node, "lineno", 0),
                    )
                )
        elif isinstance(node, ast.Assign):
            targets = tuple(_expr_text(target) for target in node.targets)
            if len(targets) == 1 and isinstance(node.targets[0], ast.Name):
                temporary_bindings.append(
                    TypedNodeSignature(
                        kind="temporary_binding",
                        value=",".join(targets),
                        line=getattr(node, "lineno", 0),
                    )
                )
            initializer_kind = _typed_initializer_kind(node)
            if initializer_kind != "other_initializer":
                typed_initializers.append(
                    TypedNodeSignature(
                        kind=initializer_kind,
                        value=_expr_text(node.value),
                        line=getattr(node, "lineno", 0),
                    )
                )
        elif isinstance(node, ast.Return):
            value_text = _expr_text(node.value)
            return_forms.append(
                TypedNodeSignature(
                    kind="named_return" if value_text in named_return_bindings else "direct_return",
                    value=value_text,
                    line=getattr(node, "lineno", 0),
                )
            )
        elif isinstance(node, ast.Compare) and len(node.ops) == 1:
            comparison_forms.append(
                TypedNodeSignature(
                    kind=_comparison_kind(node),
                    value=_expr_text(node),
                    line=getattr(node, "lineno", 0),
                )
            )

    return TypedAstSummary(
        language=language,
        parses=True,
        function_names=tuple(node.name for node in function_nodes),
        function_arity=tuple(len(node.args.args) for node in function_nodes),
        helper_names=helper_names,
        guard_conditions=tuple(guard_conditions),
        loop_headers=tuple(loop_headers),
        helper_calls=tuple(helper_calls),
        typed_initializers=tuple(typed_initializers),
        return_forms=tuple(return_forms),
        comparison_forms=tuple(comparison_forms),
        temporary_bindings=tuple(temporary_bindings),
        notes=("typed_ast_summary", "python_only_for_now"),
    )


def stable_ast_fingerprint(summary: TypedAstSummary) -> str:
    if not summary.parses:
        return "typed_ast_unavailable"
    payload = "|".join(
        (
            ",".join(summary.function_names),
            ",".join(str(item) for item in summary.function_arity),
            ",".join(item.kind for item in summary.guard_conditions),
            ",".join(item.kind for item in summary.loop_headers),
            ",".join(item.kind for item in summary.typed_initializers),
            ",".join(item.kind for item in summary.return_forms),
            ",".join(item.kind for item in summary.comparison_forms),
            ",".join(item.kind for item in summary.temporary_bindings),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
