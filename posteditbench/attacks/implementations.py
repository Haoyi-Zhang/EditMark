from __future__ import annotations

import ast
import builtins
import keyword
import re
from dataclasses import dataclass
from typing import Any

from ..models import AttackOutcome
from ..utils import language_comment_prefix, normalize_whitespace, stable_hash, strip_comments_with_reason
from .base import CodeAttack


_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_PYTHON_RESERVED = set(keyword.kwlist) | set(dir(builtins))
_BRACE_CLOSE = re.compile(r"^\}\s*;?\s*$")
_COMMON_IDENTIFIER_RESERVED = {
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "default",
    "do",
    "else",
    "enum",
    "export",
    "extends",
    "false",
    "finally",
    "for",
    "function",
    "if",
    "import",
    "in",
    "interface",
    "let",
    "new",
    "null",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "static",
    "super",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "typeof",
    "using",
    "var",
    "void",
    "while",
}
_JS_DECL_RE = re.compile(r"\b(?:let|const|var)\s+([A-Za-z_][A-Za-z0-9_]*)\b")
_GO_SHORT_DECL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:=")
_GO_VAR_DECL_RE = re.compile(r"\bvar\s+([A-Za-z_][A-Za-z0-9_]*)\b")
_RUST_LET_DECL_RE = re.compile(r"\blet(?:\s+mut)?\s+([A-Za-z_][A-Za-z0-9_]*)\b")


def _normalized_language(language: str) -> str:
    normalized = str(language or "").strip().lower()
    aliases = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "c++": "cpp",
    }
    return aliases.get(normalized, normalized)


class _PythonLocalIdentifierCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self._function_depth = 0
        self.names: list[str] = []

    def _remember(self, name: str) -> None:
        normalized = str(name).strip()
        if (
            self._function_depth <= 0
            or len(normalized) <= 2
            or normalized.startswith("__")
            or normalized in _PYTHON_RESERVED
        ):
            return
        if normalized not in self.names:
            self.names.append(normalized)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_depth += 1
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            self._remember(argument.arg)
        if node.args.vararg is not None:
            self._remember(node.args.vararg.arg)
        if node.args.kwarg is not None:
            self._remember(node.args.kwarg.arg)
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._function_depth += 1
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            self._remember(argument.arg)
        if node.args.vararg is not None:
            self._remember(node.args.vararg.arg)
        if node.args.kwarg is not None:
            self._remember(node.args.kwarg.arg)
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self._remember(node.id)


class _PythonLocalIdentifierRenamer(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping
        self._function_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        return self.visit_FunctionDef(node)

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        if self._function_depth > 0 and node.arg in self._mapping:
            node.arg = self._mapping[node.arg]
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if self._function_depth > 0 and node.id in self._mapping:
            node.id = self._mapping[node.id]
        return node


def _rename_python_local_identifiers(source: str, *, seed: int) -> tuple[str, list[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []
    collector = _PythonLocalIdentifierCollector()
    collector.visit(tree)
    if not collector.names:
        return source, []
    mapping: dict[str, str] = {}
    notes: list[str] = []
    for index, name in enumerate(collector.names):
        token = stable_hash(f"{seed}:{name}")[:6]
        replacement = f"v_{index}_{token}"
        mapping[name] = replacement
        notes.append(f"{name}->{replacement}")
    mutated_tree = _PythonLocalIdentifierRenamer(mapping).visit(tree)
    ast.fix_missing_locations(mutated_tree)
    try:
        mutated = ast.unparse(mutated_tree)
    except Exception:
        return source, []
    return (mutated, notes) if mutated != source else (source, [])


def _brace_identifier_patterns(language: str) -> tuple[re.Pattern[str], ...]:
    normalized_language = _normalized_language(language)
    if normalized_language in {"javascript", "typescript"}:
        return (_JS_DECL_RE,)
    if normalized_language == "go":
        return (_GO_VAR_DECL_RE, _GO_SHORT_DECL_RE)
    if normalized_language == "rust":
        return (_RUST_LET_DECL_RE,)
    return ()


def _strip_brace_scope_noise(line: str, *, in_block_comment: bool) -> tuple[str, bool]:
    result: list[str] = []
    index = 0
    quote: str | None = None
    escaped = False
    while index < len(line):
        current = line[index]
        next_two = line[index : index + 2]
        if in_block_comment:
            if next_two == "*/":
                in_block_comment = False
                index += 2
                continue
            index += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif current == quote:
                quote = None
            result.append(" ")
            index += 1
            continue
        if next_two == "//":
            break
        if next_two == "/*":
            in_block_comment = True
            index += 2
            continue
        if current in {"'", '"', "`"}:
            quote = current
            result.append(" ")
            index += 1
            continue
        result.append(current)
        index += 1
    return "".join(result), in_block_comment


def _collect_brace_local_identifiers(source: str, *, language: str) -> list[str]:
    patterns = _brace_identifier_patterns(language)
    if not patterns:
        return []
    candidates: list[str] = []
    brace_depth = 0
    in_block_comment = False
    for line in source.splitlines():
        sanitized, in_block_comment = _strip_brace_scope_noise(line, in_block_comment=in_block_comment)
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "#", "/*", "*", "import ", "package ", "using ", "#include")):
            brace_depth += sanitized.count("{") - sanitized.count("}")
            brace_depth = max(brace_depth, 0)
            continue
        inside_local_scope = brace_depth > 0
        for pattern in patterns:
            if not inside_local_scope:
                continue
            for match in pattern.finditer(sanitized):
                name = str(match.group(1)).strip()
                if (
                    len(name) <= 2
                    or name.startswith("__")
                    or name in _COMMON_IDENTIFIER_RESERVED
                    or name in _PYTHON_RESERVED
                ):
                    continue
                if name not in candidates:
                    candidates.append(name)
        brace_depth += sanitized.count("{") - sanitized.count("}")
        brace_depth = max(brace_depth, 0)
    return candidates


def _rewrite_brace_identifiers(source: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return source
    result: list[str] = []
    index = 0
    in_block_comment = False
    in_line_comment = False
    quote: str | None = None
    escaped = False
    while index < len(source):
        current = source[index]
        next_two = source[index : index + 2]
        if in_line_comment:
            result.append(current)
            if current == "\n":
                in_line_comment = False
            index += 1
            continue
        if in_block_comment:
            if next_two == "*/":
                result.append("*")
                result.append("/")
                index += 2
                in_block_comment = False
                continue
            result.append(current)
            index += 1
            continue
        if quote is not None:
            result.append(current)
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif current == quote:
                quote = None
            index += 1
            continue
        if next_two == "//":
            result.append("/")
            result.append("/")
            index += 2
            in_line_comment = True
            continue
        if next_two == "/*":
            result.append("/")
            result.append("*")
            index += 2
            in_block_comment = True
            continue
        if current in {"'", '"', "`"}:
            quote = current
            result.append(current)
            index += 1
            continue
        if current.isalpha() or current == "_":
            start = index
            index += 1
            while index < len(source) and (source[index].isalnum() or source[index] == "_"):
                index += 1
            token = source[start:index]
            previous = next((char for char in reversed(result) if not char.isspace()), "")
            if token in mapping and previous not in {".", ":"}:
                result.append(mapping[token])
            else:
                result.append(token)
            continue
        result.append(current)
        index += 1
    return "".join(result)


def _rename_brace_local_identifiers(source: str, *, seed: int, language: str) -> tuple[str, list[str], bool]:
    candidates = _collect_brace_local_identifiers(source, language=language)
    if not candidates:
        return source, ["unsupported_or_no_safe_local_identifier"], False
    mapping: dict[str, str] = {}
    notes: list[str] = []
    for index, name in enumerate(candidates):
        token = stable_hash(f"{seed}:{language}:{name}")[:6]
        replacement = f"v_{index}_{token}"
        mapping[name] = replacement
        notes.append(f"{name}->{replacement}")
    mutated = _rewrite_brace_identifiers(source, mapping)
    if mutated == source:
        return source, ["unsupported_or_no_safe_local_identifier"], False
    return mutated, notes, True


def _rename_identifiers(source: str, *, seed: int, language: str = "") -> tuple[str, list[str], bool]:
    normalized_language = _normalized_language(language)
    if normalized_language in {"", "python"}:
        mutated, notes = _rename_python_local_identifiers(source, seed=seed)
        if not notes:
            return source, ["unsupported_or_no_safe_local_identifier"], False
        return mutated, notes, True
    if normalized_language in {"javascript", "typescript", "go", "java", "cpp", "c", "rust"}:
        return _rename_brace_local_identifiers(source, seed=seed, language=normalized_language)
    return source, ["unsupported_language"], False


def _wrap_comment_block(seed: int, *, language: str = "") -> str:
    token = stable_hash(f"noise:{seed}")[:10]
    prefix = language_comment_prefix(language)
    return f"{prefix} artifact-noise:{token}"


def _shuffle_blocks(source: str, *, seed: int) -> tuple[str, list[str]]:
    blocks = [block for block in source.split("\n\n") if block.strip()]
    if len(blocks) <= 1:
        return source, []
    order = list(range(len(blocks)))
    for idx in range(len(order)):
        swap = (seed + idx * 7) % len(order)
        order[idx], order[swap] = order[swap], order[idx]
    shuffled = [blocks[idx] for idx in order]
    return "\n\n".join(shuffled), [f"order={order}"]


def _is_python_scaffold(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    if not isinstance(node.test, ast.Constant) or node.test.value is not True:
        return False
    if node.orelse:
        return False
    if len(node.body) != 1:
        return False
    child = node.body[0]
    if isinstance(child, ast.Pass):
        return True
    return _is_python_scaffold(child)


class _PythonScaffoldStripper(ast.NodeTransformer):
    @staticmethod
    def _strip_leading_scaffold(body: list[ast.stmt]) -> list[ast.stmt]:
        stripped = list(body)
        while stripped and _is_python_scaffold(stripped[0]):
            stripped = stripped[1:]
        return stripped

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        node.body = self._strip_leading_scaffold(list(node.body))
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.body = self._strip_leading_scaffold(list(node.body))
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        node.body = self._strip_leading_scaffold(list(node.body))
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        node.body = self._strip_leading_scaffold(list(node.body))
        return node


def _flatten_python_scaffold(source: str) -> tuple[str, list[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []
    stripped_tree = _PythonScaffoldStripper().visit(tree)
    ast.fix_missing_locations(stripped_tree)
    try:
        mutated = ast.unparse(stripped_tree)
    except Exception:  # pragma: no cover - ast.unparse should be available on py311+
        return source, []
    if mutated == source:
        return source, []
    return mutated, ["python_control_flow_flattened"]


def _brace_if_open(line: str, *, style: str) -> bool:
    stripped = line.strip()
    if style == "rust":
        return bool(re.match(r"^if\s+true\s*\{\s*$", stripped, flags=re.IGNORECASE))
    return bool(re.match(r"^if\s*\(\s*true\s*\)\s*\{\s*$", stripped, flags=re.IGNORECASE))


def _flatten_brace_scaffold(source: str, *, style: str) -> tuple[str, list[str]]:
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if "{" not in line:
            continue
        start = index + 1
        if start >= len(lines) or not _brace_if_open(lines[start], style=style):
            continue
        cursor = start
        depth = 0
        while cursor < len(lines):
            current = lines[cursor].strip()
            if _brace_if_open(lines[cursor], style=style):
                depth += 1
                cursor += 1
                continue
            if _BRACE_CLOSE.match(current):
                if depth == 0:
                    break
                depth -= 1
                cursor += 1
                if depth == 0:
                    mutated = "\n".join([*lines[:start], *lines[cursor:]])
                    if mutated != source:
                        return mutated, [f"brace_control_flow_flattened:{style}"]
                    return source, []
                continue
            break
        break
    return source, []


@dataclass(slots=True)
class CommentStripAttack(CodeAttack):
    name: str = "comment_strip"

    def apply(
        self,
        source: str,
        *,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AttackOutcome:
        language = _normalized_language((context or {}).get("language") or (metadata or {}).get("language") or "")
        mutated, unsupported_reason = strip_comments_with_reason(source, language=language)
        metadata_payload = dict(metadata or {})
        metadata_payload["supported"] = unsupported_reason is None
        if unsupported_reason is not None:
            metadata_payload["unsupported_reason"] = unsupported_reason
        return AttackOutcome(
            attack_name=self.name,
            source=mutated,
            changed=mutated != source,
            notes=("comments_removed",) if mutated != source else ((unsupported_reason or "no_comment_content"),),
            metadata=metadata_payload,
        )


@dataclass(slots=True)
class WhitespaceNormalizeAttack(CodeAttack):
    name: str = "whitespace_normalize"

    def apply(
        self,
        source: str,
        *,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AttackOutcome:
        mutated = normalize_whitespace(source)
        return AttackOutcome(
            attack_name=self.name,
            source=mutated,
            changed=mutated != source,
            notes=("whitespace_normalized",),
            metadata=metadata or {},
        )


@dataclass(slots=True)
class IdentifierRenameAttack(CodeAttack):
    name: str = "identifier_rename"

    def apply(
        self,
        source: str,
        *,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AttackOutcome:
        language = _normalized_language((context or {}).get("language") or (metadata or {}).get("language") or "")
        mutated, notes, supported = _rename_identifiers(source, seed=seed, language=language)
        metadata_payload = dict(metadata or {})
        metadata_payload["supported"] = supported
        if not supported:
            metadata_payload["unsupported_reason"] = notes[0] if notes else "unsupported_language"
        return AttackOutcome(
            attack_name=self.name,
            source=mutated,
            changed=mutated != source,
            notes=tuple(notes or ["no_rename"]),
            metadata=metadata_payload,
        )


@dataclass(slots=True)
class NoiseInsertAttack(CodeAttack):
    name: str = "noise_insert"

    def apply(
        self,
        source: str,
        *,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AttackOutcome:
        language = _normalized_language((context or {}).get("language") or (metadata or {}).get("language") or "")
        noise = _wrap_comment_block(seed, language=language)
        mutated = "\n".join([noise, source, noise])
        metadata_payload = dict(metadata or {})
        metadata_payload["supported"] = True
        return AttackOutcome(
            attack_name=self.name,
            source=mutated,
            changed=mutated != source,
            notes=("noise_comments_added",),
            metadata=metadata_payload,
        )


@dataclass(slots=True)
class BlockShuffleAttack(CodeAttack):
    name: str = "block_shuffle"

    def apply(
        self,
        source: str,
        *,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AttackOutcome:
        mutated, notes = _shuffle_blocks(source, seed=seed)
        return AttackOutcome(
            attack_name=self.name,
            source=mutated,
            changed=mutated != source,
            notes=tuple(notes or ["no_shuffle"]),
            metadata=metadata or {},
        )


@dataclass(slots=True)
class ControlFlowFlattenAttack(CodeAttack):
    name: str = "control_flow_flatten"

    def apply(
        self,
        source: str,
        *,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AttackOutcome:
        language = _normalized_language((context or {}).get("language") or (metadata or {}).get("language") or "")
        supported = True
        if language == "python":
            mutated, notes = _flatten_python_scaffold(source)
        elif language in {"javascript", "typescript", "java", "cpp", "c", "go"}:
            mutated, notes = _flatten_brace_scaffold(source, style=language)
        elif language == "rust":
            mutated, notes = _flatten_brace_scaffold(source, style="rust")
        else:
            mutated, notes = source, ["unsupported_language"]
            supported = False
        if mutated == source and notes == ["no_flattening"]:
            supported = False
        metadata_payload = dict(metadata or {})
        metadata_payload["supported"] = supported and mutated != source
        if not metadata_payload["supported"]:
            metadata_payload["unsupported_reason"] = notes[0] if notes else "no_flattening"
        return AttackOutcome(
            attack_name=self.name,
            source=mutated,
            changed=mutated != source,
            notes=tuple(notes or ["no_flattening"]),
            metadata=metadata_payload,
        )


@dataclass(slots=True)
class BudgetedAdaptiveAttack(CodeAttack):
    name: str = "budgeted_adaptive"
    candidate_order: tuple[str, ...] = (
        "comment_strip",
        "whitespace_normalize",
        "identifier_rename",
        "control_flow_flatten",
        "block_shuffle",
    )

    def _candidate_attack(self, name: str) -> CodeAttack:
        if name == "comment_strip":
            return CommentStripAttack()
        if name == "whitespace_normalize":
            return WhitespaceNormalizeAttack()
        if name == "identifier_rename":
            return IdentifierRenameAttack()
        if name == "block_shuffle":
            return BlockShuffleAttack()
        if name == "control_flow_flatten":
            return ControlFlowFlattenAttack()
        if name == "noise_insert":
            return NoiseInsertAttack()
        raise KeyError(name)

    def apply(
        self,
        source: str,
        *,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AttackOutcome:
        context = dict(context or {})
        config = dict(context.get("config") or {})
        budget = int(config.get("budget", context.get("budget", 3)))
        min_quality = float(config.get("min_quality", context.get("min_quality", 0.0)))
        candidate_order = tuple(config.get("candidate_order") or self.candidate_order)
        detector = context.get("detector")
        quality = context.get("quality")
        validate = context.get("validate")

        current = source
        current_score = float(detector(current)) if callable(detector) else 1.0
        current_quality = float(quality(current)) if callable(quality) else 1.0
        current_semantic = validate(current) if callable(validate) else None
        supported_candidate_available = False
        unsupported_candidates: list[str] = []
        curve: list[dict[str, Any]] = [
            {
                "budget": 0,
                "detector_score": round(current_score, 4),
                "quality_score": round(current_quality, 4),
                "semantic_preserving": current_semantic,
                "step_name": "start",
            }
        ]
        selected_steps: list[str] = []

        for spent in range(1, budget + 1):
            best_choice: tuple[float, float, str, str, bool | None] | None = None
            for step_name in candidate_order:
                candidate_attack = self._candidate_attack(step_name)
                candidate_outcome = candidate_attack.apply(
                    current,
                    seed=seed + spent,
                    metadata=metadata or {},
                    context=context,
                )
                candidate_supported = bool(candidate_outcome.metadata.get("supported", True))
                if not candidate_supported:
                    reason = str(candidate_outcome.metadata.get("unsupported_reason", "unsupported_candidate")).strip()
                    unsupported_candidates.append(f"{step_name}:{reason}")
                    continue
                supported_candidate_available = True
                candidate_source = candidate_outcome.source
                candidate_score = float(detector(candidate_source)) if callable(detector) else current_score
                candidate_quality = float(quality(candidate_source)) if callable(quality) else current_quality
                candidate_semantic = validate(candidate_source) if callable(validate) else None
                if candidate_semantic is False:
                    continue
                if candidate_quality < min_quality:
                    continue
                choice = (candidate_score, -candidate_quality, step_name, candidate_source, candidate_semantic)
                if best_choice is None or choice < best_choice:
                    best_choice = choice
            if best_choice is None:
                break
            candidate_score, neg_quality, step_name, candidate_source, candidate_semantic = best_choice
            candidate_quality = -neg_quality
            if candidate_score >= current_score and candidate_quality <= current_quality:
                break
            current = candidate_source
            current_score = candidate_score
            current_quality = candidate_quality
            current_semantic = candidate_semantic
            selected_steps.append(step_name)
            curve.append(
                {
                    "budget": spent,
                    "detector_score": round(current_score, 4),
                    "quality_score": round(current_quality, 4),
                    "semantic_preserving": current_semantic,
                    "step_name": step_name,
                }
            )

        outcome_metadata = dict(metadata or {})
        outcome_metadata.update(
            {
                "budget": budget,
                "selected_steps": selected_steps,
                "budget_curve": curve,
                "final_detector_score": round(current_score, 4),
                "final_quality_score": round(current_quality, 4),
                "semantic_preserving": current_semantic,
                "supported": supported_candidate_available,
                "unsupported_candidates": unsupported_candidates,
            }
        )
        if not supported_candidate_available:
            outcome_metadata["unsupported_reason"] = unsupported_candidates[0] if unsupported_candidates else "no_supported_candidates"
        notes = (f"budget={budget}", f"steps={','.join(selected_steps) or 'none'}")
        return AttackOutcome(
            attack_name=self.name,
            source=current,
            changed=current != source,
            notes=notes,
            metadata=outcome_metadata,
        )
