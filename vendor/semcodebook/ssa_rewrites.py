from __future__ import annotations

import ast


def _parse(code: str, language: str) -> ast.Module | None:
    if language.lower() != "python":
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    if not isinstance(tree, ast.Module):
        return None
    return tree


def _render(tree: ast.AST, fallback: str) -> str:
    try:
        fixed = ast.fix_missing_locations(tree)
        return ast.unparse(fixed) + "\n"
    except Exception:
        return fallback


def rewrite_accumulator_style(code: str, *, bit_value: int, language: str) -> str:
    tree = _parse(code, language)
    if tree is None or bit_value not in {0, 1}:
        return code

    class AccumulatorRewriter(ast.NodeTransformer):
        def __init__(self) -> None:
            self.changed = False

        def _rewrite_body_to_bit0(self, body: list[ast.stmt]) -> list[ast.stmt]:
            rewritten: list[ast.stmt] = []
            index = 0
            while index < len(body):
                current = body[index]
                following = body[index + 1] if index + 1 < len(body) else None
                if (
                    not self.changed
                    and isinstance(current, ast.Assign)
                    and len(current.targets) == 1
                    and isinstance(current.targets[0], ast.Name)
                    and isinstance(current.value, ast.BinOp)
                    and isinstance(current.value.op, ast.Add)
                    and isinstance(current.value.left, ast.Name)
                    and isinstance(following, ast.Assign)
                    and len(following.targets) == 1
                    and isinstance(following.targets[0], ast.Name)
                    and isinstance(following.value, ast.Name)
                    and following.value.id == current.targets[0].id
                    and following.targets[0].id == current.value.left.id
                ):
                    self.changed = True
                    rewritten.append(ast.AugAssign(target=ast.Name(id=following.targets[0].id, ctx=ast.Store()), op=ast.Add(), value=current.value.right))
                    index += 2
                    continue
                rewritten.append(current)
                index += 1
            return rewritten

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            node = self.generic_visit(node)
            if bit_value == 0:
                node.body = self._rewrite_body_to_bit0(node.body)
            return node

        def visit_For(self, node: ast.For) -> ast.AST:
            node = self.generic_visit(node)
            if bit_value == 0:
                node.body = self._rewrite_body_to_bit0(node.body)
            return node

        def visit_If(self, node: ast.If) -> ast.AST:
            node = self.generic_visit(node)
            if bit_value == 0:
                node.body = self._rewrite_body_to_bit0(node.body)
                node.orelse = self._rewrite_body_to_bit0(node.orelse)
            return node

        def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
            node = self.generic_visit(node)
            if bit_value != 1 or self.changed or not isinstance(node.target, ast.Name) or not isinstance(node.op, ast.Add):
                return node
            self.changed = True
            target_name = node.target.id
            temp_name = f"next_{target_name}"
            return [
                ast.Assign(targets=[ast.Name(id=temp_name, ctx=ast.Store())], value=ast.BinOp(left=ast.Name(id=target_name, ctx=ast.Load()), op=ast.Add(), right=node.value)),
                ast.Assign(targets=[ast.Name(id=target_name, ctx=ast.Store())], value=ast.Name(id=temp_name, ctx=ast.Load())),
            ]

    rewriter = AccumulatorRewriter()
    updated = rewriter.visit(tree)
    return _render(updated, code) if rewriter.changed else code

def rewrite_temporary_binding_style(code: str, *, bit_value: int, language: str) -> str:
    tree = _parse(code, language)
    if tree is None or bit_value not in {0, 1}:
        return code

    class TemporaryBindingRewriter(ast.NodeTransformer):
        def __init__(self) -> None:
            self.changed = False

        def _rewrite_body_to_bit0(self, body: list[ast.stmt]) -> list[ast.stmt]:
            rewritten: list[ast.stmt] = []
            index = 0
            while index < len(body):
                current = body[index]
                following = body[index + 1] if index + 1 < len(body) else None
                if (
                    not self.changed
                    and isinstance(current, ast.Assign)
                    and len(current.targets) == 1
                    and isinstance(current.targets[0], ast.Name)
                    and isinstance(following, ast.AugAssign)
                    and isinstance(following.value, ast.Name)
                    and following.value.id == current.targets[0].id
                ):
                    self.changed = True
                    rewritten.append(ast.AugAssign(target=following.target, op=following.op, value=current.value))
                    index += 2
                    continue
                rewritten.append(current)
                index += 1
            return rewritten

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            node = self.generic_visit(node)
            if bit_value == 0:
                node.body = self._rewrite_body_to_bit0(node.body)
            return node

        def visit_For(self, node: ast.For) -> ast.AST:
            node = self.generic_visit(node)
            if bit_value == 0:
                node.body = self._rewrite_body_to_bit0(node.body)
            return node

        def visit_If(self, node: ast.If) -> ast.AST:
            node = self.generic_visit(node)
            if bit_value == 0:
                node.body = self._rewrite_body_to_bit0(node.body)
                node.orelse = self._rewrite_body_to_bit0(node.orelse)
            return node

        def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
            node = self.generic_visit(node)
            if bit_value != 1 or self.changed or not isinstance(node.op, ast.Add):
                return node
            self.changed = True
            return [
                ast.Assign(targets=[ast.Name(id="current_item", ctx=ast.Store())], value=node.value),
                ast.AugAssign(target=node.target, op=node.op, value=ast.Name(id="current_item", ctx=ast.Load())),
            ]

    rewriter = TemporaryBindingRewriter()
    updated = rewriter.visit(tree)
    return _render(updated, code) if rewriter.changed else code

