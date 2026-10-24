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


def rewrite_guard_style(code: str, *, bit_value: int, language: str) -> str:
    tree = _parse(code, language)
    if tree is None or bit_value not in {0, 1}:
        return code

    class GuardRewriter(ast.NodeTransformer):
        def __init__(self) -> None:
            self.changed = False

        def visit_If(self, node: ast.If) -> ast.AST:
            node = self.generic_visit(node)
            if self.changed or not node.body:
                return node
            if bit_value == 1 and not node.orelse and len(node.body) == 1 and isinstance(node.body[0], ast.Return):
                self.changed = True
                return ast.If(test=ast.UnaryOp(op=ast.Not(), operand=node.test), body=[ast.Pass()], orelse=node.body)
            if (
                bit_value == 0
                and node.orelse
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Pass)
                and isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
            ):
                self.changed = True
                return ast.If(test=node.test.operand, body=node.orelse, orelse=[])
            return node

    rewriter = GuardRewriter()
    updated = rewriter.visit(tree)
    return _render(updated, code) if rewriter.changed else code

def rewrite_iteration_style(code: str, *, bit_value: int, language: str) -> str:
    tree = _parse(code, language)
    if tree is None or bit_value not in {0, 1}:
        return code

    class LoopRewriter(ast.NodeTransformer):
        def __init__(self) -> None:
            self.changed = False

        def visit_For(self, node: ast.For) -> ast.AST:
            node = self.generic_visit(node)
            if self.changed:
                return node
            if bit_value == 1 and isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Name):
                self.changed = True
                target_name = node.target.id
                iter_name = node.iter.id
                node.iter = ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[ast.Call(func=ast.Name(id="len", ctx=ast.Load()), args=[ast.Name(id=iter_name, ctx=ast.Load())], keywords=[])],
                    keywords=[],
                )
                node.body.insert(
                    0,
                    ast.Assign(
                        targets=[ast.Name(id=target_name, ctx=ast.Store())],
                        value=ast.Subscript(value=ast.Name(id=iter_name, ctx=ast.Load()), slice=ast.Name(id="index", ctx=ast.Load()), ctx=ast.Load()),
                    ),
                )
                node.target = ast.Name(id="index", ctx=ast.Store())
                return node
            if bit_value == 0 and isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Call):
                call = node.iter
                if (
                    isinstance(call.func, ast.Name)
                    and call.func.id == "range"
                    and len(call.args) == 1
                    and isinstance(call.args[0], ast.Call)
                    and isinstance(call.args[0].func, ast.Name)
                    and call.args[0].func.id == "len"
                    and len(call.args[0].args) == 1
                    and isinstance(call.args[0].args[0], ast.Name)
                    and node.body
                    and isinstance(node.body[0], ast.Assign)
                    and len(node.body[0].targets) == 1
                    and isinstance(node.body[0].targets[0], ast.Name)
                    and isinstance(node.body[0].value, ast.Subscript)
                ):
                    seq_name = call.args[0].args[0].id
                    first = node.body[0]
                    subscript = first.value
                    if isinstance(subscript.value, ast.Name) and subscript.value.id == seq_name:
                        self.changed = True
                        node.target = ast.Name(id=first.targets[0].id, ctx=ast.Store())
                        node.iter = ast.Name(id=seq_name, ctx=ast.Load())
                        node.body = node.body[1:]
                return node
            return node

    rewriter = LoopRewriter()
    updated = rewriter.visit(tree)
    return _render(updated, code) if rewriter.changed else code

def rewrite_helper_extraction_style(code: str, *, bit_value: int, language: str) -> str:
    tree = _parse(code, language)
    if tree is None or bit_value not in {0, 1}:
        return code

    class HelperRewriter(ast.NodeTransformer):
        def __init__(self) -> None:
            self.changed = False

        def visit_Module(self, node: ast.Module) -> ast.AST:
            node = self.generic_visit(node)
            if bit_value == 1 and self.changed:
                helper = ast.FunctionDef(
                    name="helper_transform",
                    args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="value")], kwonlyargs=[], kw_defaults=[], defaults=[]),
                    body=[ast.Return(value=ast.Name(id="value", ctx=ast.Load()))],
                    decorator_list=[],
                )
                node.body.insert(0, helper)
            elif bit_value == 0:
                filtered = [stmt for stmt in node.body if not (isinstance(stmt, ast.FunctionDef) and stmt.name == "helper_transform")]
                if len(filtered) != len(node.body):
                    self.changed = True
                node.body = filtered
            return node

        def visit_Call(self, node: ast.Call) -> ast.AST:
            node = self.generic_visit(node)
            if bit_value == 0 and isinstance(node.func, ast.Name) and node.func.id == "helper_transform" and len(node.args) == 1:
                self.changed = True
                return node.args[0]
            return node

        def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
            node = self.generic_visit(node)
            if bit_value != 1 or self.changed:
                return node
            self.changed = True
            return ast.AugAssign(target=node.target, op=node.op, value=ast.Call(func=ast.Name(id="helper_transform", ctx=ast.Load()), args=[node.value], keywords=[]))

        def visit_Assign(self, node: ast.Assign) -> ast.AST:
            node = self.generic_visit(node)
            if bit_value != 1 or self.changed or not node.targets:
                return node
            self.changed = True
            node.value = ast.Call(func=ast.Name(id="helper_transform", ctx=ast.Load()), args=[node.value], keywords=[])
            return node

    rewriter = HelperRewriter()
    updated = rewriter.visit(tree)
    return _render(updated, code) if rewriter.changed else code

