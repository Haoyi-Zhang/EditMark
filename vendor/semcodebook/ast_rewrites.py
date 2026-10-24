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


def _swap_zero_compare_operator(op: ast.cmpop) -> ast.cmpop:
    if isinstance(op, ast.Gt):
        return ast.Lt()
    if isinstance(op, ast.GtE):
        return ast.LtE()
    if isinstance(op, ast.Lt):
        return ast.Gt()
    if isinstance(op, ast.LtE):
        return ast.GtE()
    if isinstance(op, ast.Eq):
        return ast.Eq()
    if isinstance(op, ast.NotEq):
        return ast.NotEq()
    return op


def _is_zero(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == 0


def rewrite_comparison_idiom(code: str, *, bit_value: int, language: str) -> str:
    tree = _parse(code, language)
    if tree is None or bit_value not in {0, 1}:
        return code

    class ComparisonRewriter(ast.NodeTransformer):
        def __init__(self) -> None:
            self.changed = False

        def visit_Compare(self, node: ast.Compare) -> ast.AST:
            node = self.generic_visit(node)
            if self.changed or len(node.ops) != 1 or len(node.comparators) != 1:
                return node
            op = node.ops[0]
            right = node.comparators[0]
            if bit_value == 1 and _is_zero(right) and not _is_zero(node.left):
                self.changed = True
                return ast.Compare(left=ast.Constant(value=0), ops=[_swap_zero_compare_operator(op)], comparators=[node.left])
            if bit_value == 0 and _is_zero(node.left) and not _is_zero(right):
                self.changed = True
                return ast.Compare(left=right, ops=[_swap_zero_compare_operator(op)], comparators=[ast.Constant(value=0)])
            return node

    rewriter = ComparisonRewriter()
    updated = rewriter.visit(tree)
    return _render(updated, code) if rewriter.changed else code

def rewrite_initialization_idiom(code: str, *, bit_value: int, language: str) -> str:
    tree = _parse(code, language)
    if tree is None or bit_value not in {0, 1}:
        return code

    class InitializationRewriter(ast.NodeTransformer):
        def __init__(self) -> None:
            self.changed = False

        def visit_Assign(self, node: ast.Assign) -> ast.AST:
            node = self.generic_visit(node)
            if self.changed:
                return node
            if bit_value == 1:
                if isinstance(node.value, ast.Constant) and node.value.value == 0:
                    self.changed = True
                    node.value = ast.Call(func=ast.Name(id="int", ctx=ast.Load()), args=[ast.Constant(value=0)], keywords=[])
                    return node
                if isinstance(node.value, ast.Dict) and not node.value.keys:
                    self.changed = True
                    node.value = ast.Call(func=ast.Name(id="dict", ctx=ast.Load()), args=[], keywords=[])
                    return node
            if bit_value == 0:
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "int" and len(node.value.args) == 1:
                    arg = node.value.args[0]
                    if isinstance(arg, ast.Constant) and arg.value == 0:
                        self.changed = True
                        node.value = ast.Constant(value=0)
                        return node
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "dict" and not node.value.args and not node.value.keywords:
                    self.changed = True
                    node.value = ast.Dict(keys=[], values=[])
                    return node
            return node

    rewriter = InitializationRewriter()
    updated = rewriter.visit(tree)
    return _render(updated, code) if rewriter.changed else code

def rewrite_return_expression_style(code: str, *, bit_value: int, language: str) -> str:
    tree = _parse(code, language)
    if tree is None or bit_value not in {0, 1}:
        return code

    class ReturnRewriter(ast.NodeTransformer):
        def __init__(self) -> None:
            self.changed = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            node = self.generic_visit(node)
            if self.changed:
                return node
            if bit_value == 1:
                for index, stmt in enumerate(node.body):
                    if isinstance(stmt, ast.Return) and stmt.value is not None:
                        self.changed = True
                        node.body[index:index + 1] = [
                            ast.Assign(targets=[ast.Name(id="final_total", ctx=ast.Store())], value=stmt.value),
                            ast.Return(value=ast.Name(id="final_total", ctx=ast.Load())),
                        ]
                        break
                return node
            for index in range(len(node.body) - 1):
                current = node.body[index]
                following = node.body[index + 1]
                if (
                    isinstance(current, ast.Assign)
                    and len(current.targets) == 1
                    and isinstance(current.targets[0], ast.Name)
                    and isinstance(following, ast.Return)
                    and isinstance(following.value, ast.Name)
                    and current.targets[0].id == following.value.id
                ):
                    self.changed = True
                    node.body[index:index + 2] = [ast.Return(value=current.value)]
                    break
            return node

    rewriter = ReturnRewriter()
    updated = rewriter.visit(tree)
    return _render(updated, code) if rewriter.changed else code

