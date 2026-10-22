"""AD-43 — exact canonicalization for affine expression chains.

Only exact algebraic identities are normalized. No numeric fingerprints,
heuristic equivalence, or domain-specific target expressions are used.
"""
from __future__ import annotations

from dataclasses import dataclass
import ast
from typing import Optional


@dataclass(frozen=True)
class AffineForm:
    """Exact affine form a*x + b over an arbitrary symbolic base."""
    base: str
    scale: float
    offset: float


def _number(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _number(node.operand)
        return None if value is None else -value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _number(node.operand)
    return None


def _render_number(value: float) -> str:
    return f"{value:g}"


def _affine(node: ast.AST) -> Optional[AffineForm]:
    # Leaf symbolic base.
    if isinstance(node, ast.Name):
        return AffineForm(node.id, 1.0, 0.0)

    # a * expr or expr * a
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = _number(node.left)
        right = _number(node.right)
        if left is not None:
            inner = _affine(node.right)
            if inner is not None:
                return AffineForm(inner.base, inner.scale * left, inner.offset * left)
        if right is not None:
            inner = _affine(node.left)
            if inner is not None:
                return AffineForm(inner.base, inner.scale * right, inner.offset * right)

    # expr + c / expr - c
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        inner = _affine(node.left)
        c = _number(node.right)
        if inner is not None and c is not None:
            if isinstance(node.op, ast.Sub):
                c = -c
            return AffineForm(inner.base, inner.scale, inner.offset + c)

    return None


def canonicalize_affine_expression(expression: str) -> str:
    """Normalize exact affine chains; return original on unsupported syntax."""
    try:
        tree = ast.parse(expression, mode="eval").body
    except (SyntaxError, ValueError):
        return expression

    form = _affine(tree)
    if form is None:
        return expression

    base = form.base
    a = form.scale
    b = form.offset

    if a == 1.0 and b == 0.0:
        return base
    if a == 1.0:
        return f"({base}{'+' if b >= 0 else ''}{_render_number(b)})"
    if b == 0.0:
        return f"({_render_number(a)}*{base})"
    sign = "+" if b >= 0 else ""
    return f"({_render_number(a)}*{base}{sign}{_render_number(b)})"


def canonical_key(expression: str) -> str:
    """Stable exact canonical key used for structural deduplication."""
    return canonicalize_affine_expression(expression)
