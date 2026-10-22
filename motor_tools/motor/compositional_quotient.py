"""AD-43B — exact quotient for linear-model feature invariance.

This module is experimental and self-contained.  The
quotient is justified only for a downstream linear model with an intercept:
non-zero outer scale and additive offset of a feature are absorbed by the
learned coefficient/intercept.  No numeric tolerance or target expression is
used.
"""
from __future__ import annotations

import ast
from typing import Optional, Tuple

from motor_tools.motor.compositional_canonical import canonicalize_affine_expression


def _number(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _number(node.operand)
        return None if value is None else -value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _number(node.operand)
    return None


def _outer_affine(node: ast.AST) -> Optional[Tuple[float, ast.AST, float]]:
    """Return (scale, core, offset) for an exact outer a*f+b wrapper."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = _number(node.left)
        right = _number(node.right)
        if left is not None:
            inner = _outer_affine(node.right)
            if inner is not None:
                a, core, b = inner
                return left * a, core, left * b
            return left, node.right, 0.0
        if right is not None:
            inner = _outer_affine(node.left)
            if inner is not None:
                a, core, b = inner
                return right * a, core, right * b
            return right, node.left, 0.0
        return None

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        right_num = _number(node.right)
        if right_num is not None:
            inner = _outer_affine(node.left)
            if inner is not None:
                a, core, b = inner
                return a, core, b + (right_num if isinstance(node.op, ast.Add) else -right_num)
            return 1.0, node.left, right_num if isinstance(node.op, ast.Add) else -right_num

        left_num = _number(node.left)
        if left_num is not None and isinstance(node.op, ast.Add):
            inner = _outer_affine(node.right)
            if inner is not None:
                a, core, b = inner
                return a, core, b + left_num
            return 1.0, node.right, left_num
    return None


def _unparse(node: ast.AST) -> str:
    return ast.unparse(node)


def canonicalize_linear_feature(expression: str) -> str:
    """Canonicalize exact feature equivalence under downstream linear regression.

    For non-affine expressions, only the *outer* non-zero affine wrapper is
    quotiented: a*f+b -> f.  Pure affine expressions continue to use AD-43's
    exact affine canonicalizer.  Expressions with a zero feature scale are not
    quotiented because they collapse the feature to a constant.
    """
    try:
        tree = ast.parse(expression, mode="eval").body
    except (SyntaxError, ValueError):
        return expression

    # Preserve the stronger exact canonicalization for pure affine forms.
    affine = canonicalize_affine_expression(expression)
    if affine != expression or _is_pure_affine(tree):
        return affine

    outer = _outer_affine(tree)
    if outer is None:
        # No functional quotient applies, but normalize equivalent AST
        # spelling so quotient keys remain stable across generator formatting.
        return _unparse(tree)
    scale, core, _offset = outer
    if scale == 0.0:
        return expression

    # Do not alter the structure inside the nonlinear core. This keeps the
    # quotient exact and avoids recursively inventing functional identities.
    return _unparse(core)


def _is_pure_affine(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return (_number(node.left) is not None and _is_pure_affine(node.right)) or (
            _number(node.right) is not None and _is_pure_affine(node.left)
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return _is_pure_affine(node.left) and _number(node.right) is not None
    return False


def quotient_key(expression: str) -> str:
    """Stable exact quotient key for feature-space deduplication."""
    return canonicalize_linear_feature(expression)
