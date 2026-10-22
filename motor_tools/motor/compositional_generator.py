"""AD-43 — bounded compositional hypothesis generation.

This module is intentionally separate from candidate selection. It expands a
small deterministic grammar into MutationCandidates.
No domain-specific finished expression is encoded in the generator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from motor_tools.motor.mutagen import MutationCandidate


@dataclass(frozen=True)
class GeneratedNode:
    expression: str
    transform: Callable[[np.ndarray], float]
    complexity: int
    family: str


class CompositionalHypothesisGenerator:
    """Generate a bounded layered symbolic closure.

    Grammar (informal):

        state -> unary(state)
        unary(state) -> affine(unary(state))
        affine(unary(state)) -> unary(affine(unary(state)))

    The affine constants are supplied as a finite configurable grid. The
    generator is deliberately not recursive beyond these layers, keeping the
    candidate surface measurable and deterministic.
    """

    def __init__(
        self,
        *,
        max_depth: int = 4,
        max_candidates: int = 8192,
        constants: Sequence[float] = (
            -2.0, -1.0, -0.5, -0.2, -0.1, -0.05,
            0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0,
        ),
    ):
        self.max_depth = max(1, int(max_depth))
        self.max_candidates = max(1, int(max_candidates))
        self.constants = tuple(float(x) for x in constants)
        if not self.constants:
            raise ValueError("constants must not be empty")

    @staticmethod
    def _identity(index: int) -> Callable[[np.ndarray], float]:
        def fn(state: np.ndarray) -> float:
            s = np.asarray(state, dtype=np.float64).reshape(-1)
            return float(s[index])

        return fn

    @staticmethod
    def _safe_sqrt(x: float) -> float:
        return float(np.sqrt(max(0.0, x)))

    @staticmethod
    def _safe_exp(x: float) -> float:
        return float(np.exp(np.clip(x, -6.0, 6.0)))

    @staticmethod
    def _safe_log1p_abs(x: float) -> float:
        return float(np.log1p(abs(x)))

    @staticmethod
    def _unary_specs():
        return (
            ("sin", np.sin, "trig"),
            ("cos", np.cos, "trig"),
            ("tanh", np.tanh, "bounded"),
            ("exp", CompositionalHypothesisGenerator._safe_exp, "exp"),
            ("abs", np.abs, "abs"),
            ("sqrt", CompositionalHypothesisGenerator._safe_sqrt, "root"),
            ("log1p_abs", CompositionalHypothesisGenerator._safe_log1p_abs, "log"),
            ("square", lambda x: x * x, "power"),
        )

    def apply_unary(self, node: GeneratedNode):
        out = []
        for name, op, family in self._unary_specs():
            expr = f"{name}({node.expression})"
            fn = lambda state, node=node, op=op: float(op(node.transform(state)))
            out.append(GeneratedNode(expr, fn, node.complexity + 1, f"{family}_compose"))
        return out

    def apply_affine(self, node: GeneratedNode):
        out = []
        for c in self.constants:
            sign = "+" if c >= 0.0 else ""
            c_expr = f"{c:g}"
            expr_add = f"({node.expression}{sign}{c_expr})"
            add_fn = lambda state, node=node, c=c: float(node.transform(state) + c)
            out.append(GeneratedNode(expr_add, add_fn, node.complexity + 1, "affine_add"))

            expr_mul = f"({c_expr}*{node.expression})"
            mul_fn = lambda state, node=node, c=c: float(c * node.transform(state))
            out.append(GeneratedNode(expr_mul, mul_fn, node.complexity + 1, "affine_scale"))
        return out

    def seed_nodes(self, state_dim: int) -> tuple[GeneratedNode, ...]:
        return tuple(
            GeneratedNode(f"s{i}", self._identity(i), 1, "state")
            for i in range(int(state_dim))
        )

    def expand(self, nodes: Sequence[GeneratedNode]) -> tuple[GeneratedNode, ...]:
        """Expand one symbolic step without scoring or truncating."""
        out = []
        for node in nodes:
            out.extend(self.apply_unary(node))
            out.extend(self.apply_affine(node))
        return tuple(out)

    def generate(
        self,
        state_dim: int,
        active_terms: Sequence[str] = (),
    ) -> tuple[MutationCandidate, ...]:
        active = set(active_terms)
        by_expression: dict[str, MutationCandidate] = {}

        def add(node: GeneratedNode) -> None:
            if node.expression in active or node.expression in by_expression:
                return
            by_expression[node.expression] = MutationCandidate(
                name=node.expression,
                expression=node.expression,
                complexity=node.complexity,
                transform=node.transform,
                family=node.family,
            )

        bases = list(self.seed_nodes(state_dim))

        # Layer 2: primitive unary transforms.
        unary_nodes: list[GeneratedNode] = []
        for base in bases:
            for node in self.apply_unary(base):
                add(node)
                unary_nodes.append(node)

        if self.max_depth < 3:
            return self._materialize(by_expression)

        # Layer 3: affine wrappers around unary terms.
        affine_nodes: list[GeneratedNode] = []
        for node in unary_nodes:
            for wrapped in self.apply_affine(node):
                add(wrapped)
                affine_nodes.append(wrapped)

        if self.max_depth < 4:
            return self._materialize(by_expression)

        # Layer 4: outer unary transforms. This is enough to reach structures
        # such as sqrt(abs(s1)+c) without embedding that finished expression.
        for node in affine_nodes:
            for wrapped in self.apply_unary(node):
                add(wrapped)

        return self._materialize(by_expression)

    def _materialize(
        self, by_expression: dict[str, MutationCandidate]
    ) -> tuple[MutationCandidate, ...]:
        candidates = sorted(
            by_expression.values(),
            key=lambda c: (c.complexity, c.family, c.expression),
        )
        return tuple(candidates[: self.max_candidates])
