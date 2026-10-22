"""AD-43 — bounded compositional frontier search with a strict scoring budget."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import math

from motor_tools.motor.compositional_generator import CompositionalHypothesisGenerator, GeneratedNode
from motor_tools.motor.compositional_quotient import quotient_key


@dataclass(frozen=True)
class FrontierSearchResult:
    candidates: tuple[GeneratedNode, ...]
    primitive_candidates: tuple[GeneratedNode, ...]
    explored: int


class CompositionalFrontierSearch:
    """Rank candidates while guaranteeing score_fn is called at most ``budget`` times.

    ``explored`` is the exact number of score_fn calls, including calls returning
    NaN/inf. A candidate is scored at most once per search.
    """
    def __init__(self, generator: CompositionalHypothesisGenerator | None = None, *,
                 beam_width: int = 8, budget: int = 8, max_pool: int = 8192,
                 feature_contract: str = "raw", functional_quotient: bool | None = None):
        self.generator = generator or CompositionalHypothesisGenerator()
        self.beam_width = max(1, int(beam_width))
        self.budget = max(1, int(budget))
        self.max_pool = max(self.budget, int(max_pool))
        if feature_contract not in {"raw", "linear_intercept"}:
            raise ValueError("feature_contract must be 'raw' or 'linear_intercept'")
        if functional_quotient is True and feature_contract != "linear_intercept":
            raise ValueError("functional_quotient=True requires feature_contract='linear_intercept'")
        if functional_quotient is False and feature_contract == "linear_intercept":
            raise ValueError("functional_quotient=False conflicts with feature_contract='linear_intercept'")
        self.feature_contract = feature_contract
        self.functional_quotient = feature_contract == "linear_intercept"

    def _dedupe(self, nodes: Sequence[GeneratedNode]) -> tuple[GeneratedNode, ...]:
        out, seen = [], set()
        for node in nodes:
            key = quotient_key(node.expression) if self.functional_quotient else node.expression
            if key not in seen:
                seen.add(key)
                out.append(node)
        return tuple(out)

    @staticmethod
    def _sort_scored(scored):
        return tuple(sorted(scored, key=lambda item: (item[0], item[1].complexity, item[1].expression)))

    def search(self, state_dim: int, score_fn: Callable[[GeneratedNode], float], *,
               active_terms: Sequence[str] = ()) -> FrontierSearchResult:
        active = set(active_terms)
        seeds = tuple(n for n in self.generator.seed_nodes(state_dim) if n.expression not in active)
        primitive = self._dedupe(n for n in self.generator.expand(seeds) if n.expression not in active)

        # Construct structural candidates without scoring; cap structural memory separately.
        def macro_expand(nodes):
            unary_groups = tuple(self.generator.apply_unary(node) for node in nodes)
            affine_groups = tuple(self.generator.apply_affine(node) for node in nodes)
            flat_unary = tuple(c for group in unary_groups for c in group)
            flat_affine = tuple(c for group in affine_groups for c in group)
            outer_unary = tuple(child for node in flat_unary for child in self.generator.apply_unary(node))
            outer_affine = tuple(child for node in flat_affine for child in self.generator.apply_unary(node))
            return flat_unary + flat_affine + outer_unary + outer_affine

        # Reserve a portion of the hard budget for primitive ranking, which selects
        # parents. The remaining budget is available to score the full frontier.
        primitive_quota = min(len(primitive), max(1, self.budget // 2))
        scored_by_key = {}
        calls = 0

        def score_nodes(nodes, limit):
            nonlocal calls
            scored = []
            for node in nodes:
                if calls >= self.budget or len(scored) >= limit:
                    break
                key = quotient_key(node.expression) if self.functional_quotient else node.expression
                if key in scored_by_key:
                    scored.append(scored_by_key[key])
                    continue
                if calls >= self.budget:
                    break
                calls += 1
                try:
                    value = float(score_fn(node))
                except (ValueError, ArithmeticError, FloatingPointError):
                    continue
                if not math.isfinite(value):
                    continue
                pair = (value, node)
                scored_by_key[key] = pair
                scored.append(pair)
            return self._sort_scored(scored)

        primitive_ranked = score_nodes(primitive, primitive_quota)
        parents = tuple(node for _, node in primitive_ranked[:self.beam_width])
        direct_one = primitive
        direct_two = macro_expand(seeds)
        parent_one = tuple(n for n in self.generator.expand(parents) if n.expression not in active)
        parent_two = macro_expand(parents)
        pool = self._dedupe(tuple(n for n in direct_one + direct_two + parent_one + parent_two
                                  if n.expression not in active))
        if len(pool) > self.max_pool:
            pool = tuple(sorted(pool, key=lambda n: (n.complexity, n.family, n.expression))[:self.max_pool])

        # Score remaining frontier nodes, reusing cached primitive scores and never
        # invoking score_fn more than budget times total.
        frontier_ranked = score_nodes(pool, self.budget)
        combined = {}
        for value, node in tuple(scored_by_key.values()) + tuple(frontier_ranked):
            key = quotient_key(node.expression) if self.functional_quotient else node.expression
            combined[key] = (value, node)
        ranked = self._sort_scored(tuple(combined.values()))
        selected = tuple(node for _, node in ranked[:self.budget])
        return FrontierSearchResult(selected, tuple(node for _, node in primitive_ranked), calls)
