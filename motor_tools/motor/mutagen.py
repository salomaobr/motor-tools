"""MUTAGEN v3 — geração e priorização adaptativa de hipóteses estruturais.

A linguagem de mutação permanece pequena, determinística e extensível. A v3
adiciona uma política de priorização que aprende com resultados observados:
hipóteses que repetidamente produzem ganho OOS sobem na fila; hipóteses que
falham perdem prioridade, sem alterar o critério de aceitação da descoberta.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
import math
import numpy as np


@dataclass(frozen=True)
class MutationCandidate:
    name: str
    expression: str
    complexity: int
    transform: Callable[[np.ndarray], float]
    family: str = "unknown"


@dataclass
class MutationStat:
    trials: int = 0
    successes: int = 0
    gain_sum: float = 0.0
    gain_max: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.trials)

    @property
    def mean_gain(self) -> float:
        return self.gain_sum / max(1, self.trials)


class MutationPolicy:
    """Prioridade leve, determinística e persistível em memória.

    A política usa UCB sobre a expressão exata. O termo de exploração impede
    que uma descoberta antiga congele o espaço de busca.
    """

    def __init__(self, exploration: float = 0.35):
        self.exploration = max(0.0, float(exploration))
        self._stats: dict[str, MutationStat] = {}
        self._total_trials = 0

    def _stat(self, expression: str) -> MutationStat:
        return self._stats.setdefault(expression, MutationStat())

    def score(self, candidate: MutationCandidate) -> float:
        stat = self._stats.get(candidate.expression)
        if stat is None or stat.trials == 0:
            # Empate determinístico: candidatos inéditos recebem a mesma base.
            return self.exploration
        confidence = math.sqrt(
            math.log(self._total_trials + 1.0) / max(1, stat.trials)
        )
        # Sucesso importa, mas ganho médio também diferencia bons candidatos.
        return 0.70 * stat.success_rate + 0.30 * max(0.0, stat.mean_gain) + self.exploration * confidence

    def order(self, candidates: Sequence[MutationCandidate]) -> tuple[MutationCandidate, ...]:
        indexed = list(enumerate(candidates))
        indexed.sort(key=lambda item: (-self.score(item[1]), item[0]))
        return tuple(candidate for _, candidate in indexed)

    def observe(self, candidate: MutationCandidate, gain: float, accepted: bool) -> None:
        stat = self._stat(candidate.expression)
        stat.trials += 1
        stat.gain_sum += max(0.0, float(gain))
        stat.gain_max = max(stat.gain_max, max(0.0, float(gain)))
        if accepted:
            stat.successes += 1
        self._total_trials += 1

    def snapshot(self) -> Mapping[str, MutationStat]:
        return {k: MutationStat(v.trials, v.successes, v.gain_sum, v.gain_max) for k, v in self._stats.items()}


@dataclass
class ActiveSearchStat:
    trials: int = 0
    successes: int = 0
    gain_sum: float = 0.0

    @property
    def mean_gain(self) -> float:
        return self.gain_sum / max(1, self.trials)


class ActiveCompositionalSearch:
    """Selecionador ativo de candidatos composicionais com orçamento explícito.

    A seleção usa apenas evidência acumulada na busca interna. Novidade estrutural
    é medida por expressão/família nunca observada; incerteza usa um termo UCB;
    ganho observado favorece candidatos úteis; complexidade recebe penalidade.
    Empates são resolvidos pela expressão, tornando a ordenação independente da
    ordem de entrada.
    """

    def __init__(
        self,
        budget: int = 8,
        exploration: float = 0.35,
        novelty_weight: float = 0.30,
        gain_weight: float = 0.45,
        complexity_weight: float = 0.15,
    ):
        self.budget = max(1, int(budget))
        self.exploration = max(0.0, float(exploration))
        self.novelty_weight = max(0.0, float(novelty_weight))
        self.gain_weight = max(0.0, float(gain_weight))
        self.complexity_weight = max(0.0, float(complexity_weight))
        self._stats: dict[str, ActiveSearchStat] = {}
        self._seen_families: set[str] = set()
        self._total_trials = 0

    def _score(self, candidate: MutationCandidate) -> float:
        stat = self._stats.get(candidate.expression)
        if stat is None or stat.trials == 0:
            uncertainty = 1.0
            mean_gain = 0.0
        else:
            uncertainty = min(1.0, math.sqrt(
                math.log(self._total_trials + 1.0) / max(1, stat.trials)
            ))
            mean_gain = max(0.0, stat.mean_gain)
        success_rate = 0.0 if stat is None or stat.trials == 0 else stat.successes / max(1, stat.trials)
        # Ganho isolado pode ser ruído: um candidato que melhora CV mas falha
        # no gate de aceitação não deve acumular a mesma prioridade de um
        # candidato que repetidamente passa pelo gate.
        effective_gain = mean_gain * success_rate
        novelty = 1.0 if candidate.family not in self._seen_families else 0.0
        # Bounded in [0,1] so complexity cannot dominate arbitrary gain scales.
        complexity = 1.0 / max(1.0, float(candidate.complexity))
        return (
            self.novelty_weight * novelty
            + self.gain_weight * min(1.0, effective_gain)
            + self.exploration * uncertainty
            + self.complexity_weight * complexity
        )

    def rank(self, candidates: Sequence[MutationCandidate]) -> tuple[MutationCandidate, ...]:
        unique = {c.expression: c for c in candidates}
        ranked = sorted(
            unique.values(),
            key=lambda c: (-self._score(c), c.expression),
        )
        return tuple(ranked)

    def select(
        self,
        candidates: Sequence[MutationCandidate],
        budget: int | None = None,
    ) -> tuple[MutationCandidate, ...]:
        limit = self.budget if budget is None else max(1, int(budget))
        return self.rank(candidates)[:limit]

    def observe(self, candidate: MutationCandidate, gain: float, accepted: bool) -> None:
        stat = self._stats.setdefault(candidate.expression, ActiveSearchStat())
        stat.trials += 1
        stat.gain_sum += max(0.0, float(gain))
        if accepted:
            stat.successes += 1
        self._total_trials += 1
        self._seen_families.add(candidate.family)

    def snapshot(self) -> Mapping[str, ActiveSearchStat]:
        return {
            k: ActiveSearchStat(v.trials, v.successes, v.gain_sum)
            for k, v in self._stats.items()
        }


class Mutagen:
    """Propõe primitivas e composições de funções sem conhecer o domínio."""

    def __init__(self, max_candidates: int = 96, policy: MutationPolicy | None = None, composition_depth: int = 1, active_search: ActiveCompositionalSearch | None = None, active_budget: int | None = None):
        self.max_candidates = max(1, int(max_candidates))
        self.policy = policy
        # Depth 1 preserves the historical candidate surface. Larger depths
        # enable bounded symbolic closure without allowing an unbounded search.
        self.composition_depth = max(1, int(composition_depth))
        self.active_search = active_search
        self.active_budget = None if active_budget is None else max(1, int(active_budget))

    @staticmethod
    def _unary(expr: str, fn: Callable[[float], float], idx: int) -> Callable[[np.ndarray], float]:
        def wrapped(state: np.ndarray) -> float:
            s = np.asarray(state, dtype=np.float64).reshape(-1)
            return float(fn(float(s[idx])))
        wrapped.__name__ = expr
        return wrapped

    @staticmethod
    def _binary(expr: str, fn: Callable[[float, float], float], i: int, j: int) -> Callable[[np.ndarray], float]:
        def wrapped(state: np.ndarray) -> float:
            s = np.asarray(state, dtype=np.float64).reshape(-1)
            return float(fn(float(s[i]), float(s[j])))
        wrapped.__name__ = expr
        return wrapped

    def propose(
        self,
        active_terms: Sequence[str],
        state_dim: int,
        policy: MutationPolicy | None = None,
    ) -> tuple[MutationCandidate, ...]:
        active = set(active_terms)
        out: list[MutationCandidate] = []
        policy = policy or self.policy

        def add(
            name: str,
            expression: str,
            complexity: int,
            transform: Callable[[np.ndarray], float],
            family: str,
        ) -> None:
            if len(out) >= self.max_candidates or expression in active or any(c.expression == expression for c in out):
                return
            out.append(MutationCandidate(name, expression, complexity, transform, family))

        unary = (
            ("sin", np.sin, 2, "trig"),
            ("cos", np.cos, 2, "trig"),
            ("tanh", np.tanh, 2, "bounded"),
            ("exp", lambda x: np.exp(np.clip(x, -6.0, 6.0)), 2, "exp"),
            ("log1p_abs", lambda x: np.log1p(abs(x)), 2, "log"),
        )

        # Primeiro primitivas simples por variável para garantir cobertura.
        for i in range(int(state_dim)):
            for name, fn, complexity, family in unary:
                expr = f"{name}(s{i})"
                add(f"{name}_s{i}", expr, complexity, self._unary(expr, fn, i), family)

        # Depois composições unárias.
        for i in range(int(state_dim)):
            comps = (
                ("sin_sq", "sin(s{i}^2)", lambda x: np.sin(x * x), 3, "trig_poly"),
                ("cos_sq", "cos(s{i}^2)", lambda x: np.cos(x * x), 3, "trig_poly"),
                ("tanh_sq", "tanh(s{i}^2)", lambda x: np.tanh(x * x), 3, "bounded_poly"),
                ("exp_abs_neg", "exp(-abs(s{i}))", lambda x: np.exp(-abs(x)), 3, "exp_abs"),
                ("sin_tanh", "sin(tanh(s{i}))", lambda x: np.sin(np.tanh(x)), 3, "trig_bounded"),
                ("tanh_sin", "tanh(sin(s{i}))", lambda x: np.tanh(np.sin(x)), 3, "bounded_trig"),
                ("exp_half", "exp(0.5*s{i})", lambda x: np.exp(np.clip(0.5 * x, -6.0, 6.0)), 3, "exp_scaled"),
                ("exp_15", "exp(1.5*s{i})", lambda x: np.exp(np.clip(1.5 * x, -6.0, 6.0)), 3, "exp_scaled"),
                ("exp_2", "exp(2*s{i})", lambda x: np.exp(np.clip(2.0 * x, -6.0, 6.0)), 3, "exp_scaled"),
            )
            for name, tmpl, fn, complexity, family in comps:
                expr = tmpl.format(i=i)
                add(f"{name}_s{i}", expr, complexity, self._unary(expr, fn, i), family)
            add(
                f"s{i}_sin_s{i}", f"s{i}*sin(s{i})", 3,
                self._unary(f"s{i}*sin(s{i})", lambda x: x * np.sin(x), i), "self_trig"),
            add(
                f"s{i}_tanh_s{i}", f"s{i}*tanh(s{i})", 3,
                self._unary(f"s{i}*tanh(s{i})", lambda x: x * np.tanh(x), i), "self_bounded"),
            add(
                f"s{i}_exp_s{i}", f"s{i}*exp(s{i})", 3,
                self._unary(f"s{i}*exp(s{i})", lambda x: x * np.exp(np.clip(x, -6.0, 6.0)), i), "self_exp"),

        # Por fim, interações entre variáveis.
        for i in range(int(state_dim)):
            for j in range(i + 1, int(state_dim)):
                add(
                    f"sin_s{i}_times_s{j}", f"sin(s{i})*s{j}", 4,
                    self._binary(f"sin(s{i})*s{j}", lambda a, b: np.sin(a) * b, i, j), "cross_trig"),
                add(
                    f"tanh_s{i}_times_s{j}", f"tanh(s{i})*s{j}", 4,
                    self._binary(f"tanh(s{i})*s{j}", lambda a, b: np.tanh(a) * b, i, j), "cross_bounded"),
                add(
                    f"exp_s{i}_times_s{j}", f"exp(s{i})*s{j}", 4,
                    self._binary(f"exp(s{i})*s{j}", lambda a, b: np.exp(np.clip(a, -6.0, 6.0)) * b, i, j), "cross_exp"),
                add(
                    f"sin_product_s{i}_s{j}", f"sin(s{i}*s{j})", 4,
                    self._binary(f"sin(s{i}*s{j})", lambda a, b: np.sin(a * b), i, j), "cross_product"),

        # Bounded symbolic closure.  These primitives are deliberately small,
        # protected, and composable; the search depth—not a hand-written list
        # of finished expressions—controls how far discovery can expand.
        if self.composition_depth > 1:
            primitive_specs = (
                ("sin", lambda x: np.sin(x), "trig"),
                ("cos", lambda x: np.cos(x), "trig"),
                ("tanh", lambda x: np.tanh(x), "bounded"),
                ("exp", lambda x: np.exp(np.clip(x, -6.0, 6.0)), "exp"),
                ("abs", lambda x: np.abs(x), "abs"),
                ("sqrt_abs", lambda x: np.sqrt(np.abs(x) + 1e-9), "root"),
                ("log1p_abs", lambda x: np.log1p(np.abs(x)), "log"),
            )
            unary_exprs = [(f"s{i}", self._unary(f"s{i}", lambda x, i=i: x, i), 1) for i in range(int(state_dim))]
            seen = {c.expression for c in out}
            frontier = unary_exprs
            for depth in range(2, self.composition_depth + 1):
                next_frontier = []
                for base_expr, base_fn, base_complexity in frontier:
                    for name, op, family in primitive_specs:
                        expr = f"{name}({base_expr})"
                        if expr in seen or expr in active:
                            continue
                        def composed(state, base_fn=base_fn, op=op):
                            return float(op(float(base_fn(state))))
                        composed.__name__ = expr
                        complexity = base_complexity + 1
                        add(expr, expr, complexity, composed, f"composed_{family}")
                        if expr not in seen:
                            seen.add(expr)
                            next_frontier.append((expr, composed, complexity))
                frontier = next_frontier
                if not frontier or len(out) >= self.max_candidates:
                    break

        candidates = tuple(out[: self.max_candidates])
        if self.active_search is not None:
            return self.active_search.select(candidates, budget=self.active_budget)
        return policy.order(candidates) if policy is not None else candidates

    @staticmethod
    def score_extension(residual_before: np.ndarray, residual_after: np.ndarray) -> float:
        before = float(np.mean(np.abs(residual_before)))
        after = float(np.mean(np.abs(residual_after)))
        return (before - after) / max(before, 1e-9)
