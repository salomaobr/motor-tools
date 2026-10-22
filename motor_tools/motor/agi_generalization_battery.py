"""Compositional generalization evaluation battery (standalone, no agent runtime).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from motor_tools.motor.compositional_frontier import CompositionalFrontierSearch
from motor_tools.motor.compositional_generator import GeneratedNode


@dataclass(frozen=True)
class GeneralizationTask:
    task_id: str
    family: str
    train_x: np.ndarray
    train_y: np.ndarray
    holdout_x: np.ndarray
    holdout_y: np.ndarray
    transfer_x: np.ndarray
    transfer_y: np.ndarray
    target_expression: str
    target_transform: Callable[[np.ndarray], float]


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    family: str
    train_mae: float
    holdout_mae: float
    transfer_mae: float
    holdout_generalized: bool
    transfer_generalized: bool
    selected_expression: str
    explored: int


@dataclass(frozen=True)
class BatterySummary:
    tasks: int
    holdout_passes: int
    transfer_passes: int
    mean_holdout_mae: float
    mean_transfer_mae: float
    results: tuple[TaskResult, ...]

    @property
    def holdout_rate(self) -> float:
        return self.holdout_passes / self.tasks if self.tasks else 0.0

    @property
    def transfer_rate(self) -> float:
        return self.transfer_passes / self.tasks if self.tasks else 0.0


def _fit_affine(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    design = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(coef[0]), float(coef[1])


def _mae(node: GeneratedNode, x: np.ndarray, y: np.ndarray, coef: tuple[float, float]) -> float:
    a, b = coef
    pred = np.asarray([node.transform(row) for row in x], dtype=np.float64)
    return float(np.mean(np.abs(a * pred + b - y)))


def _sample(rng: np.random.Generator, n: int, *, shift: float = 0.0, scale: float = 1.0) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=(n, 2)) * scale + shift


def _make_task(task_id: str, family: str, expression: str, fn: Callable[[float], float], rng: np.random.Generator) -> GeneralizationTask:
    train_x = _sample(rng, 48)
    holdout_x = _sample(rng, 48)
    transfer_x = _sample(rng, 48, shift=0.35, scale=0.65)
    target = lambda x: float(fn(float(x[0]), float(x[1])))
    train_y = np.asarray([target(x) for x in train_x])
    holdout_y = np.asarray([target(x) for x in holdout_x])
    transfer_y = np.asarray([target(x) for x in transfer_x])
    return GeneralizationTask(task_id, family, train_x, train_y, holdout_x, holdout_y,
                              transfer_x, transfer_y, expression, target)


def default_tasks(seed: int = 4301) -> tuple[GeneralizationTask, ...]:
    """Create a fixed but target-hidden battery with held-out compositions."""
    rng = np.random.default_rng(seed)
    return (
        _make_task("alg_cos_abs", "algebraic", "cos(abs(s0)+0.2)",
                    lambda x, _: np.cos(abs(x) + 0.2), rng),
        _make_task("alg_tanh_sq", "algebraic", "tanh(square(s0)+0.2)",
                    lambda x, _: np.tanh(x * x + 0.2), rng),
        _make_task("alg_log_abs", "algebraic", "log1p(abs(s1)+0.1)",
                    lambda _, y: np.log1p(abs(y) + 0.1), rng),
        _make_task("compose_sin_abs", "compositional", "sin(abs(s0)+0.15)",
                    lambda x, _: np.sin(abs(x) + 0.15), rng),
        _make_task("compose_sqrt_abs", "compositional", "sqrt(abs(s1)+0.05)",
                    lambda _, y: np.sqrt(abs(y) + 0.05), rng),
        _make_task("compose_exp_shift", "compositional", "exp(0.5*s0+0.2)",
                    lambda x, _: np.exp(0.5 * x + 0.2), rng),
        _make_task("cross_cos_square", "cross_family", "cos(square(s0))",
                    lambda x, _: np.cos(x * x), rng),
        _make_task("cross_tanh_abs", "cross_family", "tanh(abs(s1)+0.2)",
                    lambda _, y: np.tanh(abs(y) + 0.2), rng),
    )


def _score_factory(node: GeneratedNode, x: np.ndarray, y: np.ndarray) -> tuple[float, tuple[float, float]]:
    pred = np.asarray([node.transform(row) for row in x], dtype=np.float64)
    coef = _fit_affine(pred, y)
    return _mae(node, x, y, coef), coef


def solve_task(task: GeneralizationTask, *, beam_width: int = 8, budget: int = 8, max_pool: int = 8192) -> TaskResult:
    search = CompositionalFrontierSearch(
        beam_width=beam_width,
        budget=budget,
        max_pool=max_pool,
        feature_contract="linear_intercept",
    )
    fitted: dict[str, tuple[float, float]] = {}

    def score(node: GeneratedNode) -> float:
        value, coef = _score_factory(node, task.train_x, task.train_y)
        fitted[node.expression] = coef
        return value

    result = search.search(2, score)
    if not result.candidates:
        raise RuntimeError(f"no candidates for {task.task_id}")
    selected = result.candidates[0]
    coef = fitted[selected.expression]
    train_mae = _mae(selected, task.train_x, task.train_y, coef)
    holdout_mae = _mae(selected, task.holdout_x, task.holdout_y, coef)
    transfer_mae = _mae(selected, task.transfer_x, task.transfer_y, coef)
    # The tolerance is deliberately small but not zero; this battery is about
    # generalization, not exact symbolic recovery.
    holdout_ok = holdout_mae <= 1e-3
    transfer_ok = transfer_mae <= 1e-2
    return TaskResult(task.task_id, task.family, train_mae, holdout_mae, transfer_mae,
                      holdout_ok, transfer_ok, selected.expression, result.explored)


def run_battery(tasks: Sequence[GeneralizationTask] | None = None) -> BatterySummary:
    tasks = tuple(tasks or default_tasks())
    results = tuple(solve_task(task) for task in tasks)
    return BatterySummary(
        tasks=len(results),
        holdout_passes=sum(r.holdout_generalized for r in results),
        transfer_passes=sum(r.transfer_generalized for r in results),
        mean_holdout_mae=float(np.mean([r.holdout_mae for r in results])),
        mean_transfer_mae=float(np.mean([r.transfer_mae for r in results])),
        results=results,
    )
