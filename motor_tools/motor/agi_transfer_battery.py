"""Cross-regime transfer evaluation battery (standalone, no agent runtime).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from motor_tools.domains.alien_physics import AlienPhysicsEnv


@dataclass(frozen=True)
class TransferTask:
    task_id: str
    family: str
    source_domain: str
    target_domain: str
    source_samples: tuple[tuple[np.ndarray, float, np.ndarray], ...]
    target_adaptation: tuple[tuple[np.ndarray, float, np.ndarray], ...]
    target_holdout: tuple[tuple[np.ndarray, float, np.ndarray], ...]
    expected_relation: str


@dataclass(frozen=True)
class TransferResult:
    task_id: str
    family: str
    source_mae: float
    zero_shot_mae: float
    adapted_mae: float
    adaptation_gain: float
    transferred: bool
    selected_relation: str


@dataclass(frozen=True)
class TransferSummary:
    tasks: int
    transferred: int
    mean_zero_shot_mae: float
    mean_adapted_mae: float
    mean_adaptation_gain: float
    results: tuple[TransferResult, ...]

    @property
    def transfer_rate(self) -> float:
        return self.transferred / self.tasks if self.tasks else 0.0


def _collect(world: str, seed: int, n: int, *, start: tuple[float, float, float]) -> list[tuple[np.ndarray, float, np.ndarray]]:
    # Independent one-step interventions prevent trajectory explosion in the
    # deliberately nonlinear world; each transition remains an honest causal
    # sample from the hidden environment.
    env = AlienPhysicsEnv(world, seed=seed)
    rows: list[tuple[np.ndarray, float, np.ndarray]] = []
    for i in range(n):
        state = np.asarray([
            start[0] + float(env.rng.uniform(-0.8, 0.8)),
            start[1] + float(env.rng.uniform(-0.35, 0.35)),
            start[2] if i % 2 == 0 else -start[2],
        ], dtype=np.float64)
        env.reset(tuple(state))
        force = float(env.rng.uniform(-1.0, 1.0))
        old = state.copy()
        out = env.step(force).copy()
        rows.append((old, force, out))
    return rows


def _design(rows: Sequence[tuple[np.ndarray, float, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    # Same structural representation across physical worlds, but the target
    # domain is never exposed to the fitting function.
    X = np.asarray([np.r_[s, a, 1.0] for s, a, _ in rows], dtype=np.float64)
    Y = np.asarray([o for _, _, o in rows], dtype=np.float64)
    return X, Y


def _fit(rows: Sequence[tuple[np.ndarray, float, np.ndarray]]) -> np.ndarray:
    X, Y = _design(rows)
    return np.linalg.pinv(X) @ Y


def _mae(W: np.ndarray, rows: Sequence[tuple[np.ndarray, float, np.ndarray]]) -> float:
    X, Y = _design(rows)
    return float(np.mean(np.abs(X @ W - Y)))


def _make(task_id: str, source: str, target: str, seed: int) -> TransferTask:
    source_rows = _collect(source, seed, 100, start=(0.0, 0.15, 1.0))
    target_rows = _collect(target, seed + 100, 80, start=(0.0, -0.15, 1.0))
    return TransferTask(
        task_id=task_id,
        family="causal_dynamics",
        source_domain=source,
        target_domain=target,
        source_samples=tuple(source_rows),
        target_adaptation=tuple(target_rows[:12]),
        target_holdout=tuple(target_rows[12:]),
        expected_relation="state + intervention -> next_state",
    )


def default_transfer_tasks(seed: int = 4401) -> tuple[TransferTask, ...]:
    """Paired worlds with hidden mechanisms and different target regimes."""
    return (
        _make("gravity_to_property", "gravity", "property_gravity", seed),
        _make("gravity_to_nonlinear", "gravity", "nonlinear", seed + 1),
        _make("property_to_gravity", "property_gravity", "gravity", seed + 2),
        _make("gravity_to_collision", "gravity", "collision", seed + 3),
    )


def solve_transfer(task: TransferTask) -> TransferResult:
    source_model = _fit(task.source_samples)
    zero = _mae(source_model, task.target_holdout)
    source_mae = _mae(source_model, task.source_samples)

    adapted_model = _fit(task.source_samples + task.target_adaptation)
    adapted = _mae(adapted_model, task.target_holdout)
    gain = zero - adapted

    # Transfer means the source representation remains useful and adaptation
    # improves target prediction materially without requiring a new representation.
    transferred = bool(adapted < zero * 0.80 and adapted < 0.75)
    return TransferResult(
        task.task_id,
        task.family,
        source_mae,
        zero,
        adapted,
        gain,
        transferred,
        "state + intervention -> next_state",
    )


def run_transfer_battery(tasks: Sequence[TransferTask] | None = None) -> TransferSummary:
    tasks = tuple(tasks or default_transfer_tasks())
    results = tuple(solve_transfer(task) for task in tasks)
    return TransferSummary(
        tasks=len(results),
        transferred=sum(r.transferred for r in results),
        mean_zero_shot_mae=float(np.mean([r.zero_shot_mae for r in results])),
        mean_adapted_mae=float(np.mean([r.adapted_mae for r in results])),
        mean_adaptation_gain=float(np.mean([r.adaptation_gain for r in results])),
        results=results,
    )
