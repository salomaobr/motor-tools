"""Robustness battery for cross-domain transfer.

Evaluation-only. Measures whether a transferred relation survives a target
regime shift and whether a small recovery set can restore performance without
changing the representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from motor_tools.motor.agi_transfer_battery import TransferTask, default_transfer_tasks, _fit, _mae, _collect


@dataclass(frozen=True)
class RobustnessResult:
    task_id: str
    pre_shift_mae: float
    shifted_zero_shot_mae: float
    recovered_mae: float
    retention: float
    recovery_gain: float
    retained: bool
    recovered: bool


@dataclass(frozen=True)
class RobustnessSummary:
    tasks: int
    retained: int
    recovered: int
    mean_pre_shift_mae: float
    mean_shifted_zero_shot_mae: float
    mean_recovered_mae: float
    results: tuple[RobustnessResult, ...]

    @property
    def retention_rate(self) -> float:
        return self.retained / self.tasks if self.tasks else 0.0

    @property
    def recovery_rate(self) -> float:
        return self.recovered / self.tasks if self.tasks else 0.0


def _shifted_rows(task: TransferTask, seed: int) -> list[tuple[np.ndarray, float, np.ndarray]]:
    # Hold the hidden domain fixed but move to a materially different state
    # distribution. This tests regime transfer rather than domain re-identification.
    return _collect(task.target_domain, seed, 56, start=(1.15, 0.65, -1.0))


def solve_robustness(task: TransferTask, seed: int) -> RobustnessResult:
    adapted = _fit(task.source_samples + task.target_adaptation)
    pre = _mae(adapted, task.target_holdout)
    shifted = _shifted_rows(task, seed)
    zero = _mae(adapted, shifted)
    recovery = _fit(task.source_samples + task.target_adaptation + tuple(shifted[:10]))
    recovered = _mae(recovery, tuple(shifted[10:]))
    retention = pre / zero if zero > 0 else 1.0
    recovery_gain = zero - recovered
    retained = bool(zero <= max(pre * 1.50, 0.75))
    recovered_ok = bool(recovered < zero * 0.80 and recovered < 0.75)
    return RobustnessResult(task.task_id, pre, zero, recovered, retention, recovery_gain, retained, recovered_ok)


def run_robustness_battery(tasks: Sequence[TransferTask] | None = None, seed: int = 9101) -> RobustnessSummary:
    tasks = tuple(tasks or default_transfer_tasks())
    results = tuple(solve_robustness(t, seed + i * 17) for i, t in enumerate(tasks))
    return RobustnessSummary(
        tasks=len(results),
        retained=sum(r.retained for r in results),
        recovered=sum(r.recovered for r in results),
        mean_pre_shift_mae=float(np.mean([r.pre_shift_mae for r in results])),
        mean_shifted_zero_shot_mae=float(np.mean([r.shifted_zero_shot_mae for r in results])),
        mean_recovered_mae=float(np.mean([r.recovered_mae for r in results])),
        results=results,
    )
