"""Active recovery benchmark for cross-domain regime shifts.

Evaluation-only. Compares fixed, random and disagreement-guided sample selection
when recovering a transferred linear structural model after a regime shift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from motor_tools.motor.agi_transfer_battery import TransferTask, _fit, _mae, default_transfer_tasks
from motor_tools.motor.agi_transfer_robustness import _shifted_rows

@dataclass(frozen=True)
class RecoveryResult:
    task_id: str
    policy: str
    recovery_steps: int
    initial_mae: float
    final_mae: float
    gain: float
    recovered: bool

@dataclass(frozen=True)
class RecoverySummary:
    tasks: int
    policies: tuple[str, ...]
    results: tuple[RecoveryResult, ...]

    def by_policy(self, policy: str) -> tuple[RecoveryResult, ...]:
        return tuple(r for r in self.results if r.policy == policy)

    def recovery_rate(self, policy: str) -> float:
        rows = self.by_policy(policy)
        return sum(r.recovered for r in rows) / len(rows) if rows else 0.0

    def mean_steps(self, policy: str) -> float:
        rows = [r for r in self.by_policy(policy) if r.recovered]
        return float(np.mean([r.recovery_steps for r in rows])) if rows else float("inf")


def _candidate_model(rows):
    return _fit(rows)


def _select_disagreement(base_rows, pool, budget):
    chosen = []
    remaining = list(range(len(pool)))
    current = list(base_rows)
    for _ in range(min(budget, len(remaining))):
        if not remaining:
            break
        # Bootstrap two plausible models from deterministic halves of the current
        # evidence. Select the point on which they disagree most.
        if len(current) >= 4:
            a = current[::2]
            b = current[1::2]
            if len(a) >= 2 and len(b) >= 2:
                wa, wb = _candidate_model(a), _candidate_model(b)
                scores = []
                for idx in remaining:
                    row = pool[idx]
                    s, intervention, out = row
                    x = np.r_[s, intervention, 1.0]
                    scores.append((float(np.mean(np.abs(x @ wa - x @ wb))), idx))
                _, best = max(scores, key=lambda t: (t[0], -t[1]))
            else:
                best = remaining[0]
        else:
            best = remaining[0]
        chosen.append(best)
        current.append(pool[best])
        remaining.remove(best)
    return tuple(chosen)


def _evaluate(task: TransferTask, seed: int, policy: str, budget: int = 10) -> RecoveryResult:
    base = task.source_samples + task.target_adaptation
    shifted = _shifted_rows(task, seed)
    initial = _mae(_fit(base), shifted)
    if policy == "fixed":
        indices = tuple(range(min(budget, len(shifted))))
    elif policy == "random":
        rng = np.random.default_rng(seed + 991)
        indices = tuple(int(i) for i in rng.choice(len(shifted), size=min(budget, len(shifted)), replace=False))
    elif policy == "disagreement":
        indices = _select_disagreement(base, shifted, budget)
    else:
        raise ValueError(f"unknown policy: {policy}")
    selected = tuple(shifted[i] for i in indices)
    holdout = tuple(row for i, row in enumerate(shifted) if i not in set(indices))
    final = _mae(_fit(base + selected), holdout)
    gain = initial - final
    recovered = bool(final < initial * 0.80 and final < 0.75)
    return RecoveryResult(task.task_id, policy, len(indices), initial, final, gain, recovered)


def run_active_recovery(tasks: Sequence[TransferTask] | None = None, seed: int = 9101, budget: int = 10) -> RecoverySummary:
    tasks = tuple(tasks or default_transfer_tasks())
    policies = ("fixed", "random", "disagreement")
    results = tuple(_evaluate(task, seed + i * 17, policy, budget) for i, task in enumerate(tasks) for policy in policies)
    return RecoverySummary(len(tasks), policies, results)
