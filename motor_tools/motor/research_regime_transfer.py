"""Explicit regime-transfer handler for the motor-tools research layer.

Evaluation-only. It runs a registered experiment selected by the caller after
convergent holdout evidence and returns observations; it never mutates runtime.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

from motor_tools.motor.agi_transfer_robustness import run_robustness_battery
from motor_tools.motor.research_scientist import ExperimentObservation


def run_regime_transfer(seeds: Sequence[int] = (13001, 13019, 13037, 13055, 13073)) -> tuple[ExperimentObservation, ...]:
    observations: list[ExperimentObservation] = []
    for seed in seeds:
        summary = run_robustness_battery(seed=seed)
        observations.append(
            ExperimentObservation(
                experiment="regime_transfer",
                hypothesis="H2",
                status="completed",
                evidence=f"independent regime-shift robustness battery seed={seed}",
                metrics=(
                    ("retention_rate", float(summary.retention_rate)),
                    ("recovery_rate", float(summary.recovery_rate)),
                    ("mean_pre_shift_mae", float(summary.mean_pre_shift_mae)),
                    ("mean_shifted_zero_shot_mae", float(summary.mean_shifted_zero_shot_mae)),
                    ("mean_recovered_mae", float(summary.mean_recovered_mae)),
                ),
            )
        )
    return tuple(observations)
