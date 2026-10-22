"""Experimental categorical gate for convergent observation degradation.

This lab-only adapter never combines numeric scores. It counts independent
boolean evidence channels and abstains from exposing raw novelty as strong
WORLD_SHIFT evidence only when enough observation-degradation signals agree.

Signals:
- temporal_information_insufficient
- observability_degraded
- relative_temporal_stale

Default: at least 2 of 3 degradation signals must be active.
This intentionally leaves a lone temporal-insufficiency signal insufficient to
suppress novelty, preserving the possibility of a genuinely slow OOD regime.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConvergentObservationDecision:
    raw_novelty: bool
    temporal_information_insufficient: bool
    observability_degraded: bool
    relative_temporal_stale: bool
    degradation_evidence_count: int
    minimum_degradation_evidence: int
    abstained: bool
    emit_strong_world_shift_evidence: bool


def gate_novelty_with_convergent_observation_evidence(
    *,
    raw_novelty: bool,
    temporal_information_insufficient: bool,
    observability_degraded: bool,
    relative_temporal_stale: bool,
    minimum_degradation_evidence: int = 2,
) -> ConvergentObservationDecision:
    """Gate strong novelty evidence using categorical degradation convergence."""
    minimum = int(minimum_degradation_evidence)
    if minimum < 1 or minimum > 3:
        raise ValueError("minimum_degradation_evidence must be in [1, 3]")

    raw = bool(raw_novelty)
    t = bool(temporal_information_insufficient)
    o = bool(observability_degraded)
    r = bool(relative_temporal_stale)
    count = int(t) + int(o) + int(r)
    abstain = bool(raw and count >= minimum)
    return ConvergentObservationDecision(
        raw_novelty=raw,
        temporal_information_insufficient=t,
        observability_degraded=o,
        relative_temporal_stale=r,
        degradation_evidence_count=count,
        minimum_degradation_evidence=minimum,
        abstained=abstain,
        emit_strong_world_shift_evidence=bool(raw and not abstain),
    )
