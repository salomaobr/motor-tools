"""Sequence-level attribution of world-shift and observation-transform evidence.

Experimental/lab-only. This component deliberately aggregates already-computed
boolean evidence over a fixed observation sequence. It does not assign weights,
fit thresholds from TEST data, import domains, or modify official gates.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalAttributionState:
    transform_rate: float
    world_shift_rate: float
    label: str


class TemporalAttribution:
    """Aggregate per-window evidence with a frozen majority threshold."""

    def __init__(self, majority: float = 0.50):
        if not 0.0 < majority <= 1.0:
            raise ValueError("majority must be in (0, 1]")
        self.majority = float(majority)

    def classify(self, transform_evidence: list[bool], world_shift_evidence: list[bool]) -> TemporalAttributionState:
        if len(transform_evidence) != len(world_shift_evidence) or not transform_evidence:
            raise ValueError("evidence sequences must be non-empty and equally sized")
        tr = sum(bool(x) for x in transform_evidence) / len(transform_evidence)
        wr = sum(bool(x) for x in world_shift_evidence) / len(world_shift_evidence)
        transform = tr >= self.majority
        world = wr >= self.majority
        # Conservative attribution: once a sensor transformation is present,
        # any independent temporal world-shift evidence forces abstention.
        if transform and wr > 0.0:
            label = "AMBIGUOUS"
        elif transform:
            label = "OBSERVATION_TRANSFORM"
        elif world:
            label = "WORLD_SHIFT"
        else:
            label = "NORMAL"
        return TemporalAttributionState(tr, wr, label)
