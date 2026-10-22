"""Experimental gate for novelty evidence under insufficient temporal information.

The gate is intentionally categorical: when temporal information is insufficient,
raw novelty is not emitted as strong WORLD_SHIFT evidence. This does not erase
novelty internally; it changes only the externally exposed evidence flag.

Lab-only. No domain imports. No action selection.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoveltyEvidenceDecision:
    raw_novelty: bool
    temporal_information_insufficient: bool
    emit_strong_world_shift_evidence: bool
    abstained: bool


def gate_novelty_evidence(
    *, raw_novelty: bool, temporal_information_insufficient: bool
) -> NoveltyEvidenceDecision:
    raw = bool(raw_novelty)
    temporal_low = bool(temporal_information_insufficient)
    abstain = temporal_low and raw
    return NoveltyEvidenceDecision(
        raw_novelty=raw,
        temporal_information_insufficient=temporal_low,
        emit_strong_world_shift_evidence=raw and not temporal_low,
        abstained=abstain,
    )
