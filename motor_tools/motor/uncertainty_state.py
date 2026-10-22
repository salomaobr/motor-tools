"""Generic separation of epistemic and observation uncertainty.

This module is intentionally not wired into KAIROS. It provides a small, typed
state object so experiments can keep model uncertainty and measurement noise
separate instead of collapsing them into one confidence number.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class UncertaintyState:
    """Explicit uncertainty decomposition for one prediction context.

    epistemic: uncertainty attributed to model/support limitations.
    aleatoric: uncertainty attributed to observation noise.
    total_rms: root-sum-square combination when both components are expressed
        in the same output units.
    source: provenance label for diagnostics.
    """

    epistemic: float
    aleatoric: float
    total_rms: float
    source: str = "experimental"

    def __post_init__(self) -> None:
        vals = (self.epistemic, self.aleatoric, self.total_rms)
        if not all(np.isfinite(v) and v >= 0.0 for v in vals):
            raise ValueError("uncertainties must be finite and non-negative")
        expected = float(np.hypot(self.epistemic, self.aleatoric))
        if not np.isclose(self.total_rms, expected, rtol=1e-10, atol=1e-12):
            raise ValueError("total_rms must equal hypot(epistemic, aleatoric)")

    @classmethod
    def from_components(
        cls,
        epistemic: float,
        aleatoric: float,
        *,
        source: str = "experimental",
    ) -> "UncertaintyState":
        e = float(epistemic)
        a = float(aleatoric)
        return cls(e, a, float(np.hypot(e, a)), source)

    @property
    def epistemic_fraction(self) -> float:
        if self.total_rms == 0.0:
            return 0.0
        return self.epistemic / self.total_rms

    @property
    def aleatoric_fraction(self) -> float:
        if self.total_rms == 0.0:
            return 0.0
        return self.aleatoric / self.total_rms


def combine_uncertainty(
    epistemic: float,
    observation_sigma: float,
    *,
    source: str = "experimental",
) -> UncertaintyState:
    """Create an explicit uncertainty decomposition without changing confidence."""
    return UncertaintyState.from_components(
        epistemic=float(epistemic),
        aleatoric=float(observation_sigma),
        source=source,
    )
