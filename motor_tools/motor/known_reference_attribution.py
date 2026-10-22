"""Known-reference evidence for separating observation transforms from world shifts.

Experimental/lab-only component for Phase 25D.

A known probe is an externally controlled reference sequence. During calibration,
the detector learns the distribution of probe residuals under a trusted clean
observation path. At test time it emits independent evidence that the observation
path changed. It deliberately does not infer a cause from labels or import any
domain module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class KnownReferenceState:
    residual: float
    transform_evidence: bool
    samples: int


class KnownReferenceAttribution:
    """Calibrate a known reference probe and score later reference observations."""

    def __init__(self, alpha: float = 0.01, eps: float = 1e-8):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._scale = None
        self._threshold = None
        self._dim = None

    @property
    def fitted(self) -> bool:
        return self._threshold is not None

    def fit(self, expected: Sequence[np.ndarray], observed: Sequence[np.ndarray]) -> "KnownReferenceAttribution":
        if len(expected) != len(observed) or len(expected) < 20:
            raise ValueError("at least 20 matched calibration windows are required")
        residuals: list[float] = []
        dim = None
        for e, o in zip(expected, observed):
            ee, oo = self._validate_pair(e, o)
            dim = ee.shape[1]
            residuals.append(float(np.sqrt(np.mean((ee - oo) ** 2))))
        self._dim = int(dim)
        arr = np.asarray(residuals, dtype=float)
        scale = max(float(np.median(arr) * 1.4826), self.eps)
        z = arr / scale
        self._scale = scale
        self._threshold = max(float(np.quantile(z, 1.0 - self.alpha)), 1.0)
        return self

    def evaluate(self, expected: np.ndarray, observed: np.ndarray) -> KnownReferenceState:
        if not self.fitted:
            raise RuntimeError("fit() must be called first")
        ee, oo = self._validate_pair(expected, observed)
        if ee.shape[1] != self._dim:
            raise ValueError("reference dimension does not match calibration")
        residual = float(np.sqrt(np.mean((ee - oo) ** 2)) / self._scale)
        return KnownReferenceState(
            residual=residual,
            transform_evidence=bool(residual > self._threshold),
            samples=int(len(ee)),
        )

    def _validate_pair(self, expected: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e = np.asarray(expected, dtype=float)
        o = np.asarray(observed, dtype=float)
        if e.ndim != 2 or o.ndim != 2:
            raise ValueError("expected and observed references must be 2-D")
        if e.shape != o.shape or e.shape[0] < 2:
            raise ValueError("expected and observed references must have equal non-trivial shape")
        if not np.all(np.isfinite(e)) or not np.all(np.isfinite(o)):
            raise ValueError("reference arrays must be finite")
        return e, o
