"""Generic cross-channel consistency for paired observational streams.

Experimental/lab-only. Learns a fixed affine relation B ~= A @ W.T + c from
paired calibration windows and scores held-out paired observations by normalized
residual magnitude. No domain imports and no integration with official gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RedundantObservationState:
    residual: float
    consistent: bool
    samples: int


class RedundantObservationConsistency:
    def __init__(self, alpha: float = 0.01, eps: float = 1e-8):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._W = None
        self._c = None
        self._scale = None
        self._threshold = None

    @property
    def fitted(self) -> bool:
        return self._threshold is not None

    def fit(self, pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> "RedundantObservationConsistency":
        if len(pairs) < 20:
            raise ValueError("at least 20 calibration pairs are required")
        xa, xb = [], []
        for a, b in pairs:
            aa, bb = self._validate_pair(a, b)
            xa.append(aa)
            xb.append(bb)
        A = np.vstack(xa)
        B = np.vstack(xb)
        Z = np.c_[A, np.ones(len(A))]
        coeff, *_ = np.linalg.lstsq(Z, B, rcond=None)
        self._W = coeff[:-1].T
        self._c = coeff[-1]
        pred = Z @ coeff
        resid = np.sqrt(np.mean((pred - B) ** 2, axis=1))
        self._scale = max(float(np.median(resid) * 1.4826), self.eps)
        z = resid / self._scale
        self._threshold = max(float(np.quantile(z, 1.0 - self.alpha)), 1.0)
        return self

    def evaluate(self, a: np.ndarray, b: np.ndarray) -> RedundantObservationState:
        if not self.fitted:
            raise RuntimeError("fit() must be called first")
        aa, bb = self._validate_pair(a, b)
        pred = aa @ self._W.T + self._c
        resid = float(np.sqrt(np.mean((pred - bb) ** 2)) / self._scale)
        return RedundantObservationState(resid, bool(resid <= self._threshold), int(len(aa)))

    def _validate_pair(self, a, b):
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        if aa.ndim != 2 or bb.ndim != 2:
            raise ValueError("paired observations must be 2-D")
        if len(aa) != len(bb) or aa.shape[0] < 2:
            raise ValueError("paired observations must have equal non-trivial length")
        if not np.all(np.isfinite(aa)) or not np.all(np.isfinite(bb)):
            raise ValueError("paired observations must be finite")
        return aa, bb
