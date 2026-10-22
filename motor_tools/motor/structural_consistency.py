"""Domain-agnostic structural consistency for temporal observations.

Experimental, lab-only component for Phase 24.

The detector fits a local affine one-step model
    x[t+1] ~= A x[t] + b
and compares coordinate-invariant dynamic signatures (the eigenvalue spectrum
of A) against a clean calibration set. It also tracks simple observation-scale
and observation-offset changes inside each window.

The component does *not* infer that a transformation is certainly a sensor
fault. It only emits two separate pieces of evidence:

- dynamic_consistent: the local mechanism is structurally supported;
- transform_evidence: the mechanism is supported while the observation scale
  or offset moved outside calibration support.

This distinction is deliberately kept outside the official novelty/evidence
gate until an independent benchmark validates selectivity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class StructuralConsistencyState:
    dynamic_distance: float
    dynamic_consistent: bool
    mean_shift: float
    scale_shift: float
    affine_offset: float
    transform_evidence: bool
    world_shift_evidence: bool
    sample_count: int


class StructuralConsistency:
    """Calibrate and score temporal structural consistency without domain imports."""

    def __init__(self, alpha: float = 0.01, min_window: int = 5, eps: float = 1e-8):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if int(min_window) < 3:
            raise ValueError("min_window must be >= 3")
        self.alpha = float(alpha)
        self.min_window = int(min_window)
        self.eps = float(eps)
        self._dyn = None
        self._mean = None
        self._scale = None
        self._dyn_threshold = None
        self._mean_threshold = None
        self._scale_threshold = None
        self._dim = None

    @property
    def fitted(self) -> bool:
        return self._dyn_threshold is not None

    def fit(self, windows: Sequence[np.ndarray]) -> "StructuralConsistency":
        rows = [self._validate_window(w) for w in windows]
        if len(rows) < 10:
            raise ValueError("at least 10 calibration windows are required")
        dim = rows[0].shape[1]
        if any(w.shape[1] != dim for w in rows):
            raise ValueError("all windows must have the same feature dimension")
        self._dim = dim

        dyn = []
        means = []
        scales = []
        for w in rows:
            sig, _, mean, scale = self._extract(w)
            dyn.append(sig)
            means.append(mean)
            scales.append(scale)

        self._dyn = np.asarray(dyn, dtype=float)
        self._mean = np.asarray(means, dtype=float)
        self._scale = np.asarray(scales, dtype=float)

        dyn_center = np.median(self._dyn, axis=0)
        dyn_scale = np.median(np.abs(self._dyn - dyn_center), axis=0) * 1.4826
        dyn_scale[dyn_scale < self.eps] = 1.0
        dyn_dist = np.linalg.norm((self._dyn - dyn_center) / dyn_scale, axis=1)
        self._dyn_center = dyn_center
        self._dyn_scale = dyn_scale
        self._dyn_threshold = float(np.quantile(dyn_dist, 1.0 - self.alpha))
        self._dyn_threshold = max(self._dyn_threshold, 1.0)

        mean_center = np.median(self._mean, axis=0)
        mean_scale = np.median(np.abs(self._mean - mean_center), axis=0) * 1.4826
        mean_scale[mean_scale < self.eps] = 1.0
        mean_dist = np.linalg.norm((self._mean - mean_center) / mean_scale, axis=1)
        self._mean_center = mean_center
        self._mean_scale = mean_scale
        self._mean_threshold = max(float(np.quantile(mean_dist, 1.0 - self.alpha)), 1.0)

        log_scale = np.log(np.maximum(self._scale, self.eps))
        log_center = np.median(log_scale, axis=0)
        log_scale_spread = np.median(np.abs(log_scale - log_center), axis=0) * 1.4826
        log_scale_spread[log_scale_spread < self.eps] = 1.0
        scale_dist = np.linalg.norm((log_scale - log_center) / log_scale_spread, axis=1)
        self._log_scale_center = log_center
        self._log_scale_spread = log_scale_spread
        self._scale_threshold = max(float(np.quantile(scale_dist, 1.0 - self.alpha)), 1.0)
        return self

    def evaluate(self, window: np.ndarray) -> StructuralConsistencyState:
        self._check_fit()
        w = self._validate_window(window)
        if w.shape[1] != self._dim:
            raise ValueError("window dimension does not match calibration")
        sig, offset, mean, scale = self._extract(w)

        dyn_dist = float(np.linalg.norm((sig - self._dyn_center) / self._dyn_scale))
        mean_shift = float(np.linalg.norm((mean - self._mean_center) / self._mean_scale))
        log_scale = np.log(np.maximum(scale, self.eps))
        scale_shift = float(np.linalg.norm((log_scale - self._log_scale_center) / self._log_scale_spread))
        affine_offset = float(np.linalg.norm(offset) / max(np.median(self._scale), self.eps))

        dynamic_consistent = bool(dyn_dist <= self._dyn_threshold)
        transform_evidence = bool(
            dynamic_consistent
            and (mean_shift > self._mean_threshold or scale_shift > self._scale_threshold or affine_offset > 3.0)
        )
        world_shift = bool(not dynamic_consistent)
        return StructuralConsistencyState(
            dynamic_distance=dyn_dist,
            dynamic_consistent=dynamic_consistent,
            mean_shift=mean_shift,
            scale_shift=scale_shift,
            affine_offset=affine_offset,
            transform_evidence=transform_evidence,
            world_shift_evidence=world_shift,
            sample_count=int(len(w)),
        )

    def _extract(self, w: np.ndarray):
        x = w[:-1]
        y = w[1:]
        Z = np.c_[x, np.ones(len(x))]
        coeff, *_ = np.linalg.lstsq(Z, y, rcond=None)
        A = coeff[: self._dim].T
        offset = coeff[self._dim]
        eig = np.linalg.eigvals(A)
        # Sort complex eigenvalues deterministically and encode real/imag parts.
        eig = sorted(eig.tolist(), key=lambda z: (float(np.real(z)), float(np.imag(z))))
        signature = np.asarray(
            [part for z in eig for part in (float(np.real(z)), float(np.imag(z)))],
            dtype=float,
        )
        mean = np.mean(w, axis=0)
        scale = np.std(w, axis=0)
        scale[scale < self.eps] = self.eps
        return signature, np.asarray(offset, float), mean, scale

    def _validate_window(self, window: np.ndarray) -> np.ndarray:
        w = np.asarray(window, dtype=float)
        if w.ndim != 2:
            raise ValueError("window must be a 2-D array")
        if len(w) < self.min_window:
            raise ValueError("window is shorter than min_window")
        if not np.all(np.isfinite(w)):
            raise ValueError("window must contain only finite values")
        if w.shape[1] < 1:
            raise ValueError("window must contain at least one feature")
        return w

    def _check_fit(self) -> None:
        if not self.fitted:
            raise RuntimeError("StructuralConsistency.fit() must be called first")
