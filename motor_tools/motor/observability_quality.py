"""Domain-agnostic observation/measurement quality channel.

This module deliberately measures observability degradation from the observation
mask, not novelty of the underlying numeric values. It therefore does not import
any domain module and does not modify KAIROS.

The conformal-style calibrator treats low quality as nonconformity:
    nonconformity = 1 - quality_score
    p = (1 + count(cal_nonconformity >= q)) / (n_cal + 1)
Low p means the observation is unusually degraded relative to calibration data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ObservabilityState:
    quality_score: float
    p_value: float
    coverage_ratio: float
    active_channel_ratio: float
    missing_fraction: float
    degraded: bool


class ObservabilityQuality:
    """Extract a measurement-quality signal from finite/missing observations.

    The channel intentionally ignores the numeric magnitude of observed values.
    This prevents it from becoming a second novelty detector. Missingness is
    represented by non-finite values unless an explicit mask is supplied.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        stale_penalty: float = 0.50,
        equality_tol: float = 1e-12,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not 0.0 <= stale_penalty <= 1.0:
            raise ValueError("stale_penalty must be in [0, 1]")
        if equality_tol < 0:
            raise ValueError("equality_tol must be >= 0")
        self.alpha = float(alpha)
        self.stale_penalty = float(stale_penalty)
        self.equality_tol = float(equality_tol)
        self._cal_nonconformity: Optional[np.ndarray] = None

    @property
    def fitted(self) -> bool:
        return self._cal_nonconformity is not None

    def featurize(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        previous: Optional[np.ndarray] = None,
    ) -> tuple[float, float, float, float]:
        """Return quality primitives: quality, coverage, active channels, missing."""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError("x must be a 1-D observation")
        n = arr.size
        if n == 0:
            raise ValueError("x must contain at least one feature")
        if mask is None:
            observed = np.isfinite(arr)
        else:
            observed = np.asarray(mask, dtype=bool)
            if observed.shape != arr.shape:
                raise ValueError("mask shape must match x")
        coverage = float(np.mean(observed))
        missing = 1.0 - coverage
        if n == 0:
            active = 0.0
        else:
            # Features are treated as independent channels here. A domain can
            # group dimensions externally without changing this generic module.
            active = coverage > 0.0 and float(np.mean(observed)) or 0.0

        stale = 0.0
        if previous is not None:
            prev = np.asarray(previous, dtype=np.float64)
            if prev.shape != arr.shape:
                raise ValueError("previous shape must match x")
            comparable = observed & np.isfinite(prev)
            if np.any(comparable):
                stale = float(
                    np.mean(
                        np.isclose(
                            arr[comparable],
                            prev[comparable],
                            rtol=0.0,
                            atol=self.equality_tol,
                        )
                    )
                )

        # Quality is driven by direct observability coverage. Active-channel
        # ratio remains a diagnostic output so it can be specialized later for
        # grouped/multi-dimensional sensors without changing this contract.
        quality = coverage * (1.0 - self.stale_penalty * stale)
        quality = float(np.clip(quality, 0.0, 1.0))
        return quality, coverage, active, missing

    def fit(self, calibration_observations: np.ndarray) -> "ObservabilityQuality":
        """Fit the reference distribution using clean calibration observations."""
        X = np.asarray(calibration_observations, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("calibration_observations must be 2-D")
        if len(X) < 2:
            raise ValueError("at least 2 calibration observations are required")
        # Calibration data are expected to be clean. Quality is computed only
        # from observability structure, never from numeric feature magnitudes.
        q = np.array([self.featurize(row)[0] for row in X], dtype=float)
        self._cal_nonconformity = np.sort(1.0 - q)
        return self

    def evaluate(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        previous: Optional[np.ndarray] = None,
    ) -> ObservabilityState:
        if not self.fitted:
            raise RuntimeError("ObservabilityQuality.fit() must be called first")
        quality, coverage, active, missing = self.featurize(x, mask=mask, previous=previous)
        nc = 1.0 - quality
        ref = self._cal_nonconformity
        p = float((1.0 + np.sum(ref >= nc)) / (len(ref) + 1.0))
        return ObservabilityState(
            quality_score=quality,
            p_value=p,
            coverage_ratio=coverage,
            active_channel_ratio=active,
            missing_fraction=missing,
            degraded=bool(p < self.alpha),
        )
