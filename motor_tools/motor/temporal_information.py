"""Experimental detector for insufficient temporal information.

This component estimates whether the *aggregate temporal change* of an
observation stream is unusually small relative to a calibration stream. It
does not infer the cause of low variation: slow world, global sensor freeze,
or another mechanism remain observationally unresolved here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TemporalInformationState:
    insufficient: bool
    p_value: float
    normalized_change: float
    run_length: int


class TemporalInformationInsufficiencyDetector:
    """Lower-tail detector on aggregate normalized temporal change."""

    def __init__(self, alpha: float = 0.01, min_run: int = 2) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if min_run < 1:
            raise ValueError("min_run must be >= 1")
        self.alpha = float(alpha)
        self.min_run = int(min_run)
        self._reference: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self._run_length = 0
        self._n_features: Optional[int] = None

    @property
    def fitted(self) -> bool:
        return self._reference is not None

    def fit(self, observations: np.ndarray) -> "TemporalInformationInsufficiencyDetector":
        X = np.asarray(observations, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 3:
            raise ValueError("observations must be 2-D with >=3 rows")
        if X.shape[1] < 1:
            raise ValueError("observations must have at least one channel")
        if not np.all(np.isfinite(X)):
            raise ValueError("observations must contain only finite values")
        deltas = np.abs(np.diff(X, axis=0))
        scale = np.sqrt(np.mean(deltas**2, axis=0))
        scale = np.maximum(scale, np.finfo(np.float64).eps)
        normalized = np.sqrt(np.sum((deltas / scale) ** 2, axis=1))
        self._reference = np.sort(normalized)
        self._scale = scale
        self._n_features = int(X.shape[1])
        self.reset()
        return self

    def reset(self) -> None:
        self._run_length = 0

    def evaluate(
        self,
        x: np.ndarray,
        previous: Optional[np.ndarray],
    ) -> TemporalInformationState:
        self._check_fit()
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != self._n_features:
            raise ValueError("x shape does not match fitted observations")
        if not np.all(np.isfinite(arr)):
            raise ValueError("x must be finite")
        if previous is None:
            self.reset()
            return TemporalInformationState(False, 1.0, 0.0, 0)

        prev = np.asarray(previous, dtype=np.float64)
        if prev.shape != arr.shape or not np.all(np.isfinite(prev)):
            raise ValueError("previous must match x and be finite")

        delta = np.abs(arr - prev)
        normalized_change = float(np.sqrt(np.sum((delta / self._scale) ** 2)))
        ref = self._reference
        p_value = float((1.0 + np.sum(ref <= normalized_change)) / (len(ref) + 1.0))
        low = p_value <= self.alpha
        self._run_length = self._run_length + 1 if low else 0
        insufficient = self._run_length >= self.min_run
        return TemporalInformationState(
            insufficient=bool(insufficient),
            p_value=p_value,
            normalized_change=normalized_change,
            run_length=int(self._run_length),
        )

    def _check_fit(self) -> None:
        if self._reference is None or self._scale is None or self._n_features is None:
            raise RuntimeError(
                "TemporalInformationInsufficiencyDetector.fit() must be called first"
            )
