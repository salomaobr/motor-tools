"""Experimental temporal staleness detector for measurement streams.

Domain-agnostic and intentionally separate from ObservabilityQuality.  The
signal is low temporal change per channel, calibrated on clean observations.
A stale declaration requires persistent low-change evidence for ``min_run``
consecutive samples, which avoids treating one naturally small delta as a
frozen sensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TemporalStalenessState:
    stale: bool
    channel_p_values: tuple[float, ...]
    low_change_channels: tuple[bool, ...]
    run_lengths: tuple[int, ...]


class TemporalStalenessDetector:
    """Calibrated lower-tail temporal-change detector with persistence."""

    def __init__(self, alpha: float = 0.01, min_run: int = 2) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if min_run < 1:
            raise ValueError("min_run must be >= 1")
        self.alpha = float(alpha)
        self.min_run = int(min_run)
        self._reference: Optional[np.ndarray] = None
        self._run_lengths: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None

    @property
    def fitted(self) -> bool:
        return self._reference is not None

    def fit(self, observations: np.ndarray) -> "TemporalStalenessDetector":
        X = np.asarray(observations, dtype=np.float64)
        if X.ndim != 2 or len(X) < 3:
            raise ValueError("observations must be a 2-D array with >=3 rows")
        if not np.all(np.isfinite(X)):
            raise ValueError("observations must contain only finite values")
        deltas = np.abs(np.diff(X, axis=0))
        self._reference = np.sort(deltas, axis=0)
        self._n_features = int(X.shape[1])
        self.reset()
        return self

    def reset(self) -> None:
        if self._n_features is None:
            self._run_lengths = None
            return
        self._run_lengths = np.zeros(self._n_features, dtype=np.int64)

    def evaluate(
        self,
        x: np.ndarray,
        previous: Optional[np.ndarray],
    ) -> TemporalStalenessState:
        self._check_fit()
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != self._n_features:
            raise ValueError("x shape does not match fitted observations")
        if not np.all(np.isfinite(arr)):
            raise ValueError("x must contain only finite values")
        if previous is None:
            self.reset()
            return TemporalStalenessState(
                stale=False,
                channel_p_values=tuple(1.0 for _ in range(self._n_features)),
                low_change_channels=tuple(False for _ in range(self._n_features)),
                run_lengths=tuple(int(v) for v in self._run_lengths),
            )

        prev = np.asarray(previous, dtype=np.float64)
        if prev.shape != arr.shape or not np.all(np.isfinite(prev)):
            raise ValueError("previous must match x and be finite")
        delta = np.abs(arr - prev)
        ref = self._reference
        p_values = (1.0 + np.sum(ref <= delta, axis=0)) / (len(ref) + 1.0)
        low = p_values <= self.alpha
        self._run_lengths = np.where(low, self._run_lengths + 1, 0)
        stale = bool(np.any(self._run_lengths >= self.min_run))
        return TemporalStalenessState(
            stale=stale,
            channel_p_values=tuple(float(v) for v in p_values),
            low_change_channels=tuple(bool(v) for v in low),
            run_lengths=tuple(int(v) for v in self._run_lengths),
        )

    def _check_fit(self) -> None:
        if self._reference is None or self._run_lengths is None:
            raise RuntimeError("TemporalStalenessDetector.fit() must be called first")
