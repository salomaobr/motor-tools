"""Experimental relative temporal staleness detector.

Unlike an absolute low-change detector, this component only declares a channel
stale when its change is unusually small while at least one other channel remains
within an active-change region. This is intended to separate sensor freezing of
one channel from a legitimately slow regime in which all channels move slowly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RelativeTemporalStalenessState:
    stale: bool
    channel_p_values: tuple[float, ...]
    low_change_channels: tuple[bool, ...]
    active_change_channels: tuple[bool, ...]
    run_lengths: tuple[int, ...]


class RelativeTemporalStalenessDetector:
    """Lower-tail change detector gated by cross-channel activity."""

    def __init__(
        self,
        alpha: float = 0.01,
        active_alpha: float = 0.10,
        min_run: int = 2,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not 0.0 < active_alpha < 1.0:
            raise ValueError("active_alpha must be in (0, 1)")
        if min_run < 1:
            raise ValueError("min_run must be >= 1")
        if alpha >= active_alpha:
            raise ValueError("alpha must be < active_alpha")
        self.alpha = float(alpha)
        self.active_alpha = float(active_alpha)
        self.min_run = int(min_run)
        self._reference: Optional[np.ndarray] = None
        self._run_lengths: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None

    @property
    def fitted(self) -> bool:
        return self._reference is not None

    def fit(self, observations: np.ndarray) -> "RelativeTemporalStalenessDetector":
        X = np.asarray(observations, dtype=np.float64)
        if X.ndim != 2 or len(X) < 3:
            raise ValueError("observations must be a 2-D array with >=3 rows")
        if not np.all(np.isfinite(X)):
            raise ValueError("observations must contain only finite values")
        self._reference = np.sort(np.abs(np.diff(X, axis=0)), axis=0)
        self._n_features = int(X.shape[1])
        self.reset()
        return self

    def reset(self) -> None:
        self._run_lengths = (
            None
            if self._n_features is None
            else np.zeros(self._n_features, dtype=np.int64)
        )

    def evaluate(
        self,
        x: np.ndarray,
        previous: Optional[np.ndarray],
    ) -> RelativeTemporalStalenessState:
        self._check_fit()
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != self._n_features:
            raise ValueError("x shape does not match fitted observations")
        if not np.all(np.isfinite(arr)):
            raise ValueError("x must contain only finite values")
        if previous is None:
            self.reset()
            return self._state(False, np.ones(self._n_features), np.zeros(self._n_features, bool))

        prev = np.asarray(previous, dtype=np.float64)
        if prev.shape != arr.shape or not np.all(np.isfinite(prev)):
            raise ValueError("previous must match x and be finite")

        delta = np.abs(arr - prev)
        ref = self._reference
        p_values = (1.0 + np.sum(ref <= delta, axis=0)) / (len(ref) + 1.0)
        low = p_values <= self.alpha
        active = p_values >= self.active_alpha
        for j in range(self._n_features):
            other_active = bool(np.any(np.delete(active, j)))
            relative_low = bool(low[j] and other_active)
            self._run_lengths[j] = self._run_lengths[j] + 1 if relative_low else 0
        stale = bool(np.any(self._run_lengths >= self.min_run))
        return self._state(stale, p_values, active, low)

    def _state(
        self,
        stale: bool,
        p_values: np.ndarray,
        active: np.ndarray,
        low: Optional[np.ndarray] = None,
    ) -> RelativeTemporalStalenessState:
        if low is None:
            low = np.zeros(self._n_features, dtype=bool)
        return RelativeTemporalStalenessState(
            stale=bool(stale),
            channel_p_values=tuple(float(v) for v in p_values),
            low_change_channels=tuple(bool(v) for v in low),
            active_change_channels=tuple(bool(v) for v in active),
            run_lengths=tuple(int(v) for v in self._run_lengths),
        )

    def _check_fit(self) -> None:
        if self._reference is None or self._run_lengths is None:
            raise RuntimeError("RelativeTemporalStalenessDetector.fit() must be called first")
