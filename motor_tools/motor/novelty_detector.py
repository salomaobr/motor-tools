"""Generic novelty / support detector for causal feature vectors.

The detector is deliberately domain-agnostic.  It estimates how far a query is
from the training support using standardized k-nearest-neighbour distance.
The novelty score is an empirical tail probability calibrated from leave-one-
out training distances, so it does not require labels for OOD data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class NoveltyState:
    novelty_score: float
    nearest_distance: float
    support_count: int
    regime: str


class NoveltyDetector:
    """Unsupervised support-distance novelty detector.

    Fit only on data considered in-distribution.  The score is approximately
    the fraction of leave-one-out training points whose local distance is no
    larger than the query's local distance.  Higher means less supported.
    """

    def __init__(self, k: int = 8, threshold: float = 0.95, eps: float = 1e-8):
        if k < 1:
            raise ValueError("k must be >= 1")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.k = int(k)
        self.threshold = float(threshold)
        self.eps = float(eps)
        self._mean: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None
        self._reference: Optional[np.ndarray] = None

    @property
    def fitted(self) -> bool:
        return self._X is not None

    def fit(self, X: np.ndarray) -> "NoveltyDetector":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        if len(X) < 2:
            raise ValueError("at least 2 training samples are required")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        self._mean = np.mean(X, axis=0)
        self._scale = np.std(X, axis=0)
        self._scale[self._scale < self.eps] = 1.0
        self._X = (X - self._mean) / self._scale

        k = min(self.k, len(X) - 1)
        D = self._pairwise(self._X, self._X)
        np.fill_diagonal(D, np.inf)
        kth = np.partition(D, k - 1, axis=1)[:, k - 1]
        self._reference = np.sort(kth)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._check_fit()
        Xq = np.asarray(X, dtype=np.float64)
        single = Xq.ndim == 1
        if single:
            Xq = Xq.reshape(1, -1)
        if Xq.ndim != 2 or Xq.shape[1] != self._X.shape[1]:
            raise ValueError("query feature dimension does not match fit data")
        if not np.all(np.isfinite(Xq)):
            raise ValueError("queries must contain only finite values")

        Z = (Xq - self._mean) / self._scale
        D = self._pairwise(Z, self._X)
        k = min(self.k, len(self._X))
        kth = np.partition(D, k - 1, axis=1)[:, k - 1]
        # Empirical CDF with a +1 correction keeps the score inside (0,1].
        ref = self._reference
        ranks = np.searchsorted(ref, kth, side="left")
        scores = ranks / (len(ref) + 1.0)
        return scores if not single else scores[:1]

    def nearest_distance(self, X: np.ndarray) -> np.ndarray:
        self._check_fit()
        Xq = np.asarray(X, dtype=np.float64)
        single = Xq.ndim == 1
        if single:
            Xq = Xq.reshape(1, -1)
        Z = (Xq - self._mean) / self._scale
        D = self._pairwise(Z, self._X)
        return np.min(D, axis=1) if not single else np.array([np.min(D)])

    def predict(self, X: np.ndarray) -> NoveltyState | list[NoveltyState]:
        scores = self.score(X)
        distances = self.nearest_distance(X)
        self._check_fit()
        support = int(len(self._X))
        states = [
            NoveltyState(
                novelty_score=float(s),
                nearest_distance=float(d),
                support_count=support,
                regime="novel" if float(s) >= self.threshold else "known",
            )
            for s, d in zip(scores, distances)
        ]
        return states[0] if np.asarray(X).ndim == 1 else states

    @staticmethod
    def _pairwise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A2 = np.sum(A * A, axis=1, keepdims=True)
        B2 = np.sum(B * B, axis=1, keepdims=True).T
        return np.sqrt(np.maximum(A2 + B2 - 2.0 * A @ B.T, 0.0))

    def _check_fit(self) -> None:
        if self._X is None or self._reference is None:
            raise RuntimeError("NoveltyDetector.fit() must be called first")
