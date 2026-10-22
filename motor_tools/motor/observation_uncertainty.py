"""Generic observation-noise estimation utilities.

This module is deliberately independent of any environment or domain. It estimates
sensor noise from repeated reads of the same latent state using a robust MAD
estimator. It does not modify KAIROS and is intended for experiments that need an
explicit observation-uncertainty signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class NoiseProfile:
    """Robust per-channel observation-noise estimate."""

    sigma: np.ndarray
    pooled_sigma: float
    n_repeats: int


def robust_sigma(samples: np.ndarray, axis: int = 0) -> np.ndarray:
    """Estimate Gaussian sigma from repeated observations using MAD."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError("samples must have at least two dimensions: repeats x features")
    if x.shape[axis] < 3:
        raise ValueError("at least 3 repeated observations are required")
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis)
    sigma = 1.482602218505602 * mad
    # Degenerate all-identical repeats imply zero estimated noise, which is valid.
    return np.asarray(sigma, dtype=np.float64)


def estimate_noise_profile(samples: np.ndarray, axis: int = 0) -> NoiseProfile:
    """Return robust per-feature and pooled observation-noise estimates."""
    x = np.asarray(samples, dtype=np.float64)
    sigma = robust_sigma(x, axis=axis).reshape(-1)
    pooled = float(np.sqrt(np.mean(sigma ** 2))) if sigma.size else 0.0
    n_repeats = int(x.shape[axis])
    return NoiseProfile(sigma=sigma, pooled_sigma=pooled, n_repeats=n_repeats)
