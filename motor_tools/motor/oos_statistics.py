"""Estatística OOS independente para discriminação de hipóteses.

A regra padrão implementada aqui usa teste de sinais exato, unilateral, sobre
erros absolutos pareados. Isso é evidência de dominância pela direção dos pares
(uma hipótese de sinal/mediana), não um teste exato da diferença média de MAE.
A alternativa ``permutation`` usa sign-flip pareado sobre a média e, portanto,
exige a hipótese adicional de simetria/exchangeability dos diferenciais sob o
nulo. Nenhuma das duas opções é uma garantia universal para média sob nulos
arbitrariamente assimétricos.

Hipóteses são comparadas par-a-par e a seleção do vencedor é protegida por
Bonferroni sobre todas as comparações possíveis. Isso assume que o holdout é
fixo/intocado após a seleção das hipóteses e que os pares OOS são iid/
exchangeable para a interpretação frequentista do teste.
"""
from __future__ import annotations

from math import comb
from typing import Mapping, Sequence

import numpy as np


def exact_one_sided_sign_pvalue(wins: int, losses: int) -> float:
    """P[X >= wins], X~Binomial(wins+losses, 0.5), ignorando empates."""
    wins = int(wins)
    losses = int(losses)
    if wins < 0 or losses < 0:
        raise ValueError("wins/losses must be non-negative")
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(wins, n + 1))
    return float(tail / (2 ** n))


def pairwise_sign_evidence(
    errors: Mapping[str, Sequence[float]],
    best_name: str,
    alpha: float,
) -> tuple[bool, dict[str, float], dict[str, tuple[int, int, int]], float]:
    """Testa se ``best_name`` domina cada concorrente.

    Retorna ``(supported, pvalues, counts, alpha_pair)``.
    ``counts[name]`` = (wins, losses, ties) de best contra name.
    """
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if best_name not in errors:
        return False, {}, {}, alpha

    names = list(errors)
    if len(names) < 2:
        return False, {}, {}, alpha
    lengths = {len(errors[name]) for name in names}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        return False, {}, {}, alpha

    best = np.asarray(errors[best_name], dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(best)):
        return False, {}, {}, alpha

    opponents = [name for name in names if name != best_name]
    # O vencedor é escolhido depois de observar o holdout. Portanto, a
    # seleção não pode reutilizar o mesmo alfa apenas na direção escolhida.
    # Bonferroni sobre todas as comparações direcionais protege a seleção.
    n_directional_comparisons = len(names) * (len(names) - 1)
    alpha_pair = alpha / max(1, n_directional_comparisons)
    pvalues: dict[str, float] = {}
    counts: dict[str, tuple[int, int, int]] = {}

    all_pass = True
    for name in opponents:
        other = np.asarray(errors[name], dtype=np.float64).reshape(-1)
        if other.shape != best.shape or not np.all(np.isfinite(other)):
            return False, {}, {}, alpha_pair
        # Positive diff means the competitor's error is larger, hence best wins.
        diff = other - best
        wins = int(np.count_nonzero(diff > 0.0))
        losses = int(np.count_nonzero(diff < 0.0))
        ties = int(np.count_nonzero(diff == 0.0))
        p = exact_one_sided_sign_pvalue(wins, losses)
        pvalues[name] = p
        counts[name] = (wins, losses, ties)
        if wins <= losses or p > alpha_pair:
            all_pass = False

    return all_pass, pvalues, counts, alpha_pair


def _regularized_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta via continued fraction (Numerical Recipes)."""
    from math import exp, lgamma, log, log1p
    if not (0.0 <= x <= 1.0):
        raise ValueError("beta x must be in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    eps = 3.0e-14
    fpmin = 1.0e-300
    def cf(xx: float) -> float:
        c = 1.0
        d = 1.0 - qab * xx / qap
        d = max(abs(d), fpmin) * (1.0 if d >= 0.0 else -1.0)
        d = 1.0 / d
        h = d
        for m in range(1, 10000):
            m2 = 2 * m
            aa = m * (b - m) * xx / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * xx / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < fpmin:
                d = fpmin
            c = 1.0 + aa / c
            if abs(c) < fpmin:
                c = fpmin
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                break
        return h
    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * cf(x) / a
    return 1.0 - bt * cf(1.0 - x) / b


def _student_t_sf(t_stat: float, df: int) -> float:
    """One-sided survival P[T_df >= t] for finite positive/negative t."""
    from math import atan, fabs, log1p
    if df <= 0:
        raise ValueError("df must be positive")
    if not np.isfinite(t_stat):
        return 0.0 if t_stat > 0 else 1.0
    if t_stat == 0.0:
        return 0.5
    x = df / (df + t_stat * t_stat)
    ib = _regularized_beta(df / 2.0, 0.5, x)
    return float(0.5 * ib if t_stat > 0 else 1.0 - 0.5 * ib)


def paired_t_one_sided_pvalue(
    best_errors: Sequence[float],
    other_errors: Sequence[float],
) -> float:
    """One-sided paired t p-value for mean(other-best) > 0.

    This targets the mean paired MAE difference directly. It assumes finite
    paired observations and uses the Student-t reference distribution.
    """
    a = np.asarray(best_errors, dtype=np.float64).reshape(-1)
    b = np.asarray(other_errors, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("paired error vectors must have equal length >= 2")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("paired error vectors must be finite")
    diff = b - a
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return 0.0 if mean > 0.0 else 1.0
    t_stat = mean / (sd / np.sqrt(diff.size))
    return _student_t_sf(float(t_stat), int(diff.size - 1))


def pairwise_paired_t_evidence(
    errors: Mapping[str, Sequence[float]],
    best_name: str,
    alpha: float,
) -> tuple[bool, dict[str, float], float]:
    """Multiple-comparison protected paired-t evidence on mean MAE."""
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if best_name not in errors:
        return False, {}, alpha
    names = list(errors)
    if len(names) < 2:
        return False, {}, alpha
    lengths = {len(errors[name]) for name in names}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
        return False, {}, alpha
    alpha_pair = alpha / max(1, len(names) * (len(names) - 1))
    pvalues: dict[str, float] = {}
    all_pass = True
    for name in (n for n in names if n != best_name):
        p = paired_t_one_sided_pvalue(errors[best_name], errors[name])
        pvalues[name] = p
        if p > alpha_pair:
            all_pass = False
    return all_pass, pvalues, alpha_pair


def paired_permutation_one_sided_pvalue(
    best_errors: Sequence[float],
    other_errors: Sequence[float],
    *,
    n_resamples: int = 100_000,
    seed: int = 0,
) -> float:
    """Paired sign-flip permutation p-value for mean error difference.

    The null is exchangeability of the paired assignment, implemented as
    independent sign flips of the paired differences. Positive differences
    mean the competitor has larger error, so the alternative is mean(diff)>0.
    Exact enumeration is used when feasible; otherwise a deterministic Monte
    Carlo estimate with the +1 correction is used.
    """
    a = np.asarray(best_errors, dtype=np.float64).reshape(-1)
    b = np.asarray(other_errors, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired error vectors must have equal non-zero length")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("paired error vectors must be finite")
    diff = b - a
    observed = float(np.mean(diff))
    n = int(diff.size)
    if n <= 20:
        # Exact 2^n enumeration. Zero differences do not create a useful
        # direction, but retaining them preserves the exact sign-flip model.
        total = 1 << n
        extreme = 0
        for mask in range(total):
            signed = diff.copy()
            for i in range(n):
                if (mask >> i) & 1:
                    signed[i] = -signed[i]
            if float(np.mean(signed)) >= observed - 1e-15:
                extreme += 1
        return float(extreme / total)
    rng = np.random.default_rng(int(seed))
    draws = max(1, int(n_resamples))
    extreme = 0
    block = 4096
    remaining = draws
    while remaining:
        m = min(block, remaining)
        signs = rng.integers(0, 2, size=(m, n), dtype=np.int8)
        signs = signs * 2 - 1
        stats = np.mean(signs * diff[None, :], axis=1)
        extreme += int(np.count_nonzero(stats >= observed - 1e-15))
        remaining -= m
    return float((extreme + 1) / (draws + 1))


def pairwise_permutation_evidence(
    errors: Mapping[str, Sequence[float]],
    best_name: str,
    alpha: float,
    *,
    n_resamples: int = 100_000,
    seed: int = 0,
) -> tuple[bool, dict[str, float], float]:
    """Multiple-comparison protected paired permutation evidence.

    This is a benchmarkable alternative to the sign test. It uses the mean
    paired error difference and Bonferroni over all ordered directions.
    """
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if best_name not in errors:
        return False, {}, alpha
    names = list(errors)
    if len(names) < 2:
        return False, {}, alpha
    lengths = {len(errors[name]) for name in names}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        return False, {}, alpha
    alpha_pair = alpha / max(1, len(names) * (len(names) - 1))
    pvalues = {}
    all_pass = True
    for j, name in enumerate(name for name in names if name != best_name):
        p = paired_permutation_one_sided_pvalue(
            errors[best_name], errors[name],
            n_resamples=n_resamples, seed=int(seed) + j,
        )
        pvalues[name] = p
        if p > alpha_pair:
            all_pass = False
    return all_pass, pvalues, alpha_pair
