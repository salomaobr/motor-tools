"""StructureInducer v4 — indução estrutural com validação reprodutível.

Suporta validação IID embaralhada e validação temporal walk-forward. A seleção
de termos ocorre apenas dentro dos dados de treino; a avaliação externa é
reservada para medir uma trajetória adaptativa já definida.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class StructureCandidate:
    name: str
    terms: Tuple[str, ...]
    complexity: int
    cv_mae: float
    score: float


@dataclass(frozen=True)
class StructureDiscovery:
    best: StructureCandidate
    candidates: Tuple[StructureCandidate, ...]
    active_terms: Tuple[str, ...]
    accepted: bool = True
    rejection_reason: str = ""
    coverage: float = 1.0

    @property
    def nonlinear_terms(self) -> Tuple[str, ...]:
        keys = ("^2", "^3", "*abs(", "abs(", "sign(")
        return tuple(t for t in self.active_terms if any(k in t for k in keys))

    @property
    def interaction_terms(self) -> Tuple[str, ...]:
        return tuple(t for t in self.active_terms if "*" in t and "abs" not in t and "I(" not in t)

    @property
    def gated_terms(self) -> Tuple[str, ...]:
        return tuple(t for t in self.active_terms if "I(" in t or "relu(" in t)


@dataclass(frozen=True)
class CandidateTransform:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    complexity: int = 1


def feature_library(state: np.ndarray, action: float, extra_terms=()) -> Tuple[np.ndarray, Tuple[str, ...]]:
    s = np.asarray(state, dtype=np.float64).reshape(-1)
    a = float(action)
    cols = [*s.tolist(), a, 1.0]
    names = [f"s{i}" for i in range(len(s))] + ["a", "bias"]
    for i, value in enumerate(s):
        cols.extend([value * value, value * value * value, value * abs(value),
                     abs(value), np.sign(value), max(value, 0.0)])
        names.extend([f"s{i}^2", f"s{i}^3", f"s{i}*abs(s{i})",
                      f"abs(s{i})", f"sign(s{i})", f"relu(s{i})"])
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            cols.append(s[i] * s[j])
            names.append(f"s{i}*s{j}")
    for i, value in enumerate(s):
        cols.append(value * a)
        names.append(f"s{i}*a")
    for i in range(len(s)):
        for j in range(len(s)):
            if i == j:
                continue
            cols.append(max(s[i], 0.0) * float(s[j] > 0.0))
            names.append(f"relu(s{i})*I(s{j}>0)")
    for term, fn in extra_terms:
        z = np.asarray(fn(s), dtype=np.float64).reshape(-1)
        if z.size != 1:
            raise ValueError(f"extra transform must return scalar: {term}")
        cols.append(float(z[0]))
        names.append(str(term))
    return np.asarray(cols, dtype=np.float64), tuple(names)


def _fit(X, Y, cols):
    A = X[:, cols]
    ridge = 1e-6 * np.eye(len(cols))
    return np.linalg.solve(A.T @ A + ridge, A.T @ Y)


class StructureInducer:
    def __init__(
        self, folds=5, max_extra_terms=6, min_relative_gain=0.01,
        complexity_penalty=1e-5, max_cv_mae=0.20, max_cv_ratio=1.25,
        prune_relative_tolerance=0.02, cv_strategy="iid", seed=0,
    ):
        self.folds = max(3, int(folds))
        self.max_extra_terms = max(0, int(max_extra_terms))
        self.min_relative_gain = max(0.0, float(min_relative_gain))
        self.complexity_penalty = max(0.0, float(complexity_penalty))
        self.max_cv_mae = float(max_cv_mae)
        self.max_cv_ratio = float(max_cv_ratio)
        self.prune_relative_tolerance = max(0.0, float(prune_relative_tolerance))
        if cv_strategy not in {"iid", "time_series"}:
            raise ValueError("cv_strategy must be 'iid' or 'time_series'")
        self.cv_strategy = cv_strategy
        self.seed = int(seed)

    def _make_splits(self, n, *, strategy=None, seed=None, folds=None):
        """Return (train_indices, test_indices); temporal splits are forward-only."""
        strategy = strategy or self.cv_strategy
        folds = max(3, min(int(folds or self.folds), max(3, n // 4)))
        if n < folds * 2:
            raise ValueError("not enough samples for cross-validation")
        if strategy == "iid":
            rng = np.random.default_rng(self.seed if seed is None else int(seed))
            indices = rng.permutation(n)
            test_parts = np.array_split(indices, folds)
            all_indices = np.arange(n)
            splits = []
            for test in test_parts:
                mask = np.ones(n, dtype=bool)
                mask[test] = False
                splits.append((all_indices[mask], np.sort(test)))
            return tuple(splits)
        if strategy == "time_series":
            # Initial prefix is training-only; every test block is strictly later.
            initial = max(4, n // (folds + 1))
            if initial >= n - 1:
                raise ValueError("not enough samples for temporal validation")
            remaining = np.arange(initial, n)
            test_parts = np.array_split(remaining, min(folds, len(remaining)))
            splits = []
            for test in test_parts:
                if len(test) == 0:
                    continue
                train = np.arange(int(test[0]))
                if len(train) < 4:
                    continue
                splits.append((train, test))
            if len(splits) < 2:
                raise ValueError("not enough samples for temporal validation")
            return tuple(splits)
        raise ValueError("unknown CV strategy")

    def _select(self, X, Y, train, base_count):
        train = np.asarray(train)
        if train.dtype == bool:
            train_idx = np.flatnonzero(train)
        else:
            train_idx = train.astype(int)
        base = list(range(base_count))
        remaining = list(range(base_count, X.shape[1]))
        selected = base.copy()

        def mae(cols):
            W = _fit(X[train_idx], Y[train_idx], cols)
            return float(np.mean(np.abs(X[train_idx][:, cols] @ W - Y[train_idx])))

        for _ in range(self.max_extra_terms):
            current = mae(selected)
            best_err = current
            best_col = None
            for col in remaining:
                err = mae(selected + [col])
                if np.isfinite(err) and err < best_err:
                    best_err, best_col = err, col
            if best_col is None or (current - best_err) / max(current, 1e-9) < self.min_relative_gain:
                break
            selected.append(best_col)
            remaining.remove(best_col)
        return selected

    @staticmethod
    def _cv_mae_splits(X, Y, cols, splits):
        errors = []
        cols = list(cols)
        for train_idx, test_idx in splits:
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            W = _fit(X[train_idx], Y[train_idx], cols)
            pred = X[test_idx][:, cols] @ W
            err = float(np.mean(np.abs(pred - Y[test_idx])))
            if np.isfinite(err):
                errors.append(err)
        return float(np.mean(errors)) if errors else float("inf")

    def _prune(self, X, Y, selected, splits, base_count):
        """Prune terms only against the supplied internal CV splits."""
        selected = list(selected)
        if len(selected) <= base_count or self.prune_relative_tolerance <= 0.0:
            return selected
        current = self._cv_mae_splits(X, Y, selected, splits)
        while len(selected) > base_count:
            trials = []
            for pos in range(base_count, len(selected)):
                trial = selected[:pos] + selected[pos + 1:]
                err = self._cv_mae_splits(X, Y, trial, splits)
                trials.append((err, pos, trial))
            err, _, trial = min(trials, key=lambda row: (row[0], row[1]))
            if err <= current * (1.0 + self.prune_relative_tolerance):
                selected, current = trial, err
            else:
                break
        return selected

    @staticmethod
    def _arrays(samples, extra_terms):
        X = np.stack([feature_library(s, a, extra_terms)[0] for s, a, _ in samples])
        names = feature_library(samples[0][0], samples[0][1], extra_terms)[1]
        states = np.stack([np.asarray(s, dtype=np.float64).reshape(-1) for s, _, _ in samples])
        outcomes = np.stack([np.asarray(o, dtype=np.float64).reshape(-1) for _, _, o in samples])
        if states.shape != outcomes.shape:
            raise ValueError("state and outcome dimensions must match")
        Y = outcomes - states
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
            raise ValueError("samples contain non-finite values")
        return X, names, states, Y

    def fit(self, samples: Sequence[Tuple[np.ndarray, float, np.ndarray]], extra_terms=(), *, seed=None, cv_strategy=None) -> StructureDiscovery:
        effective_folds = min(self.folds, max(3, len(samples) // 4))
        if len(samples) < effective_folds * 4:
            raise ValueError("not enough samples")
        X, names, states, Y = self._arrays(samples, extra_terms)
        base_count = states.shape[1] + 2
        strategy = cv_strategy or self.cv_strategy
        splits = self._make_splits(len(samples), strategy=strategy, seed=seed)
        errors = []
        for train_idx, test_idx in splits:
            # Internal pruning is done only within this fold's training data.
            local_splits = []
            for tr, te in self._make_splits(len(train_idx), strategy=strategy, seed=seed):
                local_splits.append((tr, te))
            Xtr, Ytr = X[train_idx], Y[train_idx]
            selected = self._prune(Xtr, Ytr, self._select(Xtr, Ytr, np.arange(len(train_idx)), base_count), local_splits, base_count)
            W = _fit(X[train_idx], Y[train_idx], selected)
            pred = X[test_idx][:, selected] @ W
            errors.append(float(np.mean(np.abs(pred - Y[test_idx]))))
        cv_mae = float(np.mean(errors))
        full_splits = self._make_splits(len(samples), strategy=strategy, seed=seed)
        selected_full = self._select(X, Y, np.arange(len(samples)), base_count)
        selected_full = self._prune(X, Y, selected_full, full_splits, base_count)
        base_mae = float(np.mean(np.abs(Y)))
        extra = max(0, len(selected_full) - base_count)
        score = cv_mae + self.complexity_penalty * extra
        terms = tuple(names[i] for i in selected_full)
        ratio = cv_mae / max(base_mae, 1e-9)
        accepted = bool(np.isfinite(cv_mae) and cv_mae <= self.max_cv_mae and ratio <= self.max_cv_ratio)
        reason = "" if accepted else (
            f"insufficient_out_of_sample_explanation: cv_mae={cv_mae:.6f}, "
            f"baseline={base_mae:.6f}, ratio={ratio:.3f}"
        )
        coverage = float(max(0.0, min(1.0, 1.0 - ratio / 2.0)))
        best = StructureCandidate("sparse_terms", terms, len(selected_full), cv_mae, score)
        return StructureDiscovery(best, (best,), terms, accepted, reason, coverage)

    def evaluate_holdout(self, train_samples, test_samples, discovery, extra_terms=()):
        """Fit coefficients on train_samples and score a fixed external holdout once."""
        if not train_samples or not test_samples:
            raise ValueError("train and test samples must be non-empty")
        Xtr, names, _, Ytr = self._arrays(train_samples, extra_terms)
        Xte, test_names, _, Yte = self._arrays(test_samples, extra_terms)
        if names != test_names:
            raise ValueError("feature schema mismatch")
        selected = [names.index(term) for term in discovery.active_terms if term in names]
        if not selected:
            raise ValueError("discovery has no terms available in feature schema")
        W = _fit(Xtr, Ytr, selected)
        pred = Xte[:, selected] @ W
        return float(np.mean(np.abs(pred - Yte)))

    def fit_open(self, samples: Sequence[Tuple[np.ndarray, float, np.ndarray]], *, mutagen=None,
                 max_rounds: int = 3, min_gain: float = 0.01, outer_folds: int | None = None,
                 seed=None, cv_strategy=None) -> StructureDiscovery:
        """Nested discovery: adaptation on inner data, one score per outer fold."""
        if len(samples) < 24:
            raise ValueError("not enough samples for open discovery")
        if mutagen is None:
            from motor_tools.motor.mutagen import Mutagen
            mutagen = Mutagen()
        rounds = max(1, int(max_rounds))
        strategy = cv_strategy or self.cv_strategy
        folds = max(3, min(int(outer_folds or self.folds), len(samples) // 8))
        outer_splits = self._make_splits(len(samples), strategy=strategy, seed=seed, folds=folds)
        outer_errors, outer_baselines, final_terms, final_complexities = [], [], [], []

        def fit_trajectory(train, local_seed):
            current = self.fit(train, seed=local_seed, cv_strategy=strategy)
            for _ in range(rounds):
                candidates = mutagen.propose(current.active_terms, int(np.asarray(train[0][0]).size))
                if not candidates:
                    break
                extras = tuple((c.expression, c.transform) for c in candidates)
                trial = self.fit(train, extra_terms=extras, seed=local_seed, cv_strategy=strategy)
                gain = (current.best.cv_mae - trial.best.cv_mae) / max(abs(current.best.cv_mae), 1e-9)
                if gain < float(min_gain):
                    break
                current = trial
            return current

        for k, (train_idx, test_idx) in enumerate(outer_splits):
            train = [samples[int(i)] for i in train_idx]
            test = [samples[int(i)] for i in test_idx]
            current = fit_trajectory(train, None if seed is None else int(seed) + k)
            # Reconstruct candidate transforms by expression, only for final scoring.
            proposed = mutagen.propose((), int(np.asarray(train[0][0]).size))
            by_expr = {c.expression: c.transform for c in proposed}
            base_names = feature_library(train[0][0], train[0][1], ())[1]
            extras = tuple((term, by_expr[term]) for term in current.active_terms if term not in base_names and term in by_expr)
            outer_errors.append(self.evaluate_holdout(train, test, current, extras))
            outer_baselines.append(float(np.mean([
                np.mean(np.abs(np.asarray(o, dtype=np.float64).reshape(-1) -
                                np.asarray(s, dtype=np.float64).reshape(-1)))
                for s, _, o in test
            ])))
            final_terms.append(current.active_terms)
            final_complexities.append(current.best.complexity)

        # Full-data refit is for the deployable candidate only; it does not alter
        # the already-computed outer estimate.
        full = fit_trajectory(samples, seed)
        outer_mae = float(np.mean(outer_errors))
        outer_baseline = float(np.mean(outer_baselines))
        ratio = outer_mae / max(outer_baseline, 1e-9)
        accepted = bool(np.isfinite(outer_mae) and outer_mae <= self.max_cv_mae and ratio <= self.max_cv_ratio)
        reason = "" if accepted else f"open_oos_rejection: outer_cv_mae={outer_mae:.6f}, baseline={outer_baseline:.6f}, ratio={ratio:.3f}"
        best = StructureCandidate("open_sparse_terms", full.active_terms, int(np.mean(final_complexities)),
                                  outer_mae, outer_mae + self.complexity_penalty * max(
                                      0, len(full.active_terms) - (int(np.asarray(samples[0][0]).size) + 2)))
        base_mae = float(np.mean([np.mean(np.abs(np.asarray(o).reshape(-1) - np.asarray(s).reshape(-1)))
                                  for s, _, o in samples]))
        return StructureDiscovery(best, (best,), full.active_terms, accepted, reason,
                                  float(max(0.0, 1.0 - outer_mae / max(base_mae, 1e-9))))
