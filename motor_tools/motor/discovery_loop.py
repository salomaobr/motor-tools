"""Closed-loop structural discovery with a sealed external holdout.

Candidate selection uses development data only. The external holdout is scored
only after the search has selected its final candidate, preventing repeated
adaptive comparisons against the same holdout.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence
import math
import numpy as np

from motor_tools.motor.structure_inducer import StructureInducer, StructureDiscovery, StructureCandidate
from motor_tools.motor.mutagen import Mutagen, MutationCandidate, MutationPolicy


@dataclass(frozen=True)
class AcceptedMutation:
    candidate: MutationCandidate
    gain: float
    before_mae: float
    after_mae: float


@dataclass(frozen=True)
class DiscoveryResult:
    initial: StructureDiscovery
    final: StructureDiscovery
    accepted_mutations: tuple[AcceptedMutation, ...]
    status: str  # accepted | unknown | mutated
    candidate_trials: int = 0


class ClosedLoopDiscovery:
    """Search mutations internally; use external data only for final confirmation."""

    def __init__(self, inducer: StructureInducer | None = None,
                 mutagen: Mutagen | None = None, min_gain: float = 0.05,
                 policy: MutationPolicy | None = None, max_trials: int | None = None):
        self.inducer = inducer or StructureInducer()
        self.policy = policy or MutationPolicy()
        self.mutagen = mutagen or Mutagen(policy=self.policy)
        self.min_gain = max(0.0, float(min_gain))
        self.max_trials = None if max_trials is None else max(1, int(max_trials))

    def _working_inducer(self, n):
        # Keep a minimum of 3 folds while allowing small historical datasets.
        folds = min(self.inducer.folds, max(3, int(n) // 4))
        return StructureInducer(
            folds=folds,
            max_extra_terms=self.inducer.max_extra_terms,
            min_relative_gain=self.inducer.min_relative_gain,
            complexity_penalty=self.inducer.complexity_penalty,
            max_cv_mae=self.inducer.max_cv_mae,
            max_cv_ratio=self.inducer.max_cv_ratio,
            prune_relative_tolerance=self.inducer.prune_relative_tolerance,
            cv_strategy=self.inducer.cv_strategy,
            seed=self.inducer.seed,
        )

    def _split(self, samples, seed):
        n = len(samples)
        # Reserve a final 20% (at least 4 rows) for one external assessment.
        n_test = max(4, int(math.ceil(n * 0.20)))
        if n - n_test < 12:
            raise ValueError("closed-loop discovery requires at least 16 samples")
        if self.inducer.cv_strategy == "time_series":
            return list(samples[:-n_test]), list(samples[-n_test:])
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(n)
        test_idx = set(int(i) for i in order[:n_test])
        train = [sample for i, sample in enumerate(samples) if i not in test_idx]
        test = [sample for i, sample in enumerate(samples) if i in test_idx]
        return train, test

    @staticmethod
    def _with_external_metric(discovery, mae, baseline_mae, inducer):
        ratio = float(mae) / max(float(baseline_mae), 1e-9)
        accepted = bool(np.isfinite(mae) and mae <= inducer.max_cv_mae and ratio <= inducer.max_cv_ratio)
        reason = "" if accepted else (
            f"external_holdout_rejection: mae={mae:.6f}, "
            f"baseline={baseline_mae:.6f}, ratio={ratio:.3f}"
        )
        best = replace(discovery.best, cv_mae=float(mae),
                       score=float(mae) + inducer.complexity_penalty * max(0, discovery.best.complexity))
        return StructureDiscovery(best, discovery.candidates, discovery.active_terms,
                                  accepted, reason, float(max(0.0, min(1.0, 1.0-ratio/2.0))))

    def run(self, samples: Sequence[tuple[np.ndarray, float, np.ndarray]], *, seed=None) -> DiscoveryResult:
        if len(samples) < 16:
            raise ValueError("closed-loop discovery requires at least 16 samples")
        run_seed = self.inducer.seed if seed is None else int(seed)
        development, holdout = self._split(samples, run_seed)
        inducer = self._working_inducer(len(development))
        initial_internal = inducer.fit(development, seed=run_seed)
        baseline_external = inducer.evaluate_holdout(development, holdout, initial_internal)
        initial = self._with_external_metric(initial_internal, baseline_external,
                                             float(np.mean([np.mean(np.abs(np.asarray(o).reshape(-1)-np.asarray(s).reshape(-1))) for s,_,o in holdout])),
                                             inducer)

        # If baseline is adequate, no adaptive search is needed.
        if initial_internal.accepted:
            status = "accepted" if initial.accepted else "unknown"
            return DiscoveryResult(initial, initial, (), status, 0)

        state_dim = int(np.asarray(development[0][0]).size)
        candidates = self.mutagen.propose(initial_internal.active_terms, state_dim, policy=self.policy)
        if self.max_trials is not None:
            candidates = candidates[:self.max_trials]

        trials = 0
        scored = []
        for candidate in candidates:
            trials += 1
            try:
                extra = ((candidate.expression, candidate.transform),)
                trial = inducer.fit(development, extra_terms=extra, seed=run_seed)
                gain = (initial_internal.best.cv_mae - trial.best.cv_mae) / max(abs(initial_internal.best.cv_mae), 1e-9)
                success = bool(trial.accepted and gain >= self.min_gain and np.isfinite(gain))
                self.policy.observe(candidate, float(gain), success)
                active_search = getattr(self.mutagen, "active_search", None)
                if active_search is not None:
                    active_search.observe(candidate, float(gain), success)
                scored.append((trial.best.cv_mae, -gain, candidate.expression, candidate, trial, float(gain)))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                self.policy.observe(candidate, 0.0, False)

        if not scored:
            return DiscoveryResult(initial, initial, (), "unknown", trials)

        # Select solely on development CV. External holdout has not influenced ranking.
        _, _, _, winner, trial_internal, internal_gain = min(scored, key=lambda x: (x[0], x[1], x[2]))
        extra = ((winner.expression, winner.transform),)
        final_external_mae = inducer.evaluate_holdout(development, holdout, trial_internal, extra)
        final = self._with_external_metric(trial_internal, final_external_mae,
                                           float(np.mean([np.mean(np.abs(np.asarray(o).reshape(-1)-np.asarray(s).reshape(-1))) for s,_,o in holdout])),
                                           inducer)
        external_gain = (baseline_external - final_external_mae) / max(abs(baseline_external), 1e-9)
        if final.accepted and np.isfinite(external_gain) and external_gain >= self.min_gain:
            accepted = AcceptedMutation(winner, float(external_gain), baseline_external, final_external_mae)
            return DiscoveryResult(initial, final, (accepted,), "mutated", trials)
        # Candidate failed independent confirmation: keep baseline as final.
        return DiscoveryResult(initial, initial, (), "unknown", trials)
