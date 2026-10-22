"""Audit-only evidence aggregation for the experimental Research Scientist layer.

The ledger never promotes candidates or edits runtime state. It aggregates explicit
ExperimentObservation records and reports whether repeated evidence is convergent,
conflicting, or insufficient under a small deterministic contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from motor_tools.motor.research_scientist import ExperimentObservation


@dataclass(frozen=True)
class EvidenceSummary:
    hypothesis: str
    experiment: str
    observations: int
    completed: int
    failed: int
    statuses: tuple[str, ...]
    metric_means: tuple[tuple[str, float], ...]
    state: str
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


class EvidenceLedger:
    """Aggregate observations without deciding truth or promotion."""

    def summarize(
        self,
        observations: Iterable[ExperimentObservation],
        *,
        min_completed: int = 2,
        agreement_tolerance: float = 0.10,
    ) -> EvidenceSummary:
        obs = tuple(observations)
        if not obs:
            raise ValueError("at least one observation is required")
        if min_completed < 1:
            raise ValueError("min_completed must be >= 1")
        if agreement_tolerance < 0:
            raise ValueError("agreement_tolerance must be >= 0")

        hypothesis = obs[0].hypothesis
        experiment = obs[0].experiment
        if any(o.hypothesis != hypothesis for o in obs):
            raise ValueError("observations must target one hypothesis")
        if any(o.experiment != experiment for o in obs):
            raise ValueError("observations must target one experiment")

        completed = tuple(o for o in obs if o.status == "completed")
        failed = tuple(o for o in obs if o.status == "failed")
        statuses = tuple(o.status for o in obs)

        names = sorted({name for o in completed for name, _ in o.metrics})
        means = []
        for name in names:
            values = [float(v) for o in completed for n, v in o.metrics if n == name]
            means.append((name, sum(values) / len(values)))

        if failed:
            state = "conflicting"
            rationale = f"{len(failed)} observation(s) failed; evidence requires review."
        elif len(completed) < min_completed:
            state = "insufficient"
            rationale = f"{len(completed)} completed observation(s); {min_completed} required."
        else:
            state = "convergent"
            for name in names:
                vals = [float(v) for o in completed for n, v in o.metrics if n == name]
                if len(vals) >= 2:
                    scale = max(max(abs(v) for v in vals), 1e-12)
                    if (max(vals) - min(vals)) / scale > agreement_tolerance:
                        state = "conflicting"
                        rationale = f"metric '{name}' varies beyond tolerance."
                        break
            else:
                rationale = "completed observations agree within the declared tolerance."

        return EvidenceSummary(
            hypothesis=hypothesis,
            experiment=experiment,
            observations=len(obs),
            completed=len(completed),
            failed=len(failed),
            statuses=statuses,
            metric_means=tuple(means),
            state=state,
            rationale=rationale,
        )
