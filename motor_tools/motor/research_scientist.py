"""Research Scientist: hypothesis triage and experiment orchestration.

Experimental, read-only layer. It never edits runtime code or promotes mutations.
The scientist can select a discriminator and execute a caller-supplied experiment
handler. Handlers are explicit capabilities: the scientist cannot invent tools,
write files, or mutate any runtime by itself.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping

from motor_tools.motor.research_loop import ResearchDecision


@dataclass(frozen=True)
class ResearchHypothesis:
    id: str
    statement: str
    discriminator: str
    experiment: str
    priority: float


@dataclass(frozen=True)
class ScientistDecision:
    trigger: str
    selected_hypothesis: str | None
    rationale: str
    next_experiment: str
    alternatives: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentObservation:
    experiment: str
    hypothesis: str
    status: str
    evidence: str
    metrics: tuple[tuple[str, float], ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ScientistCycle:
    decision: ScientistDecision
    observation: ExperimentObservation

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.to_dict(),
            "observation": self.observation.to_dict(),
        }


ExperimentHandler = Callable[[ResearchHypothesis], ExperimentObservation]


class ResearchScientist:
    """Converts evidence into competing, testable hypotheses.

    The policy is deterministic and conservative. It uses only recorded evidence
    and does not claim that a hypothesis is true.
    """

    def propose(self, result: ResearchDecision) -> tuple[ResearchHypothesis, ...]:
        if result.status == "mutated":
            return (
                ResearchHypothesis("H1", "the candidate is real rather than search noise",
                                   "fresh_seed_and_independent_holdout", "replicate_oos", 1.0),
                ResearchHypothesis("H2", "the candidate is regime-specific",
                                   "cross_regime_transfer", "regime_transfer", 0.8),
                ResearchHypothesis("H3", "an equivalent representation exists",
                                   "functional_equivalence", "quotient_check", 0.6),
            )
        if result.status == "accepted":
            return (
                ResearchHypothesis("H1", "the existing library explains the signal",
                                   "independent_holdout", "independent_holdout", 1.0),
                ResearchHypothesis("H2", "the explanation is regime-specific",
                                   "regime_shift", "regime_transfer", 0.9),
            )
        return (
            ResearchHypothesis("H1", "search coverage is insufficient",
                               "expanded_budget", "search_expansion", 1.0),
            ResearchHypothesis("H2", "the current representation is inadequate",
                               "representation_change", "representation_probe", 0.9),
            ResearchHypothesis("H3", "the signal is unstable or noisy",
                               "fresh_seed_repeat", "stability_probe", 0.8),
        )

    def decide(self, result: ResearchDecision) -> ScientistDecision:
        hs = self.propose(result)
        selected = hs[0] if hs else None
        rationale = (
            f"status={result.status}; gain={result.gain:.6g}; "
            f"evidence={result.evidence_strength}. Selected the highest-priority "
            "discriminator without treating it as proof."
        )
        return ScientistDecision(
            trigger=result.experiment,
            selected_hypothesis=selected.id if selected else None,
            rationale=rationale,
            next_experiment=selected.experiment if selected else "manual_review",
            alternatives=tuple(h.id for h in hs[1:]),
        )

    def execute(
        self,
        result: ResearchDecision,
        handlers: Mapping[str, ExperimentHandler],
    ) -> ScientistCycle:
        """Select and execute exactly one explicitly registered experiment.

        This is an orchestration boundary, not an autonomous capability grant:
        callers decide which experiment handlers exist and what they are allowed
        to do. The scientist never falls back to an unregistered action.
        """
        decision = self.decide(result)
        hypotheses = self.propose(result)
        if not hypotheses:
            raise RuntimeError("scientist produced no hypothesis")
        selected = hypotheses[0]
        handler = handlers.get(selected.experiment)
        if handler is None:
            raise KeyError(f"no experiment handler registered: {selected.experiment}")
        observation = handler(selected)
        if observation.experiment != selected.experiment:
            raise ValueError("experiment handler returned a mismatched experiment")
        if observation.hypothesis != selected.id:
            raise ValueError("experiment handler returned a mismatched hypothesis")
        return ScientistCycle(decision=decision, observation=observation)
