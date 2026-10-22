"""Research Loop: supervisor experimental para ciclos de descoberta.

A camada é deliberadamente externa a qualquer runtime. Ela transforma
um conjunto determinístico de amostras em um registro auditável:
hipótese -> baseline -> descoberta -> evidência -> decisão -> próxima ação.

Não altera o alvo, não promove mutações automaticamente e não grava estado global.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence
import json
import math
import time

import numpy as np

from motor_tools.motor.discovery_loop import ClosedLoopDiscovery, DiscoveryResult
from motor_tools.motor.mutagen import Mutagen, MutationPolicy
from motor_tools.motor.structure_inducer import StructureInducer

Sample = tuple[np.ndarray, float, np.ndarray]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    hypothesis: str
    expected_signal: str
    min_gain: float = 0.05
    max_trials: int | None = None
    seed: int = 0


@dataclass(frozen=True)
class EvidenceRecord:
    status: str
    initial_mae: float
    final_mae: float
    gain: float
    accepted_terms: tuple[str, ...]
    candidate_trials: int
    elapsed_s: float
    evidence_strength: str


@dataclass(frozen=True)
class ResearchDecision:
    experiment: str
    hypothesis: str
    baseline_mae: float
    final_mae: float
    gain: float
    status: str
    decision: str
    evidence_strength: str
    next_action: str
    accepted_terms: tuple[str, ...]
    candidate_trials: int
    elapsed_s: float

    def to_dict(self) -> dict:
        return asdict(self)


class ResearchLoop:
    """Executa experiências de descoberta e produz decisões auditáveis.

    A decisão é puramente descritiva:
      accepted -> "retain_candidate_for_review"
      unknown  -> "expand_experiment_or_keep_unknown"

    Nenhuma decisão deste objeto promove código ou altera o runtime.
    """

    def __init__(
        self,
        inducer_factory: Callable[[], StructureInducer] | None = None,
        mutagen_factory: Callable[[], Mutagen] | None = None,
    ) -> None:
        self.inducer_factory = inducer_factory or (
            lambda: StructureInducer(
                folds=4,
                max_extra_terms=4,
                min_relative_gain=0.001,
                max_cv_mae=0.5,
                max_cv_ratio=2.0,
            )
        )
        self.mutagen_factory = mutagen_factory or (
            lambda: Mutagen(max_candidates=48, composition_depth=2)
        )

    @staticmethod
    def _baseline(result: DiscoveryResult) -> float:
        return float(result.initial.best.cv_mae)

    @staticmethod
    def _final(result: DiscoveryResult) -> float:
        return float(result.final.best.cv_mae)

    @staticmethod
    def _strength(result: DiscoveryResult, gain: float) -> str:
        if result.status == "mutated" and gain >= 0.20:
            return "strong"
        if result.status == "mutated" and gain > 0.0:
            return "moderate"
        return "insufficient"

    def run(
        self,
        spec: ExperimentSpec,
        samples: Sequence[Sample],
    ) -> ResearchDecision:
        if len(samples) < 16:
            raise ValueError("research loop requires at least 16 samples")
        start = time.perf_counter()
        result = ClosedLoopDiscovery(
            inducer=self.inducer_factory(),
            mutagen=self.mutagen_factory(),
            min_gain=spec.min_gain,
            max_trials=spec.max_trials,
        ).run(samples, seed=spec.seed)
        elapsed = time.perf_counter() - start
        before = self._baseline(result)
        after = self._final(result)
        gain = (before - after) / max(abs(before), 1e-9)
        strength = self._strength(result, gain)
        accepted_terms = tuple(m.candidate.expression for m in result.accepted_mutations)

        if result.status == "mutated":
            decision = "retain_candidate_for_review"
            next_action = "repeat_on_fresh_seed_and_holdout"
        elif result.status == "accepted":
            decision = "library_already_explains_signal"
            next_action = "test_transfer_or_regime_shift"
        else:
            decision = "keep_unknown_no_promotion"
            next_action = "expand_search_or_collect_discriminating_evidence"

        return ResearchDecision(
            experiment=spec.name,
            hypothesis=spec.hypothesis,
            baseline_mae=before,
            final_mae=after,
            gain=float(gain),
            status=result.status,
            decision=decision,
            evidence_strength=strength,
            next_action=next_action,
            accepted_terms=accepted_terms,
            candidate_trials=int(result.candidate_trials),
            elapsed_s=float(elapsed),
        )

    @staticmethod
    def report(results: Sequence[ResearchDecision]) -> str:
        payload = {
            "schema": "motor_tools.research_loop.v1",
            "n_experiments": len(results),
            "results": [r.to_dict() for r in results],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
