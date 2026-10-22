"""Camada de adaptação JSON <-> motor.*. Sem estado, sem efeitos colaterais."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from motor_tools.motor.compositional_canonical import canonicalize_affine_expression
from motor_tools.motor.compositional_quotient import quotient_key
from motor_tools.motor.convergent_observation_gate import gate_novelty_with_convergent_observation_evidence
from motor_tools.motor.known_reference_attribution import KnownReferenceAttribution
from motor_tools.motor.novelty_detector import NoveltyDetector
from motor_tools.motor.novelty_evidence_gate import gate_novelty_evidence
from motor_tools.motor.observation_uncertainty import estimate_noise_profile
from motor_tools.motor.oos_statistics import (
    exact_one_sided_sign_pvalue,
    pairwise_paired_t_evidence,
    pairwise_permutation_evidence,
    pairwise_sign_evidence,
)
from motor_tools.motor.redundant_observation_consistency import RedundantObservationConsistency
from motor_tools.motor.relative_temporal_staleness import RelativeTemporalStalenessDetector
from motor_tools.motor.structural_consistency import StructuralConsistency
from motor_tools.motor.temporal_attribution import TemporalAttribution
from motor_tools.motor.temporal_information import TemporalInformationInsufficiencyDetector
from motor_tools.motor.temporal_staleness import TemporalStalenessDetector
from motor_tools.motor.uncertainty_state import combine_uncertainty

from motor_tools.motor.agi_active_recovery import run_active_recovery
from motor_tools.motor.agi_generalization_battery import run_battery as run_generalization_battery
from motor_tools.motor.agi_transfer_robustness import run_robustness_battery
from motor_tools.motor.observability_quality import ObservabilityQuality
from motor_tools.motor.research_regime_transfer import run_regime_transfer
from motor_tools.motor.discovery_loop import ClosedLoopDiscovery
from motor_tools.motor.mutagen import Mutagen
from motor_tools.motor.research_evidence import EvidenceLedger
from motor_tools.motor.research_loop import ExperimentSpec, ResearchDecision, ResearchLoop
from motor_tools.motor.research_scientist import ExperimentObservation, ResearchScientist
from motor_tools.motor.structure_inducer import StructureInducer


def sign_test(wins: int, losses: int) -> Dict[str, Any]:
    """P-valor unilateral exato do teste de sinais (empates já excluídos)."""
    p = exact_one_sided_sign_pvalue(wins, losses)
    return {"wins": int(wins), "losses": int(losses), "p_value": p}


def compare_paired_errors(
    errors: Dict[str, List[float]],
    best: str,
    alpha: float = 0.05,
    method: str = "sign",
    n_resamples: int = 100_000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Testa se o método `best` tem erro menor que todos os outros (pareado).

    errors: {nome_do_método: [erro por item]}, mesmo tamanho e mesma ordem.
    method: 'sign' (dominância por sinais), 't' (média, t pareado) ou
    'permutation' (média, sign-flip). Bonferroni sobre todas as comparações.
    """
    if method == "sign":
        ok, pv, counts, a = pairwise_sign_evidence(errors, best, alpha)
        return {
            "method": method, "supported": bool(ok), "alpha_per_pair": a,
            "p_values": pv,
            "counts_wins_losses_ties": {k: list(v) for k, v in counts.items()},
        }
    if method == "t":
        ok, pv, a = pairwise_paired_t_evidence(errors, best, alpha)
    elif method == "permutation":
        ok, pv, a = pairwise_permutation_evidence(
            errors, best, alpha, n_resamples=n_resamples, seed=seed
        )
    else:
        raise ValueError("method must be 'sign', 't' or 'permutation'")
    return {"method": method, "supported": bool(ok), "alpha_per_pair": a, "p_values": pv}


def novelty_gate(raw_novelty: bool, temporal_information_insufficient: bool) -> Dict[str, Any]:
    """Só emite evidência forte de mudança de mundo se houver informação temporal suficiente."""
    d = gate_novelty_evidence(
        raw_novelty=raw_novelty,
        temporal_information_insufficient=temporal_information_insufficient,
    )
    return {
        "raw_novelty": d.raw_novelty,
        "temporal_information_insufficient": d.temporal_information_insufficient,
        "emit_strong_world_shift_evidence": d.emit_strong_world_shift_evidence,
        "abstained": d.abstained,
    }


def staleness_check(
    calibration: List[List[float]],
    stream: List[List[float]],
    alpha: float = 0.01,
    min_run: int = 2,
) -> Dict[str, Any]:
    """Detecta sensor/canal 'congelado'.

    calibration: matriz (>=3 linhas) de observações limpas, colunas = canais.
    stream: observações a avaliar, em ordem temporal.
    """
    det = TemporalStalenessDetector(alpha=alpha, min_run=min_run).fit(np.asarray(calibration, float))
    prev: Optional[np.ndarray] = None
    steps = []
    for row in stream:
        x = np.asarray(row, float)
        st = det.evaluate(x, prev)
        steps.append({
            "stale": st.stale,
            "channel_p_values": list(st.channel_p_values),
            "run_lengths": list(st.run_lengths),
        })
        prev = x
    return {"any_stale": any(s["stale"] for s in steps), "steps": steps}


def structural_check(
    calibration_windows: List[List[List[float]]],
    window: List[List[float]],
    alpha: float = 0.01,
    min_window: int = 5,
) -> Dict[str, Any]:
    """Compara a dinâmica de uma janela temporal com janelas de calibração limpas.

    Distingue 'mecanismo mudou' (world_shift_evidence) de 'só escala/offset da
    observação mudou' (transform_evidence). Precisa de >= 10 janelas de calibração.
    """
    sc = StructuralConsistency(alpha=alpha, min_window=min_window)
    sc.fit([np.asarray(w, float) for w in calibration_windows])
    s = sc.evaluate(np.asarray(window, float))
    return {
        "dynamic_distance": s.dynamic_distance,
        "dynamic_consistent": s.dynamic_consistent,
        "mean_shift": s.mean_shift,
        "scale_shift": s.scale_shift,
        "affine_offset": s.affine_offset,
        "transform_evidence": s.transform_evidence,
        "world_shift_evidence": s.world_shift_evidence,
        "sample_count": s.sample_count,
    }


def novelty_score(
    calibration: List[List[float]],
    queries: List[List[float]],
    k: int = 8,
    threshold: float = 0.95,
) -> Dict[str, Any]:
    """Detecta novidade / fora-de-suporte por distância k-NN padronizada.

    calibration: pontos considerados in-distribution.
    queries: pontos a avaliar.
    score alto = menos suportado. regime = 'novel' se score >= threshold.
    """
    det = NoveltyDetector(k=k, threshold=threshold).fit(np.asarray(calibration, float))
    Xq = np.asarray(queries, float)
    if Xq.ndim == 1:
        Xq = Xq.reshape(1, -1)
    states = det.predict(Xq)
    if not isinstance(states, list):
        states = [states]
    return {
        "results": [
            {
                "novelty_score": s.novelty_score,
                "nearest_distance": s.nearest_distance,
                "support_count": s.support_count,
                "regime": s.regime,
            }
            for s in states
        ]
    }


def temporal_information_check(
    calibration: List[List[float]],
    stream: List[List[float]],
    alpha: float = 0.01,
    min_run: int = 2,
) -> Dict[str, Any]:
    """Detecta informação temporal agregada insuficientemente baixa.

    calibration: observações limpas (>=3 linhas).
    stream: sequência temporal a avaliar.
    """
    det = TemporalInformationInsufficiencyDetector(alpha=alpha, min_run=min_run)
    det.fit(np.asarray(calibration, float))
    prev: Optional[np.ndarray] = None
    steps = []
    for row in stream:
        x = np.asarray(row, float)
        st = det.evaluate(x, prev)
        steps.append({
            "insufficient": st.insufficient,
            "p_value": st.p_value,
            "normalized_change": st.normalized_change,
            "run_length": st.run_length,
        })
        prev = x
    return {
        "any_insufficient": any(s["insufficient"] for s in steps),
        "steps": steps,
    }


def redundant_consistency_check(
    calibration_pairs: List[List[List[List[float]]]],
    pair: List[List[List[float]]],
    alpha: float = 0.01,
) -> Dict[str, Any]:
    """Verifica consistência entre duas observações redundantes (A, B).

    calibration_pairs: lista de pares [A, B] limpos (>=20).
    pair: [A, B] a avaliar (cada um é uma matriz temporal).
    """
    if len(pair) != 2:
        raise ValueError("pair must be [A, B]")
    pairs = []
    for p in calibration_pairs:
        if len(p) != 2:
            raise ValueError("each calibration pair must be [A, B]")
        pairs.append((np.asarray(p[0], float), np.asarray(p[1], float)))
    roc = RedundantObservationConsistency(alpha=alpha)
    roc.fit(pairs)
    st = roc.evaluate(np.asarray(pair[0], float), np.asarray(pair[1], float))
    return {
        "residual": st.residual,
        "consistent": st.consistent,
        "samples": st.samples,
    }


def relative_staleness_check(
    calibration: List[List[float]],
    stream: List[List[float]],
    alpha: float = 0.01,
    active_alpha: float = 0.10,
    min_run: int = 2,
) -> Dict[str, Any]:
    """Detecta canal congelado de forma relativa (só se outros canais ainda se movem).

    Diferente de staleness_check: exige que pelo menos um outro canal esteja ativo
    para declarar um canal stale — evita confundir regime lento global com sensor morto.
    """
    det = RelativeTemporalStalenessDetector(
        alpha=alpha, active_alpha=active_alpha, min_run=min_run
    ).fit(np.asarray(calibration, float))
    prev: Optional[np.ndarray] = None
    steps = []
    for row in stream:
        x = np.asarray(row, float)
        st = det.evaluate(x, prev)
        steps.append({
            "stale": st.stale,
            "channel_p_values": list(st.channel_p_values),
            "low_change_channels": list(st.low_change_channels),
            "active_change_channels": list(st.active_change_channels),
            "run_lengths": list(st.run_lengths),
        })
        prev = x
    return {"any_stale": any(s["stale"] for s in steps), "steps": steps}


def estimate_noise(
    samples: List[List[float]],
) -> Dict[str, Any]:
    """Estima ruído de observação por canal via MAD robusto.

    samples: matriz repeats x features (pelo menos 3 repetições do mesmo estado latente).
    """
    profile = estimate_noise_profile(np.asarray(samples, float), axis=0)
    return {
        "sigma": [float(v) for v in profile.sigma],
        "pooled_sigma": float(profile.pooled_sigma),
        "n_repeats": int(profile.n_repeats),
    }


def convergent_novelty_gate(
    raw_novelty: bool,
    temporal_information_insufficient: bool,
    observability_degraded: bool,
    relative_temporal_stale: bool,
    minimum_degradation_evidence: int = 2,
) -> Dict[str, Any]:
    """Gate de novidade com convergência de evidências de degradação observacional.

    Só absteve evidência forte de WORLD_SHIFT se pelo menos
    `minimum_degradation_evidence` sinais de degradação estiverem ativos.
    """
    d = gate_novelty_with_convergent_observation_evidence(
        raw_novelty=raw_novelty,
        temporal_information_insufficient=temporal_information_insufficient,
        observability_degraded=observability_degraded,
        relative_temporal_stale=relative_temporal_stale,
        minimum_degradation_evidence=minimum_degradation_evidence,
    )
    return {
        "raw_novelty": d.raw_novelty,
        "temporal_information_insufficient": d.temporal_information_insufficient,
        "observability_degraded": d.observability_degraded,
        "relative_temporal_stale": d.relative_temporal_stale,
        "degradation_evidence_count": d.degradation_evidence_count,
        "minimum_degradation_evidence": d.minimum_degradation_evidence,
        "abstained": d.abstained,
        "emit_strong_world_shift_evidence": d.emit_strong_world_shift_evidence,
    }


def combine_uncertainties(
    epistemic: float,
    observation_sigma: float,
    source: str = "experimental",
) -> Dict[str, Any]:
    """Combina incerteza epistêmica e aleatória (ruído de observação) em RMS."""
    u = combine_uncertainty(epistemic, observation_sigma, source=source)
    return {
        "epistemic": u.epistemic,
        "aleatoric": u.aleatoric,
        "total_rms": u.total_rms,
        "epistemic_fraction": u.epistemic_fraction,
        "aleatoric_fraction": u.aleatoric_fraction,
        "source": u.source,
    }


def temporal_attribution_check(
    transform_evidence: List[bool],
    world_shift_evidence: List[bool],
    majority: float = 0.50,
) -> Dict[str, Any]:
    """Agrega evidências booleanas de transform vs world-shift ao longo de uma sequência.

    label: NORMAL | OBSERVATION_TRANSFORM | WORLD_SHIFT | AMBIGUOUS
    """
    st = TemporalAttribution(majority=majority).classify(
        transform_evidence, world_shift_evidence
    )
    return {
        "transform_rate": st.transform_rate,
        "world_shift_rate": st.world_shift_rate,
        "label": st.label,
    }


def known_reference_check(
    calibration_expected: List[List[List[float]]],
    calibration_observed: List[List[List[float]]],
    expected: List[List[float]],
    observed: List[List[float]],
    alpha: float = 0.01,
) -> Dict[str, Any]:
    """Compara referência conhecida esperada vs observada (calibração + avaliação).

    calibration_*: listas de janelas (>=20) esperadas e observadas.
    """
    kra = KnownReferenceAttribution(alpha=alpha)
    kra.fit(
        [np.asarray(w, float) for w in calibration_expected],
        [np.asarray(w, float) for w in calibration_observed],
    )
    st = kra.evaluate(np.asarray(expected, float), np.asarray(observed, float))
    return {
        "residual": st.residual,
        "transform_evidence": st.transform_evidence,
        "samples": st.samples,
    }


def canonicalize_expression(expression: str) -> Dict[str, Any]:
    """Canonicaliza expressão afim (ex.: '2*x+1') para forma normal e chave de quociente."""
    canonical = canonicalize_affine_expression(expression)
    qk = quotient_key(expression)
    return {
        "expression": expression,
        "canonical": canonical,
        "quotient_key": qk,
        "changed": canonical != expression,
    }



def _summary_to_dict(obj) -> Dict[str, Any]:
    """Converte dataclass / objeto de resultado em dict JSON-serializável."""
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
        return asdict(obj)
    if hasattr(obj, '__dict__'):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
    return {"value": str(obj)}


def agi_robustness_battery(seed: int = 9101) -> Dict[str, Any]:
    """Bateria de robustez a mudança de regime (transfer + recovery).

    Usa o domínio AlienPhysics embutido. Retorna taxas de retenção/recuperação e MAE.
    """
    s = run_robustness_battery(seed=int(seed))
    d = _summary_to_dict(s)
    # results may be nested dataclasses
    if 'results' in d:
        d['results'] = [_summary_to_dict(r) if not isinstance(r, dict) else r for r in d['results']]
    return d


def agi_active_recovery_battery(seed: int = 9101, budget: int = 10) -> Dict[str, Any]:
    """Compara políticas de recuperação ativa após shift (fixed/random/disagreement)."""
    s = run_active_recovery(seed=int(seed), budget=int(budget))
    d = _summary_to_dict(s)
    if 'results' in d:
        d['results'] = [_summary_to_dict(r) if not isinstance(r, dict) else r for r in d['results']]
    if 'policies' in d and not isinstance(d['policies'], list):
        d['policies'] = list(d['policies'])
    return d


def agi_generalization_battery() -> Dict[str, Any]:
    """Bateria de generalização composicional (holdout + transfer)."""
    s = run_generalization_battery()
    d = _summary_to_dict(s)
    if 'results' in d:
        d['results'] = [_summary_to_dict(r) if not isinstance(r, dict) else r for r in d['results']]
    return d


def regime_transfer_battery(seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    """Experimento de transferência entre regimes (várias seeds)."""
    if seeds is None:
        obs = run_regime_transfer()
    else:
        obs = run_regime_transfer(seeds=seeds)
    items = []
    for o in obs:
        items.append({
            "experiment": o.experiment,
            "hypothesis": o.hypothesis,
            "status": o.status,
            "evidence": o.evidence,
            "metrics": dict(o.metrics) if o.metrics is not None else {},
        })
    return {"observations": items, "n": len(items)}


def observability_check(
    calibration: List[List[float]],
    stream: List[List[float]],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Qualidade de observabilidade (staleness/penalties) em stream calibrado."""
    det = ObservabilityQuality(alpha=alpha).fit(np.asarray(calibration, float))
    prev = None
    steps = []
    for row in stream:
        x = np.asarray(row, float)
        st = det.evaluate(x, previous=prev)
        steps.append(_summary_to_dict(st) if hasattr(st, '__dataclass_fields__') else {
            "score": getattr(st, 'score', None),
            "degraded": getattr(st, 'degraded', None),
        })
        # fill common fields if dataclass-like
        if hasattr(st, '__dict__'):
            steps[-1] = {k: (list(v) if hasattr(v, 'tolist') else v)
                         for k, v in st.__dict__.items() if not k.startswith('_')}
        prev = x
    return {"steps": steps}


# ─── Descoberta simbólica e evidência científica (v0.8.0) ────────────────────

_HYPOTHESIS_STATUSES = ("mutated", "accepted", "unknown")
_EVIDENCE_STRENGTHS = ("strong", "moderate", "insufficient")
_CV_STRATEGIES = ("iid", "time_series")


def _fail(exc: Exception) -> Dict[str, Any]:
    """Resposta fail-soft padrão das tools de descoberta."""
    return {"supported": False, "error": f"{type(exc).__name__}: {exc}"}


def _check_cv_strategy(cv_strategy: Any) -> str:
    """Valida a estratégia de validação: "iid" ou "time_series"."""
    if not isinstance(cv_strategy, str) or cv_strategy not in _CV_STRATEGIES:
        raise ValueError("cv_strategy must be 'iid' or 'time_series'")
    return cv_strategy


def _check_seed(seed: Any) -> int:
    """Valida a semente: inteiro >= 0 (bool, texto, nulo e fracionários são rejeitados)."""
    ok = isinstance(seed, (int, float)) and not isinstance(seed, bool)
    if ok and isinstance(seed, float):
        ok = seed.is_integer()  # rejeita 1.5, NaN e infinitos
    if not ok or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(seed)


def _to_samples(payload: Any) -> List[tuple]:
    """Converte [[state, action, outcome], ...] em amostras NumPy validadas.

    state e outcome são vetores de mesma dimensão (outcome = próximo estado),
    action é um escalar. Valores não finitos são rejeitados.
    """
    if not isinstance(payload, (list, tuple)) or len(payload) == 0:
        raise ValueError("samples must be a non-empty list of [state, action, outcome]")
    out: List[tuple] = []
    for i, item in enumerate(payload):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"samples[{i}] must be [state, action, outcome]")
        s, a, o = item
        state = np.asarray(s, dtype=np.float64).reshape(-1)
        outcome = np.asarray(o, dtype=np.float64).reshape(-1)
        action = float(a)
        if state.size == 0 or state.size != outcome.size:
            raise ValueError(f"samples[{i}]: state and outcome must be non-empty and have the same size")
        if not (np.isfinite(state).all() and np.isfinite(outcome).all() and np.isfinite(action)):
            raise ValueError(f"samples[{i}] contains non-finite values")
        out.append((state, action, outcome))
    if len({s.size for s, _, _ in out}) != 1:
        raise ValueError("all samples must have the same state dimension")
    return out


def _discovery_to_dict(d) -> Dict[str, Any]:
    return {
        "accepted": bool(d.accepted),
        "rejection_reason": str(d.rejection_reason),
        "coverage": float(d.coverage),
        "active_terms": list(d.active_terms),
        "best": {
            "name": str(d.best.name),
            "terms": list(d.best.terms),
            "complexity": int(d.best.complexity),
            "cv_mae": float(d.best.cv_mae),
            "score": float(d.best.score),
        },
    }


def induce_structure(
    samples: list,
    folds: int = 5,
    seed: int = 0,
    cv_strategy: str = "iid",
) -> Dict[str, Any]:
    """Descobre a estrutura simbólica (termos) que explica as amostras.

    samples: lista de [state, action, outcome], com pelo menos folds*4 itens.
    folds: número de dobras da validação cruzada (>= 2).
    seed: inteiro >= 0 (padrão 0). Em cv_strategy="iid" define o embaralhamento
    das dobras; mesma semente + mesmos dados = mesmo resultado.
    cv_strategy: "iid" (padrão, dobras embaralhadas) ou "time_series"
    (walk-forward: cada dobra de teste é posterior ao treino; envie as amostras
    em ordem cronológica, da mais antiga para a mais nova).
    Retorna os termos aceitos e o erro de validação cruzada.
    """
    try:
        n_folds = int(folds)
        if n_folds < 2:
            raise ValueError("folds must be >= 2")
        run_seed = _check_seed(seed)
        strategy = _check_cv_strategy(cv_strategy)
        result = StructureInducer(
            folds=n_folds, cv_strategy=strategy, seed=run_seed
        ).fit(_to_samples(samples))
        return {
            "supported": True,
            "seed": run_seed,
            "cv_strategy": strategy,
            **_discovery_to_dict(result),
        }
    except Exception as e:  # fail-soft: nunca propaga
        return _fail(e)


def closed_loop_discovery(
    samples: list,
    min_gain: float = 0.05,
    max_trials: int = 48,
    composition_depth: int = 2,
    seed: int = 0,
    cv_strategy: str = "iid",
) -> Dict[str, Any]:
    """Induz estrutura e aplica mutações simbólicas até aceitar melhoria.

    samples: lista de [state, action, outcome], com pelo menos 16 itens.
    Os candidatos são escolhidos só com dados de desenvolvimento (validação
    cruzada); um holdout externo selado (20% das amostras, mínimo 4) mede apenas
    o baseline e o candidato final. Retorna o status (accepted | mutated |
    unknown), as mutações aceitas e o antes/depois do erro (MAE) nesse holdout.
    max_trials: 1..500. composition_depth: 1..3.
    seed: inteiro >= 0 (padrão 0). Em cv_strategy="iid" define o embaralhamento
    das dobras e a escolha do holdout; mesma semente + mesmos dados = mesmo resultado.
    cv_strategy: "iid" (padrão) ou "time_series" (walk-forward: cada teste é
    posterior ao treino e o holdout são as últimas amostras; envie as amostras
    em ordem cronológica, da mais antiga para a mais nova).
    """
    try:
        trials = int(max_trials)
        depth = int(composition_depth)
        if not 1 <= trials <= 500:
            raise ValueError("max_trials must be between 1 and 500")
        if not 1 <= depth <= 3:
            raise ValueError("composition_depth must be between 1 and 3")
        run_seed = _check_seed(seed)
        strategy = _check_cv_strategy(cv_strategy)
        loop = ClosedLoopDiscovery(
            inducer=StructureInducer(cv_strategy=strategy, seed=run_seed),
            mutagen=Mutagen(max_candidates=trials, composition_depth=depth),
            min_gain=float(min_gain),
        )
        result = loop.run(_to_samples(samples), seed=run_seed)
        return {
            "supported": True,
            "seed": run_seed,
            "cv_strategy": strategy,
            "status": str(result.status),
            "candidate_trials": int(result.candidate_trials),
            "initial": _discovery_to_dict(result.initial),
            "final": _discovery_to_dict(result.final),
            "accepted_mutations": [
                {
                    "expression": str(m.candidate.expression),
                    "complexity": int(m.candidate.complexity),
                    "family": str(m.candidate.family),
                    "gain": float(m.gain),
                    "before_mae": float(m.before_mae),
                    "after_mae": float(m.after_mae),
                }
                for m in result.accepted_mutations
            ],
        }
    except Exception as e:  # fail-soft: nunca propaga
        return _fail(e)


def summarize_evidence(
    observations: list,
    min_completed: int = 2,
    agreement_tolerance: float = 0.10,
) -> Dict[str, Any]:
    """Agrega observações de UM experimento/hipótese em evidência convergent | conflicting | insufficient.

    observations: lista de dicts com experiment, hypothesis, status ('completed' | 'failed'),
    evidence (str, opcional) e metrics (dict nome -> valor, opcional).
    Todas as observações devem ter o mesmo experiment e a mesma hypothesis.
    """
    try:
        if not isinstance(observations, (list, tuple)):
            raise ValueError("observations must be a list of objects")
        obs = []
        for i, o in enumerate(observations):
            if not isinstance(o, dict):
                raise ValueError(f"observations[{i}] must be an object")
            metrics = o.get("metrics") or {}
            if not isinstance(metrics, dict):
                raise ValueError(f"observations[{i}].metrics must be an object")
            pairs = tuple((str(k), float(v)) for k, v in metrics.items())
            if not all(math.isfinite(v) for _, v in pairs):
                raise ValueError(f"observations[{i}].metrics contains non-finite values")
            obs.append(
                ExperimentObservation(
                    experiment=str(o["experiment"]),
                    hypothesis=str(o["hypothesis"]),
                    status=str(o["status"]),
                    evidence=str(o.get("evidence", "")),
                    metrics=pairs,
                )
            )
        summary = EvidenceLedger().summarize(
            obs,
            min_completed=int(min_completed),
            agreement_tolerance=float(agreement_tolerance),
        )
        return {"supported": True, **summary.to_dict()}
    except Exception as e:  # fail-soft: nunca propaga
        return _fail(e)


def propose_hypotheses(
    status: str,
    gain: float = 0.0,
    evidence_strength: str = "insufficient",
) -> Dict[str, Any]:
    """Gera hipóteses concorrentes (H1/H2/H3) e o discriminador de cada uma para o próximo passo.

    status: 'mutated' | 'accepted' | 'unknown' (resultado de closed_loop_discovery).
    evidence_strength: 'strong' | 'moderate' | 'insufficient'.
    """
    try:
        if status not in _HYPOTHESIS_STATUSES:
            raise ValueError(f"status must be one of {list(_HYPOTHESIS_STATUSES)}, got {status!r}")
        if evidence_strength not in _EVIDENCE_STRENGTHS:
            raise ValueError(
                f"evidence_strength must be one of {list(_EVIDENCE_STRENGTHS)}, got {evidence_strength!r}"
            )
        decision = ResearchDecision(
            experiment="manual_trigger",
            hypothesis="H0",
            baseline_mae=0.0,
            final_mae=0.0,
            gain=float(gain),
            status=status,
            decision="manual",
            evidence_strength=evidence_strength,
            next_action="manual",
            accepted_terms=(),
            candidate_trials=0,
            elapsed_s=0.0,
        )
        hypotheses = ResearchScientist().propose(decision)
        return {
            "supported": True,
            "hypotheses": [
                {
                    "id": str(h.id),
                    "statement": str(h.statement),
                    "discriminator": str(h.discriminator),
                    "experiment": str(h.experiment),
                    "priority": float(h.priority),
                }
                for h in hypotheses
            ],
        }
    except Exception as e:  # fail-soft: nunca propaga
        return _fail(e)


def run_research_experiment(
    samples: list,
    experiment: str = "default",
    hypothesis: str = "H1",
    min_gain: float = 0.05,
) -> Dict[str, Any]:
    """Roda um experimento completo de descoberta auditável.

    samples: lista de [state, action, outcome], com pelo menos 16 itens.
    Retorna status, ganho, termos aceitos e próxima ação sugerida.
    """
    try:
        spec = ExperimentSpec(
            name=str(experiment),
            hypothesis=str(hypothesis),
            expected_signal="sparse_terms",
            min_gain=float(min_gain),
        )
        decision = ResearchLoop().run(spec, _to_samples(samples))
        return {"supported": True, **decision.to_dict()}
    except Exception as e:  # fail-soft: nunca propaga
        return _fail(e)


TOOLS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "sign_test": sign_test,
    "compare_paired_errors": compare_paired_errors,
    "novelty_gate": novelty_gate,
    "staleness_check": staleness_check,
    "structural_check": structural_check,
    "novelty_score": novelty_score,
    "temporal_information_check": temporal_information_check,
    "redundant_consistency_check": redundant_consistency_check,
    "relative_staleness_check": relative_staleness_check,
    "estimate_noise": estimate_noise,
    "convergent_novelty_gate": convergent_novelty_gate,
    "combine_uncertainties": combine_uncertainties,
    "temporal_attribution_check": temporal_attribution_check,
    "known_reference_check": known_reference_check,
    "canonicalize_expression": canonicalize_expression,
    "agi_robustness_battery": agi_robustness_battery,
    "agi_active_recovery_battery": agi_active_recovery_battery,
    "agi_generalization_battery": agi_generalization_battery,
    "regime_transfer_battery": regime_transfer_battery,
    "observability_check": observability_check,
    # descoberta simbólica e evidência científica (v0.8.0)
    "induce_structure": induce_structure,
    "closed_loop_discovery": closed_loop_discovery,
    "summarize_evidence": summarize_evidence,
    "propose_hypotheses": propose_hypotheses,
    "run_research_experiment": run_research_experiment,
}


def call_tool(name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Chama uma ferramenta pelo nome com argumentos JSON.

    Retorna {"supported": False, "error": "..."} em caso de erro.
    NUNCA levanta exceção para o caller (fail-soft).
    """
    if not isinstance(name, str) or name not in TOOLS:
        return {
            "supported": False,
            "error": f"unknown tool {name!r}",
            "available": sorted(TOOLS),
        }
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return {
            "supported": False,
            "error": f"TypeError: arguments must be an object, got {type(arguments).__name__}",
            "tool": name,
        }
    try:
        return TOOLS[name](**arguments)
    except Exception as e:  # fronteira fail-soft: qualquer falha vira resposta
        return {
            "supported": False,
            "error": f"{type(e).__name__}: {e}",
            "tool": name,
        }
