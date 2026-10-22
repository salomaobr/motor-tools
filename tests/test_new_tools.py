import math

import numpy as np
import pytest

from motor_tools import call_tool, TOOLS
from motor_tools.api import _to_samples
from motor_tools.motor.compositional_frontier import CompositionalFrontierSearch
from motor_tools.motor.discovery_loop import ClosedLoopDiscovery
from motor_tools.motor.mutagen import Mutagen
from motor_tools.motor.structure_inducer import StructureInducer

NEW_TOOLS = (
    "induce_structure",
    "closed_loop_discovery",
    "summarize_evidence",
    "propose_hypotheses",
    "run_research_experiment",
)


def _fake_samples(n=40):
    """Oscilador harmônico discretizado: x' = x + 0.1*v, v' = v - 0.1*x."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        state = rng.normal(0, 1, 2)
        x, v = state
        next_x = x + 0.1 * v
        next_v = v - 0.1 * x
        out.append([state.tolist(), 0.0, [next_x, next_v]])
    return out


def _cos_samples(n=120, seed=1):
    """Sinal fora da biblioteca inicial: s1' = s1 + 4*cos(s1) (s0 constante)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = rng.normal(0, 2, 2)
        out.append([s.tolist(), 0.0, [float(s[0]), float(s[1] + 4 * np.cos(s[1]))]])
    return out


def test_tools_has_25():
    assert len(TOOLS) == 25
    for name in NEW_TOOLS:
        assert name in TOOLS


def test_induce_structure_returns_dict():
    r = call_tool("induce_structure", {"samples": _fake_samples(60)})
    assert r["supported"] is True
    assert "active_terms" in r
    assert "best" in r
    assert r["best"]["cv_mae"] >= 0


def test_induce_structure_fail_soft_on_bad_input():
    r = call_tool("induce_structure", {"samples": []})
    assert r["supported"] is False
    assert "error" in r


def test_induce_structure_too_few_samples():
    r = call_tool("induce_structure", {"samples": _fake_samples(10)})
    assert r["supported"] is False
    assert "not enough samples" in r["error"]


def test_induce_structure_rejects_non_finite_values():
    samples = _fake_samples(30)
    samples[0][0][0] = float("nan")
    r = call_tool("induce_structure", {"samples": samples})
    assert r["supported"] is False
    assert "non-finite" in r["error"]


def test_induce_structure_rejects_malformed_items():
    for bad in ([[1, 2]] * 30, [[[1.0, 2.0], 0.0, [1.0]]] * 30, "abc", 7):
        r = call_tool("induce_structure", {"samples": bad})
        assert r["supported"] is False, bad
        assert "error" in r


def test_closed_loop_discovery_returns_status():
    r = call_tool("closed_loop_discovery",
                  {"samples": _fake_samples(60), "max_trials": 12})
    assert r["supported"] is True
    assert r["status"] in ("accepted", "mutated", "unknown")
    assert {"initial", "final", "accepted_mutations", "candidate_trials"} <= set(r)


def test_closed_loop_discovery_finds_mutation():
    r = call_tool("closed_loop_discovery", {"samples": _cos_samples()})
    assert r["supported"] is True
    assert r["status"] == "mutated"
    assert r["initial"]["accepted"] is False
    m = r["accepted_mutations"][0]
    assert m["expression"] == "cos(s1)"
    assert m["after_mae"] < m["before_mae"] and m["gain"] >= 0.05
    assert r["final"]["best"]["cv_mae"] < r["initial"]["best"]["cv_mae"]
    assert r["candidate_trials"] >= 1


@pytest.mark.parametrize("kw", [
    {"max_trials": 0}, {"max_trials": 501},
    {"composition_depth": 0}, {"composition_depth": 4},
])
def test_closed_loop_discovery_rejects_out_of_range_params(kw):
    r = call_tool("closed_loop_discovery", {"samples": _fake_samples(60), **kw})
    assert r["supported"] is False
    assert "error" in r


def test_summarize_evidence_convergent():
    obs = [
        {"experiment": "exp1", "hypothesis": "H1", "status": "completed",
         "metrics": {"mae": 0.10}},
        {"experiment": "exp1", "hypothesis": "H1", "status": "completed",
         "metrics": {"mae": 0.11}},
    ]
    r = call_tool("summarize_evidence", {"observations": obs})
    assert r["supported"] is True
    assert r["state"] == "convergent"


def test_summarize_evidence_conflicting_and_insufficient():
    base = {"experiment": "exp1", "hypothesis": "H1", "status": "completed"}
    far = [dict(base, metrics={"mae": 0.1}), dict(base, metrics={"mae": 0.5})]
    assert call_tool("summarize_evidence", {"observations": far})["state"] == "conflicting"
    one = [dict(base, metrics={"mae": 0.1})]
    assert call_tool("summarize_evidence", {"observations": one})["state"] == "insufficient"
    failed = [dict(base, status="failed"), dict(base)]
    assert call_tool("summarize_evidence", {"observations": failed})["state"] == "conflicting"


def test_summarize_evidence_fail_soft():
    r = call_tool("summarize_evidence", {"observations": []})
    assert r["supported"] is False


def test_summarize_evidence_rejects_bad_observations():
    mixed = [
        {"experiment": "a", "hypothesis": "H1", "status": "completed"},
        {"experiment": "b", "hypothesis": "H1", "status": "completed"},
    ]
    for bad in (mixed, ["x"], [{"experiment": "a"}],
                [{"experiment": "a", "hypothesis": "H1", "status": "completed",
                  "metrics": {"mae": float("inf")}}]):
        r = call_tool("summarize_evidence", {"observations": bad})
        assert r["supported"] is False, bad
        assert "error" in r


def test_propose_hypotheses_mutated():
    r = call_tool("propose_hypotheses",
                  {"status": "mutated", "gain": 0.25,
                   "evidence_strength": "strong"})
    assert r["supported"] is True
    assert len(r["hypotheses"]) >= 2
    assert r["hypotheses"][0]["id"] == "H1"


@pytest.mark.parametrize("status,n", [("mutated", 3), ("accepted", 2), ("unknown", 3)])
def test_propose_hypotheses_per_status(status, n):
    r = call_tool("propose_hypotheses", {"status": status})
    assert r["supported"] is True
    assert len(r["hypotheses"]) == n


def test_propose_hypotheses_fail_soft():
    r = call_tool("propose_hypotheses", {})
    assert r["supported"] is False


def test_propose_hypotheses_rejects_invalid_enums():
    assert call_tool("propose_hypotheses", {"status": "bogus"})["supported"] is False
    r = call_tool("propose_hypotheses", {"status": "mutated", "evidence_strength": "weak"})
    assert r["supported"] is False


def test_run_research_experiment_returns_decision():
    r = call_tool("run_research_experiment",
                  {"samples": _fake_samples(60), "min_gain": 0.01})
    assert r["supported"] is True
    assert "status" in r
    assert "decision" in r
    assert "next_action" in r


def test_run_research_experiment_fail_soft():
    r = call_tool("run_research_experiment", {"samples": []})
    assert r["supported"] is False


def test_run_research_experiment_needs_16_samples():
    r = call_tool("run_research_experiment", {"samples": _fake_samples(10)})
    assert r["supported"] is False
    assert "16" in r["error"]


@pytest.mark.http
def test_new_tools_over_http():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from motor_tools.http_server import app

    c = TestClient(app)
    listed = set(c.get("/tools").json())
    assert len(listed) == 25 and set(NEW_TOOLS) <= listed
    samples = _fake_samples(60)

    r = c.post("/tools/induce_structure", json={"samples": samples})
    assert r.status_code == 200 and r.json()["supported"] is True
    r = c.post("/tools/closed_loop_discovery", json={"samples": samples, "max_trials": 8})
    assert r.status_code == 200 and r.json()["status"] in ("accepted", "mutated", "unknown")
    r = c.post("/tools/closed_loop_discovery", json={"samples": _cos_samples()})
    assert r.status_code == 200 and r.json()["status"] == "mutated"
    assert r.json()["accepted_mutations"][0]["expression"] == "cos(s1)"
    r = c.post("/tools/run_research_experiment", json={"samples": samples, "min_gain": 0.01})
    assert r.status_code == 200 and r.json()["supported"] is True
    obs = [{"experiment": "e", "hypothesis": "H1", "status": "completed", "metrics": {"mae": 0.1}}] * 2
    r = c.post("/tools/summarize_evidence", json={"observations": obs})
    assert r.status_code == 200 and r.json()["state"] == "convergent"
    r = c.post("/tools/propose_hypotheses", json={"status": "unknown"})
    assert r.status_code == 200 and r.json()["hypotheses"][0]["id"] == "H1"
    # fail-soft nas tools novas também sobre HTTP: 200 + supported=false
    r = c.post("/tools/induce_structure", json={"samples": []})
    assert r.status_code == 200 and r.json()["supported"] is False


def test_compositional_frontier_enforces_hard_scoring_budget():
    calls = []
    search = CompositionalFrontierSearch(beam_width=3, budget=3, max_pool=100)
    result = search.search(1, lambda node: calls.append(node.expression) or float(len(calls)))
    assert len(calls) <= 3
    assert result.explored == len(calls)
    assert len(set(calls)) == len(calls)


def test_structure_inducer_temporal_splits_are_forward_only_and_seeded():
    temporal = StructureInducer(folds=3, cv_strategy="time_series")
    splits = temporal._make_splits(30, strategy="time_series")
    assert len(splits) >= 2
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0 and len(test_idx) > 0
        assert int(np.max(train_idx)) < int(np.min(test_idx))

    iid = StructureInducer(folds=3, cv_strategy="iid")
    a = iid._make_splits(30, strategy="iid", seed=123)
    b = iid._make_splits(30, strategy="iid", seed=123)
    c = iid._make_splits(30, strategy="iid", seed=456)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    assert any(not np.array_equal(x[1], y[1]) for x, y in zip(a, c))


# --- Achado A: holdout externo selado (sem vazamento na selecao) -------------

def _trace_closed_loop(monkeypatch, samples, *, seed=0, cv_strategy="iid"):
    """Roda ClosedLoopDiscovery espionando fit() e evaluate_holdout().

    Devolve (resultado, chamadas de fit, chamadas de evaluate_holdout). As
    amostras sao identificadas por id(): _split() reparte os mesmos objetos.
    """
    fits, evals = [], []
    with monkeypatch.context() as m:
        real_fit = StructureInducer.fit
        real_eval = StructureInducer.evaluate_holdout

        def spy_fit(self, train, *args, **kwargs):
            result = real_fit(self, train, *args, **kwargs)
            extra = kwargs.get("extra_terms", args[0] if args else ())
            fits.append({"ids": {id(s) for s in train},
                         "extra": tuple(term for term, _ in extra),
                         "cv_mae": result.best.cv_mae})
            return result

        def spy_eval(self, train, test, discovery, extra_terms=()):
            mae = real_eval(self, train, test, discovery, extra_terms)
            evals.append({"train_ids": {id(s) for s in train},
                          "test_ids": {id(s) for s in test},
                          "extra": tuple(term for term, _ in extra_terms),
                          "mae": mae})
            return mae

        m.setattr(StructureInducer, "fit", spy_fit)
        m.setattr(StructureInducer, "evaluate_holdout", spy_eval)
        loop = ClosedLoopDiscovery(
            inducer=StructureInducer(cv_strategy=cv_strategy, seed=seed),
            mutagen=Mutagen(max_candidates=48, composition_depth=2),
        )
        result = loop.run(samples, seed=seed)
    return result, fits, evals


@pytest.mark.parametrize("strategy", ["iid", "time_series"])
def test_closed_loop_split_is_disjoint_complete_and_forward_only(strategy):
    samples = _to_samples(_cos_samples())
    loop = ClosedLoopDiscovery(inducer=StructureInducer(cv_strategy=strategy))
    pos = {id(s): i for i, s in enumerate(samples)}

    def rows(split):
        return {pos[id(s)] for s in split}

    dev, hold = loop._split(samples, 0)
    assert rows(dev).isdisjoint(rows(hold))
    assert rows(dev) | rows(hold) == set(range(len(samples)))
    assert len(hold) == max(4, math.ceil(len(samples) * 0.20))
    if strategy == "time_series":
        # holdout = bloco final: nenhuma linha do futuro entra no desenvolvimento
        assert max(rows(dev)) < min(rows(hold))
    else:
        # IID: holdout deterministico por semente
        assert rows(hold) == rows(loop._split(samples, 0)[1])
        assert rows(hold) != rows(loop._split(samples, 1)[1])


def test_closed_loop_holdout_is_never_fitted_and_scored_at_most_twice(monkeypatch):
    samples = _to_samples(_cos_samples())
    result, fits, evals = _trace_closed_loop(monkeypatch, samples)
    assert result.status == "mutated" and result.candidate_trials > 2
    holdout = evals[0]["test_ids"]
    # baseline + candidato final: o holdout nao e reutilizado a cada tentativa
    assert len(evals) == 2
    assert all(e["test_ids"] == holdout and e["train_ids"].isdisjoint(holdout)
               for e in evals)
    # nenhuma busca/validacao cruzada viu uma linha do holdout
    assert len(fits) > 2
    assert all(f["ids"].isdisjoint(holdout) for f in fits)


def test_closed_loop_selection_is_invariant_to_holdout_labels(monkeypatch):
    clean = _to_samples(_cos_samples())
    _, fits_a, evals_a = _trace_closed_loop(monkeypatch, clean)
    pos = {id(s): i for i, s in enumerate(clean)}
    holdout_rows = {pos[i] for i in evals_a[0]["test_ids"]}
    # No holdout a "lei" muda (s1' = s1 + 4*sin(s1) em vez de + 4*cos(s1)): um seletor
    # que espiasse o holdout passaria a preferir sin(s1); a busca selada nao muda.
    corrupted = [
        (s, a, np.array([o[0], s[1] + 4.0 * np.sin(s[1])])) if i in holdout_rows else (s, a, o)
        for i, (s, a, o) in enumerate(clean)
    ]
    _, fits_b, evals_b = _trace_closed_loop(monkeypatch, corrupted)
    # a busca so enxerga desenvolvimento: mesma trilha de CV e mesmo vencedor
    assert ([(f["extra"], f["cv_mae"]) for f in fits_a]
            == [(f["extra"], f["cv_mae"]) for f in fits_b])
    assert evals_a[-1]["extra"] == evals_b[-1]["extra"] == ("cos(s1)",)
    # ...e a corrupcao mexeu de fato na metrica externa (o teste nao e vacuoso)
    assert evals_b[-1]["mae"] > evals_a[-1]["mae"] + 1.0


# --- seed e cv_strategy expostos nas tools (Python, HTTP e MCP) --------------

def _drop(d, *keys):
    return {k: v for k, v in d.items() if k not in keys}


def test_induce_structure_exposes_seed_and_cv_strategy():
    samples = _cos_samples(80)

    def run(**kw):
        return call_tool("induce_structure", {"samples": samples, **kw})

    default = run()
    assert default["supported"] is True
    assert (default["seed"], default["cv_strategy"]) == (0, "iid")
    assert default == run(seed=0, cv_strategy="iid")  # padroes inalterados
    assert run(seed=1) == run(seed=1)  # reprodutivel
    assert run(seed=7.0) == run(seed=7)  # inteiro vindo como float de JSON
    assert run(seed=1)["best"]["cv_mae"] != run(seed=2)["best"]["cv_mae"]
    temporal = run(cv_strategy="time_series")
    assert temporal["supported"] is True and temporal["cv_strategy"] == "time_series"
    assert temporal["best"]["cv_mae"] != default["best"]["cv_mae"]
    # walk-forward nao embaralha: a semente nao muda o resultado
    assert _drop(temporal, "seed") == _drop(run(cv_strategy="time_series", seed=99), "seed")


def test_closed_loop_discovery_exposes_seed_and_cv_strategy():
    samples = _cos_samples()

    def run(**kw):
        return call_tool("closed_loop_discovery", {"samples": samples, **kw})

    default = run()
    assert default["supported"] is True
    assert (default["seed"], default["cv_strategy"]) == (0, "iid")
    assert default == run(seed=0, cv_strategy="iid")  # padroes inalterados
    assert run(seed=3) == run(seed=3)  # reprodutivel
    # outra semente IID = outro holdout externo = outra metrica externa
    assert run(seed=3)["initial"]["best"]["cv_mae"] != run(seed=4)["initial"]["best"]["cv_mae"]
    t1 = run(cv_strategy="time_series", seed=1)
    t2 = run(cv_strategy="time_series", seed=2)
    assert t1["supported"] is True and t1["cv_strategy"] == "time_series"
    assert t1["status"] == "mutated"
    assert t1["accepted_mutations"][0]["expression"] == "cos(s1)"
    assert _drop(t1, "seed") == _drop(t2, "seed")  # holdout temporal nao usa semente


@pytest.mark.parametrize("tool", ["induce_structure", "closed_loop_discovery"])
@pytest.mark.parametrize("kw,field", [
    ({"cv_strategy": "walk_forward"}, "cv_strategy"),
    ({"cv_strategy": "temporal"}, "cv_strategy"),
    ({"cv_strategy": None}, "cv_strategy"),
    ({"cv_strategy": 1}, "cv_strategy"),
    ({"seed": -1}, "seed"),
    ({"seed": 1.5}, "seed"),
    ({"seed": "7"}, "seed"),
    ({"seed": True}, "seed"),
    ({"seed": None}, "seed"),
])
def test_seed_and_cv_strategy_are_validated(tool, kw, field):
    r = call_tool(tool, {"samples": _cos_samples(), **kw})
    assert r["supported"] is False
    assert field in r["error"]


@pytest.mark.http
def test_seed_and_cv_strategy_over_http():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from motor_tools.http_server import app

    c = TestClient(app)
    samples = _cos_samples()
    for tool in ("induce_structure", "closed_loop_discovery"):
        r = c.post(f"/tools/{tool}",
                   json={"samples": samples, "seed": 5, "cv_strategy": "time_series"})
        body = r.json()
        assert r.status_code == 200 and body["supported"] is True
        assert (body["seed"], body["cv_strategy"]) == (5, "time_series")
        # valor invalido: fail-soft (200 + supported=false), nao 422
        r = c.post(f"/tools/{tool}", json={"samples": samples, "cv_strategy": "walk_forward"})
        assert r.status_code == 200 and r.json()["supported"] is False
        assert "cv_strategy" in r.json()["error"]
    # quem descobre as tools pelo OpenAPI (GPT Actions etc.) ve os parametros na descricao
    paths = c.get("/openapi.json").json()["paths"]
    for tool in ("induce_structure", "closed_loop_discovery"):
        desc = paths[f"/tools/{tool}"]["post"]["description"]
        assert "seed" in desc and "cv_strategy" in desc and "time_series" in desc
