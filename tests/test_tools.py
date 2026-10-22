import numpy as np
import pytest

from motor_tools import call_tool, TOOLS


def test_sign_test_known_value():
    r = call_tool("sign_test", {"wins": 8, "losses": 2})
    assert r["p_value"] == pytest.approx(56 / 1024)


def test_compare_methods_detect_clear_winner():
    rng = np.random.default_rng(0)
    good = rng.uniform(0.0, 0.1, 40)
    errs = {"a": good.tolist(), "b": (good + 0.2).tolist(), "c": (good + 0.15).tolist()}
    for m in ("sign", "t", "permutation"):
        r = call_tool("compare_paired_errors",
                      {"errors": errs, "best": "a", "alpha": 0.05, "method": m, "n_resamples": 4000})
        assert r["supported"] is True, m


def test_compare_no_difference_not_supported():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=30)
    errs = {"a": x.tolist(), "b": (x + rng.normal(0, 0.01, 30)).tolist()}
    r = call_tool("compare_paired_errors", {"errors": errs, "best": "a", "method": "sign"})
    assert r["supported"] is False


def test_compare_bad_method():
    r = call_tool("compare_paired_errors",
                  {"errors": {"a": [1, 2], "b": [2, 3]},
                   "best": "a", "method": "x"})
    assert r["supported"] is False
    assert "error" in r


def test_novelty_gate_truth_table():
    assert call_tool("novelty_gate", {"raw_novelty": True, "temporal_information_insufficient": False})[
        "emit_strong_world_shift_evidence"] is True
    r = call_tool("novelty_gate", {"raw_novelty": True, "temporal_information_insufficient": True})
    assert r["abstained"] is True and r["emit_strong_world_shift_evidence"] is False


def test_staleness_detects_frozen_channel():
    rng = np.random.default_rng(2)
    cal = np.cumsum(rng.normal(size=(200, 2)), axis=0).tolist()
    live = np.cumsum(rng.normal(size=(10, 2)), axis=0)
    live[4:, 0] = live[4, 0]  # canal 0 congela
    r = call_tool("staleness_check", {"calibration": cal, "stream": live.tolist()})
    assert r["any_stale"] is True
    clean = np.cumsum(rng.normal(size=(10, 2)), axis=0).tolist()
    assert call_tool("staleness_check", {"calibration": cal, "stream": clean, "alpha": 0.001})["any_stale"] is False


def _ar_window(rng, a, n=40, scale=1.0, offset=0.0):
    x = np.zeros((n, 2))
    for t in range(1, n):
        x[t] = a @ x[t - 1] + rng.normal(0, 0.3, 2)
    return x * scale + offset


def test_structural_world_shift_and_transform():
    rng = np.random.default_rng(3)
    A = np.array([[0.8, 0.1], [-0.1, 0.7]])
    cal = [_ar_window(rng, A).tolist() for _ in range(30)]
    same = call_tool("structural_check", {"calibration_windows": cal, "window": _ar_window(rng, A).tolist()})
    assert same["dynamic_consistent"] is True
    B = np.array([[-0.9, 0.0], [0.0, 0.95]])
    shifted = call_tool("structural_check", {"calibration_windows": cal, "window": _ar_window(rng, B).tolist()})
    assert shifted["world_shift_evidence"] is True
    scaled = call_tool("structural_check",
                       {"calibration_windows": cal, "window": _ar_window(rng, A, scale=8.0, offset=20.0).tolist()})
    assert scaled["transform_evidence"] is True


def test_unknown_tool():
    r = call_tool("nope")
    assert r["supported"] is False
    assert "error" in r


@pytest.mark.http
def test_http_endpoints():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from motor_tools.http_server import app
    c = TestClient(app)
    assert set(c.get("/tools").json()) == set(TOOLS)
    r = c.post("/tools/sign_test", json={"wins": 5, "losses": 0})
    assert r.status_code == 200 and r.json()["p_value"] == pytest.approx(1 / 32)
    assert c.post("/tools/compare_paired_errors", json={"errors": {"a": [1], "b": [2]}, "best": "a", "method": "x"}).status_code == 422
    # entrada insuficiente não levanta erro: a biblioteca devolve supported=False
    assert c.post("/tools/compare_paired_errors", json={"errors": {"a": [1], "b": [2]}, "best": "a", "method": "t"}).json()["supported"] is False
    assert "/tools/novelty_gate" in c.get("/openapi.json").json()["paths"]


def test_novelty_score_detects_outlier():
    rng = np.random.default_rng(4)
    cal = rng.normal(size=(80, 3)).tolist()
    in_dist = rng.normal(size=(5, 3)).tolist()
    far = (rng.normal(size=(3, 3)) + 8.0).tolist()
    r_in = call_tool("novelty_score", {"calibration": cal, "queries": in_dist, "threshold": 0.9})
    r_far = call_tool("novelty_score", {"calibration": cal, "queries": far, "threshold": 0.9})
    assert all(x["regime"] == "known" for x in r_in["results"]) or np.mean([x["novelty_score"] for x in r_in["results"]]) < 0.9
    assert any(x["regime"] == "novel" for x in r_far["results"])


def test_temporal_information_low_change():
    rng = np.random.default_rng(5)
    cal = np.cumsum(rng.normal(size=(100, 2)), axis=0).tolist()
    # stream with almost no change
    base = rng.normal(size=2)
    stream = [base.tolist()] + [(base + 1e-9 * i).tolist() for i in range(1, 8)]
    r = call_tool("temporal_information_check", {"calibration": cal, "stream": stream, "alpha": 0.05, "min_run": 2})
    assert r["any_insufficient"] is True


def test_redundant_consistency():
    rng = np.random.default_rng(6)
    pairs = []
    for _ in range(25):
        a = rng.normal(size=(12, 2))
        b = a * 1.5 + 0.3 + rng.normal(0, 0.05, size=a.shape)
        pairs.append([a.tolist(), b.tolist()])
    a_ok = rng.normal(size=(12, 2))
    b_ok = a_ok * 1.5 + 0.3 + rng.normal(0, 0.05, size=a_ok.shape)
    r_ok = call_tool("redundant_consistency_check", {
        "calibration_pairs": pairs,
        "pair": [a_ok.tolist(), b_ok.tolist()],
    })
    assert r_ok["consistent"] is True
    b_bad = a_ok * 1.5 + 0.3 + 5.0
    r_bad = call_tool("redundant_consistency_check", {
        "calibration_pairs": pairs,
        "pair": [a_ok.tolist(), b_bad.tolist()],
    })
    assert r_bad["consistent"] is False


def test_relative_staleness_one_frozen_channel():
    rng = np.random.default_rng(7)
    cal = np.cumsum(rng.normal(size=(150, 2)), axis=0).tolist()
    live = np.cumsum(rng.normal(size=(12, 2)), axis=0)
    live[5:, 0] = live[5, 0]  # canal 0 congela; canal 1 continua
    r = call_tool("relative_staleness_check", {
        "calibration": cal, "stream": live.tolist(), "alpha": 0.05, "min_run": 2
    })
    assert r["any_stale"] is True


def test_estimate_noise():
    rng = np.random.default_rng(8)
    # 10 repeats of same latent + noise
    base = np.array([1.0, -2.0, 0.5])
    samples = (base + rng.normal(0, 0.1, size=(10, 3))).tolist()
    r = call_tool("estimate_noise", {"samples": samples})
    assert r["n_repeats"] == 10
    assert len(r["sigma"]) == 3
    assert r["pooled_sigma"] > 0
    # identical repeats → zero noise
    same = [[1.0, 2.0]] * 5
    r0 = call_tool("estimate_noise", {"samples": same})
    assert r0["pooled_sigma"] == 0.0



def test_convergent_novelty_gate():
    r = call_tool("convergent_novelty_gate", {
        "raw_novelty": True,
        "temporal_information_insufficient": True,
        "observability_degraded": True,
        "relative_temporal_stale": False,
        "minimum_degradation_evidence": 2,
    })
    assert r["abstained"] is True
    assert r["emit_strong_world_shift_evidence"] is False
    r2 = call_tool("convergent_novelty_gate", {
        "raw_novelty": True,
        "temporal_information_insufficient": True,
        "observability_degraded": False,
        "relative_temporal_stale": False,
    })
    assert r2["emit_strong_world_shift_evidence"] is True


def test_combine_uncertainties():
    r = call_tool("combine_uncertainties", {"epistemic": 3.0, "observation_sigma": 4.0})
    assert r["total_rms"] == pytest.approx(5.0)
    assert r["epistemic_fraction"] == pytest.approx(0.6)


def test_temporal_attribution_check():
    r = call_tool("temporal_attribution_check", {
        "transform_evidence": [True, True, False],
        "world_shift_evidence": [False, False, False],
    })
    assert r["label"] == "OBSERVATION_TRANSFORM"


def test_known_reference_check():
    rng = np.random.default_rng(9)
    cal_e, cal_o = [], []
    for _ in range(25):
        e = rng.normal(size=(8, 2))
        o = e + rng.normal(0, 0.05, size=e.shape)
        cal_e.append(e.tolist())
        cal_o.append(o.tolist())
    e = rng.normal(size=(8, 2))
    o_ok = (e + rng.normal(0, 0.05, size=e.shape)).tolist()
    r_ok = call_tool("known_reference_check", {
        "calibration_expected": cal_e,
        "calibration_observed": cal_o,
        "expected": e.tolist(),
        "observed": o_ok,
    })
    assert r_ok["transform_evidence"] is False
    o_bad = (e + 5.0).tolist()
    r_bad = call_tool("known_reference_check", {
        "calibration_expected": cal_e,
        "calibration_observed": cal_o,
        "expected": e.tolist(),
        "observed": o_bad,
    })
    assert r_bad["transform_evidence"] is True


def test_canonicalize_expression():
    r = call_tool("canonicalize_expression", {"expression": "2*x+1"})
    assert r["changed"] is True or "x" in r["canonical"]
    assert r["quotient_key"] is not None


def test_tools_registry_v05():
    assert len(TOOLS) >= 20
    assert "agi_robustness_battery" in TOOLS


def test_agi_robustness_battery():
    r = call_tool("agi_robustness_battery", {"seed": 2})
    assert "tasks" in r and r["tasks"] >= 1
    assert "results" in r


def test_agi_active_recovery_battery():
    r = call_tool("agi_active_recovery_battery", {"seed": 2, "budget": 2})
    assert "results" in r and len(r["results"]) >= 1


def test_regime_transfer_battery():
    r = call_tool("regime_transfer_battery", {"seeds": [13001]})
    assert r["n"] >= 1
    assert "metrics" in r["observations"][0]


def test_tools_count_v07():
    assert len(TOOLS) >= 19
