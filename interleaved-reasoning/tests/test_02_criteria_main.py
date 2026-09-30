"""Criteria 1-4 and 6 for the three main (MLP) runs, plus the rule that
recomputed test numbers must match the JSON (no drift)."""
import pytest

from src.metrics import full_test_eval
from tests.conftest import load_run

MAIN = {"arithmetic": "A_mlp_main", "logic": "B_mlp_main", "recall": "C_mlp_main"}
TOL = 1.05


def _test_eval(run_id, make_test_set):
    m, j = load_run(run_id)
    task = j["config"]["task"]
    c = j["config"]
    ds = make_test_set(task, c)
    # full_test_eval is what produced the JSON numbers (hard+soft passes,
    # c3 KL from both sides) — recompute the same way so drift checks work.
    res = full_test_eval(m, ds, tau=c["tau"], semantics=c["semantics"],
                         batch=64, noise_seed=7)["hard"]
    return m, j, res


@pytest.mark.parametrize("task,run", list(MAIN.items()))
def test_c1_non_degenerate(task, run, make_test_set):
    _, j, res = _test_eval(run, make_test_set)
    assert res["c1_var_p"] > 0.1, (
        f"{run}: Var(p_in)={res['c1_var_p']:.3f} <= 0.1 -> mode collapse")
    # JSON must agree with recomputation
    assert abs(res["c1_var_p"] - j["test"]["hard"]["c1_var_p"]) < 1e-4


@pytest.mark.parametrize("task,run", list(MAIN.items()))
def test_c2_adaptive_measured(task, run, make_test_set):
    _, j, res = _test_eval(run, make_test_set)
    assert res["c2_r_measured"] > 0.3, (
        f"{run}: corr(measured, emits)={res['c2_r_measured']:.3f} <= 0.3")
    assert abs(res["c2_r_measured"] - j["test"]["hard"]["c2_r_measured"]) < 1e-4
    # label correlation is reported (not required to pass) but must exist
    assert "c2_r_label" in j["test"]["hard"]


@pytest.mark.parametrize("task,run", list(MAIN.items()))
def test_c3_train_inference_parity(task, run, make_test_set):
    _, j, res = _test_eval(run, make_test_set)
    assert res["c3_kl"] < 0.1, (
        f"{run}: KL(soft||hard)={res['c3_kl']:.3f} >= 0.1 -> soft-hard gap")
    assert abs(res["c3_kl"] - j["test"]["hard"]["c3_kl"]) < 1e-4


@pytest.mark.parametrize("task,run", list(MAIN.items()))
def test_c4_no_accuracy_loss(task, run, make_test_set):
    _, j, res = _test_eval(run, make_test_set)
    emit_id = run.replace("_mlp_main", "_emit_base")
    latent_id = run.replace("_mlp_main", "_latent_base")
    _, je = load_run(emit_id)   # load_run returns (model, json)
    _, jl = load_run(latent_id)
    ce_emit = je["test"]["hard"]["ce"]
    ce_latent = jl["test"]["hard"]["ce"]
    r_emit = res["ce"] / ce_emit
    r_latent = res["ce"] / ce_latent
    assert r_emit <= TOL, f"{run}: CE ratio vs emit baseline {r_emit:.3f} > {TOL}"
    assert r_latent <= TOL, f"{run}: CE ratio vs latent baseline {r_latent:.3f} > {TOL}"


@pytest.mark.parametrize("task,run", list(MAIN.items()))
def test_c6_param_budget(task, run):
    m, j = load_run(run)
    pg = m.param_groups()
    assert pg["overhead_pct"] < 5.0, f"{run}: overhead {pg['overhead_pct']:.2f}% >= 5%"
    assert j["params"]["overhead_pct"] == pytest.approx(pg["overhead_pct"])
