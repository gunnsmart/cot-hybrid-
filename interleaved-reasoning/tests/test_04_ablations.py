"""Negative results are first class: each core ablation MUST reproduce its
expected failure signature (otherwise the "fix is necessary" claim is false
and this suite fails on purpose)."""
import pytest

from scripts import common


def done(run_id):
    import os
    p = common.run_json_path(run_id)
    return os.path.exists(p) and common.read_json(p).get("status") == "done"


def hard(run_id):
    return common.read_json(common.run_json_path(run_id))["test"]["hard"]


def test_failure3_token_spam_without_price():
    if not done("A_abl_noprice"):
        pytest.skip("run not done yet")
    h = hard("A_abl_noprice")
    n = common.read_json(common.run_json_path("A_abl_noprice"))["config"]["n"]
    assert h["mean_emits"] > 0.8 * n, (
        f"without price, expected spam (emits>={0.8*n:.1f}); got {h['mean_emits']:.2f}")


def test_failure1_collapse_without_nondegeneracy():
    if not done("A_abl_nondeg"):
        pytest.skip("run not done yet")
    h = hard("A_abl_nondeg")
    assert h["c1_var_p"] <= 0.1, (
        f"without non-degeneracy term, expected collapse (var<=0.1); "
        f"got {h['c1_var_p']:.3f}")


def test_failure2_gap_without_commitment():
    if not done("A_abl_nocommit"):
        pytest.skip("run not done yet")
    h = hard("A_abl_nocommit")
    assert h["c3_kl"] >= 0.1, (
        f"without commitment, expected soft-hard gap (KL>=0.1); "
        f"got {h['c3_kl']:.3f}")


def test_bare_reproduces_multiple_failures():
    if not done("A_abl_bare"):
        pytest.skip("run not done yet")
    h = hard("A_abl_bare")
    n = common.read_json(common.run_json_path("A_abl_bare"))["config"]["n"]
    bad = sum([h["c1_var_p"] <= 0.1,      # collapse
               h["c3_kl"] >= 0.1,         # gap
               h["mean_emits"] > 0.8 * n])  # spam
    assert bad >= 2, f"bare model should show >=2 of {3} failures; showed {bad}"


def test_failure4_latent_blackout_check_sig0():
    """With a PERFECT latent channel (sigma=0) emission stops being adaptive:
    the measured-difficulty correlation collapses relative to the full recipe
    (latent blackout: the latent branch carries everything, modes degenerate)."""
    if not done("A_abl_sig0") or not done("A_mlp_main"):
        pytest.skip("runs not done yet")
    h0 = hard("A_abl_sig0")
    hb = hard("A_mlp_main")
    assert h0["c2_r_measured"] < 0.5 * max(0.3, hb["c2_r_measured"]), (
        f"sigma=0 should kill adaptive emission: corr={h0['c2_r_measured']:.3f} "
        f"vs full-recipe {hb['c2_r_measured']:.3f}")


def test_hybrid_modes_are_both_load_bearing():
    """Failure-4 detector on the full-recipe models: forcing either single
    mode must lose accuracy vs the hard-mode hybrid (both gaps > 0)."""
    for run in ("A_mlp_main", "B_mlp_main", "C_mlp_main"):
        if not done(run):
            pytest.skip("run not done yet")
        j = common.read_json(common.run_json_path(run))
        t = j["test"]
        assert t["latent_gap"] > 0, f"{run}: latent stages contribute nothing"
        assert t["emit_gap"] > 0, f"{run}: emissions contribute nothing"
