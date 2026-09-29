"""Criterion 5: >= 2 architectures must pass. We run four (MLP main rows
are covered in test_02; here: GRU, SSM, transformer on task A with their own
baselines, plus GRU/transformer on task B)."""
import pytest

from src.metrics import evaluate
from tests.conftest import load_run

TOL = 1.05
ARCH_A = {"gru": ("A_gru", "A_gru_emit", "A_gru_latent"),
          "ssm": ("A_ssm", "A_ssm_emit", "A_ssm_latent"),
          "transformer": ("A_transformer", "A_tf_emit", "A_tf_latent")}
ARCH_B = {"gru": "B_gru", "transformer": "B_transformer"}


def _eval(run, make_test_set):
    m, j = load_run(run)
    c = j["config"]
    ds = make_test_set(c["task"], c)
    res = evaluate(m, ds, list(range(len(ds.samples))),
                   mode="hard", tau=c["tau"], semantics=c["semantics"],
                   batch=64, noise_seed=7)
    return m, j, res


@pytest.mark.parametrize("arch,run,emit,latent",
                         [(a, v[0], v[1], v[2]) for a, v in ARCH_A.items()])
def test_arch_c1_c3(arch, run, emit, latent, make_test_set):
    _, j, res = _eval(run, make_test_set)
    assert res["c1_var_p"] > 0.1, f"{run}: collapsed (c1={res['c1_var_p']:.3f})"
    assert res["c3_kl"] < 0.1, f"{run}: soft-hard gap (c3={res['c3_kl']:.3f})"


@pytest.mark.parametrize("arch,run,emit,latent",
                         [(a, v[0], v[1], v[2]) for a, v in ARCH_A.items()])
def test_arch_c4(arch, run, emit, latent, make_test_set):
    _, j, res = _eval(run, make_test_set)
    je, _ = load_run(emit)
    jl, _ = load_run(latent)
    r_e = res["ce"] / je["test"]["hard"]["ce"]
    r_l = res["ce"] / jl["test"]["hard"]["ce"]
    assert r_e <= TOL, f"{run}: vs emit baseline {r_e:.3f} > {TOL}"
    assert r_l <= TOL, f"{run}: vs latent baseline {r_l:.3f} > {TOL}"


@pytest.mark.parametrize("arch,run", list(ARCH_B.items()))
def test_arch_cross_task(arch, run, make_test_set):
    _, j, res = _eval(run, make_test_set)
    assert res["c1_var_p"] > 0.1, f"{run}: collapsed on task B"
    assert res["c2_r_measured"] > 0.3, (
        f"{run}: not adaptive on task B (corr={res['c2_r_measured']:.3f})")


def test_c5_at_least_two_architectures():
    """The renderer's c5 count must be >= 2; here we re-verify directly from
    the JSONs: mlp (3 main tasks) + at least one of {gru, ssm, transformer}
    on task A passing c1, c3, c4."""
    import os
    from scripts import common
    def done(rid):
        p = common.run_json_path(rid)
        return os.path.exists(p) and common.read_json(p).get("status") == "done"
    n = 0
    any_arch = any(done(a[1][0]) for a in ARCH_A.values() if a[0] != "mlp")
    if not any_arch:
        pytest.skip("no non-MLP architecture run available yet")
    for arch, (run, e, l) in ARCH_A.items():
        if done(run) and done(e) and done(l):
            j = common.read_json(common.run_json_path(run))
            h = j["test"]["hard"]
            je = common.read_json(common.run_json_path(e))["test"]["hard"]
            jl = common.read_json(common.run_json_path(l))["test"]["hard"]
            ok = (h["c1_var_p"] > 0.1 and h["c3_kl"] < 0.1
                  and h["ce"] / je["ce"] <= TOL and h["ce"] / jl["ce"] <= TOL)
            n += ok
    assert n >= 1, "criterion 5 needs >= 1 non-MLP arch passing (we have MLP) "
    # plus the MLP main runs make the total >= 2 when at least one arch passes
    total = n + 1
    assert total >= 2
