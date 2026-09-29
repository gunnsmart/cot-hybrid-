"""Every results JSON must satisfy the schema the measurement rules rely on."""
import os
import pytest

from scripts import common

REQUIRED_TOP = ["run_id", "status", "config", "params", "curves", "test"]
REQUIRED_CFG = ["task", "arch", "d", "n", "vocab", "steps", "seed", "sigma",
                "lam_price", "lam_commit", "lam_nd", "tau", "semantics",
                "force", "push_every"]
REQUIRED_TEST = ["hard", "soft", "force_emit", "force_latent",
                 "latent_gap", "emit_gap"]
REQUIRED_HARD = ["ce", "acc", "c1_var_p", "c2_r_measured", "c2_r_label",
                 "c3_kl", "p_soft", "p_hard", "mean_emits", "stage_profile",
                 "emits_by_difficulty"]


def _jsons():
    if not os.path.isdir(common.RESULTS):
        return []
    return sorted(f for f in os.listdir(common.RESULTS)
                  if f.endswith(".json") and f != "_meta.json")


@pytest.mark.parametrize("fname", _jsons() or ["(none)"])
def test_schema(fname):
    if fname == "(none)":
        pytest.skip("no results yet")
    j = common.read_json(os.path.join(common.RESULTS, fname))
    for k in REQUIRED_TOP:
        assert k in j, f"{fname}: missing top-level '{k}'"
    for k in REQUIRED_CFG:
        assert k in j["config"], f"{fname}: missing config '{k}'"
    if j["status"] == "done":
        for k in REQUIRED_TEST:
            assert k in j["test"], f"{fname}: missing test.{k}"
        for k in REQUIRED_HARD:
            assert k in j["test"]["hard"], f"{fname}: missing test.hard.{k}"
        # both correlations present (measurement rule)
        assert "c2_r_measured" in j["test"]["hard"] and "c2_r_label" in j["test"]["hard"]
        # push rule config
        assert j["config"]["push_every"] == 500, "push rule is exactly 500 steps"
        # param budget recorded
        assert set(j["params"]) >= {"stages", "mechanism", "base", "total",
                                     "overhead_pct"}
