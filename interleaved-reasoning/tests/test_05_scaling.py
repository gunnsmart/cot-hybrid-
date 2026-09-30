"""Scaling in N = 4, 8, 12, 24, 48 (task A, mlp, d=96)."""
import pytest

from scripts import common

SCALE = ["A_scale_n4", "A_scale_n8", "A_scale_n12", "A_scale_n24", "A_scale_n48"]


def hard(rid):
    return common.read_json(common.run_json_path(rid))["test"]["hard"]


def _n_of(run):
    """A_scale_n48 -> 48 (id suffix after the final 'n')."""
    return int(run.rsplit("n", 1)[-1])


@pytest.mark.parametrize("run", SCALE)
def test_scaling_runs_exist(run):
    import os
    p = common.run_json_path(run)
    if not os.path.exists(p):
        pytest.skip(f"run {run} not available yet")
    assert os.path.exists(p), f"missing {run}.json"
    j = common.read_json(common.run_json_path(run))
    assert j["status"] == "done"
    assert j["config"]["n"] == _n_of(run)


@pytest.mark.parametrize("run", SCALE)
def test_scaling_c1_c3(run):
    """Non-degeneracy and parity must survive all N (the mechanism, not the
    depth, produces them)."""
    import os
    if not os.path.exists(common.run_json_path(run)):
        pytest.skip(f"run {run} not available yet")
    h = hard(run)
    assert h["c1_var_p"] > 0.1, f"{run}: collapsed at N={_n_of(run)}"
    assert h["c3_kl"] < 0.1, f"{run}: soft-hard gap at N={_n_of(run)}"


def test_scaling_accuracy_not_catastrophic():
    """Accuracy at every N must stay above 50% (the tasks are solvable at all
    depths in this range; deeper != broken)."""
    import os
    if not all(os.path.exists(common.run_json_path(r)) for r in SCALE):
        pytest.skip("scaling runs not available yet")
    for run in SCALE:
        h = hard(run)
        assert h["acc"] > 0.5, f"{run}: acc={h['acc']:.3f} < 0.5"
