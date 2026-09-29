"""Scaling in N = 4, 8, 12, 24, 48 (task A, mlp, d=96)."""
import pytest

from scripts import common

SCALE = ["A_scale_n4", "A_scale_n8", "A_scale_n12", "A_scale_n24", "A_scale_n48"]


def hard(rid):
    return common.read_json(common.run_json_path(rid))["test"]["hard"]


@pytest.mark.parametrize("run", SCALE)
def test_scaling_runs_exist(run):
    import os
    assert os.path.exists(common.run_json_path(run)), f"missing {run}.json"
    j = common.read_json(common.run_json_path(run))
    assert j["status"] == "done"
    assert j["config"]["n"] == int(run[-2:])


@pytest.mark.parametrize("run", SCALE)
def test_scaling_c1_c3(run):
    """Non-degeneracy and parity must survive all N (the mechanism, not the
    depth, produces them)."""
    h = hard(run)
    assert h["c1_var_p"] > 0.1, f"{run}: collapsed at N={h and int(run[-2:])}"
    assert h["c3_kl"] < 0.1, f"{run}: soft-hard gap at N={int(run[-2:])}"


def test_scaling_accuracy_not_catastrophic():
    """Accuracy at every N must stay above 50% (the tasks are solvable at all
    depths in this range; deeper != broken)."""
    for run in SCALE:
        h = hard(run)
        assert h["acc"] > 0.5, f"{run}: acc={h['acc']:.3f} < 0.5"
