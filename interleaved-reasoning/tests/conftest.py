import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.data import make_dataset          # noqa: E402
from src.models import InterleavedProcessor  # noqa: E402
from src.metrics import load_checkpoint     # noqa: E402
from scripts import common                  # noqa: E402

common.setup_paths()
torch.set_num_threads(2)


@pytest.fixture(scope="session")
def datasets():
    return {
        t: {
            "test": make_dataset(t, "test", 1024),
            "dev": make_dataset(t, "dev", 512),
        }
        for t in ("arithmetic", "logic", "recall")
    }


def load_run(run_id):
    """Load a finished run's best checkpoint into a fresh model.
    Skips the calling test if the run has not been produced yet
    (the experiment matrix fills these in as it runs)."""
    p = common.run_json_path(run_id)
    if not os.path.exists(p):
        pytest.skip(f"run {run_id} not available yet ({p} missing)")
    j = common.read_json(p)
    assert j.get("status") == "done", f"run {run_id} is not done"
    c = j["config"]
    mechanism = c.get("force", "none") == "none"
    m = InterleavedProcessor(d=c["d"], n_stages=c["n"], vocab=c["vocab"],
                             arch=c["arch"], mechanism=mechanism,
                             sigma=c["sigma"],
                             reader_layers=c.get("reader_layers", 1),
                             q_conf=c.get("q_conf", 0.0),
                             t_conf=c.get("t_conf", 0.1))
    load_checkpoint(m, common.best_pt_path(run_id))
    return m, j
