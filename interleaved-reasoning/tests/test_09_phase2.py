"""PHASE 2 -- pressure regimes (experiments/PHASE2.md).

Unit tests for the new model paths (gradual reader, forced schedule) run
immediately. Run-level tests are SKIPPED until the corresponding results
JSON exists -- the experiment matrix fills them in. Once a result exists:
the new hybrids must stay non-degenerate and train/inference-aligned
(c1 > 0.1, c3 < 0.1), the hybrid must beat the emit specialist (<= 1.05x)
in every regime, and schedule runs must emit exactly N/k tokens (+/-0.5).

Beating the LATENT specialist is the measurement of this phase, not an
assertion -- it is reported in FINDINGS.md ("Phase 2 - pressure regimes").
"""
import os

import pytest
import torch

from scripts import common
from src.models import InterleavedProcessor
from tests.conftest import load_run

TOL = 1.05
# hybrid run -> (emit specialist, latent specialist) re-trained in the regime
REGIMES = {
    "A_s03": ("A_s03_emit", "A_s03_latent"),
    "A_s05": ("A_s05_emit", "A_s05_latent"),
    "A_bneck": ("A_bneck_emit", "A_bneck_latent"),
}
# schedule run -> k (emit every k-th stage)
SCHEDULES = {"A_sched4": 4, "A_sched2": 2}
ALL_PHASE2 = ["A_s03", "A_s03_emit", "A_s03_latent",
              "A_s05", "A_s05_emit", "A_s05_latent",
              "A_bneck", "A_bneck_emit", "A_bneck_latent",
              "A_sched4", "A_sched2"]


def _done(rid):
    p = common.run_json_path(rid)
    return os.path.exists(p) and common.read_json(p).get("status") == "done"


# ---------------------------------------------------------------------------
# Unit: gradual reader (H2)
# ---------------------------------------------------------------------------

def test_gradual_shapes_and_param_accounting():
    d, n, B, L = 32, 6, 4, 16
    m_grad = InterleavedProcessor(d=d, n_stages=n, vocab=64, arch="mlp",
                                  mechanism=True, sigma=0.1,
                                  reader_mode="gradual")
    m_full = InterleavedProcessor(d=d, n_stages=n, vocab=64, arch="mlp",
                                  mechanism=True, sigma=0.1,
                                  reader_mode="full", reader_layers=2)
    x = torch.randint(0, 64, (B, L))

    ce = m_grad._chunk_embeds(x)
    assert ce.shape == (B, n + 1, d)
    assert m_grad.encode(x).shape == (B, d)
    h, P, trace, G = m_grad.forward(x, mode="soft")
    assert h.shape == (B, d) and P.shape == (B, n)
    assert trace.shape == (B, n) and G.shape == (B, n)
    logits, _, _, _ = m_grad.predict(x, mode="hard", tau=0.5)
    assert logits.shape == (B, 64)

    # zero-init: at step 0 the gradual model receives no input after chunk 0
    assert float(m_grad.chunk_proj[0].weight.abs().sum()) == 0.0
    assert float(m_grad.chunk_proj[-1].weight.abs().sum()) == 0.0

    # gradients reach the chunk projections
    m_grad.predict(x, mode="soft")[0].sum().backward()
    assert m_grad.chunk_proj[-1].weight.grad is not None

    # parameter accounting: chunk_proj is task interface (group `base`),
    # never mechanism. (No c6<5% assert here: on a toy model the fixed-size
    # mechanism inflates the percentage by construction.)
    pg_g, pg_f = m_grad.param_groups(), m_full.param_groups()
    n_chunk = sum(p.numel() for p in m_grad.chunk_proj.parameters())
    assert n_chunk == (n + 1) * d * d
    embed = sum(p.numel() for p in m_grad.embed.parameters())
    readout = sum(p.numel() for p in m_grad.readout.parameters())
    assert pg_g["base"] == embed + readout + n_chunk
    assert pg_g["total"] == pg_g["stages"] + pg_g["mechanism"] + pg_g["base"]
    # the regime changes the task interface, NOT the mechanism size
    assert pg_g["mechanism"] == pg_f["mechanism"]
    mech_manual = (sum(p.numel() for p in m_grad.mode_norm.parameters())
                   + sum(p.numel() for p in m_grad.mode_head.parameters()))
    assert pg_g["mechanism"] == mech_manual
    assert pg_g["stages"] == pg_f["stages"]


def test_gradual_chunk_pad_mask():
    d, n = 16, 3
    m = InterleavedProcessor(d=d, n_stages=n, vocab=32, arch="mlp",
                             reader_mode="gradual")
    # L=8, K=4 chunks -> boundaries linspace(0,8,5) = [0,2,4,6,8]
    x = torch.randint(1, 32, (2, 8))
    x[:, 6:] = 0                                # last chunk all <pad>
    ce = m._chunk_embeds(x)
    assert torch.all(ce[:, 3] == 0)             # all-pad chunk = zero vector
    assert torch.any(ce[:, 0] != 0)

    # a pad token never contributes to the mean: chunk [0,2) = [5, <pad>]
    x3 = torch.tensor([[5, 0, 7, 8, 9, 10, 11, 12]])
    ce3 = m._chunk_embeds(x3)
    assert torch.allclose(ce3[0, 0], m.embed.weight[5], atol=1e-6)

    # short sequence: empty-span chunks are zero vectors
    # L=2, K=4 -> linspace(0,2,5).round() = [0,0,1,2,2]: chunks 0 and 3 empty
    x2 = torch.randint(1, 32, (2, 2))
    ce2 = m._chunk_embeds(x2)
    assert torch.all(ce2[:, 0] == 0) and torch.all(ce2[:, 3] == 0)
    # forward still runs end-to-end on it
    h, _, _, _ = m.forward(x2, mode="soft")
    assert torch.isfinite(h).all()


# ---------------------------------------------------------------------------
# Unit: forced schedule (H3)
# ---------------------------------------------------------------------------

def test_schedule_pattern_soft_and_hard():
    d, n = 32, 12
    x = torch.randint(0, 64, (5, 10))
    for k in (4, 2):
        m = InterleavedProcessor(d=d, n_stages=n, vocab=64, arch="mlp",
                                 mechanism=False, q_conf=0.30,
                                 schedule_every=k)
        # no mode head is trained (or exists) for a schedule run
        assert m.mode_head is None
        want = torch.tensor([1.0 if (l + 1) % k == 0 else 0.0
                             for l in range(n)])
        for mode in ("soft", "hard"):
            _, P, _, _ = m.forward(x, mode=mode, tau=0.5, force="schedule")
            assert torch.equal(P, want.unsqueeze(0).expand(5, n)), \
                f"mode={mode}: P must be the exact 0/1 schedule for k={k}"

        # gate bypassed: soft gate values are all 1 even with q_conf > 0
        _, _, _, G = m.forward(x, mode="soft", force="schedule")
        assert torch.all(G == 1.0)

        # hard trace: writes exactly at the scheduled stages, every sample
        _, _, trace, _ = m.forward(x, mode="hard", tau=0.5, force="schedule")
        emitted = trace >= 0
        for l in range(n):
            if (l + 1) % k == 0:
                assert emitted[:, l].all(), f"k={k}: stage {l} must emit"
            else:
                assert not emitted[:, l].any(), f"k={k}: stage {l} must not emit"
        assert float(emitted.sum(1).float().mean()) == pytest.approx(n / k)


def test_full_reader_path_regression():
    """The default (phase-1) configuration behaves exactly as before."""
    torch.manual_seed(0)
    m = InterleavedProcessor(d=32, n_stages=4, vocab=64, arch="mlp",
                             sigma=0.1, reader_layers=2, q_conf=0.30,
                             content="expectation")
    assert m.reader_mode == "full" and m.chunk_proj is None
    assert m.schedule_every is None and m.mode_head is not None
    x = torch.randint(0, 64, (3, 9))
    h, P, trace, G = m.forward(x, mode="soft")
    assert h.shape == (3, 32) and P.shape == (3, 4)
    logits, _, _, _ = m.predict(x, mode="hard", tau=0.5)
    assert torch.isfinite(logits).all()
    pg = m.param_groups()
    assert set(pg) == {"stages", "mechanism", "base", "total", "overhead_pct"}


# ---------------------------------------------------------------------------
# Run-level (skipped until the matrix produces the results)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rid", ALL_PHASE2)
def test_phase2_runs_done(rid):
    if not _done(rid):
        pytest.skip(f"run {rid} not done yet")
    j = common.read_json(common.run_json_path(rid))
    assert j["status"] == "done"


@pytest.mark.parametrize("rid", sorted(REGIMES))
def test_phase2_hybrids_keep_c1_c3(rid):
    """Pressure must not break the mechanism itself (collapse / soft-hard gap)."""
    if not _done(rid):
        pytest.skip(f"run {rid} not done yet")
    h = common.read_json(common.run_json_path(rid))["test"]["hard"]
    assert h["c1_var_p"] > 0.1, \
        f"{rid}: mode collapse under pressure (c1={h['c1_var_p']:.3f})"
    assert h["c3_kl"] < 0.1, \
        f"{rid}: soft-hard gap under pressure (c3={h['c3_kl']:.3f})"


@pytest.mark.parametrize("rid", sorted(REGIMES))
def test_phase2_hybrid_beats_emit_specialist(rid):
    """Asserted half of c4: the hybrid still wins against full-emit,
    in every new regime, within the 1.05 tolerance."""
    emit_id = REGIMES[rid][0]
    if not (_done(rid) and _done(emit_id)):
        pytest.skip(f"runs {rid}/{emit_id} not done yet")
    jh = common.read_json(common.run_json_path(rid))
    je = common.read_json(common.run_json_path(emit_id))
    r = jh["test"]["hard"]["ce"] / je["test"]["hard"]["ce"]
    assert r <= TOL, \
        f"{rid}: hybrid lost to the emit specialist in its own regime (r={r:.3f})"


@pytest.mark.parametrize("rid,k", sorted(SCHEDULES.items()))
def test_schedule_emits_exact(rid, k):
    """A fixed schedule must produce exactly N/k emissions (+/-0.5)."""
    if not _done(rid):
        pytest.skip(f"run {rid} not done yet")
    j = common.read_json(common.run_json_path(rid))
    assert j["config"]["force"] == "schedule"
    assert j["config"]["schedule_every"] == k
    n = j["config"]["n"]
    emits = j["test"]["hard"]["mean_emits"]
    assert abs(emits - n / k) <= 0.5, \
        f"{rid}: emits={emits:.2f}, expected N/k={n / k:.1f}"


@pytest.mark.parametrize("rid", ["A_bneck", "A_sched4"])
def test_phase2_checkpoints_load(rid):
    """load_run must rebuild the new architectures from their JSON config."""
    m, j = load_run(rid)  # skips while the run is missing
    if rid == "A_bneck":
        assert m.reader_mode == "gradual" and m.chunk_proj is not None
    else:
        assert m.mode_head is None and m.schedule_every == 4
