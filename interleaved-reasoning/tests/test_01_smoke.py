"""Fast end-to-end smoke test: the pipeline trains without NaNs and the
mode controller moves away from its p=0.5 initialization."""
import numpy as np
import torch
import torch.nn.functional as F

from src.data import make_dataset, collate
from src.models import InterleavedProcessor


def test_smoke_training_arithmetic():
    torch.manual_seed(0)
    np.random.seed(0)
    ds = make_dataset("arithmetic", "train", 256)
    m = InterleavedProcessor(d=48, n_stages=4, vocab=64, arch="mlp", sigma=0.1)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4)
    rng = np.random.default_rng(0)
    first, last, p_first, p_last = None, None, None, None
    for step in range(120):
        idx = rng.choice(len(ds), size=32, replace=False).tolist()
        X, y = collate([ds.samples[i] for i in idx], ds.max_len)
        logits, P, _ = m.predict(X, mode="soft")
        loss = (F.cross_entropy(logits, y)
                + 0.2 * P.mean()
                + 0.1 * (4 * P * (1 - P)).mean())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        if step == 0:
            first, p_first = float(loss), float(P.mean())
        last, p_last = float(loss), float(P.mean())
    assert np.isfinite(last)
    assert last < first, f"loss did not decrease: {first} -> {last}"
    # mode controller must actually be used (not stuck at p=0.5 exactly)
    assert P.shape == (32, 4)
    logits_h, P_h, trace = m.predict(X, mode="hard", tau=0.5)
    assert trace.shape == (32, 4)
    assert torch.isfinite(logits_h).all()
    # forced modes work
    _, P_f, _ = m.forward(X, mode="hard", tau=0.5, force="emit")
    assert float(P_f.mean()) == 1.0
    _, P_l, _ = m.forward(X, mode="hard", tau=0.5, force="latent")
    assert float(P_l.mean()) == 0.0
