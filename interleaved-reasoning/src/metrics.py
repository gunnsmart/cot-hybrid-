"""Evaluation harness and the six success criteria.

Metric definitions (formal statement in PROBLEM.md):

  c1  Var across inputs of the per-input mean emit probability > 0.1
      (soft mode, test set). Detects mode collapse.
  c2  Pearson(mode_count, measured_difficulty) > 0.3, AND the correlation
      with the *label* difficulty is reported separately (they may disagree).
      mode_count = number of hard emissions at threshold tau.
  c3  KL( Bern(p_soft) || Bern(p_hard) ) < 0.1 nats, where p_soft is the mean
      emit probability over (input, stage) and p_hard the hard-emit frequency
      -- the train/inference mode-distribution parity.
  c4  test CE under hard mode <= 1.05 * baseline CE, for BOTH the full-emit
      and the full-latent separately-trained baseline (explicit tolerance 1.05).
  c5  the above holds for >= 2 different stage architectures.
  c6  mode-mechanism parameters (mode_norm + mode_head) < 5% of stage params.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .data import collate


def _kl_bernoulli(p, q, eps=1e-12):
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def evaluate(model, dataset, indices, mode="hard", tau=0.5, force=None,
             semantics="additive", batch=64, device="cpu",
             noise_seed=7, sigma_eval=None):
    """Run `dataset[indices]` through the model in the given mode.

    Returns a dict with losses, per-sample p / emissions and convenience
    arrays for the criteria. Channel noise during eval is deterministic
    (fresh generator seeded by noise_seed), so results are reproducible.
    """
    model.eval()
    if sigma_eval is None:
        sigma_eval = model.sigma
    old_sigma = model.sigma
    model.sigma = sigma_eval

    ces, ps, counts, stage_p = [], [], [], []
    trace_tokens_all = []
    correct = 0
    with torch.no_grad():
        for s in range(0, len(indices), batch):
            idx = indices[s:s + batch]
            X, y = collate([dataset.samples[i] for i in idx], dataset.max_len)
            X, y = X.to(device), y.to(device)
            gen = torch.Generator(device="cpu").manual_seed(noise_seed + s)
            if mode == "hard":
                gen = gen  # deterministic per (noise_seed, batch position)
            logits, P, trace = model.predict(X, mode=mode, tau=tau, force=force,
                                             semantics=semantics, noise_gen=gen)
            ce = torch.nn.functional.cross_entropy(logits, y, reduction="mean").item()
            pred = logits.argmax(-1)
            correct += int((pred == y).sum().item())
            ces.append(ce * len(idx))
            ps.append(P.cpu().numpy())
            if mode == "hard":
                counts.append((P >= tau).sum(1).cpu().numpy())
                trace_tokens_all.append(trace.cpu().numpy())
            else:
                counts.append(P.sum(1).cpu().numpy())  # expected emissions
            stage_p.append(P.cpu().numpy())
    model.sigma = old_sigma

    n = len(indices)
    P = np.concatenate(ps, 0)                    # (n, N)
    counts = np.concatenate(counts, 0)           # (n,)
    samples = [dataset.samples[i] for i in indices]
    measured = np.array([s.measured for s in samples], dtype=float)
    label = np.array([s.label for s in samples], dtype=float)
    ce = sum(ces) / n
    acc = correct / n

    # c1: variance across inputs of per-input mean p
    p_in = P.mean(axis=1)
    c1_var = float(np.var(p_in))

    # c3: parity of soft vs hard mode distribution (aggregate Bernoulli KL)
    p_soft = float(P.mean())
    p_hard = float((P >= tau).mean())
    c3_kl = _kl_bernoulli(p_soft, p_hard)

    # c2: correlations (hard mode counts)
    def _pearson(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    c2_measured = _pearson(counts, measured)
    c2_label = _pearson(counts, label)

    # emit count by measured-difficulty bucket (analysis)
    buckets = {}
    for m in np.unique(measured):
        sel = measured == m
        buckets[int(m)] = {
            "n": int(sel.sum()),
            "mean_emits": float(counts[sel].mean()),
            "mean_p": float(P[sel].mean()),
        }

    return {
        "ce": ce, "acc": acc,
        "c1_var_p": c1_var,
        "c2_r_measured": c2_measured, "c2_r_label": c2_label,
        "c3_kl": c3_kl,
        "p_soft": p_soft, "p_hard": p_hard,
        "mean_emits": float(counts.mean()),
        "counts": counts.tolist(),
        "stage_profile": P.mean(axis=0).tolist(),
        "emits_by_difficulty": buckets,
    }


def full_test_eval(model, dataset, tau=0.5, semantics="additive", device="cpu",
                   noise_seed=7, batch=64):
    """Everything a results JSON needs about the test set, for one model."""
    idx = list(range(len(dataset)))
    hard = evaluate(model, dataset, idx, mode="hard", tau=tau, force=None,
                    semantics=semantics, device=device, noise_seed=noise_seed)
    soft = evaluate(model, dataset, idx, mode="soft", tau=tau, force=None,
                    semantics=semantics, device=device, noise_seed=noise_seed,
                    sigma_eval=0.0)
    fe = evaluate(model, dataset, idx, mode="hard", tau=tau, force="emit",
                  semantics=semantics, device=device, noise_seed=noise_seed)
    fl = evaluate(model, dataset, idx, mode="hard", tau=tau, force="latent",
                  semantics=semantics, device=device, noise_seed=noise_seed)
    return {
        "hard": hard, "soft": soft,
        "force_emit": {"ce": fe["ce"], "acc": fe["acc"]},
        "force_latent": {"ce": fl["ce"], "acc": fl["acc"]},
        "latent_gap": hard["acc"] - fl["acc"],
        "emit_gap": hard["acc"] - fe["acc"],
    }


def save_checkpoint(model, path, dtype=torch.float16):
    sd = {k: v.detach().to(dtype) for k, v in model.state_dict().items()}
    torch.save(sd, path)


def load_checkpoint(model, path, dtype=torch.float32):
    sd = torch.load(path, map_location="cpu")
    sd = {k: v.to(dtype) for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(dtype)
    return model
