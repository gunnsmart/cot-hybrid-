# Ablations — which component is necessary?

Full-recipe run on task A (d=96, N=12, 2500 steps): `A_mlp_main` (d=128) is
the headline; the d=96 rows share everything else so each ablation isolates
exactly one factor.

| run | changed | expected signature (failure mode) |
|---|---|---|
| `A_abl_noprice`  | λ_price = 0 | **Failure 3 — token spam**: emits/input → N (≈12); the controller buys compute greedily. |
| `A_abl_nocommit` | λ_commit = 0 | **Failure 2 — soft-hard gap**: p sits near 0.5, KL(Bern(p_soft)‖Bern(p_hard)) ≥ 0.1. |
| `A_abl_nondeg`   | λ_nd = 0 | **Failure 1 — mode collapse**: Var_inputs(mean p) ≤ 0.1 (one mode wins). |
| `A_abl_bare`     | all λ = 0 | ≥ 2 of the three signatures simultaneously. |
| `A_abl_sig0`     | σ = 0 | **Failure 4 — latent blackout**: with a perfect latent channel, re-emitting an anchor is never worth its price, so adaptive emission vanishes (corr(measured, emits) collapses). |
| `A_abl_reset`    | semantics = reset (reference sketch's blend) | informational: state-replacing vs additive emission. |
| `A_abl_nogate`   | q_conf = 0 (confidence gate off) | **Failure 2 — soft-hard content gap**: diffuse-soft vs bold-hard emission; hard CE regresses toward the ungated calibration value (2.34x full-latent). |
| `A_abl_redundant`| p_trivial = 0.4 (oracle-simplifiable `x 1` ops injected) | **difficulty-axis test**: with redundant surface, `label` (surface ops) and `measured` (nontrivial ops) diverge. Report `corr(label, emits)` vs `corr(measured, emits)`: does the controller track model-experienced difficulty or the oracle axis? |

Interpretation template (from the JSON, in FINDINGS.md):

- If `A_abl_noprice` spams while the full recipe does not → the **price** is
  the anti-spam component.
- If `A_abl_nondeg` collapses while the full recipe does not → the
  **non-degeneracy floor** is the anti-collapse component (it targets the
  exact quantity of criterion 1, so this is a by-construction detector; the
  ablation verifies the detector fires when the fix is removed).
- If `A_abl_nocommit` shows the gap → **commitment** (not temperature
  annealing — none is used) is what closes the soft-hard gap.
- If `A_abl_sig0` kills adaptivity → the **lossy latent channel** is what
  makes emission useful; without it, the latent branch is a lossless highway
  and the controller has no reason to ever write to the trace.

Negative results are first-class: if any expected signature does NOT appear,
FINDINGS.md records `refuted` and that is reported as a finding (it would
mean the corresponding failure mode does not materialize under our setup,
which is itself information about the mechanism).
