# CONSTRAINT.md — Hard Constraints

The implementation must satisfy all of the following. Each constraint lists
how this repository honors it and where to verify.

## 1. Differentiable only
No RL, no policy gradient, no sampling in the objective.
- The mode decision is relaxed by `p_emit = sigmoid(Δz)` and the soft forward
  is the **exact expectation** of the hard rule (a soft-assignment blend),
  which is a plain differentiable function. No Gumbel, no straight-through,
  no sampled actions, no reward shaping.
- The three regularizers (price, commitment, non-degeneracy) are
  pointwise/batch statistics of `p_emit` — ordinary losses.
- Verify: `src/models.py` (forward), `scripts/train.py` (loss).

## 2. Single-phase training
No curriculum, no stage gates, no scheduled knobs.
- All hyperparameters (λ₁, λ₂, λ₃, σ, τ, lr) are **constant** over the whole
  run; there is exactly one training loop with one loss.
- No temperature annealing (commitment, not annealing, closes the
  soft-hard gap).
- Best-checkpoint selection on the held-out dev set is standard model
  selection, not a training phase; the JSON records which step was chosen.
- Verify: `scripts/train.py` (no schedules anywhere; `--force` baselines are
  separate runs, not a phase of one run).

## 3. No external supervision on mode selection
- The only loss on task outputs is cross-entropy on the final answer token.
- Gold intermediate traces (data generators) are stored for *analysis only*
  and never enter any loss.
- The regularizers are unsupervised statistics of `p_emit` itself (a budget
  and a commitment), not labels.
- Verify: `scripts/train.py` loss block; `src/data.py` (traces unused in
  training).

## 4. No changes to the underlying stage architecture
- The four stage blocks (MLP as in the reference, GRU cell, diagonal SSM
  cell, pre-LN cross-attention block) are standard; the mode mechanism
  (norm + 2-logit head), the emission interface (shared embedding) and the
  channel noise act **between/around** stages and never modify stage
  parameters or structure.
- Verify: `src/models.py` STAGES; parameter-accounting groups in
  `InterleavedProcessor.param_groups()`.

## 5. Must scale to N ≥ 12 stages
- Runs at N ∈ {4, 8, 12, 24, 48} (task A) — see `results/A_scale_*.json`.
- No architecture has any state that depends on N at construction time
  beyond the stage list itself (the transformer stage's trace buffer is
  O(emissions) and is a function of the run, not of N at init).

## 6. Same semantics at train and inference time
- Same model, same forward, same `p_emit`, same channel noise `σ` (seeded
  generator at eval for reproducibility).
- The only difference is the decision rule: soft expectation (training) vs
  hard threshold `τ = 0.5` (inference). The gap this creates is exactly
  criterion 3, which we measure and bound (< 0.1 nats) and close with the
  commitment term.

## 7. Push rule (operational, not algorithmic)
- Commit + push every 500 steps; refuse to continue on push failure;
  `scripts/recover_git.sh` refuses unpushed state. See `PROBLEM.md` §7.

## Interpretation notes (edge cases, decided and recorded)

- **"Differentiable" vs surrogate gradients.** We use none. The soft forward
  is a true derivative path (expectation), unlike STE/Gumbel-Softmax
  straight-through, which some might consider policy-gradient-adjacent.
- **Channel noise σ.** Applied identically at train and inference (same
  semantics). It is part of the problem setup (an imperfect latent channel),
  not regularization that is removed at test time. Ablation `A_abl_sig0`
  reports the σ = 0 world.
- **Vocabulary / embedding sharing.** One embedding table for input tokens
  and emitted tokens: the trace and the input live in the same vocabulary
  (a value token denotes the same value in both places). This is standard
  and keeps criterion 6 honest (the table is counted once, in the base).
- **Baselines.** "Full-emit" / "full-latent" baselines are separately
  trained models with the mode forced for the whole run (and no mode head),
  identical otherwise. Forced-mode evaluations *of the hybrid model* are
  additionally reported for the Failure-4 contribution analysis.
