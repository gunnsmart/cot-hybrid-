# PROBLEM — Technical Specification

## 1. Setup

A sequential processor maps an input to an answer while passing a hidden state
`h ∈ R^d` through `N` ordered stages:

```
h_0 → h_1 → h_2 → ... → h_N
```

`h_0 = Reader(input)` encodes the full input (a depth-as-time GRU read over the
input tokens; the reader is a task interface, not a stage). Each stage has two
modes:

- **Latent mode** (silent computation):
  `h_l = stage_l(h_{l-1})`
- **Emit mode** (write to the output trace):
  `t_l = readout(h_l)`  (hard: argmax over the vocabulary)
  `h_l = embed(t_l) + stage_l(h_{l-1})`   (additive — per problem statement)

The **mode controller** produces per stage `p_emit(l) = sigmoid(z_emit − z_latent)`
from a 2-logit head on the LayerNorm'd state. Training is end-to-end
differentiable; inference uses hard mode selection at threshold `τ = 0.5`.

**Soft training forward** (exact expectation of the additive hard rule — no
sampling, fully differentiable, deterministic):

```
h' = h + p_emit · E_{t ~ softmax(readout(h))} [ embed(t) ]
```

**Channel noise.** After every stage the latent state passes through
`h ← h + σ·N(0, I)` with fixed `σ > 0`, identically at train and inference
("same semantics at train and inference time"). This models imperfect
retention of the latent channel — the physical reason a clean re-anchored
token can beat a drifted latent value. `σ` is a channel property, not a
curriculum knob; it is never scheduled.

**Emission semantics (ablation).** The reference sketch in the task uses a
state-replacing blend `(1−p)h + p·Ē` / `h ← embed(t)`. We implement the
**spec (additive)** by default and expose the sketch's variant as
`--semantics reset` for the semantics ablation.

## 2. Objective

Learn a non-degenerate, input-adaptive allocation of compute between silent
latent stages and emitted trace tokens, **without any mode supervision**
(no labels for modes, no RL, no policy gradients, no sampling in the
objective). Success = the six criteria below, all of which must pass.

## 3. The four failure modes (all unsolved in literature)

### Failure 1 — Mode collapse
The controller learns to always pick one mode (all-latent or all-emit).
*Mechanism:* gradient asymmetry between the two branches early in training —
the branch that happens to help the task first receives the mode-head
gradient, the other branch starves (`p(1−p) → 0` and the branch weight `1−p`
vanish together), and the collapse becomes self-reinforcing.
*Detector:* criterion 1, `Var_inputs(mean_stage p_emit) ≤ 0.1`.

### Failure 2 — Soft-hard gap
Training uses the soft mixture; inference uses hard selection. Training loss
looks good while inference degrades.
*Detector:* criterion 3, `KL(Bern(p_soft) ‖ Bern(p_hard)) ≥ 0.1` nats, plus
criterion 4 (hard-mode loss within 5% of both baselines).

### Failure 3 — Token spam
Without an emission penalty the controller emits greedily to "buy" more
compute (each emission re-anchors the state and can only look helpful
locally), defeating the purpose of latent reasoning.
*Detector:* `mean emissions/input ≈ N` while the full-latent baseline is
comparable or better.

### Failure 4 — Latent blackout
The latent mode contributes nothing measurable to the output, so end-to-end
gradient through latent stages carries no signal and the model silently
degenerates to "emit everything" or "emit nothing useful".
*Detector:* on the trained hybrid model, forced-latent and forced-emit
evaluations must both be *worse* than the hard-mode hybrid by a measurable
margin (`latent_gap`, `emit_gap` in the JSON); and ablation `A_abl_sig0`
shows what happens when the latent channel becomes perfect (emission loses
its benefit and mode selection degenerates).

## 4. Success criteria (ALL must pass)

Measured on the **test split** (1024 samples, deterministic generation) of
each task, with the best-dev checkpoint, hard mode at `τ = 0.5`:

1. **Non-degenerate mode selection.**
   `Var over inputs of (mean over stages of p_emit) > 0.1`.
2. **Input-adaptive mode allocation.**
   `Pearson(mode_count, measured_difficulty) > 0.3`, where `mode_count` is
   the number of hard emissions. **Both** correlations are always reported:
   `corr(measured_difficulty, mode_count)` and `corr(label, mode_count)` —
   they may disagree (see §6).
3. **Train-inference parity.**
   `KL( Bern(p_soft) ‖ Bern(p_hard) ) < 0.1` nats, where `p_soft` is the mean
   emit probability over (input, stage) and `p_hard` the hard-emit frequency
   on the same set.
4. **No accuracy loss.**
   Test CE under hard mode `≤ 1.05 ×` the full-emit baseline CE **and**
   `≤ 1.05 ×` the full-latent baseline CE. Baselines are separately trained
   models of identical capacity with the mode forced (no mode head). The
   tolerance **1.05** is explicit and is printed everywhere the ratio is
   reported.
5. **Generalization.** The criteria hold for ≥ 2 stage architectures among
   {MLP, GRU, SSM, transformer}. (We run all four.)
6. **Parameter budget.** Mode-mechanism parameters (`mode_norm` +
   `mode_head`) `< 5%` of stage parameters. The shared embedding table is
   counted in the base model (it serves input tokens and emitted trace tokens
   from one vocabulary); the answer `readout` is task interface, present in
   every model including baselines.

## 5. Tasks (synthetic benchmarks)

All three tasks reduce to a chain of value updates, so the only decision is
*where to anchor the working value* — in the lossy latent channel or on the
trace. Single final answer token; loss = cross-entropy on `readout(h_N)`.
Gold intermediate traces are produced for **analysis only** (never used as
supervision).

| task | input | success | label difficulty | measured difficulty |
|---|---|---|---|---|
| A arithmetic | `(3+4)x(2+5)` over digits 1–9, values ≤ 49 | correct final value | number of operators in the surface expression | number of operators that are not trivially `x 1` (oracle simplification) |
| B logic | facts `A->B. B->C. ...` + distractor facts, query `Q A` | correct endpoint entity | number of facts given (chain edges + distractors) | number of hops actually needed (distractors are redundant) |
| C recall | `start v; op1 ... opD` (dbl/add1/add2/sub1/sub2/half/nop) | correct final value | number of instructions given | number of non-`nop` instructions |

Depth ranges: A: 2–7 ops; B: 2–8 hops; C: 2–8 instructions. Trivial
components (`x 1`, distractor facts, `nop`) are injected stochastically, so
`label` and `measured` genuinely differ on a substantial fraction of samples
and the two correlations in criterion 2 can disagree.

Splits: train 4096 / dev 512 / test 1024, deterministic from data seed 1234.

## 6. Measurement rules (mandatory)

- Every number is reproducible from `results/*.json`. `FINDINGS.md` is
  generated from JSON by `scripts/render_findings.py`; a test re-renders and
  diffs byte-for-byte. No prose claim may drift from JSON.
- Report **both** `corr(measured_difficulty, mode_count)` and
  `corr(label, mode_count)` — they may disagree.
- Report criterion 4 with the explicit tolerance (1.05) on both comparisons.
- The pytest collection count printed in FINDINGS.md must match live
  `pytest --collect-only` (tested).
- Seeds: data seed fixed at 1234; model seed per run (default 0) recorded in
  the run JSON. Channel noise at eval uses a seeded generator per (run,
  batch) so test numbers are bit-reproducible.

## 7. Push rule (mandatory)

This deliverable is tracked in the session repository (the outer git repo
containing this directory).

- Commit + push **every 250 training steps** by default (the spec's
  "every 500 steps / never run > 500 without pushing" is the *floor*; the
  finer cadence means a push still lands at 500/1000/1500...): the run JSON
  (`results/<id>.json` — progress curves + metrics) and, when present, the
  current best checkpoint (`checkpoints/<id>.pt`). A full fp16 snapshot of
  the latest weights is written to `runs/<id>/latest.pt` on disk at the same
  moment (scratch dir, used for resume).
- Never run > 500 steps without pushing: the trainer **exits** if a push
  fails.
- `scripts/recover_git.sh` **refuses** (exit 1) if any unpushed commit exists
  or if the upstream of the current branch is missing; run it before resuming
  work.
- Final checkpoints (fp32) are committed under `checkpoints/` at run end and
  are what the test suite loads.
