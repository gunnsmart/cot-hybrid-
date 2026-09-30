# PHASE 2 — pressure regimes

## Why phase 2

Phase 1 ended with a clean negative result: the hybrid **beats the full-emit
specialist on every task** but **loses to the full-latent specialist on every
task** (criterion-4 latent ratio ≈ 1.07–2.78× across tasks A/B/C; see
`FINDINGS.md`, rendered from `results/*.json`).

The diagnosis: the phase-1 regime never *forces* the model to externalize
state. Concretely,

- channel noise was low (σ = 0.1), so the latent state barely drifts and a
  re-anchoring token buys almost nothing;
- the reader consumes the **entire input before stage 0** (`h_0 =
  Reader(input)`), so no stage ever needs to remember an input fact it has
  not seen yet;
- the tasks are shallow (depth ≤ 4–5), so little has to be carried across
  many stages.

Under those conditions "keep everything latent" is a legitimately good
strategy, and the controller's emissions are a small tax. PHASE 2 tests that
conclusion directly: change the regime so that externalizing state *should*
pay, and watch what happens to the hybrid-vs-specialist ratios.

## Hypotheses (all falsifiable)

| H | change | runs | prediction (falsifiable) |
|---|---|---|---|
| **H1 — lossy channel** | σ ∈ {0.3, 0.5} (was 0.1) | `A_s03(_emit/_latent)`, `A_s05(_emit/_latent)` | higher noise hurts the latent specialist more than the hybrid (token re-anchoring becomes worth its price) → `r_latent` decreases with σ, and there should be a σ with c4 ≤ 1.05 |
| **H2 — information over time** | `--reader-mode gradual`: the input is split into N+1 chunks — chunk 0 builds `h_0`, chunk l+1 is injected into the state **after stage l** through per-stage `nn.Linear(d, d, bias=False)` projections, zero-initialized. No stage ever sees the full input again. A chunk is the mean embedding of its span (`<pad>` = token id 0 masked out; an empty chunk is the zero vector). | `A_bneck(_emit/_latent)` | information arrives piecewise → the latent-only model must remember facts across many noisy stages, while emissions re-anchor → the hybrid gains an edge over the latent specialist; c2 should strengthen |
| **H3 — does the controller actually earn its keep?** | `--force schedule --schedule-every k`: unconditional emission every k-th stage — **no confidence gate** (to isolate "learned when" from the gate mechanism) and the mode head is not trained | `A_sched4` (3 emits ≈ hybrid's 2), `A_sched2` (6 emits) | the controller must beat a fixed schedule at a comparable emit budget; if it loses, the controller itself adds no value over positional heuristics (a major finding, not a bug) |
| H4 (fallback) | deep tasks (depth 8–12, d = 192) | not yet run | only if H1/H2 show signal |

Design notes:

- **Regime-matched baselines.** For H1/H2 the emit/latent specialists are
  *re-trained inside the same regime* (`A_s03_emit`, `A_bneck_latent`, …),
  so every c4 ratio is apples-to-apples. Phase-1 specialists belong to the
  σ = 0.1 / full-reader regime and are never mixed in.
- **Parameter accounting.** The gradual reader's chunk projections are task
  interface — they replace the GRU reader every baseline also needs — so they
  count in `base`, not `mechanism` (criterion 6 is untouched by the regime).
- **Schedule = gate bypassed.** `force="schedule"` emits exactly at stages
  k, 2k, 3k, … regardless of readout confidence, and no mode head exists.
  This isolates the controller's *decision* from the gate's *content*
  protection.
- **Half of c4 that is asserted vs reported.** Winning against the emit
  specialist is asserted by the test suite (`tests/test_09_phase2.py`);
  closing the gap to the latent specialist is the *measurement* of this phase
  — winning or losing there is a result, so it is reported in `FINDINGS.md`
  ("Phase 2 - pressure regimes") and never asserted.

## The 11 runs (task A / arithmetic, mlp, d=128, N=12, depth ≤ 4,
p_trivial 0.0, 2500 steps, phase-1 recipe: `--sigma 0.1 --lam-price 0.2
--lam-commit 0.1 --lam-nd 20 --reader-layers 2 --batch 96 --q-conf 0.30
--content expectation --push-every 100`)

| run | regime delta |
|---|---|
| `A_s03` / `A_s03_emit` / `A_s03_latent` | `--sigma 0.3` |
| `A_s05` / `A_s05_emit` / `A_s05_latent` | `--sigma 0.5` |
| `A_bneck` / `A_bneck_emit` / `A_bneck_latent` | `--reader-mode gradual` |
| `A_sched4` | `--force schedule --schedule-every 4` (3 emits) |
| `A_sched2` | `--force schedule --schedule-every 2` (6 emits) |

Defined in `scripts/run_matrix.sh` (PHASE 2 block, after the phase-1 matrix)
and in the MATRIX2 cells of `kaggle.ipynb` / `colab.ipynb`. Override flags
(σ, reader-mode, force) are placed **after** the shared recipe array — the
phase-1 lesson where a flag hidden *before* the recipe was silently
overridden and wasted six ablation runs.

## How to read the results

Everything renders into `FINDINGS.md` § "Phase 2 - pressure regimes" from
`results/*.json` only (`python -m scripts.render_findings`):

- **r_emit / r_latent** = hybrid test CE ÷ specialist test CE *in the same
  regime* (tolerance 1.05, as in criterion 4). Phase-1 reference point:
  r_latent(A) = 1.237.
- **H1 confirmed** if r_latent falls as σ rises and some σ reaches
  c4 ≤ 1.05 (both halves of c4). Refuted if the full data shows neither.
- **H2 confirmed** if r_latent improves over phase 1 *and* c2 (corr of
  emissions with measured difficulty) strengthens.
- **H3 confirmed** if the hybrid's CE beats `A_sched4`'s (the
  budget-matched schedule, ≈3 emits vs the hybrid's ≈2) within 1.05;
  `A_sched2` shows the budget trade-off at 6 emits. If a schedule wins,
  that is reported as a first-class negative result about the controller.
- Run-level gates asserted by the suite: the new hybrids must stay
  non-degenerate and train/inference-aligned (c1 > 0.1, c3 < 0.1) and must
  still beat the emit specialist (≤ 1.05×) in every regime; schedule runs
  must emit exactly N/k tokens (± 0.5).
