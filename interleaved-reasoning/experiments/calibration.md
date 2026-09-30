# Calibration — how the recipe was found

All numbers below are read from `results/A_cal*.json` (and the two
`*_base_cal2` baselines) at the time of writing; regenerate the summary with
`scripts/render_findings.py`. Calibration runs share the headline geometry
(d=128, N=12, task A, depth 2–4, p_trivial=0, σ=0.2, R2L reader) and differ
only in the knobs under test.

## 0. Baselines (the c4 yardstick)

| run | acc | CE |
|---|---|---|
| `A_latent_base_cal2` (force latent) | 0.716 | 0.811 |
| `A_emit_base_cal2` (force emit) | 0.443 | 2.948 |

c4 ceiling for any hybrid: CE ≤ 1.05 × 0.811 = **0.851** (latent is the
binding side).

## 1. The pre-fix-generator era (discarded)

`A_cal1` / `A_cal2` were run before the task-A generator fix (depth
semantics + triviality detection). Their "measured" axis was
oracle-defined and anti-correlated with the model's actual difficulty, so
they were deleted from `results/` (history remains in git). **Do not cite
their numbers.** The two cal2-named baselines above postdate the fix and are
valid.

## 2. The confidence gate (c4)

With plain expectation content and no gate, hard-mode emissions inject bold
argmax tokens the model was trained on diffuse mixtures — the *content*
half of the soft-hard gap. Ungated hard mode measured 2.34× the full-latent
CE (A_cal2-era, pre-fix data). The fix: gate emissions on the readout's
peak probability `maxp` (soft: `g = sigmoid((maxp−q)/t)`, hard: `maxp ≥ q`),
same `(q,t)` both modes.

## 3. Canary series (post-fix data, d128 R2L depth≤4 p_trivial=0 σ=0.2)

| run | price | commit | t | steps | acc | CE | c1 | c2m | c3 | c4 ratio (latent) |
|---|---|---|---|---|---|---|---|---|---|---|
| `A_cal3` | 0.2 | 0.1 | 0.1 | 2500 | 0.740 | 0.909 | 0.110 | 0.284 | 0.000 | 1.121 ✗ |
| `A_cal4` | 0.15 | 0.1 | 0.05 | 3000 | 0.732 | 0.813 | 0.125 | 0.158 | 0.002 | 1.002 ✓ |
| `A_cal5` | 0.2 | 0.3 | 0.05 | 3000 | 0.724 | 0.909 | 0.124 | −0.364 | 0.000 | 1.121 ✗ |
| `A_cal6`* | 0.3 | 0.2 | 0.05 | 3000 | 0.702 | 0.856 | 0.123 | −0.344 | 0.000 | 1.055 ✗ |
| `A_cal7` | 0.2 | 0.1 | 0.1 | 4000 | 0.753 | 0.748 | 0.220 | 0.220 | 0.001 | 0.922 ✓ |

\* `A_cal6` is the first run after the price was corrected to the expected
number of tokens actually written, `E[p·g]`.

**Readings.**

- The gate's sharpness trades c4 against c2: soft gate (t=0.1) keeps
  emissions tracking difficulty (c2m 0.28) but leaves CE above the ceiling;
  sharp gate (t=0.05) nails c4 but inverts the difficulty trend, because
  readout confidence is *anti*-correlated with difficulty (easy inputs are
  the confident ones).
- Raising the commitment (A_cal5) makes the inversion worse: the model
  emits aggressively on easy inputs (4.49/input at m=1) and shuts off on
  hard ones (0.00 at m=4) — the task loss rewards emission where the
  model is confident-and-correct, which is the easy tier.
- **Per-tier diagnostic (A_cal4 weights):** emissions are net-helpful on
  *every* tier — hard mode beats the same weights forced latent by
  +13.9/+3.7/+6.6/+9.5 acc points at m=1..4. The c2 problem is pure
  frequency allocation, not content quality.
- A_cal7 (soft gate, 4000 steps, effective price) passes c1/c3/c4 with room
  (CE 0.748 < 0.851; better than the latent specialist) but c2m=0.220.

## 4. Same content in both modes (c2)

The remaining c2 gap is attributed to the *content* mismatch: with
expectation content, the mode head's gradient points at a diffuse mixture
instead of the specific token that will be written at inference. The
`--content argmax` mode trains soft emission with exactly the hard-mode
token (only the *decision* is relaxed: `p·g` vs the threshold), giving the
mode head a clean per-token utility signal. Canary `A_cal8` tests this.

## 5. Final recipe (set by the passing canary)

Recorded here from the winning canary's JSON so the matrix is fully
reproducible; the matrix script reads the same values.
