# FINDINGS

Living document -- **regenerated from `results/*.json` only** by `scripts/render_findings.py`. Numbers below are not hand-editable.

## TL;DR

| task | acc (hard) | emits/input | c1 var(p) | c2 corr(measured) | c3 KL | c4 ratio emit/latent (tol 1.05) |
|---|---|---|---|---|---|---|
| arithmetic | 0.733 | 1.99 | 0.227 | 0.308 | 0.003 | 0.444/1.237 (FAIL) |
| logic | 0.574 | 0.11 | 0.045 | 0.235 | 0.002 | 0.812/1.066 (FAIL) |
| recall | 0.961 | 0.94 | 0.152 | 0.121 | 0.002 | 0.530/2.775 (FAIL) |

## Negative results (first class)

What broke when we removed a component, and whether the expected failure mode appeared. `confirmed` means the ablation reproduced the failure signature on the test set.

| ablation | expected failure | observed (test) | verdict |
|---|---|---|---|
| A_abl_noprice | removing the emission price  (expect: Failure 3, token spam) | pending | pending |
| A_abl_nocommit | removing mode commitment     (expect: Failure 2, soft-hard gap) | pending | pending |
| A_abl_nondeg | removing the non-degeneracy term (expect: Failure 1, mode collapse) | pending | pending |
| A_abl_bare | removing ALL three regularizers (expect: all failures) | pending | pending |
| A_abl_sig0 | zero channel noise sigma=0   (expect: emission stops being useful) | pending | pending |
| A_abl_reset | reset semantics instead of additive (informational variant) | acc=0.580, emits=0.27, KL=0.003 | info |
| A_abl_redundant | redundant surface p_trivial=0.4 (oracle 'measured' vs model-experienced difficulty) | acc=0.694, emits=2.73, KL=0.003 | info |
| A_abl_nogate | confidence gate off (q_conf=0)  (expect: Failure 2 content gap -> c4 vs latent) | pending | pending |

Interpretation is written in `experiments/ABLATIONS.md` from the same JSON; nothing here is hand-edited after rendering.

## Success criteria scoreboard

Tolerance for criterion 4 is **1.05** (explicit). c5 requires >= 2 architectures passing; c6 threshold is 5% of stage parameters.

| run | arch | N | c1 | c2 measured | c2 label | c3 | c4 emit | c4 latent | c6 |
|---|---|---|---|---|---|---|---|---|---|
| A_mlp_main | mlp | 12 | 0.227 (PASS) | 0.308 (PASS) | 0.259 | 0.003 (PASS) | 0.444 | 1.237 | 0.26% (PASS) |
| B_mlp_main | mlp | 12 | 0.045 (FAIL) | 0.235 (FAIL) | 0.376 | 0.002 (PASS) | 0.812 | 1.066 | 0.26% (PASS) |
| C_mlp_main | mlp | 12 | 0.152 (PASS) | 0.121 (FAIL) | 0.121 | 0.002 (PASS) | 0.530 | 2.775 | 0.26% (PASS) |
| A_gru | gru | 12 | 0.218 (PASS) | 0.137 (FAIL) | 0.112 | 0.011 (PASS) | 0.514 | 1.263 | 0.06% (PASS) |
| A_ssm | ssm | 12 | 0.225 (PASS) | 0.225 (FAIL) | 0.198 | 0.004 (PASS) | 0.530 | 1.448 | 0.35% (PASS) |
| A_transformer | transformer | 12 | 0.218 (PASS) | -0.026 (FAIL) | 0.006 | 0.003 (PASS) | 0.552 | 1.069 | 0.06% (PASS) |

Criterion 5: **0** architecture rows pass all row criteria (requirement: >= 2) -> FAIL.

## Scaling in N (task A, mlp, d=96)

| N | acc | emits/input | c1 | c2 measured | c3 | p_eff(hard) |
|---|---|---|---|---|---|---|
| 4 | 0.648 | 0.55 | 0.227 | 0.133 | 0.006 | 0.136 |
| 8 | 0.548 | 2.16 | 0.221 | -0.359 | 0.000 | 0.270 |
| 12 | 0.624 | 0.98 | 0.218 | 0.144 | 0.018 | 0.081 |
| 24 | 0.443 | 5.14 | 0.187 | -0.284 | 0.002 | 0.214 |
| 48 | 0.532 | 2.00 | 0.170 | 0.297 | 0.039 | 0.042 |

## Architecture generalization

Task A (with per-architecture baselines for c4) and task B (c1-c3 only):

| run | arch | task | acc | c1 | c2 measured | c3 | c4 (emit/latent) |
|---|---|---|---|---|---|---|---|
| A_gru | gru | arithmetic | 0.703 | 0.218 | 0.137 | 0.011 | 0.514/1.263 (FAIL) |
| A_ssm | ssm | arithmetic | 0.609 | 0.225 | 0.225 | 0.004 | 0.530/1.448 (FAIL) |
| A_transformer | transformer | arithmetic | 0.630 | 0.218 | -0.026 | 0.003 | 0.552/1.069 (FAIL) |
| B_gru | gru | logic | 0.191 | 0.069 | 0.000 | 0.124 | - |
| B_transformer | transformer | logic | 0.615 | 0.104 | 0.002 | 0.006 | - |

## Mode analysis (main runs)

### arithmetic (A_mlp_main)

acc=0.733, emits/input=1.99, p_eff(soft)=0.193, p_eff(hard)=0.166, c3 KL=0.003

mean p_emit per stage:

`0.48 0.44 0.43 0.43 0.43 0.43 0.43 0.43 0.44 0.44 0.44 0.44`

emissions by measured difficulty:

| measured | n | emits | mean p |
|---|---|---|---|
| 1 | 108 | 0.90 | 0.193 |
| 2 | 516 | 1.45 | 0.325 |
| 3 | 284 | 2.81 | 0.599 |
| 4 | 116 | 3.35 | 0.770 |

corr(measured, emits) = 0.308 ; corr(label, emits) = 0.259 (agree).
Contribution of each mode (hybrid hard vs forced): latent gap = 0.061, emit gap = 0.383.

### logic (B_mlp_main)

acc=0.574, emits/input=0.11, p_eff(soft)=0.017, p_eff(hard)=0.009, c3 KL=0.002

mean p_emit per stage:

`0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05`

emissions by measured difficulty:

| measured | n | emits | mean p |
|---|---|---|---|
| 2 | 244 | 0.00 | 0.007 |
| 3 | 251 | 0.00 | 0.005 |
| 4 | 255 | 0.00 | 0.003 |
| 5 | 274 | 0.42 | 0.182 |

corr(measured, emits) = 0.235 ; corr(label, emits) = 0.376 (disagree).
Contribution of each mode (hybrid hard vs forced): latent gap = -0.002, emit gap = 0.353.

### recall (C_mlp_main)

acc=0.961, emits/input=0.94, p_eff(soft)=0.094, p_eff(hard)=0.078, c3 KL=0.002

mean p_emit per stage:

`0.22 0.24 0.23 0.23 0.23 0.22 0.22 0.20 0.19 0.19 0.18 0.18`

emissions by measured difficulty:

| measured | n | emits | mean p |
|---|---|---|---|
| 2 | 274 | 0.67 | 0.151 |
| 3 | 253 | 0.77 | 0.186 |
| 4 | 263 | 1.10 | 0.239 |
| 5 | 234 | 1.26 | 0.278 |

corr(measured, emits) = 0.121 ; corr(label, emits) = 0.121 (agree).
Contribution of each mode (hybrid hard vs forced): latent gap = 0.085, emit gap = 0.521.

## Reproducibility

- All numbers above are rendered by `scripts/render_findings.py` from `results/*.json`. Regenerate with `python -m scripts.render_findings`.
- Test suite: `pytest -q` (collection count reported here: **n/a** ; a test asserts this matches live `pytest --collect-only`).
- Data is generated deterministically from fixed seeds (data seed 1234); each run's model seed is in its JSON config.
- Push rule: every run commits+pushes every 250 steps (spec floor 500; `--push-every` in each run's JSON); `scripts/recover_git.sh` refuses unpushed state (git history shows the cadence).
- Reproduce one run: `python -m scripts.train --run-id A_mlp_main --task arithmetic --arch mlp --d 128 --n 12 --vocab 64 --steps 4000 --batch 96 --sigma 0.1 --lam-price 0.2 --lam-commit 0.1 --lam-nd 20.0 --q-conf 0.3 --t-conf 0.1 --tau 0.5 --semantics additive --force none --depth-max 4 --p-trivial 0.0 --reader-layers 2 --train-n 4096 --dev-n 512 --test-n 1024` (push rule on; add `--no-git` to skip).

---
*Negative results are first-class: a refuted expectation above is a finding, not a bug.*
