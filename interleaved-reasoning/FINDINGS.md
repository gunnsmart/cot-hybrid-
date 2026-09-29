# FINDINGS

Living document -- **regenerated from `results/*.json` only** by `scripts/render_findings.py`. Numbers below are not hand-editable.

## TL;DR

| task | acc (hard) | emits/input | c1 var(p) | c2 corr(measured) | c3 KL | c4 ratio emit/latent (tol 1.05) |
|---|---|---|---|---|---|---|
| arithmetic | pending | | | | | |
| logic | pending | | | | | |
| recall | pending | | | | | |

## Negative results (first class)

What broke when we removed a component, and whether the expected failure mode appeared. `confirmed` means the ablation reproduced the failure signature on the test set.

| ablation | expected failure | observed (test) | verdict |
|---|---|---|---|
| A_abl_noprice | removing the emission price  (expect: Failure 3, token spam) | pending | pending |
| A_abl_nocommit | removing mode commitment     (expect: Failure 2, soft-hard gap) | pending | pending |
| A_abl_nondeg | removing the non-degeneracy term (expect: Failure 1, mode collapse) | pending | pending |
| A_abl_bare | removing ALL three regularizers (expect: all failures) | pending | pending |
| A_abl_sig0 | zero channel noise sigma=0   (expect: emission stops being useful) | pending | pending |
| A_abl_reset | reset semantics instead of additive (informational variant) | pending | pending |
| A_abl_redundant | redundant surface p_trivial=0.4 (oracle 'measured' vs model-experienced difficulty) | pending | pending |

Interpretation is written in `experiments/ABLATIONS.md` from the same JSON; nothing here is hand-edited after rendering.

## Success criteria scoreboard

Tolerance for criterion 4 is **1.05** (explicit). c5 requires >= 2 architectures passing; c6 threshold is 5% of stage parameters.

| run | arch | N | c1 | c2 measured | c2 label | c3 | c4 emit | c4 latent | c6 |
|---|---|---|---|---|---|---|---|---|---|
| A_mlp_main | mlp | ? | pending |
| B_mlp_main | mlp | ? | pending |
| C_mlp_main | mlp | ? | pending |
| A_gru | gru | ? | pending |
| A_ssm | ssm | ? | pending |
| A_transformer | transformer | ? | pending |

Criterion 5: **0** architecture rows pass all row criteria (requirement: >= 2) -> FAIL.

## Scaling in N (task A, mlp, d=96)

| N | acc | emits/input | c1 | c2 measured | c3 | p_eff(hard) |
|---|---|---|---|---|---|---|
| n4 | pending |
| n8 | pending |
| 12 | pending |
| 24 | pending |
| 48 | pending |

## Architecture generalization

Task A (with per-architecture baselines for c4) and task B (c1-c3 only):

| run | arch | task | acc | c1 | c2 measured | c3 | c4 (emit/latent) |
|---|---|---|---|---|---|---|---|
| A_gru | gru | ? | pending |
| A_ssm | ssm | ? | pending |
| A_transformer | transformer | ? | pending |
| B_gru | gru | ? | pending |
| B_transformer | transformer | ? | pending |

## Mode analysis (main runs)

### arithmetic: pending

### logic: pending

### recall: pending

## Reproducibility

- All numbers above are rendered by `scripts/render_findings.py` from `results/*.json`. Regenerate with `python -m scripts.render_findings`.
- Test suite: `pytest -q` (collection count reported here: **n/a** ; a test asserts this matches live `pytest --collect-only`).
- Data is generated deterministically from fixed seeds (data seed 1234); each run's model seed is in its JSON config.
- Push rule: every run commits+pushes every 250 steps (spec floor 500; `--push-every` in each run's JSON); `scripts/recover_git.sh` refuses unpushed state (git history shows the cadence).
- Reproduce one run: `python -m scripts.train --run-id A_mlp_main ...` (see the run's JSON config once it exists).

---
*Negative results are first-class: a refuted expectation above is a finding, not a bug.*
