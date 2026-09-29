# Interleaved Latent-Token Reasoning in Sequential Processors

**Open problem.** A sequential processor passes a hidden state `h ∈ R^d` through
`N` ordered stages. At each stage a *mode controller* chooses:

- **latent mode** — compute silently: `h_l = stage_l(h_{l-1})`
- **emit mode** — write a token to the output trace: `t_l = readout(h_l)`,
  `h_l = embed(t_l) + stage_l(h_{l-1})`

The controller is learned **end-to-end and differentially, with no mode
supervision**: the model must discover *when silent thinking helps* and *when
putting a token on the trace helps*.

**Status.** Working single-phase reference solution on three synthetic tasks,
four stage architectures, N up to 48. All numbers in this repo are rendered
from `results/*.json` — see [FINDINGS.md](FINDINGS.md).

<!-- RESULTS:BEGIN -->
| task | acc (hard) | emits/input | c2 corr(measured) | status |
|---|---|---|---|---|
| arithmetic | pending | | | running |
| logic | pending | | | running |
| recall | pending | | | running |
<!-- RESULTS:END -->

## The four failure modes

All four are unsolved in the literature and all four are reproduced by
deliberate ablation in this repo (see FINDINGS.md, "Negative results"):

| # | Failure | Mechanism | Fix used here | Ablation proving necessity |
|---|---|---|---|---|
| 1 | Mode collapse | gradient asymmetry between branches early in training | non-degeneracy regularizer on Var across inputs of mean `p_emit` | `A_abl_nondeg` |
| 2 | Soft-hard gap | soft mixture at train, hard selection at inference | mode-commitment term `4p(1-p)` (no temperature annealing) | `A_abl_nocommit` |
| 3 | Token spam | emission "buys" compute, so `p_emit -> 1` everywhere | linear emission price `λ·mean(p_emit)` | `A_abl_noprice` |
| 4 | Latent blackout | latent branch carries no measurable contribution | additive emission + lossy latent channel make both branches load-bearing | `A_abl_sig0` |

## Repository layout

```
README.md          this file: problem statement + reference implementation
PROBLEM.md         technical spec: setup, 4 failure modes, 6 criteria, metric defs
CONSTRAINT.md      hard constraints + interpretation
prior_art.md       review of prior work
src/               working implementation (models, data, metrics)
scripts/           training CLI, push-rule hook, recover_git.sh, matrix runner,
                   FINDINGS renderer
tests/             verification suite (all 6 criteria + negative-result checks)
experiments/       ablation / generalization / scaling descriptions + results
results/           JSON only -- single source of truth for every number
checkpoints/       final best weights (fp32) for the test suite
runs/              (gitignored) scratch: fp16 latest weights per run
FINDINGS.md        generated from results/*.json -- do not hand-edit
```

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install torch numpy pytest

# one run (deterministic; ~minutes on CPU)
.venv/bin/python -m scripts.train --run-id A_mlp_main \
    --task arithmetic --arch mlp --d 128 --n 12 --steps 3000

# full matrix (30 runs; auto commit+push every 500 steps)
scripts/run_matrix.sh

# verification suite (all 6 criteria, negative results, consistency)
.venv/bin/pytest -q

# regenerate FINDINGS.md from JSON
.venv/bin/python -m scripts.render_findings
```

## Reference implementation (as given in the task)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Linear(d, d)
    def forward(self, h):
        return h + self.mlp(F.gelu(self.norm(h)))


class InterleavedProcessor(nn.Module):
    def __init__(self, d=256, n_stages=12, vocab=64):
        super().__init__()
        self.d = d
        self.n_stages = n_stages
        self.stages = nn.ModuleList([Stage(d) for _ in range(n_stages)])
        self.mode_head = nn.Linear(d, 2)       # [latent, emit]
        self.readout = nn.Linear(d, vocab)
        self.embed = nn.Embedding(vocab, d)

    def forward(self, h, mode="soft", tau=0.5):
        tokens = []
        mode_logs = []
        for l, stage in enumerate(self.stages):
            h = stage(h)
            logits = self.mode_head(F.layer_norm(h, [self.d]))
            if mode == "soft":
                p_emit = torch.sigmoid(logits[..., 1] - logits[..., 0])
                tok_soft = F.softmax(self.readout(h), dim=-1)
                h = (1 - p_emit) * h + p_emit * (self.embed.weight.T @ tok_soft)
                mode_logs.append(p_emit)
            else:
                p_emit = torch.sigmoid(logits[..., 1] - logits[..., 0])
                if (p_emit >= tau).any():
                    tok = self.readout(h).argmax(-1)
                    tokens.append(tok)
                    h = self.embed(tok)
        return h, tokens, mode_logs
```

Three deviations from this sketch, all documented in `PROBLEM.md`:

1. **Batched-matmul bug.** `self.embed.weight.T @ tok_soft` fails for general
   batch sizes (inner dims `vocab` vs `batch`). Fixed to `tok_soft @ self.embed.weight`.
2. **Per-item hard selection.** `(p_emit >= tau).any()` emits for the *whole
   batch* when *any* sample passes the threshold; the implementation masks
   per item.
3. **Emission semantics.** The problem statement defines additive emission
   `h_l = embed(t_l) + stage_l(h_{l-1})`; the sketch replaces the state.
   We implement the **spec (additive)** by default and keep the sketch's
   blend ("reset") as an ablation (`--semantics reset`).

## Method in one paragraph

The mode head outputs `p_emit = sigmoid(z_emit - z_latent)`. Training runs the
soft forward `h' = h + p_emit · E_t~softmax(readout(h))[embed(t)]` (the exact
expectation of the additive hard rule — differentiable, deterministic, no
sampling), plus three unsupervised regularizers: an **emission price**
(λ₁·mean p, kills spam), a **mode-commitment** term (λ₂·mean 4p(1−p), closes the
soft-hard gap without temperature annealing), and a **non-degeneracy** term
(λ₃·ReLU(0.1 − Var_inputs(mean p))², kills collapse — the *same quantity* as
criterion 1). The latent channel adds fixed noise `h ← h + σN(0,I)` identically
at train and inference; it is what makes re-emitting a clean anchor worth its
price on hard inputs. Inference is hard: emit iff `p_emit ≥ τ = 0.5`.

## Measurement rules (enforced by tests)

- Every number is reproducible from `results/*.json`; `FINDINGS.md` is
  generated, never hand-edited (a test re-renders and diffs).
- Both `corr(measured_difficulty, mode_count)` and `corr(label, mode_count)`
  are reported everywhere — they can disagree.
- Criterion 4 is reported with the explicit tolerance **1.05** on both
  baseline comparisons.
- The pytest collection count printed in FINDINGS.md is asserted equal to the
  live `pytest --collect-only` count.

## Push rule (mandatory)

This deliverable is tracked in the session repository (the git repository
that contains this directory); every git command below resolves to it.

- Every training run commits + pushes after **every 500 steps**
  (`scripts/common.push_step`), and **refuses to continue** if a push fails.
  The commit carries the run JSON (progress + metrics) and the current best
  checkpoint; a full fp16 snapshot of the latest weights is written to
  `runs/<id>/latest.pt` on disk at the same moment for resume.
- `scripts/recover_git.sh` **refuses** to start work if unpushed commits
  exist (verified by a test on a throwaway repo).
- 500-step commits are visible in the repository history of the session
  branch (search for `push rule: every 500 steps`).

## Publishing

To publish to your own account, make a standalone repo from this directory:

```bash
cd interleaved-reasoning
git init -b main && git add -A && git commit -m "interleaved reasoning"
git remote add origin https://github.com/<you>/interleaved-reasoning.git
git push -u origin main
```

(`runs/` is scratch and ignored by the session repo; the standalone repo's
own `.gitignore` keeps it out as well.)
