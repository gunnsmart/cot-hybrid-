# prior_art.md — Review of Prior Work

Scope: systems that decide *how much* and *in what form* a sequential
processor thinks. The joint problem here — **learnable, per-stage,
end-to-end, differentiable selection between latent (silent) computation and
emitted trace tokens, with no mode supervision** — is not solved by any of
the following.

| Work | Year | What it decides | Limitation for this problem |
|---|---|---|---|
| Chain of Thought (Wei et al., arXiv:2201.11903) | 2022 | — | All reasoning is emitted tokens; no latent mode, no choice of when to emit. |
| Pause Tokens (Goyal et al., "Think before you speak", arXiv:2310.10139) | 2023 | fixed pause count (learned per input) | The pause *count* is learned but the pause is silent and fixed in *form*; no per-step decision to emit *contentful* tokens vs stay silent; count is a single scalar, not a per-stage policy over a trace. |
| Quiet-STaR (Zelikman et al., arXiv:2403.09629) | 2024 | — | Distills rationales **post-hoc** into hidden states; not an end-to-end differentiable per-stage mode decision at inference. |
| Coconut (Meta, "Continuous CoT", arXiv:2412.06769) | 2024 | — | **All latent**: states are continuous embeddings with no emitted tokens at all; the opposite degenerate extreme of this problem's mode space. |
| o1 / R1 (OpenAI o1 System Card 2024; DeepSeek-R1 arXiv:2501.12948) | 2024/25 | — | CoT only, no hybrid; the "think longer" decision is not a per-stage learned mode with a silent branch, and training uses RL rather than the differentiable-only constraint here. |
| PonderNet (Banino et al., "Learning to Ponder", arXiv:2107.05407) | 2021 | **halting** (when to stop the recurrence) | Closest in spirit: a learned per-step halting probability trained end-to-end with a halting-distribution regularizer. But it only decides *stop vs continue-the-same-computation*; there is no second output channel (no emission), so no latent/emit choice — halting is a one-bit decision with one mode on both sides. |
| Universal Transformer (Dehghani et al., arXiv:1806.05165) | 2018/19 | shared-depth recurrence with a halting distribution | Depth recurrence only; every step is the same transformation, no emitted trace, no per-stage choice between two semantically different modes. |
| Mixture-of-Depths (Raposo et al., arXiv:2404.02258) | 2024 | **static** routing of tokens across layers by a learned router | Decides which *tokens* skip which *layers* (compute budget), with a fixed per-layer budget; the decision is about token routing, not about the hidden state's mode, and there is no silent-computation branch vs emitted-computation branch — it is a budget allocator, not a mode controller. |

## Why the joint problem remains open

Every line of work above optimizes **one** of the following:

1. **amount** of computation (PonderNet, Universal Transformer, MoD, pause
   tokens) — but with a single computational mode;
2. **form** of computation (CoT = all tokens, Coconut = all latent) — but
   fixed for the whole trajectory;
3. **post-hoc** alignment of latent and token spaces (Quiet-STaR) — but not
   differentiable end-to-end control at inference.

None learns, per stage, a **two-mode** decision (silent transform vs
emitting a token that re-anchors the state) with the four failure modes of
this problem in play: mode collapse (gradient asymmetry), soft-hard gap
(soft training / hard inference), token spam (unpriced emission), and latent
blackout (unmeasurable latent contribution). This repository takes that
joint problem as its contribution target and provides, for each failure
mode, a differentiable single-phase fix plus the ablation that proves the
fix is necessary (see FINDINGS.md "Negative results").

## Notation bridge

- PonderNet's halting probability ↔ our `p_emit`: both are sigmoid heads on
  the state, trained end-to-end with a distribution regularizer (PonderNet:
  KL to a target halting distribution — which is arguably *external
  supervision on the schedule*; ours: an unsupervised variance floor + price
  + commitment, i.e., no target distribution at all).
- MoD's per-layer router ↔ our per-stage mode head: same shape, different
  decision (which tokens get compute vs which *mode* the state is in).
- Coconut's continuous state ↔ our latent mode: we keep the token
  vocabulary reachable from the state, which is what makes "emit" a
  meaningful alternative to "stay latent".
