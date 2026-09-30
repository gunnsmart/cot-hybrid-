# Architecture generalization

The mode mechanism must be architecture-agnostic: identical mode head,
identical regularizers, identical tasks — only the stage block changes.

| arch | stage block | notes |
|---|---|---|
| `mlp` | `h + Linear(GELU(LN(h)))` (reference Stage) | baseline architecture |
| `gru` | one GRU cell with the state as its own input (depth-as-time) | recurrent, no extra sequence needed |
| `ssm` | diagonal S4-style cell `a·h + (1−a)·Wh`, `a=sigmoid(·) ∈ (0,1)^d` | linear-recurrence state-space cell, one step |
| `transformer` | pre-LN cross-attention of the state (query) over the emitted trace (keys/values) + MLP | the only stage that *reads back* the trace; emission has a structural benefit here |

Runs:

- Task A with **per-architecture baselines** (so criterion 4 is a controlled
  comparison inside each architecture): `A_gru(+emit/latent)`,
  `A_ssm(+emit/latent)`, `A_transformer(+A_tf_emit/A_tf_latent)`.
- Task B on `gru` and `transformer` (c1/c2 only — cross-task generalization,
  no attached baselines).

Criterion 5 passes when ≥ 2 architecture rows satisfy c1, c3 and c4
(MLP counts via the main runs).
