# experiments/

Descriptions of the three experiment families. **All numbers live in
`results/*.json` and are surfaced in `FINDINGS.md`** (rendered, never
hand-edited) — the files in this directory are design documents, not
results.

| family | question | runs | spec |
|---|---|---|---|
| ablations | which component of the recipe is necessary? | `A_abl_*` | [ABLATIONS.md](ABLATIONS.md) |
| scaling | does the mechanism survive N = 4…48? | `A_scale_n*` | [SCALING.md](SCALING.md) |
| generalization | does it work on other stage architectures? | `A_gru/A_ssm/A_transformer`, `B_*` | [GENERALIZATION.md](GENERALIZATION.md) |

Run everything with `scripts/run_matrix.sh` (commit+push every 500 steps per
run; `scripts/recover_git.sh` guards each start).
