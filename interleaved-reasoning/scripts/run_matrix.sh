#!/usr/bin/env bash
# run_matrix.sh -- run the full experiment matrix sequentially.
#
# The push rule is enforced inside every training run (commit+push every 500
# steps, refuse to continue on push failure). This runner additionally runs
# scripts/recover_git.sh before each run so a crashed/unpushed state is
# never built upon.
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

run() { # run <run-id> <args...>
  local id="$1"; shift
  if [ -f "results/$id.json" ] && grep -q '"status": "done"' "results/$id.json"; then
    echo "SKIP $id (already done)"
    return 0
  fi
  echo "=== $id ==="
  scripts/recover_git.sh || { echo "ABORT: recover_git.sh refused (unpushed commits)."; exit 1; }
  "$PY" -m scripts.train --run-id "$id" "$@" || { echo "FAILED $id"; exit 1; }
}

# ---- main runs (d=128, N=12) ----------------------------------------------
run A_mlp_main   --task arithmetic --arch mlp --d 128 --n 12 --steps 3000
run B_mlp_main   --task logic      --arch mlp --d 128 --n 12 --steps 3000
run C_mlp_main   --task recall     --arch mlp --d 128 --n 12 --steps 3000

# ---- baselines for criterion 4 (same task/d/n) -----------------------------
run A_emit_base   --task arithmetic --arch mlp --d 128 --n 12 --steps 3000 --force emit
run A_latent_base --task arithmetic --arch mlp --d 128 --n 12 --steps 3000 --force latent
run B_emit_base   --task logic      --arch mlp --d 128 --n 12 --steps 3000 --force emit
run B_latent_base --task logic      --arch mlp --d 128 --n 12 --steps 3000 --force latent
run C_emit_base   --task recall     --arch mlp --d 128 --n 12 --steps 3000 --force emit
run C_latent_base --task recall     --arch mlp --d 128 --n 12 --steps 3000 --force latent

# ---- ablations (task A, d=96, N=12): which component is necessary? ---------
run A_abl_noprice  --task arithmetic --d 96 --n 12 --steps 2500 --lam-price 0.0
run A_abl_nocommit --task arithmetic --d 96 --n 12 --steps 2500 --lam-commit 0.0
run A_abl_nondeg   --task arithmetic --d 96 --n 12 --steps 2500 --lam-nd 0.0
run A_abl_bare     --task arithmetic --d 96 --n 12 --steps 2500 --lam-price 0.0 --lam-commit 0.0 --lam-nd 0.0
run A_abl_sig0     --task arithmetic --d 96 --n 12 --steps 2500 --sigma 0.0
run A_abl_reset    --task arithmetic --d 96 --n 12 --steps 2500 --semantics reset

# ---- scaling: N = 4, 8, 12, 24, 48 (task A, mlp, d=96; 12 = A_abl row) -----
run A_scale_n4  --task arithmetic --d 96 --n 4  --steps 2500
run A_scale_n8  --task arithmetic --d 96 --n 8  --steps 2500
run A_scale_n24 --task arithmetic --d 96 --n 24 --steps 2500
run A_scale_n48 --task arithmetic --d 96 --n 48 --steps 2500

# ---- architecture generalization (task A + baselines; d=96, N=12) ----------
run A_gru         --task arithmetic --arch gru         --d 96 --n 12 --steps 2500
run A_ssm         --task arithmetic --arch ssm         --d 96 --n 12 --steps 2500
run A_transformer --task arithmetic --arch transformer --d 96 --n 12 --steps 2500
run A_gru_emit    --task arithmetic --arch gru         --d 96 --n 12 --steps 2500 --force emit
run A_gru_latent  --task arithmetic --arch gru         --d 96 --n 12 --steps 2500 --force latent
run A_ssm_emit    --task arithmetic --arch ssm         --d 96 --n 12 --steps 2500 --force emit
run A_ssm_latent  --task arithmetic --arch ssm         --d 96 --n 12 --steps 2500 --force latent
run A_tf_emit     --task arithmetic --arch transformer --d 96 --n 12 --steps 2500 --force emit
run A_tf_latent   --task arithmetic --arch transformer --d 96 --n 12 --steps 2500 --force latent

# ---- cross-task check on two other architectures ----------------------------
run B_gru         --task logic --arch gru         --d 96 --n 12 --steps 2500
run B_transformer --task logic --arch transformer --d 96 --n 12 --steps 2500

echo "MATRIX COMPLETE"
