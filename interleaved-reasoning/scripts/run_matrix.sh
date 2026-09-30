#!/usr/bin/env bash
# run_matrix.sh -- run the full experiment matrix sequentially.
#
# The push rule is enforced inside every training run (commit+push every
# 250 steps by default -- the spec floor is 500 -- and the run refuses to
# continue on push failure). This runner additionally runs
# scripts/recover_git.sh before each run so a crashed/unpushed state is
# never built upon.
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="../.venv/bin/python"   # venv at the outer repo root
[ -x "$PY" ] || PY="python3"

# shared recipe (single phase, no curriculum) — the A_cal27 calibration
# winner (sigma 0.1, gate q 0.30, expectation content; see experiments/calibration.md):
REC=( --sigma 0.1 --lam-price 0.2 --lam-commit 0.1 --lam-nd 20 --reader-layers 2 --batch 96 --q-conf 0.30 --content expectation --push-every 100 )

# หมายเหตุ: ธงที่ต้องการทับค่า REC ต้องอยู่ "หลัง" "${REC[@]}" (argparse ตัวหลังชนะ)
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
run A_mlp_main   --task arithmetic --arch mlp --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 4000 "${REC[@]}"
run B_mlp_main   --task logic      --arch mlp --d 128 --n 12 --depth-max 5 --steps 3000 "${REC[@]}"
run C_mlp_main   --task recall     --arch mlp --d 128 --n 12 --depth-max 5 --p-trivial 0.0 --steps 3000 "${REC[@]}"

# ---- baselines for criterion 4 (same task/d/n) -----------------------------
run A_emit_base   --task arithmetic --arch mlp --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 3000 --force emit "${REC[@]}"
run A_latent_base --task arithmetic --arch mlp --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 3000 --force latent "${REC[@]}"
run B_emit_base   --task logic      --arch mlp --d 128 --n 12 --depth-max 5 --steps 3000 --force emit "${REC[@]}"
run B_latent_base --task logic      --arch mlp --d 128 --n 12 --depth-max 5 --steps 3000 --force latent "${REC[@]}"
run C_emit_base   --task recall     --arch mlp --d 128 --n 12 --depth-max 5 --p-trivial 0.0 --steps 3000 --force emit "${REC[@]}"
run C_latent_base --task recall     --arch mlp --d 128 --n 12 --depth-max 5 --p-trivial 0.0 --steps 3000 --force latent "${REC[@]}"

# ---- ablations (task A, same recipe, 2500 steps): which component is needed?
run A_abl_noprice  --task arithmetic --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 "${REC[@]}" --lam-price 0.0
run A_abl_nocommit --task arithmetic --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 "${REC[@]}" --lam-commit 0.0
run A_abl_nondeg   --task arithmetic --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 "${REC[@]}" --lam-nd 0.0
run A_abl_bare     --task arithmetic --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 "${REC[@]}" --lam-price 0.0 --lam-commit 0.0 --lam-nd 0.0
run A_abl_sig0     --task arithmetic --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 "${REC[@]}" --sigma 0.0
run A_abl_reset    --task arithmetic --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 --semantics reset "${REC[@]}"
run A_abl_redundant --task arithmetic --d 128 --n 12 --depth-max 4 --p-trivial 0.4 --steps 2500 "${REC[@]}"
run A_abl_nogate  --task arithmetic --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 "${REC[@]}" --q-conf 0.0

# ---- scaling: N = 4, 8, 12, 24, 48 (task A, mlp, d=96) ----------------------
run A_scale_n4  --task arithmetic --d 96 --n 4  --depth-max 4 --p-trivial 0.0 --steps 2000 "${REC[@]}"
run A_scale_n8  --task arithmetic --d 96 --n 8  --depth-max 4 --p-trivial 0.0 --steps 2000 "${REC[@]}"
run A_scale_n12 --task arithmetic --d 96 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2000 "${REC[@]}"
run A_scale_n24 --task arithmetic --d 96 --n 24 --depth-max 4 --p-trivial 0.0 --steps 2000 "${REC[@]}"
run A_scale_n48 --task arithmetic --d 96 --n 48 --depth-max 4 --p-trivial 0.0 --steps 2000 "${REC[@]}"

# ---- architecture generalization (task A + per-arch baselines; d=96) -------
run A_gru         --task arithmetic --arch gru         --d 96 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2000 "${REC[@]}"
run A_ssm         --task arithmetic --arch ssm         --d 96 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2000 "${REC[@]}"
run A_transformer --task arithmetic --arch transformer --d 96 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2000 "${REC[@]}"
run A_gru_emit    --task arithmetic --arch gru         --d 96 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2000 --force emit "${REC[@]}"
run A_gru_latent  --task arithmetic --arch gru         --d 96 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2000 --force latent "${REC[@]}"
run A_ssm_emit    --task arithmetic --arch ssm         --d 96 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2000 --force emit "${REC[@]}"
run A_ssm_latent  --task arithmetic --arch ssm         --d 96 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2000 --force latent "${REC[@]}"
run A_tf_emit     --task arithmetic --arch transformer --d 96 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2000 --force emit "${REC[@]}"
run A_tf_latent   --task arithmetic --arch transformer --d 96 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2000 --force latent "${REC[@]}"

# ---- cross-task check on two other architectures ----------------------------
run B_gru         --task logic --arch gru         --d 96 --n 12 --depth-max 5 --steps 2000 "${REC[@]}"
run B_transformer --task logic --arch transformer --d 96 --n 12 --depth-max 5 --steps 2000 "${REC[@]}"

# ---- PHASE 2: pressure regimes (experiments/PHASE2.md) ----------------------
# Regimes that SHOULD force the hybrid to externalize state. Same recipe as
# the phase-1 A ablations (2500 steps, d=128, N=12, depth<=4, p_trivial 0).
# NOTE: override flags stay AFTER "${REC[@]}" (argparse: last one wins).

# H1 — lossy channel: sigma 0.3 / 0.5 (phase-1 was 0.1), with
#      regime-matched re-trained baselines so c4 stays apples-to-apples.
run A_s03        --task arithmetic --arch mlp --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 "${REC[@]}" --sigma 0.3
run A_s03_emit   --task arithmetic --arch mlp --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 --force emit "${REC[@]}" --sigma 0.3
run A_s03_latent --task arithmetic --arch mlp --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 --force latent "${REC[@]}" --sigma 0.3
run A_s05        --task arithmetic --arch mlp --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 "${REC[@]}" --sigma 0.5
run A_s05_emit   --task arithmetic --arch mlp --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 --force emit "${REC[@]}" --sigma 0.5
run A_s05_latent --task arithmetic --arch mlp --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 --force latent "${REC[@]}" --sigma 0.5

# H2 — information over time: input arrives chunk-by-chunk (gradual reader);
#      no stage ever sees the full input.
run A_bneck        --task arithmetic --arch mlp --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 "${REC[@]}" --reader-mode gradual
run A_bneck_emit   --task arithmetic --arch mlp --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 --force emit "${REC[@]}" --reader-mode gradual
run A_bneck_latent --task arithmetic --arch mlp --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 --force latent "${REC[@]}" --reader-mode gradual

# H3 — learned controller vs fixed emission schedules (gate bypassed,
#      mode head not trained): 3 emits ~= hybrid's ~2, and 6 emits.
run A_sched4 --task arithmetic --arch mlp --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 "${REC[@]}" --force schedule --schedule-every 4
run A_sched2 --task arithmetic --arch mlp --d 128 --n 12 --depth-max 4 --p-trivial 0.0 --steps 2500 "${REC[@]}" --force schedule --schedule-every 2

echo "MATRIX COMPLETE"
