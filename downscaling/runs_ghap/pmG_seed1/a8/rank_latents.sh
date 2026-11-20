#!/bin/bash
# A8: one srun task (inside the container) = the latents of this rank's days, exactly the command of runs_ghap/p1r_G.sh
# step 2 (same day list per rank, same seeds, same --n-per-radius), for one split or for both splits at the same time.
#   bash rank_latents.sh RUN RUNS_ROOT STEP DIR NPR MODE [NDAY]
#   MODE = both | v3odd | v3even ; NDAY > 0 keeps only the first NDAY days of the rank's list (smoke only)
set -uo pipefail
RUN=$1; RG=$2; STEP=$3; DIR=$4; NPR=$5; MODE=$6; NDAY=${7:-0}
source "$CDFFM_ENVSH"
cd "$CDFFM_W"
export PYTHONDONTWRITEBYTECODE=1
days() {   # $1 = parity of k (1 -> v3odd = fit days, 0 -> v3even = check days), as p1r_G.sh
  python3 -c "
import os
r = int(os.environ['SLURM_PROCID']); par = $1; nday = $NDAY
ks = [3 * k for k in range(120) if k % 2 == par and (3 * k) % 15]
d = ks[r::8]
if nday:
    d = d[:nday]
print(','.join(str(x) for x in d))"
}
run_split() {
  local SP=$1 PAR DAYS
  PAR=$([ "$SP" = v3odd ] && echo 1 || echo 0)
  DAYS=$(days $PAR)
  python -u -m cdffm_aq_x.copula_x latents --run "$RUN" --runs-root "$RG" --ckpt "$STEP" --dir "$DIR" \
         --splits "${SP}_r${SLURM_PROCID}" --day-idx "$DAYS" --n-per-radius "$NPR"
}
case "$MODE" in
  both)
    run_split v3odd & p1=$!
    run_split v3even & p2=$!
    wait $p1; r1=$?
    wait $p2; r2=$?
    [ $r1 = 0 ] && [ $r2 = 0 ] || { echo "[rank_latents r$SLURM_PROCID] v3odd rc=$r1 v3even rc=$r2"; exit 1; }
    ;;
  v3odd|v3even)
    run_split "$MODE"
    ;;
  *) echo "rank_latents: unknown mode $MODE"; exit 2 ;;
esac
