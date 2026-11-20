#!/bin/bash
# One srun task per GCD, INSIDE the container (started by points_cal_job.sh): the issue days of this rank, one process per
# day as realtime_1km/replay/code/km_rank.py does (children see one rank and device 0 of the visible set).
#   points_rank.sh <list file> <out dir> <raw dir | none> <tau> <knee | none> [pair]
# pair (proof of the copy): this rank gets its own MIOpen database (the ranks of a node otherwise share one, and the
# convolution algorithm that a process picks decides the last bits of the result); for each day the ORIGINAL
# realtime_1km/replay/code/infer_points.py is run first (outputs redirected to <out dir>/orig, nothing written under the replay
# directory), then the copy with tau = 1 (<out dir>/copy), in the same cache state.
set -uo pipefail
LIST=$1; OUT=$2; RAW=$3; TAU=$4; KNEE=$5; PAIR=${6:-}
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
source $W/env.sh
export HIP_VISIBLE_DEVICES=$SLURM_LOCALID PYTHONDONTWRITEBYTECODE=1
RANK=$SLURM_PROCID; WORLD=$SLURM_NTASKS
CK=$W/runs_ghap/pmG_final/ckpt/best.ckpt
mapfile -t ALL < <(grep -v '^[[:space:]]*$' "$LIST" | sort)
if [ "$PAIR" = pair ]; then
  export MIOPEN_USER_DB_PATH=$CDFFM_TMP/miopen_r$RANK
  export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
  mkdir -p "$MIOPEN_USER_DB_PATH"
fi
rc=0
for ((i = RANK; i < ${#ALL[@]}; i += WORLD)); do
  d=${ALL[$i]}
  t0=$(date +%s)
  if [ "$PAIR" = pair ]; then
    cd $W/realtime_1km/replay/code
    SLURM_PROCID=0 SLURM_NTASKS=1 SLURM_LOCALID=0 python -u infer_points.py --ckpt $CK --out "$OUT/orig/km" --raw "$OUT/orig/raw" "$d" \
        || { echo "[rank $RANK] $d original FAILED"; rc=1; }
    cd $W/runs_ghap/pmG_cal/replay_code
    SLURM_PROCID=0 SLURM_NTASKS=1 SLURM_LOCALID=0 python -u infer_points_cal.py --ckpt $CK --out "$OUT/copy/km" --raw "$OUT/copy/raw" \
        --tau 1.0 --knee none "$d" || { echo "[rank $RANK] $d copy FAILED"; rc=1; }
  else
    cd $W/runs_ghap/pmG_cal/replay_code
    SLURM_PROCID=0 SLURM_NTASKS=1 SLURM_LOCALID=0 python -u infer_points_cal.py --ckpt $CK --out "$OUT" --raw "$RAW" --tau "$TAU" \
        --knee "$KNEE" "$d" || { echo "[rank $RANK] $d FAILED"; rc=1; }
  fi
  echo "[rank $RANK] $d wall $(( $(date +%s) - t0 )) s (process start to end)"
done
exit $rc
