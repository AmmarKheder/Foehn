#!/bin/bash
#SBATCH --job-name=o1k_inf
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/logs/o1k_inf_%j.out
# infer.py + quicklook.py only, on an already prepared day (realtime_1km/work/<day>/ with chan_L1..5.npz): for several test
# checkpoints of the same day in parallel without re-running prep.py (whose IFS fetch dir is shared and removed at its end).
#   sbatch rt1km/infer_only_job.sh <yyyy-mm-dd> <checkpoint.ckpt> <tag>
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
O=$W/realtime_1km
S=$W/logs/RT1KM_STATUS.md
DATE=$1; CKPT=$2; TAG=$3
note() { printf '%s\n' "- $(date '+%F %H:%M') [rt1km] job $SLURM_JOB_ID (infer only): $1" >> "$S"; }
source $W/env.sh
cd $O
[ -f "$CKPT" ] || { note "checkpoint $CKPT not found"; exit 2; }
for L in 1 2 3 4 5; do [ -f work/$DATE/chan_L$L.npz ] || { note "work/$DATE/chan_L$L.npz missing: run rt1km_job.sh"; exit 2; }; done
echo "start $(date -u +%FT%TZ) infer-only $DATE ckpt=$CKPT tag=$TAG"
T0=$(date +%s)
srun --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B $CDFFM_BIND $CDFFM_SIF \
  bash -c "source $CDFFM_ENVSH; cd $O; python -u infer.py $DATE --ckpt $CKPT --tag $TAG" || { note "infer $DATE $TAG FAILED"; exit 1; }
srun -n 1 --ntasks-per-node=1 --cpus-per-task=7 singularity exec -B $CDFFM_BIND $CDFFM_SIF \
  bash -c "source $CDFFM_ENVSH; cd $O; python -u quicklook.py $DATE --tag $TAG" || { note "quicklook $DATE $TAG FAILED"; exit 1; }
note "$DATE ($TAG) done in $(( $(date +%s) - T0 )) s -> realtime_1km/out/$DATE/$TAG/ (local, not published)"
