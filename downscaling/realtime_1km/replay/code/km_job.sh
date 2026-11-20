#!/bin/bash
#SBATCH --job-name=rp_km
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay/logs/%x_%j.out
# WP1a steps B1(pm) + B2 + C: one day stream per GCD (km_rank.py). Environment = realtime_1km/rt1km_job.sh (env.sh container).
#   sbatch km_job.sh <file with one issue day per line> [--keep-fg] [--overwrite]
# Each task is confined to its own GCD with HIP_VISIBLE_DEVICES=$SLURM_LOCALID (the five FLEXGREEN start days of a day
# then share that GCD instead of five GCDs: same code, same seeds, same result).
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1   # no .pyc next to the imported production modules
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
R=$W/realtime_1km/replay
CKPT=$W/runs_ghap/pmG_final/ckpt/best.ckpt
LIST=$1; shift
source $W/env.sh
BIND=""; [ "${SLURM_NTASKS:-1}" = 8 ] && [ "${SLURM_JOB_PARTITION:-}" = standard-g ] && BIND="--cpu-bind=$CDFFM_CPU_BIND_8"
echo "start $(date -u +%FT%TZ) km replay list=$LIST ($(wc -l < $LIST) days) args=$* (job $SLURM_JOB_ID, $SLURM_NTASKS tasks, bind '${BIND:0:20}')"
T0=$(date +%s)
srun $BIND singularity exec -B $CDFFM_BIND $CDFFM_SIF \
  bash -c "source $CDFFM_ENVSH; export HIP_VISIBLE_DEVICES=\$SLURM_LOCALID; cd $R/code; python -u km_rank.py --ckpt $CKPT --list $LIST $*"
rc=$?
echo "end $(date -u +%FT%TZ) rc=$rc wall $(( $(date +%s) - T0 )) s; km files: $(ls $R/km/*.npz 2>/dev/null | wc -l)"
exit $rc
