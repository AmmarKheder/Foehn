#!/bin/bash
#SBATCH --account=project_462001804
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=bf_era5_fix
#SBATCH --partition=small
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/bf_era5_fix_%A_%a.out
# Backfill: repair of the ERA5 2026 stores after the create race of array 23422009 (task 0 = daily 30-ch, 1 = extra
# 7-ch): rebuild the attrs from the chunks on disk, then fill the missing months SEQUENTIALLY (one process per store).
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/GLOBAL_PREP_STATUS.md
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=1 WORLD_DATA=/scratch/project_462001804/ammar/eccv/data/world
note() { printf '%s\n' "- $(date '+%F %H:%M') [backfill] job $SLURM_JOB_ID: $1" >> "$S"; }
kind=$([ "$SLURM_ARRAY_TASK_ID" = 0 ] && echo daily || echo extra)
trap 'note "ERA5 2026 $kind repair/fill FAILED, logs/bf_era5_fix_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"' ERR
LAST=$(cat $W/data/processed/global/backfill/arco_last_day.txt)
cd $W/scripts
$PY -I era5_2026_repair.py $kind
$PY -I dl_era5_daily_arco.py $kind 2026 --months 1 2 3 4 5 6 7 8 9 10 11 --last-day $LAST --threads 16
$PY -I era5_2026_repair.py $kind
note "ERA5 2026 $kind store repaired (attrs rebuilt from the chunks after the create race of array 23422009) and filled sequentially to $LAST"
