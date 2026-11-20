#!/bin/bash
#SBATCH --account=project_462001804
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=bf_era5
#SBATCH --partition=small
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-43%10
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/bf_era5_%A_%a.out
# Backfill ERA5 from ARCO, one month per task: 0-10 daily 30-ch 2026-01..11 (era5_global_daily/2026.zarr), 11-21 extra
# 7-ch 2026-01..11 (era5_extra_daily/2026.zarr), 22-43 era5_B 3-D FLEXGREEN fields 2025-02 .. 2026-11
# (data/processed/global/wb2/). Days bounded by backfill/arco_last_day.txt; the last month is partial.
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/GLOBAL_PREP_STATUS.md
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=1 WORLD_DATA=/scratch/project_462001804/ammar/eccv/data/world
note() { printf '%s\n' "- $(date '+%F %H:%M') [backfill] job $SLURM_JOB_ID: $1" >> "$S"; }
i=$SLURM_ARRAY_TASK_ID
LAST=$(cat $W/data/processed/global/backfill/arco_last_day.txt)
cd $W/scripts
if [ $i -le 10 ]; then
  m=$((i + 1)); $PY -I dl_era5_daily_arco.py daily 2026 --months $m --last-day $LAST --threads 16
elif [ $i -le 21 ]; then
  m=$((i - 10)); $PY -I dl_era5_daily_arco.py extra 2026 --months $m --last-day $LAST --threads 16
else
  j=$((i - 22)); if [ $j -lt 11 ]; then y=2025; m=$((j + 2)); else y=2026; m=$((j - 10)); fi
  first=$(date -d "$y-$m-01" +%F)
  if [ "$first" \> "$LAST" ]; then echo "era5_B $y-$m: after $LAST, nothing to do"; exit 0; fi
  lim=0
  if [ "$(date -d "$first +1 month -1 day" +%F)" \> "$LAST" ]; then lim=$(( ($(date -d $LAST +%s) - $(date -d $first +%s)) / 86400 + 1 )); fi
  $PY -I dl_era5_B_arco.py $y $m --threads 16 --limit $lim
fi
