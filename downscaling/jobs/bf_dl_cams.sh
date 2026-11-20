#!/bin/bash
#SBATCH --account=project_462001804
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=bf_dl_cams
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=2-12:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/bf_dl_cams_%j.out
# Backfill: CAMS 00 UTC forecasts 2026 (pm2p5, pm10, 5 lead days) month by month from the ADS into
# eccv/data/world/cams_forecast_daily/2026.zarr (partial current month); resumable (done_days attr).
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/GLOBAL_PREP_STATUS.md
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=1 WORLD_DATA=/scratch/project_462001804/ammar/eccv/data/world
note() { printf '%s\n' "- $(date '+%F %H:%M') [backfill] job $SLURM_JOB_ID: $1" >> "$S"; }
trap 'note "CAMS 2026 download FAILED (resubmit: resumes by month), logs/bf_dl_cams_$SLURM_JOB_ID.out"' ERR
cd $W/scripts
$PY -I backfill_2025_2026.py cams-download --year 2026
L=$($PY -I -c "import zarr; print(zarr.open_array('$WORLD_DATA/cams_forecast_daily/2026.zarr', mode='r').attrs.get('last_day'))")
note "CAMS 2026 downloaded from the ADS into cams_forecast_daily/2026.zarr, last day $L ($(du -sh $WORLD_DATA/cams_forecast_daily/2026.zarr | cut -f1))"
