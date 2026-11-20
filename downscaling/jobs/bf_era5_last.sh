#!/bin/bash
#SBATCH --account=project_462001804
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=bf_era5_last
#SBATCH --partition=small
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/bf_era5_last_%j.out
# Backfill: last finite ERA5 day in ARCO -> backfill/arco_last_day.txt; ARCO daily builder validated on 3 days of 2025.
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/GLOBAL_PREP_STATUS.md
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=1 WORLD_DATA=/scratch/project_462001804/ammar/eccv/data/world
note() { printf '%s\n' "- $(date '+%F %H:%M') [backfill] job $SLURM_JOB_ID: $1" >> "$S"; }
trap 'note "ERA5 ARCO last-day/validation FAILED, logs/bf_era5_last_$SLURM_JOB_ID.out"' ERR
cd $W/scripts
$PY -I dl_era5_daily_arco.py last
$PY -I dl_era5_daily_arco.py validate --threads 16
note "ARCO-ERA5 last finite day $(cat $W/data/processed/global/backfill/arco_last_day.txt); daily/extra builders validated vs the 2025 archives (backfill/era5_arco_validation.json)"
