#!/bin/bash
#SBATCH --account=project_462001804
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=bf_dl_gfas
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/bf_dl_gfas_%j.out
# Backfill: GFAS monthly files 2025-01 .. this month (tpmfire, frpfire, injh) from the ADS into data/raw/gfas_injh/
# (<y>-<m>.nc, partial month <y>-<m>_to<dd>.nc); resumable.
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/GLOBAL_PREP_STATUS.md
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=1 WORLD_DATA=/scratch/project_462001804/ammar/eccv/data/world
note() { printf '%s\n' "- $(date '+%F %H:%M') [backfill] job $SLURM_JOB_ID: $1" >> "$S"; }
trap 'note "GFAS download FAILED (resubmit: resumes by month), logs/bf_dl_gfas_$SLURM_JOB_ID.out"' ERR
cd $W/scripts
$PY -I backfill_2025_2026.py gfas-download --workers 3
note "GFAS 2025-01 .. $(ls $W/data/raw/gfas_injh/2026-*nc | tail -1 | xargs -n1 basename) downloaded from the ADS (3 variables) into data/raw/gfas_injh/"
