#!/bin/bash
#SBATCH --account=project_462001804
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=bf_build
#SBATCH --partition=small
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/bf_build_%j.out
# Backfill: coarse/{met,fires_injh,cams_pm25_025}_2025_2026.npy from the archives (rows available, NaN beyond) + coverage.
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/GLOBAL_PREP_STATUS.md
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=1 WORLD_DATA=/scratch/project_462001804/ammar/eccv/data/world
note() { printf '%s\n' "- $(date '+%F %H:%M') [backfill] job $SLURM_JOB_ID: $1" >> "$S"; }
trap 'note "coarse 2025-2026 build FAILED, logs/bf_build_$SLURM_JOB_ID.out"' ERR
cd $W/scripts
for k in cams fires met; do P=32; [ $k = fires ] && P=8; echo "=== $k $(date +%T) procs $P"; $PY -I backfill_2025_2026.py $k --procs $P; done   # fires: one GFAS month (~2.4 GB) cached per process -> 8 procs (32 went OOM at 96G, job 23436117)
$PY -I backfill_2025_2026.py coverage > /dev/null
note "coarse 2025-2026 stores built: $($PY -I -c "import json; c=json.load(open('$W/data/processed/global/COVERAGE_2025_2026.json'))['inputs']; print(', '.join(f\"{k} to {c[k]['last_complete_day']}\" for k in ('cams','met','fires')))") (coarse/{cams_pm25_025,met,fires_injh}_2025_2026.npy, row = d - 2025-01-01)"
