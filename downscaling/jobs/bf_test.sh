#!/bin/bash
#SBATCH --account=project_462001804
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=bf_test
#SBATCH --partition=small
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/bf_test_%j.out
# Backfill: reference rows from the sources (world venv) -> GlobalStore/GhapStore unit test (container) -> coverage json.
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/GLOBAL_PREP_STATUS.md
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=1 WORLD_DATA=/scratch/project_462001804/ammar/eccv/data/world
note() { printf '%s\n' "- $(date '+%F %H:%M') [backfill] job $SLURM_JOB_ID: $1" >> "$S"; }
trap 'note "store backfill TEST FAILED, logs/bf_test_$SLURM_JOB_ID.out"' ERR
cd $W/scripts
$PY -I backfill_2025_2026.py coverage > /dev/null
$PY -I backfill_2025_2026.py refs
source $W/env.sh
cdffm_run python -u $W/scripts/test_store_backfill.py
$PY -I backfill_2025_2026.py coverage > /dev/null
L=$($PY -I -c "import json; print(json.load(open('$W/data/processed/global/COVERAGE_2025_2026.json'))['last_complete_day_all_inputs'])")
note "READY for A3: GlobalStore/GhapStore serve raw_inputs(d) for 2025-01-01 .. $L (all inputs; data/processed/global/COVERAGE_2025_2026.json); unit test PASSED (logs/test_store_backfill.json: $($PY -I -c "import json; r=json.load(open('$W/logs/test_store_backfill.json')); print(len(r['days']))") reference days bit-identical to the sources, KeyError beyond, 2017-2024 unchanged). Days after $L belong to the real-time chain (realtime_1km/prep.py)."
