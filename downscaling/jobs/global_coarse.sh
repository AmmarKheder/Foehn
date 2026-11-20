#!/bin/bash
#SBATCH --job-name=cdffm_g_coarse
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/global_coarse_%j.out
# Global coarse inputs (PLAN.md 2.7 items 1-3): met/fires 2023-2024, CAMS 0.25 deg 2017-2024, non-regression, days.npz.
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/BUILD_STATUS.md
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=1
note() { printf '%s\n' "- $(date '+%F %H:%M') [global-build] job $SLURM_JOB_ID: $1" >> "$S"; }
trap 'note "coarse FAILED, see logs/global_coarse_$SLURM_JOB_ID.out"' ERR
for k in met fires cams; do echo "=== $k $(date +%T)"; $PY -I $W/scripts/build_global_coarse.py $k --procs 32; done
note "coarse met_2023_2024, fires_2023_2024, cams_pm25_025 written"
echo "=== nonreg $(date +%T)"; $PY -I $W/scripts/build_global_coarse.py nonreg
note "non-regression PASSED (met/fires rows 2022-12-31, 2017-01-01, 2020-06-15 bit-identical to Foehn; CAMS = smoke store)"
$PY -I $W/scripts/build_global_coarse.py days
$PY -I $W/scripts/build_global_coarse.py check
note "coarse DONE (days.npz, CAMS checks)"
echo "=== done $(date +%T)"
