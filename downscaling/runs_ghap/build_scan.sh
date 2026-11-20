#!/bin/bash
#SBATCH --job-name=ghap_scan
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-23
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/logs/scan_%A_%a.out
# PROTOCOL.md A2, pass 1 of the GHAP store: validity scan 2017-2021 + training-day harmonic sums, one latitude band per task
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
$PY -I $W/scripts/build_ghap.py scan --band ${SLURM_ARRAY_TASK_ID} ${GHAP_TEST:-}
