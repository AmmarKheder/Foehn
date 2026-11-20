#!/bin/bash
#SBATCH --job-name=ghap_points
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-23
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/logs/points_%A_%a.out
# PROTOCOL.md A2, pass 2 of the GHAP store: training points 2017-2020, 2021 hold-out full fields, cell means; one band per task
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=4
$PY -I $W/scripts/build_ghap.py points --band ${SLURM_ARRAY_TASK_ID} ${GHAP_TEST:-}
