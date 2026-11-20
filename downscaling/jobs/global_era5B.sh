#!/bin/bash
#SBATCH --job-name=cdffm_g_era5B
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --array=0-25%9
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/global_era5B_%A_%a.out
# ERA5 3-D FLEXGREEN fields from ARCO (PLAN.md 2.7 item 4): task 0 = full-month validation of 2022-12 against WB2,
# tasks 1..25 = months 2023-01 .. 2025-01 into data/processed/global/wb2/.
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=1
i=$SLURM_ARRAY_TASK_ID
if [ "$i" = 0 ]; then
  $PY -I $W/scripts/dl_era5_B_arco.py validate --threads 32
else
  y=$(( 2023 + (i - 1) / 12 )); m=$(( (i - 1) % 12 + 1 ))
  $PY -I $W/scripts/dl_era5_B_arco.py $y $m --threads 32
fi
