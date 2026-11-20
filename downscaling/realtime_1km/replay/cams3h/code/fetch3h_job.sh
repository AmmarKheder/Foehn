#!/bin/bash
#SBATCH --job-name=c3h_fetch
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay/cams3h/logs/fetch3h_%j.out
# cams3h step 1 on a CPU node: the 3-hourly CAMS forecast steps of the 53 cycles (see fetch3h.py). Evaluation data only.
#   sbatch fetch3h_job.sh [--deadline 2026-11-20T04:15] [--groups ...]
unset PYTHONPATH; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cd /tmp
exec /projappl/project_462001804/envs/world/bin/python -I -B -u \
  /scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay/cams3h/code/fetch3h.py "$@"
