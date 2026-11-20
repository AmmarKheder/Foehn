#!/bin/bash
#SBATCH --job-name=ghap_finish
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/logs/finish_%j.out
# PROTOCOL.md A2: GHAP store, step 'finish' of scripts/build_ghap.py
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/GLOBAL_PREP_STATUS.md
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
note() { printf '%s\n' "- $(date '+%F %H:%M') [ghap] job $SLURM_JOB_ID: $1" >> "$S"; }
trap 'note "store step finish FAILED (runs_ghap/logs/finish_$SLURM_JOB_ID.out)"' ERR
$PY -I $W/scripts/build_ghap.py finish
note "store step finish done (runs_ghap/logs/finish_$SLURM_JOB_ID.out)"
