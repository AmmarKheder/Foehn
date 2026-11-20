#!/bin/bash
#SBATCH --job-name=ghap_bcsd
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/logs/bcsd_%j.out
# PROTOCOL.md A2(d): B2 BCSD fitted on the GHAP training days 2017-2020 (runs_ghap/fit_bcsd_ghap.py fit + check)
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
note() { printf '%s\n' "- $(date '+%F %H:%M') [ghap] job $SLURM_JOB_ID: $1" >> $W/logs/GLOBAL_PREP_STATUS.md; }
trap 'note "GHAP BCSD FAILED (runs_ghap/logs/bcsd_$SLURM_JOB_ID.out)"' ERR
$PY -I $W/runs_ghap/fit_bcsd_ghap.py fit
$PY -I $W/runs_ghap/fit_bcsd_ghap.py check
note "GHAP BCSD fitted on 2017-2020 + checked -> data/processed/ghap/ghap_clim/bcsd_G.npz"
