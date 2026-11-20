#!/bin/bash
#SBATCH --job-name=ghap_test
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/logs/test_store_%j.out
# PROTOCOL.md A2: unit test of GhapStore + LGHAP store non-regression (runs_ghap/test_store_ghap.py)
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
source $W/env.sh
export PYTHONDONTWRITEBYTECODE=1
singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "source $CDFFM_ENVSH; cd $CDFFM_W; python -u runs_ghap/test_store_ghap.py"
rc=$?
printf '%s\n' "- $(date '+%F %H:%M') [ghap] job $SLURM_JOB_ID GhapStore unit test rc=$rc (runs_ghap/logs/test_store_ghap.json)" >> $W/logs/GLOBAL_PREP_STATUS.md
rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
exit $rc
