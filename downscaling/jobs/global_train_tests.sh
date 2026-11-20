#!/bin/bash
#SBATCH --job-name=gt_tests
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/global_train_tests_%j.out
# CPU checks of HCF on the global store (global-train; 0 GPU-h): scripts/test_global_hcf.py [checks...]
#   sbatch jobs/global_train_tests.sh [shapes resfree dateline injh]
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/GLOBAL_STATUS.md
source $W/env.sh
export PYTHONDONTWRITEBYTECODE=1
cd $W
CHECKS=${*:-shapes resfree dateline injh}
cdffm_run python -u scripts/test_global_hcf.py $CHECKS
rc=$?
printf '%s\n' "- $(date '+%F %H:%M') [global-train] job $SLURM_JOB_ID: test_global_hcf ($CHECKS) rc=$rc: $(grep 'GLOBAL HCF SUMMARY' $W/logs/global_train_tests_$SLURM_JOB_ID.out | tail -1)" >> $S
exit $rc
