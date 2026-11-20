#!/bin/bash
#SBATCH --job-name=gp_cellmeans
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/global_cellmeans_%j.out
# hcf z cell-mean table of the global store (training + validation days only), then the direct-recompute check.
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/GLOBAL_PREP_STATUS.md
note() { printf '%s\n' "- $(date '+%F %H:%M') [global-prep] job $SLURM_JOB_ID: $1" >> "$S"; }
source $W/env.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
cd $W
rc=0
cdffm_run python -u scripts/build_global_cellmeans.py build --procs 32 || rc=1
if [ $rc = 0 ]; then
  cdffm_run python -u scripts/build_global_cellmeans.py check || rc=1
fi
if [ $rc = 0 ]; then note "cell-mean table built + check PASSED (lghap_clim/cellmean_z_A.npz, _check.json)"
else note "cell-mean table FAILED, see logs/global_cellmeans_$SLURM_JOB_ID.out"; fi
exit $rc
