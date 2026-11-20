#!/bin/bash
#SBATCH --job-name=gp_bcsd
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/global_bcsd_%j.out
# B2 BCSD baseline (PROTOCOL.md 3), fitted on the 1,270 split-A training days only -> lghap_clim/bcsd_A.npz, + check.
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/GLOBAL_PREP_STATUS.md
note() { printf '%s\n' "- $(date '+%F %H:%M') [global-prep] job $SLURM_JOB_ID: $1" >> "$S"; }
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=8
cd $W/scripts
rc=0
$PY -I fit_bcsd.py fit || rc=1
[ $rc = 0 ] && { $PY -I fit_bcsd.py check || rc=1; }
if [ $rc = 0 ]; then note "B2 BCSD fitted on training days + check PASSED (lghap_clim/bcsd_A.npz, bcsd_A.json, bcsd_A_check.json)"
else note "B2 BCSD FAILED, see logs/global_bcsd_$SLURM_JOB_ID.out"; fi
exit $rc
