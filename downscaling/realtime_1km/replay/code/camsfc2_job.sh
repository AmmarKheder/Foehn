#!/bin/bash
#SBATCH --job-name=rp_camsfc2
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay/logs/%x_%j.out
# WP1a, decision A of the coordinator, second version: ONE request for all the cycles that lack the ten leads of the CAMS
# forecast (camsfc_fetch2.py; a request already queued, running or done is re-attached, never sent twice). CPU, internet.
#   sbatch camsfc2_job.sh [--no-check]
set -uo pipefail
R=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=1
PY=/projappl/project_462001804/envs/world/bin/python
cd $R/code
echo "start $(date -u +%FT%TZ) camsfc2 $* (job $SLURM_JOB_ID)"
$PY -u camsfc_fetch2.py "$@" 2>&1 | grep --line-buffered -v -i -E "api_key|token|authorization"
rc=${PIPESTATUS[0]}
echo "end $(date -u +%FT%TZ) rc=$rc"
exit $rc
