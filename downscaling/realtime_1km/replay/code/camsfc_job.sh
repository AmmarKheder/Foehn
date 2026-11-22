#!/bin/bash
#SBATCH --job-name=rp_camsfc
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay/logs/%x_%j.out
# WP1a: the ten leads of the CAMS operational forecast for the cycles that only have 24 h
# and 120 h, into replay/camsfc/<d>/ (one request after the other; the chain's own request and client). CPU, internet.
#   sbatch camsfc_job.sh            (check on 2026-10-12, then every target day with < 10 stored leads)
set -uo pipefail
R=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=1
PY=/projappl/project_462001804/envs/world/bin/python
cd $R/code
echo "start $(date -u +%FT%TZ) camsfc (job $SLURM_JOB_ID)"
$PY -u camsfc_fetch.py --check 2026-10-12 --missing 2>&1 | grep --line-buffered -v -i -E "key|token|authorization"
rc=${PIPESTATUS[0]}
echo "end $(date -u +%FT%TZ) rc=$rc"
exit $rc
