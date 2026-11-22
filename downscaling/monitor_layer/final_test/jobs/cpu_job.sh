#!/bin/bash
#SBATCH --job-name=ftM_cpu
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cpu_%j.out
# CPU steps of the final test (draft Addendum 8 of A11), 0 GCD-hours, the record scorer's Python (world env, numpy only):
#   sbatch jobs/cpu_job.sh 'score_final.py --tests'      known-answer tests of the frozen scorer (no observation file)
#   sbatch jobs/cpu_job.sh 'test_final.py cpu'           leak / causality / firewall tests of the inputs
#   sbatch jobs/cpu_job.sh 'score_final.py'              THE DECISION (refuses unless FILED holds the scorer's SHA-256)
set -uo pipefail
FT=${FOEHN_WORK}/global_model_forecasting_air_pollution/a11/stageM/final_test
PY=${FOEHN_ENVS}/world/bin/python
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 FT_TMP=/tmp
unset PYTHONPATH
cd $FT
echo "[cpu] start $(date -u +%FT%TZ) cmd=$1"
$PY -I -B -W ignore $1
rc=$?
echo "[cpu] end $(date -u +%FT%TZ) rc=$rc"
exit $rc
