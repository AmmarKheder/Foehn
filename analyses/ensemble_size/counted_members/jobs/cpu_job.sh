#!/bin/bash
#SBATCH --job-name=a14E_cpu
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# copy of a14/jobs/cpu_job.sh for A14 class E: usage (from a14/classE): sbatch [--job-name N --time T] jobs/cpu_job.sh "<command>"
# CPU only (partition small, no GPU). Changes from the original: output folder, working folder a14/classE, more cores and
# memory, and the python of the world venv instead of the training container (the env in which a13/warn16/score16.py and
# extrap16.py ran, jobs/score_job.sh of a13/warn16: numpy 2.4, scipy, zarr), so that score16 is imported in its own env.
set -uo pipefail
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH=${FOEHN_ENVS}/world/bin:$PATH
cd ${FOEHN_WORK}/global_model_forecasting_air_pollution/a14/classE
echo "start $(date) host $(hostname) cmd=$1"
bash -c "$1"
rc=$?
echo "end $(date) rc=$rc"
exit $rc
