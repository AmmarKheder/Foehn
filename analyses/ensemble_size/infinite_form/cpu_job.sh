#!/bin/bash
#SBATCH --job-name=a14u
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# A14 class U, CPU only (partition small, no GPU): sbatch [--job-name N --time T --mem M] jobs/cpu_job.sh <script.py> [args]
# world venv (numpy, zarr), as analyses/reliability/scripts/run_small.sh; no bytecode written anywhere (-B).
set -uo pipefail
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=${FOEHN_ENVS}/.cache MPLCONFIGDIR=/tmp/mpl-$USER-${SLURM_JOB_ID:-0}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} PYTHONDONTWRITEBYTECODE=1
cd ${FOEHN_WORK}/global_model_forecasting_air_pollution/a14/classU/scripts
echo "start $(date) host $(hostname) cmd $*"
${FOEHN_ENVS}/world/bin/python -u -B "$@"
rc=$?
echo "end $(date) rc=$rc"
exit $rc
