#!/bin/bash
#SBATCH --job-name=a16_score22
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/score22.out
#SBATCH --open-mode=append
# usage (from a16/score22): sbatch [--time T] [--output F] cpu22.sh "<python command run in a16/score22>"
#   CPU only, the job container of a15/jobs/cpu.sh (lumi-pytorch-rocm-6.1.3, torchenv, numpy 1.26.3), 0 GCD-hours.
set -uo pipefail
unset PYTHONPATH
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
cd ${FOEHN_WORK}/global_model_forecasting_air_pollution/a16/score22
echo "=== A16 score22 cpu job ${SLURM_JOB_ID} start $(date) cmd=$1"
singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate; export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src;
  export MPLCONFIGDIR=/tmp/mpl-\$USER; export OMP_NUM_THREADS=4; export PYTHONDONTWRITEBYTECODE=1; $1"
rc=$?
echo "=== A16 score22 cpu job ${SLURM_JOB_ID} end $(date) rc=$rc"
exit $rc
