#!/bin/bash
#SBATCH --job-name=a14_r2cpu
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=../logs/%x_%j.out
# usage: sbatch jobs/cpu_job.sh "<python command run in a14/r2>"   (A14 run R2, CPU only, the training container's env)
set -uo pipefail
unset PYTHONPATH
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
cd ${FOEHN_WORK}/global_model_forecasting_air_pollution/a14/r2
echo "start $(date) cmd=$1"
singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate; export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src;
  export MPLCONFIGDIR=/tmp/mpl-\$USER; export OMP_NUM_THREADS=4; export PYTHONDONTWRITEBYTECODE=1; $1"
rc=$?
echo "end $(date) rc=$rc"
exit $rc
