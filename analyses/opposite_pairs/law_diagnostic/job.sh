#!/bin/bash
#SBATCH --job-name=a17_diag
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=190G
#SBATCH --time=01:30:00
# usage (from anywhere): sbatch --output=<log> a17/diag/job.sh "<python command run in a17/diag>"   (CPU only; no GPU)
# The container of P/a15/jobs/cpu.sh (numpy 1.26.3). Writes only under P/a17/diag.
set -uo pipefail
unset PYTHONPATH
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
cd ${FOEHN_WORK}/global_model_forecasting_air_pollution/a17/diag
echo "=== A17 diag job ${SLURM_JOB_ID} start $(date) cmd=$1"
singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate; export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src;
  export MPLCONFIGDIR=/tmp/mpl-\$USER; export OMP_NUM_THREADS=4; export PYTHONDONTWRITEBYTECODE=1; $1"
rc=$?
echo "=== A17 diag job ${SLURM_JOB_ID} end $(date) rc=$rc"
exit $rc
