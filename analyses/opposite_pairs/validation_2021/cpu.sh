#!/bin/bash
#SBATCH --job-name=a17_cpu
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=out/a17_cpu.out
#SBATCH --open-mode=append
# A17 CPU jobs (a17/A17_draft.md), CPU only, in the job container (numpy 1.26.3); 0 GCD-hours. Usage from a17/code:
#   sbatch cpu.sh 'python -u check17.py; python -u score17.py --tests'   (checks 6.1 and the scorer's null / known-answer tests)
#   sbatch --time=00:40:00 cpu.sh 'python -u score17.py'                 (after the main run: the decision of section 7)
set -uo pipefail
unset PYTHONPATH
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
cd ${FOEHN_WORK}/global_model_forecasting_air_pollution/a17/code
echo "=== A17 cpu job ${SLURM_JOB_ID} start $(date) cmd=$1"
singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate; export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src;
  export MPLCONFIGDIR=/tmp/mpl-\$USER; export OMP_NUM_THREADS=8; export PYTHONDONTWRITEBYTECODE=1; $1"
rc=$?
echo "=== A17 cpu job ${SLURM_JOB_ID} end $(date) rc=$rc"
exit $rc
