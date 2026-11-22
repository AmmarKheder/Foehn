#!/bin/bash
#SBATCH --job-name=a19_cpu
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=out/a19_cpu.out
#SBATCH --open-mode=append
# A19 CPU jobs (a19/A19_draft.md), CPU only, in the job container of a16/jobs/cpu.sh (numpy 1.26.3); 0 GCD-hours. From a19/code:
#   sbatch --time=01:00:00 cpu.sh 'python -u -B check19.py'      (checks 6.1, before the smoke)
#   sbatch cpu.sh 'python -u -B score19.py --tests'              (identity checks and null / known-answer tests, stored data only)
#   sbatch cpu.sh 'python -u -B score19.py'                      (after the main run: the decision of section 7)
set -uo pipefail
unset PYTHONPATH
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
cd ${FOEHN_WORK}/global_model_forecasting_air_pollution/a19/code
echo "=== A19 cpu job ${SLURM_JOB_ID} start $(date) cmd=$1"
singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate; export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src;
  export MPLCONFIGDIR=/tmp/mpl-\$USER; export OMP_NUM_THREADS=8; export PYTHONDONTWRITEBYTECODE=1; $1"
rc=$?
echo "=== A19 cpu job ${SLURM_JOB_ID} end $(date) rc=$rc"
exit $rc
