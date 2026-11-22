#!/bin/bash
#SBATCH --job-name=a15_cpu
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/a15.out
#SBATCH --open-mode=append
# usage: sbatch [--time T] jobs/cpu.sh "<python command run in a15>"   (A15, CPU only, the job container's env)
#   sbatch jobs/cpu.sh 'python -u check_cpu.py && python -u prep.py'      (limit 1:00)
#   sbatch --time=00:30:00 jobs/cpu.sh 'python -u calib.py'               (after run C)
#   sbatch --time=01:30:00 jobs/cpu.sh 'python -u score.py'               (after run M)
set -uo pipefail
unset PYTHONPATH
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
cd ${FOEHN_WORK}/global_model_forecasting_air_pollution/a15
echo "=== A15 cpu job ${SLURM_JOB_ID} start $(date) cmd=$1"
singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate; export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src;
  export MPLCONFIGDIR=/tmp/mpl-\$USER; export OMP_NUM_THREADS=4; export PYTHONDONTWRITEBYTECODE=1; $1"
rc=$?
echo "=== A15 cpu job ${SLURM_JOB_ID} end $(date) rc=$rc"
exit $rc
