#!/bin/bash
#SBATCH --job-name=a20_cpu
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=210G
#SBATCH --time=01:45:00
#SBATCH --output=out/a20_cpu.out
#SBATCH --open-mode=append
# A20 CPU jobs (a20/A20_draft.md), CPU only, in the job container of a19/code/cpu.sh (numpy 1.26.3); 0 GCD-hours.
# From a20/code (the resources of each job are given on the sbatch line, A20_draft.md section 5):
#   sbatch cpu.sh 'python -u -B stpos20.py && python -u -B check20.py && python -u -B collect20.py --workers 22'
#   sbatch --cpus-per-task=16 --mem=120G --time=01:00:00 cpu.sh 'python -u -B fit20.py && python -u -B fit20.py --tests'
#   sbatch --mem=200G --time=01:30:00 cpu.sh 'python -u -B score20.py --tests'
#   after the filing only:  sbatch --mem=200G --time=01:00:00 cpu.sh 'python -u -B score20.py'
set -uo pipefail
unset PYTHONPATH
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
cd ${FOEHN_WORK}/global_model_forecasting_air_pollution/a20/code
echo "=== A20 cpu job ${SLURM_JOB_ID} start $(date) cpus=${SLURM_CPUS_PER_TASK} mem=${SLURM_MEM_PER_NODE:-?} cmd=$1"
sha256sum common20.py stpos20.py check20.py collect20.py fit20.py score20.py cpu.sh 2>/dev/null
singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate; export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src;
  export MPLCONFIGDIR=/tmp/mpl-\$USER; export OMP_NUM_THREADS=${A20_OMP:-1}; export OPENBLAS_NUM_THREADS=${A20_OMP:-1};
  export MKL_NUM_THREADS=${A20_OMP:-1}; export PYTHONDONTWRITEBYTECODE=1; $1"
rc=$?
echo "=== A20 cpu job ${SLURM_JOB_ID} end $(date) rc=$rc"
exit $rc
