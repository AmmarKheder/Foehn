#!/bin/bash
#SBATCH --job-name=a18_cpu
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=190G
#SBATCH --time=01:30:00
#SBATCH --output=out/a18_cpu.out
#SBATCH --open-mode=append
# A18, CPU only (no GPU), the job container and numpy (1.26.3) of a15/jobs/cpu.sh. Runs "$1" in a18/code.
#   sbatch cpu.sh 'python -B -u collect18.py --part members --npix 20000 && python -B -u score18.py --plumb; python -B -u score18.py --tests; rm -rf ../work'
#   the A18 score (section 9 of A18_draft.md):
#   sbatch cpu.sh 'python -B -u collect18.py --part all && python -B -u score18.py && rm -rf ../work'
set -uo pipefail
unset PYTHONPATH
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
cd ${FOEHN_WORK}/global_model_forecasting_air_pollution/a18/code
echo "=== A18 cpu job ${SLURM_JOB_ID} start $(date) cmd=$1"
sha256sum collect18.py score18.py
singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate;
  export MPLCONFIGDIR=/tmp/mpl-\$USER; export OMP_NUM_THREADS=4; export PYTHONDONTWRITEBYTECODE=1; $1"
rc=$?
echo "=== A18 cpu job ${SLURM_JOB_ID} end $(date) rc=$rc"
exit $rc
