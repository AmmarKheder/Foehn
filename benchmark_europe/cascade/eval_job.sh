#!/bin/bash
#SBATCH --job-name=cascade_eval
#SBATCH --account=project_462001140
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/cascade/logs/%x_%j.out
# usage: sbatch eval_job.sh <run dir> <out dir> [evaluate.py args]   (paths relative to experiments/cascade)
set -uo pipefail
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
RUN=$1; OUT=$2; shift 2
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source /projappl/project_462001140/envs/torchenv/bin/activate;
  export PYTHONPATH=/scratch/project_462001140/ammar/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_PROCID"
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain
echo "start $(date) run=$RUN out=$OUT args=$*"
srun --cpu-bind=$CPU_BIND singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV;
  python -u evaluate.py --cascade ../cascade/$RUN --out ../cascade/$OUT --weather hres $*"
singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python evaluate.py --merge ../cascade/$OUT"
echo "end $(date)"
