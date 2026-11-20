#!/bin/bash
#SBATCH --job-name=rt12
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime/logs/%x_%j.out
# RT 1.2 fine-tuning on one node (8 GCDs, DDP), the environment of cams_job.sh. usage: sbatch finetune_job.sh "<finetune.py args>"
set -uo pipefail
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source /projappl/project_462001804/envs/torchenv/bin/activate;
  export PYTHONPATH=/scratch/project_462001804/ammar/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_PROCID"
cd /scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime
echo "start $(date -u +%FT%TZ) finetune.py $1"
srun --cpu-bind=$CPU_BIND singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python -u finetune.py $1"
rc=$?; echo "end $(date -u +%FT%TZ) rc=$rc"; exit $rc
