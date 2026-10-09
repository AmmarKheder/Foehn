#!/bin/bash
#SBATCH --job-name=dv_train
#SBATCH --account=project_462001140
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/dejavu/logs/%x_%j.out
# usage: sbatch train_job.sh --source dejavu [--epochs 60 ...]
set -euo pipefail
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}; export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3; export NCCL_NET_GDR_LEVEL=PHB; export OMP_NUM_THREADS=1
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/dejavu
srun --cpu-bind=$CPU_BIND singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "
  mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source /projappl/project_462001140/envs/torchenv/bin/activate; export MPLCONFIGDIR=/tmp/mpl-\$USER;
  python -u dejavu.py train $*"
