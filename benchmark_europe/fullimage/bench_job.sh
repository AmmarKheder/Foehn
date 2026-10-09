#!/bin/bash
#SBATCH --job-name=fi_bench
#SBATCH --account=project_462001140
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/fullimage/%x_%j.out
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}; mkdir -p $MIOPEN_USER_DB_PATH
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "source /opt/miniconda3/bin/activate pytorch;
  export MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH; cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/fullimage; python -u bench_attention.py"
