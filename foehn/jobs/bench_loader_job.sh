#!/bin/bash
#SBATCH --job-name=g1km_bench
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/logs/%x_%j.out
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km
singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source /projappl/project_462001140/envs/torchenv/bin/activate; export PYTHONPATH=/scratch/project_462001140/ammar/eccv/cran-pm/src;
  export MPLCONFIGDIR=/tmp/mpl-\$USER; python -u bench_loader.py"
