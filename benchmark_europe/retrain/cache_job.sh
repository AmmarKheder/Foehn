#!/bin/bash
#SBATCH --job-name=cran_cache
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain/logs/cache_%x_%a.out
source /scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain/torchenv.sh
torch_exec "cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain; python -u build_era5_cache.py ${1:-era5} $((2017 + SLURM_ARRAY_TASK_ID))"
