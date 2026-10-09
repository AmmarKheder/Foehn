#!/bin/bash
#SBATCH --job-name=dv_build
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/dejavu/logs/%x_%j.out
source /scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain/torchenv.sh
torch_exec "cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/dejavu; python -u build_coarse_data.py"
