#!/bin/bash
#SBATCH --job-name=fi_static
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/fullimage/logs/%x_%j.out
source /scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain/torchenv.sh
torch_exec "cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/fullimage; python -u build_static.py"
