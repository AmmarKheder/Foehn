#!/bin/bash
#SBATCH --job-name=cran_stats
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain/logs/stats_%j.out
source /scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain/torchenv.sh
torch_exec "python -u -m cranpm.data.europe_dataset_v2 ${1:-stats}"
