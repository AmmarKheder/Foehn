#!/bin/bash
#SBATCH --job-name=fi_ghap
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/fullimage/logs/%x_%a.out
source /scratch/project_462001140/ammar/eccv/world/env.sh
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/fullimage
python -u build_ghap1km.py $((2017 + SLURM_ARRAY_TASK_ID))
