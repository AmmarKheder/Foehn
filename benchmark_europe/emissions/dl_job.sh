#!/bin/bash
#SBATCH --job-name=cams_ant
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/emissions/logs/%x_%a.out
source /scratch/project_462001140/ammar/eccv/world/env.sh
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/emissions
python -u dl_cams_glob_ant.py $((2017 + SLURM_ARRAY_TASK_ID))
