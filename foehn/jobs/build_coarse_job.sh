#!/bin/bash
#SBATCH --job-name=g1km_coarse
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/logs/%x_%A_%a.out
# usage: sbatch build_coarse_job.sh stats | sbatch --array=0-15 build_coarse_job.sh fill | sbatch build_coarse_job.sh meta
source /scratch/project_462001140/ammar/eccv/world/env.sh
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km
if [ "$1" = "fill" ]; then python -u build_coarse.py $SLURM_ARRAY_TASK_ID 16; else python -u build_coarse.py $1; fi
