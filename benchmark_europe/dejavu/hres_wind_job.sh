#!/bin/bash
#SBATCH --job-name=dv_hwind
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=1-12
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/dejavu/logs/%x_%a.out
source /scratch/project_462001140/ammar/eccv/world/env.sh
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/dejavu
for i in 1 2 3; do python -u dl_hres_wind.py $SLURM_ARRAY_TASK_ID && break; echo "retry $i"; sleep 30; done
