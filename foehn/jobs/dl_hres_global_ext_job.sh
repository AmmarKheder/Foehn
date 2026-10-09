#!/bin/bash
#SBATCH --job-name=g1km_hres_ext
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/logs/%x_%A_%a.out
# usage: sbatch --array=1-12 dl_hres_global_job.sh      (one month per task; LIMIT=1 for a test)
source /scratch/project_462001140/ammar/eccv/world/env.sh
mkdir -p /scratch/project_462001140/ammar/eccv/data/world/global1km/hres
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km
/usr/bin/time -v python -u dl_hres_global_ext.py ${SLURM_ARRAY_TASK_ID:-1} 2>&1 | grep -v "^\s*\(Page\|Swap\|File\|Socket\|Signal\|Vol\|Invol\|Minor\|Major\|Average\|Exit\|Percent\|System time\|User time\)"
