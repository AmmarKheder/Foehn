#!/bin/bash
#SBATCH --job-name=wb2
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/logs/%x_%A_%a.out
# usage: sbatch [--array=1-12] dl_wb2_job.sh <hres|era5|static> <year> [month or "array"] [dl_wb2.py options]
#   with --array, the month is SLURM_ARRAY_TASK_ID
source /scratch/project_462001140/ammar/eccv/world/env.sh
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km
KIND=$1; YEAR=$2; MONTH=$3; shift 3
[ "$MONTH" = "array" ] && MONTH=$SLURM_ARRAY_TASK_ID
echo "start $(date) $KIND $YEAR $MONTH $*"
/usr/bin/time -v python -u dl_wb2.py $KIND $YEAR $MONTH "$@" 2>&1 | grep -v "^\s*\(Page\|Swap\|File\|Socket\|Signal\|Vol\|Invol\|Minor\|Major\|Average\|Exit\|Percent\|System time\|User time\)"
echo "end $(date)"
