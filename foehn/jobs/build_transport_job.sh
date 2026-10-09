#!/bin/bash
#SBATCH --job-name=g1km_transp
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/logs/%x_%A_%a.out
# usage: sbatch build_transport_job.sh ant | sbatch --array=0-15 build_transport_job.sh fires|wind
source /scratch/project_462001140/ammar/eccv/world/env.sh
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km
if [ "$1" = "fires" ] || [ "$1" = "wind" ]; then python -u build_transport.py $1 $SLURM_ARRAY_TASK_ID 16; else python -u build_transport.py $1; fi
