#!/bin/bash
#SBATCH --job-name=wb2_hres
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/logs/%x_%A_%a.out
# one array over (year, month): task k -> year 2017 + (k-1)//12 (2022: 3-D fields only), month 1 + (k-1)%12
source /scratch/project_462001140/ammar/eccv/world/env.sh
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km
K=$SLURM_ARRAY_TASK_ID; Y=$((2017 + (K - 1) / 12)); M=$((1 + (K - 1) % 12)); OPT=""; [ $Y -eq 2022 ] && OPT="--no-a"
echo "start $(date) hres $Y $M $OPT"
python -u dl_wb2.py hres $Y $M $OPT
echo "end $(date)"
