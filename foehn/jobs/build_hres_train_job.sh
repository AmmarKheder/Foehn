#!/bin/bash
#SBATCH --job-name=build_hres_train
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/logs/%x_%j.out
source /scratch/project_462001140/ammar/eccv/world/env.sh
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km
echo "start $(date)"; ulimit -v unlimited 2>/dev/null; ulimit -a | grep -i virtual
python -u build_hres_train.py; rc=$?
echo "end $(date)"; exit $rc
