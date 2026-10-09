#!/bin/bash
#SBATCH --job-name=g1km_stclim
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/logs/%x_%j.out
source /scratch/project_462001140/ammar/eccv/world/env.sh
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km
srun --ntasks=64 bash -c 'python -u station_ghap_history.py $SLURM_PROCID 64'
python -u station_ghap_history.py merge
