#!/bin/bash
#SBATCH --job-name=sc_score
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay/score/logs/score_%j.out
# WP1b scoring on a CPU node: sbatch run_all_job.sh [POINTS_DIR] [OUT_DIR] [ARMS] [FOCUS]   (see run_all.sh)
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
bash /scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay/score/run_all.sh "$@"
