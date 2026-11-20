#!/bin/bash
#SBATCH --job-name=o1k_calib_blh
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/o1k_calib_blh_%j.out
# RT1KM: calibration of the BLH substitute (encoder channels 31-33) on ERA5 training-period days -> realtime_1km/blh_calib.json
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
source $W/env.sh
srun singularity exec -B $CDFFM_BIND $CDFFM_SIF bash -c "source $CDFFM_ENVSH; python -I $W/realtime_1km/calib_blh.py"
printf '%s\n' "- $(date '+%F %H:%M') [rt1km] job $SLURM_JOB_ID: BLH calibration written (realtime_1km/blh_calib.json)" >> $W/logs/RT1KM_STATUS.md
