#!/bin/bash
#SBATCH --job-name=gp_injh_check
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/global_injh_check_%j.out
# GlobalStore(fires='injh') opens, channels 0-1 bit-identical to mami, channel 2 = injh, a real training batch is finite.
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
source $W/env.sh
cdffm_run python -u "$1"
