#!/bin/bash
#SBATCH --job-name=cdffm_fg_check
#SBATCH --account=project_462001804
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/flexgreen_check_%j.out
# flexgreen_pm.py lead 1 on two 2021 start days (doy 100, 364) vs the Foehn green/era5 store (bit equality), timing.
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
source $W/env.sh
T=$W/data/processed/global_fgtest
rm -rf $T
srun singularity exec -B $CDFFM_BIND $CDFFM_SIF bash -c "source $CDFFM_ENVSH; cd $W/transport; python -u flexgreen_pm.py run --src era5 --year 2021 --days 100 364 --max-lead 1 --out-root $T --fires-new $W/data/processed/global/coarse/fires_2023_2024.npy --verbose"
srun singularity exec -B $CDFFM_BIND $CDFFM_SIF bash -c "source $CDFFM_ENVSH; python -u $W/scripts/check_flexgreen_pm.py $T 2021 100 364"
rc=$?
rm -rf $T
exit $rc
