#!/bin/bash
#SBATCH --job-name=o1k_gfas_injh
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/gfas_injh_%j.out
# The fires input of the 1 km model used in real time: GFAS injh 2017-2024 (ADS, monthly) -> coarse/fires_injh_{2017_2022,2023_2024}.npy -> checks.
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/RT1KM_STATUS.md
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=1
note() { printf '%s\n' "- $(date '+%F %H:%M') [rt1km] job $SLURM_JOB_ID: $1" >> "$S"; }
trap 'note "gfas_injh FAILED, see logs/gfas_injh_$SLURM_JOB_ID.out"' ERR
cd $W/scripts
rm -f $W/data/raw/gfas_injh/*.part
$PY -I dl_gfas_injh.py download --workers 4
note "GFAS injh 2017-2024 downloaded ($(du -sh $W/data/raw/gfas_injh | cut -f1), data/raw/gfas_injh/)"
$PY -I dl_gfas_injh.py build --procs 16
$PY -I dl_gfas_injh.py check
note "fires_injh stores built and checked (data/raw/gfas_injh/check_report.json)"
