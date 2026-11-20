#!/bin/bash
#SBATCH --job-name=a7_prep
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/test2022_official/logs/prep_%j.out
# A7 step 1 (CPU, world venv): station positions and rings; CAMS analyses and CAMS forecast fields on the model grids.
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
O=$W/test2022_official
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
note() { printf '%s\n' "- $(date '+%F %H:%M') job $SLURM_JOB_ID (prep): $1" >> $O/STATUS.md; }
trap 'note "FAILED (logs/prep_$SLURM_JOB_ID.out)"' ERR
T0=$(date +%s)
$PY -I $O/code/prep_positions.py
$PY -I $O/code/prep_inputs.py --procs 16
note "positions (pos/st_ring_official.npz, st_meta_official.npz, st_qc_official.json) and inputs (inputs/cams_an_2022.npy, camsfc025.npy, inputs.json) written in $(( $(date +%s) - T0 )) s"
