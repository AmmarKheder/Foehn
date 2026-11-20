#!/bin/bash
#SBATCH --account=project_462001804
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=bf_t6
#SBATCH --partition=small
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/bf_t6_%j.out
# Backfill: T6 transport channels 2025, 2026 (NaN after the last FLEXGREEN/CAMS day) + coverage json.
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/GLOBAL_PREP_STATUS.md
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=1 WORLD_DATA=/scratch/project_462001804/ammar/eccv/data/world
note() { printf '%s\n' "- $(date '+%F %H:%M') [backfill] job $SLURM_JOB_ID: $1" >> "$S"; }
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
trap 'note "T6 2025-2026 FAILED, logs/bf_t6_$SLURM_JOB_ID.out"' ERR
run() { singularity exec -B /scratch,/pfs $SIF bash -c "unset PYTHONPATH; export PYTHONNOUSERSITE=1; source /opt/miniconda3/bin/activate pytorch; python -I $W/scripts/build_global_t6.py $*"; }
run --years 2025 --threads 8 --allow-missing > $W/logs/bf_t6_${SLURM_JOB_ID}_2025.log 2>&1 &
run --years 2026 --threads 8 --allow-missing > $W/logs/bf_t6_${SLURM_JOB_ID}_2026.log 2>&1 &
wait
tail -n 1 $W/logs/bf_t6_${SLURM_JOB_ID}_2025.log $W/logs/bf_t6_${SLURM_JOB_ID}_2026.log; rm -f $W/logs/bf_t6_${SLURM_JOB_ID}_202?.log
cd $W/scripts && $PY -I backfill_2025_2026.py coverage > /dev/null
note "T6 2025-2026 built (transport/t6_2025.npy, t6_2026.npy; last complete days $($PY -I -c "import json; c=json.load(open('$W/data/processed/global/COVERAGE_2025_2026.json'))['inputs']['t6']; print(', '.join(f\"{y} {v['last_complete_day']}\" for y, v in c.items()))"))"
