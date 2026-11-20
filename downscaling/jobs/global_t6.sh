#!/bin/bash
#SBATCH --job-name=cdffm_g_t6
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/global_t6_%j.out
# T6 transport channels per target year (PLAN.md 2.7 item 6). usage: sbatch jobs/global_t6.sh 2017 2018 ... [--check-smoke]
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/BUILD_STATUS.md
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
note() { printf '%s\n' "- $(date '+%F %H:%M') [global-build] job $SLURM_JOB_ID: $1" >> "$S"; }
trap 'note "t6 FAILED ($*), see logs/global_t6_$SLURM_JOB_ID.out"' ERR
run() { singularity exec -B /scratch,/pfs $SIF bash -c "unset PYTHONPATH; export PYTHONNOUSERSITE=1; source /opt/miniconda3/bin/activate pytorch; python -I $W/scripts/build_global_t6.py $*"; }
pids=()
for y in "$@"; do
  case $y in --*) continue ;; esac
  run --years $y --threads 4 > $W/logs/global_t6_${SLURM_JOB_ID}_$y.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
for y in "$@"; do case $y in --*) ;; *) tail -2 $W/logs/global_t6_${SLURM_JOB_ID}_$y.log; rm -f $W/logs/global_t6_${SLURM_JOB_ID}_$y.log ;; esac; done
case " $* " in *" --check-smoke "*) run --check-smoke ;; esac
note "t6 DONE for $*"
