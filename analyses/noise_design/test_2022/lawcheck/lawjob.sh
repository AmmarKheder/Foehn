#!/bin/bash
#SBATCH --job-name=a16_law
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=lawjob.out
#SBATCH --open-mode=append
# A16 law check, CPU only, reads only (writes under a16/lawcheck). usage: sbatch lawjob.sh "noise" "pop 24" "ncp 24"
set -uo pipefail
cd ${FOEHN_WORK}/global_model_forecasting_air_pollution/a16/lawcheck
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=${FOEHN_ENVS}/world/bin/python
for step in "$@"; do
  echo "=== a16_law ${SLURM_JOB_ID} step '$step' start $(date)"
  $PY -u -B lawcheck.py $step
  rc=$?
  echo "=== a16_law ${SLURM_JOB_ID} step '$step' end $(date) rc=$rc"
  [ $rc -eq 0 ] || exit $rc
done
