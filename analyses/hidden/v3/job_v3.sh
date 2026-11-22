#!/bin/bash
#SBATCH --job-name=hid_v3
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# who is warned, version 3 (persistence, exact factors, centred 0.25 deg blocks), 358 forecasts -> v3/data/*.npz
: "${FOEHN_WORK:?set FOEHN_WORK (see the README of this folder)}"
: "${FOEHN_ENVS:?set FOEHN_ENVS (see the README of this folder)}"
cd ${FOEHN_WORK}/eccv/cran-pm/paper/natcomms/analyses/hidden/v3
unset PYTHONPATH; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${FOEHN_ENVS}/world/bin/python
echo "start $(date -u +%FT%TZ)"
$PY -u -B figdata_hidden_v3.py 32
echo "end $(date -u +%FT%TZ)"
