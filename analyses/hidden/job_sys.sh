#!/bin/bash
#SBATCH --job-name=hid_sys
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/hidden/logs/%x_%j.out
# people exposed / hidden / warned at the populated 1 km pixels, 2022: Foehn, Foehn at 40 km, CAMS forecast, Aurora
# (figdata_hidden_systems.py, 32 forked workers) -> data/hidden_systems_pop.npz
cd /scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/hidden
unset PYTHONPATH; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/projappl/project_462001804/envs/world/bin/python
echo "start $(date -u +%FT%TZ)"
/usr/bin/time -v $PY -u -B figdata_hidden_systems.py 32 2>&1 | grep -v "^\s*\(Page\|Swaps\|File\|Socket\|Signals\|Voluntary\|Involuntary\|Average\|Major\|Minor\|Percent\)"
echo "end $(date -u +%FT%TZ)"
