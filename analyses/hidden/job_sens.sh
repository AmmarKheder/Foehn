#!/bin/bash
#SBATCH --job-name=hid_sens
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --mem=220G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/hidden/logs/%x_%j.out
# sensitivity of the count of hidden people (figdata_exposure_sens.py): 16 ranks over the days of 2022, then merge
cd /scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/hidden
unset PYTHONPATH; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/projappl/project_462001804/envs/world/bin/python
echo "start $(date -u +%FT%TZ)"
[ -f data/pop_pos.npz ] || $PY -u -B figdata_exposure_sens.py prep
srun $PY -u -B figdata_exposure_sens.py
$PY -u -B figdata_exposure_sens.py merge
echo "end $(date -u +%FT%TZ)"
