#!/bin/bash
#SBATCH --job-name=hid_test
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/hidden/logs/%x_%j.out
# smoke tests of the two data scripts (nothing kept except data/pop_pos.npz and data/hidden_systems_pop_test.npz)
cd /scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/hidden
unset PYTHONPATH; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/projappl/project_462001804/envs/world/bin/python
echo "start $(date -u +%FT%TZ)"
/usr/bin/time -v $PY -u -B figdata_hidden_systems.py 4 0,64 2>&1 | grep -v "^\s*\(Page\|Swaps\|File\|Socket\|Signals\|Voluntary\|Involuntary\|Average\|Exit\|Major\|Minor\|Percent\)"
echo "---- prep $(date -u +%FT%TZ)"
/usr/bin/time -v $PY -u -B figdata_exposure_sens.py prep 2>&1 | grep -v "^\s*\(Page\|Swaps\|File\|Socket\|Signals\|Voluntary\|Involuntary\|Average\|Exit\|Major\|Minor\|Percent\)"
echo "---- one day $(date -u +%FT%TZ)"
DAYS=0,180 /usr/bin/time -v $PY -u -B figdata_exposure_sens.py 2>&1 | grep -v "^\s*\(Page\|Swaps\|File\|Socket\|Signals\|Voluntary\|Involuntary\|Average\|Exit\|Major\|Minor\|Percent\)"
echo "end $(date -u +%FT%TZ)"
