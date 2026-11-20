#!/bin/bash
#SBATCH --job-name=hid_bc_test
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/hidden/v2/logs/%x_%j.out
# smoke test of v2/figdata_hidden_bc.py on 4 forecasts (start days 0 and 64, leads 1 and 3) + identity check of the
# recomputed systems against ../data/hidden_systems_pop.npz (job 23864988)
cd /scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/hidden/v2
unset PYTHONPATH; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/projappl/project_462001804/envs/world/bin/python
$PY -u -B figdata_hidden_bc.py 4 0,64
$PY -u -B check_identity.py data/hidden_bc_pop_test.npz
