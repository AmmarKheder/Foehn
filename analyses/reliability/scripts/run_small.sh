#!/bin/bash
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/reliability/logs/%x_%j.out
# CPU job of the calibration analysis (world venv, no GPU): sbatch --job-name=<name> run_small.sh <script.py> [args]
unset PYTHONPATH; export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache MPLCONFIGDIR=/tmp/mpl-$USER-$SLURM_JOB_ID
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/reliability/scripts
echo "start $(date -u +%FT%TZ) $*"
/usr/bin/time -v /projappl/project_462001804/envs/world/bin/python -u -B "$@" 2>&1 | grep -v "findfont"
echo "end $(date -u +%FT%TZ) exit ${PIPESTATUS[0]}"
