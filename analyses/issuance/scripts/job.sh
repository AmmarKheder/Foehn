#!/bin/bash
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/issuance/logs/%x_%j.out
# usage: sbatch --job-name=<name> [--export=ALL,ISSUANCE=d1,DUMP=1] job.sh <script.py> [args]
cd /scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/issuance/scripts
unset PYTHONPATH; export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache MPLCONFIGDIR=/tmp/mpl-$USER-$SLURM_JOB_ID
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
F=$1; shift
echo "start $(date -u +%FT%TZ) $F $* ISSUANCE=${ISSUANCE:-} DUMP=${DUMP:-}"
/usr/bin/time -v /projappl/project_462001804/envs/world/bin/python -B -u $F "$@" 2>&1 | grep --line-buffered -v "findfont"
echo "end $(date -u +%FT%TZ) exit ${PIPESTATUS[0]}"
