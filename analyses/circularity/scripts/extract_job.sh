#!/bin/bash
#SBATCH --job-name=circ_extract
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/circularity/logs/%x_%j.out
# CPU only: reads stored forecasts (no network is run). usage: sbatch extract_job.sh <script.py> [args]
set -uo pipefail
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
PY=/projappl/project_462001804/envs/world/bin/python
cd /scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/circularity/scripts
echo "start $(date) $*"
$PY -u "$@"
rc=$?
echo "end $(date) rc=$rc"
exit $rc
