#!/bin/bash
#SBATCH --job-name=emis_recompute
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/emissions/logs/recompute_%j.out
# gains of the emissions and of the transport at equal training, from the 2022 evaluation files (read only): sbatch recompute_job.sh
cd /scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/emissions
unset PYTHONPATH; export PYTHONNOUSERSITE=1
echo "start $(date -u +%FT%TZ)"
/usr/bin/time -v /projappl/project_462001804/envs/world/bin/python -B -u recompute.py
echo "end $(date -u +%FT%TZ) exit $?"
