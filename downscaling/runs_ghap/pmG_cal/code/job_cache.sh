#!/bin/bash
#SBATCH --job-name=cal_cache
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_cal/logs/%x_%j.out
# PROTOCOL.md A6(c): the validation sets (truth values of 2021 hold-out days and of training-day points; no model).
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
C=$W/runs_ghap/pmG_cal
source $W/env.sh
cd $W
echo "start $(date) node=$SLURMD_NODENAME"
singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "source $CDFFM_ENVSH; cd $CDFFM_W; export PYTHONPATH=$C/code PYTHONDONTWRITEBYTECODE=1; \
    python -u -m calib.diag_cal cache --run pmG_ref --device cpu"
rc=$?
rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
ls -la $C/cache
echo "end $(date) rc=$rc"
exit $rc
