#!/bin/bash
#SBATCH --job-name=cal_diag
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_cal/logs/%x_%j.out
#SBATCH --time=00:40:00
# one or several calib.diag_cal tasks on one node; ARGS = ';'-separated argument lists, e.g.
#   sbatch --export=ALL,ARGS="tapergrid --run pmG_ref --ckpt 22000 --tag tapergrid_frozen" job_diag.sh
source /scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_cal/code/job_head.sh
echo "start $(date) node=$SLURMD_NODENAME args=$ARGS"
note "diag started on $SLURMD_NODENAME: $ARGS"
IFS=';' read -ra LIST <<< "$ARGS"
rc=0
for A in "${LIST[@]}"; do
  [ -z "$A" ] && continue
  run8 "python -u -m calib.diag_cal $A" || rc=1
done
rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
note "diag ended rc=$rc after $(elapsed) node-min"
echo "end $(date) rc=$rc node-min $(elapsed)"
exit $rc
