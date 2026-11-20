#!/bin/bash
#SBATCH --job-name=cal_d0
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
# PROTOCOL.md A6(c) reference, A6(d) diagnostics and A6(e) temperature fit/grid, all on the FROZEN system (no training).
source /scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_cal/code/job_head.sh
echo "start $(date) node=$SLURMD_NODENAME"
note "D0 + temperature started on $SLURMD_NODENAME"
[ -f $C/results/d0_frozen.json ] || run8 "python -u -m calib.diag_cal d0 --run pmG_ref --ckpt 22000 --tag d0_frozen" || { note "d0 FAILED"; exit 1; }
[ -f $C/results/tfit_frozen.json ] || run8 "python -u -m calib.diag_cal tfit --run pmG_ref --ckpt 22000 --tag tfit_frozen" || { note "tfit FAILED"; exit 1; }
[ -f $C/results/tgrid_frozen.json ] || run8 "python -u -m calib.diag_cal tgrid --run pmG_ref --ckpt 22000 --tag tgrid_frozen" || { note "tgrid FAILED"; exit 1; }
rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
note "D0 + temperature ended after $(elapsed) node-min -> results/{d0,tfit,tgrid}_frozen.json"
echo "end $(date) node-min $(elapsed)"
