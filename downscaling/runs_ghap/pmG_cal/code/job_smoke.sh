#!/bin/bash
#SBATCH --job-name=cal_smoke
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_cal/logs/%x_%j.out
# PROTOCOL.md A6(j) smoke test: time per step and memory of a velocity-net arm and of the all-weights arm (30 steps each,
# weights discarded, nothing scored), the validation on the frozen weights (8 SEL days), the eval_x wrapper on 8 SEL days
# at every 64th pixel (frozen system).
source /scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_cal/code/job_head.sh
echo "start $(date) node=$SLURMD_NODENAME"
note "smoke started on $SLURMD_NODENAME"
rm -rf $C/runs/smoke_V $C/runs/smoke_A $C/runs/smoke_U
t=$(date +%s)
run8 "python -u -m calib.train_cal --name smoke_V $TRAIN_COMMON --train-part velocity --crps-space z --crps-fm 0.1 --lr 1e-4 \
      --max-steps 3000 --stop-at-step 30 --no-final-val --time-steps 30 --log-every 10 --val-sets ''"
echo "SMOKE_V rc=$? wall $(( $(date +%s) - t )) s"
t=$(date +%s)
run8 "python -u -m calib.train_cal --name smoke_U $TRAIN_COMMON --train-part velocity --crps-space ug --crps-fm 0.1 --lr 1e-4 \
      --max-steps 3000 --stop-at-step 12 --no-final-val --time-steps 12 --log-every 10 --val-sets ''"
echo "SMOKE_U rc=$? wall $(( $(date +%s) - t )) s"
t=$(date +%s)
run8 "python -u -m calib.train_cal --name smoke_A $TRAIN_COMMON --train-part all --crps-space z --crps-fm 1 --lr 3e-5 \
      --max-steps 2000 --stop-at-step 30 --no-final-val --time-steps 30 --log-every 10 --val-sets ''"
echo "SMOKE_A rc=$? wall $(( $(date +%s) - t )) s"
rm -rf $C/runs/smoke_V $C/runs/smoke_A $C/runs/smoke_U
t=$(date +%s)
run8 "python -u -m calib.diag_cal score --run pmG_ref --ckpt 22000 --sets sel,probe --max-days 8 --tag smoke_val"
echo "SMOKE_VAL rc=$? wall $(( $(date +%s) - t )) s"
t=$(date +%s)
run8 "python -u -m calib.eval_cal --run pmG_ref --runs-root $C/runs --ckpt 22000 --members 8 --ode-steps 10 \
      --days 0,15,30,45,60,75,90,105 --pixel-stride 64 --prior-file $PRIOR --z-bound $ZB --pmax-ug $PMAX --z0-trunc $T --tag smoke_s64"
echo "SMOKE_EVAL rc=$? wall $(( $(date +%s) - t )) s"
rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
note "smoke ended after $(elapsed) node-min (log logs/cal_smoke_$SLURM_JOB_ID.out)"
echo "end $(date) node-min $(elapsed)"
