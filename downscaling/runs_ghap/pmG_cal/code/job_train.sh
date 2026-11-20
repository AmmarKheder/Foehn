#!/bin/bash
#SBATCH --job-name=cal_train
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_cal/logs/%x_%j.out
#SBATCH --time=01:00:00
# PROTOCOL.md A6(b): one fine-tuning arm.   sbatch --export=ALL,ARM=V2 --job-name=cal_V2 job_train.sh
source /scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_cal/code/job_head.sh
ARM=${ARM:?set ARM (V1 V2 V3 V4 A1)}
case $ARM in
  V1) OPT="--train-part velocity --crps-space z --crps-fm 1 --lr 1e-4 --max-steps 3000" ;;
  V2) OPT="--train-part velocity --crps-space z --crps-fm 0.1 --lr 1e-4 --max-steps 3000" ;;
  V3) OPT="--train-part velocity --crps-space z --crps-fm 0.01 --lr 1e-4 --max-steps 3000" ;;
  V4) OPT="--train-part velocity --crps-space ug --crps-scale 6 --crps-fm 0.1 --lr 1e-4 --max-steps 3000" ;;
  A1) OPT="--train-part all --crps-space z --crps-fm 1 --lr 3e-5 --max-steps 2000" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
[ -f $C/runs/$ARM/metrics.jsonl ] && { echo "$ARM already has a run"; exit 3; }
echo "start $(date) node=$SLURMD_NODENAME arm=$ARM opt=$OPT"
note "arm $ARM started on $SLURMD_NODENAME ($OPT)"
run8 "python -u -m calib.train_cal --name $ARM $TRAIN_COMMON $OPT --log-every 50 --val-sets sel,probe,ins2020"
rc=$?
rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
LAST=$(python3 -c "
import json;r=[json.loads(l) for l in open('$C/runs/$ARM/metrics.jsonl') if l.strip()][-1];u=r['sets']['sel']['ug']
print('step %d: V100k crps %.4f rmseM %.4f ssF %.4f, hot-spot %d' % (r['step'], u['crps_fair'], u['rmse_ensmean'], u['spread_skill_fortin'], r['sets']['probe']['sanity']['n_pix_gt_TG']))" 2>/dev/null || echo "?")
note "arm $ARM ended rc=$rc after $(elapsed) node-min; last validation $LAST (log logs/${SLURM_JOB_NAME}_$SLURM_JOB_ID.out)"
echo "end $(date) rc=$rc node-min $(elapsed)"
exit $rc
