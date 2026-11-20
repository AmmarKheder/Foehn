#!/bin/bash
#SBATCH --job-name=a8_train
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=03:00:00
# protocol_amendments/A8_seed_replicates.md (c): the training of the frozen system (runs_ghap/train_G.sh, pmG_base) with
# another seed and nothing else changed, in the frozen run's two segments (0 -> 2,000 as job 23246364, then --resume to
# 24,000 as job 23246365), in one job. Run directory runs_ghap/pmG_seed<SEED>.
#   sbatch --export=ALL,SEED=1 --output=<...>/pmG_seed1/a8/logs/train_seed1_%j.out train_job.sh
set -uo pipefail
source /scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_seed1/a8/a8_lib.sh
SEED=${SEED:?give SEED=1 or SEED=2}
case "$SEED" in 1|2) ;; *) echo "A8 trains seeds 1 and 2 only (got '$SEED')"; exit 2 ;; esac
NAME=pmG_seed$SEED
LOGD=$RGH/$NAME/a8/logs
mkdir -p $LOGD
export A8_WHAT="train seed $SEED"
export A8_NOTES=$LOGD/notes_train_seed$SEED.md
export A8_MEMLOG=$LOGD/mem_train_seed${SEED}_$SLURM_JOB_ID.txt
TJ=$(date +%s)
memmon_start "$A8_MEMLOG"
trap 'memmon_stop; rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"' EXIT
note "job started on $SLURMD_NODENAME: $NAME"

seg() {   # seg STOP BUDGET : one training segment; one retry if it dies in its first 4 minutes (start-up failure)
  local STOP=$1 BUDGET=$2 t0 rc
  t0=$(date +%s)
  train_seg $NAME $RGH $SEED $STOP $BUDGET; rc=$?
  if [ $rc != 0 ] && [ $(( $(date +%s) - t0 )) -lt 240 ]; then
    note "segment to $STOP failed at start-up (rc=$rc): one retry in 60 s"
    sleep 60
    train_seg $NAME $RGH $SEED $STOP $BUDGET; rc=$?
  fi
  return $rc
}
if [ "$(last_step $RGH/$NAME)" -lt 2000 ]; then
  seg 2000 22 || { note "segment 1 FAILED"; exit 1; }
fi
[ "$(last_step $RGH/$NAME)" -ge 2000 ] || { note "segment 1 ended before step 2,000 (time budget?)"; exit 1; }
note "segment 1 complete at +$(( $(date +%s) - TJ )) s"
if [ "$(last_step $RGH/$NAME)" -lt 24000 ]; then
  seg 24000 185 || { note "segment 2 FAILED"; exit 1; }
fi
[ "$(last_step $RGH/$NAME)" -ge 24000 ] || { note "training ended before step 24,000"; exit 1; }
note "training of $NAME complete after $(( $(date +%s) - TJ )) s of job time"
echo "end $(date) rc=0"
