#!/bin/bash
#SBATCH --job-name=a11M_train
#SBATCH --account=<project>
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=03:40:00
#SBATCH --output=logs/train_%x_%j.out
# A11 experiment M (Addendum 2 point 1): the pmS_base recipe (runs_ghap/train_S.sh, PROTOCOL.md A3(c): the pmG recipe +
# --stations data/processed/assim, dropout 0-100 %) with ONE addition, the lag option: for a training day t the station
# features are those of the assimilation monitors of day t - lag (lag uniform in 1..5; the same-day lag 0 is off: M_LAG0_PROB=0 in m_lib.sh),
# with their anomaly and the lag. From scratch, seed 0, 8 GCDs x 2 days x 16,384 points, to step 24,000 (pmS_base's cap);
# validation every 2,000 steps on the 24 SEL days of 2021 against the 1 km estimates (lags 1 and 5; no held-out monitor).
#   sbatch jobs/train_M.sh [STOP=24000] [BUDGET_MIN=200]        (8 GCDs x 3 h 40 = 29.3 GCD-hours requested)
#   a second segment resumes from last.ckpt:  sbatch jobs/train_M.sh 24000 200
# NAME: M_RUN (default pmM_base). Submitted together with jobs/val_lag_job.sh (Slurm dependency afterok).
set -uo pipefail
source ${FOEHN_WORK}/global_model_forecasting_air_pollution/a11/stageM/jobs/m_lib.sh
export M_WHAT="train"
STOP=${1:-24000}; BUDGET=${2:-200}
NAME=${M_RUN:-pmM_base}
trap 'rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"' EXIT
[ -f $W/data/processed/assim/assim_2021.npz ] || { note "assimilation files missing: not started"; exit 1; }
RES=""; [ -f $RUNS/$NAME/ckpt/last.ckpt ] && RES="--resume"
echo "start $(date) node=$SLURMD_NODENAME name=$NAME stop=$STOP budget=$BUDGET min resume='$RES' lag='$M_LAG'"
note "training $NAME to step $STOP started on $SLURMD_NODENAME (budget $BUDGET min, '$RES', $M_LAG)"
T0=$(date +%s)
srun --ntasks=8 --cpus-per-task=7 --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" \
     bash -c "$INC; python -u -m cdffm_aq_m.train_aq --name $NAME --runs-root $RUNS $M_COMMON $M_LAG $RES \
              --stop-at-step $STOP --time-budget-min $BUDGET"
rc=$?
LAST=$(last_step $RUNS/$NAME)
VAL=$(python3 -c "
import json;r=[json.loads(l) for l in open('$RUNS/$NAME/metrics.jsonl') if l.strip()][-1];u=r['model']['ug'];s=r['lag']['scores']
g=lambda lb,lg,c:('%.3f' % s[lb][lg][c]['crps_fair']) if s[lb][lg][c] else '-'
print('SEL crps (lags 1+5) %.4f rmseM %.3f ssF %.2f | lag 1: 0-10 km %s (hidden %s), 10-50 km %s (hidden %s) | lag 5: 0-10 km %s, 10-50 km %s | %.0f ms/step' % (
 u['crps_fair'], u['rmse_ensmean'], u.get('spread_skill_fortin', float('nan')), g('model','1','0-10km'), g('model|nostations','1','0-10km'),
 g('model','1','10-50km'), g('model|nostations','1','10-50km'), g('model','5','0-10km'), g('model','5','10-50km'), r['train_window'].get('ms_per_step', float('nan'))))" 2>/dev/null || echo "?")
note "training segment of $NAME ended rc=$rc after $(( ($(date +%s) - T0) / 60 )) node-min; last validated step $LAST: $VAL"
echo "end $(date) rc=$rc last_step=$LAST"
# the chained validation (afterok) must only run on a finished training
[ $rc = 0 ] && [ "$LAST" -ge "$STOP" ] || { note "training of $NAME not complete (rc=$rc, last validated step $LAST < $STOP): resubmit jobs/train_M.sh to resume"; exit 1; }
exit 0
