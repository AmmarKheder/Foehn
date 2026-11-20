#!/bin/bash
#SBATCH --job-name=ghap_train
#SBATCH --account=project_462001804
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/logs/train_%x_%j.out
# PROTOCOL.md A2(c): pmG_base = the frozen K0 recipe (pmA_baseK0 flags, --sigma-k 0) on the GHAP store (--store ghap,
# split G: train 2017-2020, SEL = 24 days of 2021). Segments (as A1(e)):
#   sbatch -p dev-g -t 0:30:00 train_G.sh 2000 22          smoke: 2,000 steps + 1 SEL validation (~1 GPU-h)
#   sbatch -p standard-g -t 3:30:00 train_G.sh 24000 180   resume to 24,000 steps (~11 GPU-h)
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
RG=$W/runs_ghap
S=$W/logs/GLOBAL_PREP_STATUS.md
STOP=$1; BUDGET=$2
NAME=${GHAP_RUN:-pmG_base}
COMMON="--store ghap --fires injh --split G --grid 025 --transport t6 --batch-size 2 --n-points 16384 --seed 0 \
 --max-steps 60000 --val-every 2000 --val-days $(seq -s, 0 15 345) --val-pixels 100000 --val-members 8 \
 --val-ode-steps 10 --log-every 100 --sigma-k 0"
note() { printf '%s\n' "- $(date '+%F %H:%M') [ghap] job $SLURM_JOB_ID (train $NAME): $1" >> "$S"; }
source $W/env.sh
cd $W
export PYTHONDONTWRITEBYTECODE=1
RES=""; [ -f $RG/$NAME/ckpt/last.ckpt ] && RES="--resume"
echo "start $(date) node=$SLURMD_NODENAME stop=$STOP budget=$BUDGET min resume='$RES'"
note "started on $SLURMD_NODENAME: $NAME to step $STOP (budget $BUDGET min, $RES)"
T0=$(date +%s)
srun --ntasks=8 --cpus-per-task=7 --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" \
     bash -c "source $CDFFM_ENVSH; cd $CDFFM_W; python -u -m cdffm_aq.train_aq --name $NAME --runs-root $RG $COMMON $RES \
              --stop-at-step $STOP --time-budget-min $BUDGET"
rc=$?
rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
LAST=$(python3 -c "import json;print(max(json.loads(l)['step'] for l in open('$RG/$NAME/metrics.jsonl') if l.strip()))" 2>/dev/null || echo "?")
VAL=$(python3 -c "
import json;r=[json.loads(l) for l in open('$RG/$NAME/metrics.jsonl') if l.strip()][-1]['model']['ug']
print('SEL crps %.3f rmseM %.3f ssF %.2f' % (r['crps_fair'], r['rmse_ensmean'], r.get('spread_skill_fortin', float('nan'))))" 2>/dev/null || echo "?")
note "segment ended rc=$rc after $(( ($(date +%s) - T0) / 60 )) node-min; last validated step $LAST: $VAL (log runs_ghap/logs/train_ghap_train_$SLURM_JOB_ID.out)"
echo "end $(date) rc=$rc last_step=$LAST"
exit $rc
