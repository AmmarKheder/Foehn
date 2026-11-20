# sourced by the A6 GPU jobs: paths, environment, run helpers (one srun task per GCD, as runs_ghap/train_G.sh / p1r_G.sh)
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
RG=$W/runs_ghap
C=$RG/pmG_cal
S=$C/STATUS.md
PYW=/projappl/project_462001804/envs/world/bin/python
ZB=10; T=3
PMAX=$(python3 -c "import json;print(json.load(open('$W/data/processed/ghap/manifest.json'))['P_max_ug'])")
CENTRES=45.0:10.0,36.5:116.0,26.5:81.0,13.0:2.0,-11.0:-62.0,39.5:-85.0
SEL_DAYS=$(seq -s, 0 15 345)
PRIOR=$RG/pmG_final/prior.pt:P3
source $W/env.sh
cd $W
export PYTHONDONTWRITEBYTECODE=1
INC="source $CDFFM_ENVSH; cd $CDFFM_W; export PYTHONPATH=$C/code PYTHONDONTWRITEBYTECODE=1"
run1() { srun --ntasks=1 --cpus-per-task=7 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "$INC; $*"; }
run8() { srun --ntasks=8 --cpus-per-task=7 --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "$INC; $*"; }
note() { printf '%s\n' "- $(date '+%F %H:%M') job $SLURM_JOB_ID ($SLURM_JOB_NAME, $SLURM_JOB_PARTITION): $1" >> "$S"; }
TRAIN_COMMON="--runs-root $C/runs --store ghap --fires injh --split G --grid 025 --transport t6 --sigma-k 0 --batch-size 2 \
 --n-points 16384 --seed 0 --ema 0.99 --warmup 100 --val-every 250 --val-members 8 --val-ode-steps 10 \
 --init-from $RG/pmG_final/ckpt/best.ckpt"
EVAL_COMMON="--members 8 --ode-steps 10 --days $SEL_DAYS --prior-file $PRIOR --z-bound $ZB --pmax-ug $PMAX --z0-trunc $T"
JOB_T0=$(date +%s)
elapsed() { echo $(( ($(date +%s) - JOB_T0) / 60 )); }
