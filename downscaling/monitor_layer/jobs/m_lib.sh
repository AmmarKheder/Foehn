# m_lib.sh -- sourced by the job scripts of A11 experiment M (host side of a Slurm job on LUMI-G).
# protocol_amendments/A11_retraining_two_stages.md, Addendum 2 point 1; a11/stageM/README.md.
# Made from runs_ghap/train_S.sh (PROTOCOL.md A3(c): the pmS_base settings) with
#   * the copy of the package: python -m cdffm_aq_m.train_aq, run from a11/stageM (the original package is not used for
#     a new run and is not modified);
#   * the lag option (M_LAG below);
#   * every output under a11/stageM/runs/<NAME>, notes in a11/stageM/logs/notes.md (never logs/GLOBAL_PREP_STATUS.md);
#     nothing is written to runs_ghap, nothing is frozen, no production script is called.
: "${FOEHN_WORK:?set FOEHN_WORK (see the README of this folder)}"
W=${FOEHN_WORK}/global_model_forecasting_air_pollution
SM=$W/a11/stageM
RUNS=$SM/runs
source $W/env.sh
cd $SM
export PYTHONDONTWRITEBYTECODE=1
M_NOTES=${M_NOTES:-$SM/logs/notes.md}
M_WHAT=${M_WHAT:-job}
note() { printf '%s\n' "- $(date '+%F %H:%M:%S') job ${SLURM_JOB_ID:-?} ($M_WHAT): $1" | tee -a "$M_NOTES"; }
# inside the container, from a11/stageM (so that cdffm_aq_m is the package that is imported)
INC="source $CDFFM_ENVSH; cd $SM"
run1() { srun --ntasks=1 --cpus-per-task=7 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "$INC; $*"; }
run8() { srun --ntasks=8 --cpus-per-task=7 --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "$INC; $*"; }

# the pmS_base settings, word for word (runs_ghap/train_S.sh, COMMON)
M_COMMON="--store ghap --fires injh --split G --grid 025 --transport t6 --batch-size 2 --n-points 16384 --seed 0 \
 --max-steps 60000 --val-every 2000 --val-days $(seq -s, 0 15 345) --val-pixels 100000 --val-members 8 \
 --val-ode-steps 10 --log-every 100 --sigma-k 0 --stations $W/data/processed/assim"
# the lag option of experiment M: lags 1..5 uniformly. By design: the same-day lag 0 is
# OFF (M_LAG0_PROB default 0): no value of a valid day enters the training of M, as Addendum 2 states (lag = lead day).
# in-training validation at lags 1 and 5 (+ every monitor hidden); the full lag table is val_lag.py's
M_LAG="--stn-lag 1,2,3,4,5 --stn-lag0-prob ${M_LAG0_PROB:-0} --val-lags 1,5"

last_step() { python3 -c "import json;print(max(json.loads(l)['step'] for l in open('$1/metrics.jsonl') if l.strip()))" 2>/dev/null || echo 0; }
