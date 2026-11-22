#!/bin/bash
# Sourced by every driver of the S1 CPU chain (README.md). Requires FOEHN_DESIGN=<D> (e.g. ar): the final model's folders
# RUN/<eval>_<D> are read, every derived file carries the label (crps_e01_<D>, data_<D>/, results_<D>/, figs_<D>/), and
# every script runs through run.py, whose write guard refuses any write to a path without the label.
set -uo pipefail
: "${FOEHN_DESIGN:?set FOEHN_DESIGN=<label>, e.g. FOEHN_DESIGN=ar}"
[[ "$FOEHN_DESIGN" =~ ^[a-z0-9]+$ ]] || { echo "bad FOEHN_DESIGN=$FOEHN_DESIGN"; exit 2; }
export FOEHN_DESIGN
D=$FOEHN_DESIGN; L=crps_e01_$D
ROOT=${FOEHN_WORK}/eccv/cran-pm
N=$ROOT/paper/natcomms
S1=$N/rebuild/s1_chain
FIG=$ROOT/experiments/figures
RUNS=$ROOT/experiments/global1km/runs
RUN=green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4
RD=$RUNS/$RUN
A=green_flow_shift25_global_L7_noant_whres_ghres:eval_final              # base model A (class e): read as stored
C=green_flow_shift25_global_L7_noant_whres_notransport:eval_final        # transport ablation C (class e): read as stored
CIRC=$N/analyses/circularity/scripts
ISS=$N/analyses/issuance/scripts
REL=$N/analyses/reliability/scripts
HID=$N/analyses/hidden
OFF=${FOEHN_WORK}/global_model_forecasting_air_pollution/test2022_official
FIGS_D=$N/rebuild/figs_$D
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=${FOEHN_ENVS}/.cache
export MPLCONFIGDIR=/tmp/mpl-$USER-${SLURM_JOB_ID:-$$}
PY="${FOEHN_ENVS}/world/bin/python -I -B"
# step <script.py> [args]: one script through the write guard; stops the driver on the first failure
step() {
  echo; echo "===== $(date -u +%FT%TZ) $* (FOEHN_DESIGN=$D)"; local t0; t0=$(date +%s)
  $PY $S1/run.py "$@" 2>&1 | grep --line-buffered -v findfont; local rc=${PIPESTATUS[0]}
  echo "===== rc=$rc $(( $(date +%s) - t0 )) s $(basename "$1") ${2:-}" | cut -c1-200; [ "$rc" -eq 0 ] || exit "$rc"; }
# bg <log> <script.py> [args]: the same in the background (wait with waitall)
PIDS=()
bg() { local log=$1; shift; ( step "$@" ) > "$log" 2>&1 & PIDS+=($!); echo "started $(basename "$1") ${2:-} -> $log"; }
waitall() { local bad=0; for p in "${PIDS[@]}"; do wait "$p" || bad=1; done; PIDS=(); [ $bad -eq 0 ] || { echo "a background step failed"; exit 4; }; }
# need <path> ...: inputs that must exist before the step starts
need() { for p in "$@"; do [ -e "$p" ] || { echo "missing input: $p"; exit 3; }; done; }
mkdir -p "$FIGS_D"
