#!/bin/bash
#SBATCH --job-name=a11M_vallag
#SBATCH --account=<project>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/vallag_%j.out
# A11 experiment M: val_lag.py on ONE GCD (1 GCD-hour requested): every saved and validated checkpoint of the run, the
# naive baseline (runs_ghap/pmS_final fed with lagged monitors) and the station-free runs_ghap/pmG_final, on the 24 SEL
# days of 2021 x 100,000 pixels, 8 members x 10 Euler, lags 0 (same-day reference) and 1..5, by distance class.
# Truth = the 1 km estimates; no held-out monitor is read.
#   sbatch --dependency=afterok:<training job> jobs/val_lag_job.sh [RUN=pmM_base] [CKPT=all] [LAGS=0,1,2,3,4,5]
set -uo pipefail
source ${FOEHN_WORK}/global_model_forecasting_air_pollution/a11/stageM/jobs/m_lib.sh
export M_WHAT="val_lag"
RUN=${1:-${M_RUN:-pmM_base}}; CKPT=${2:-all}; LAGS=${3:-0,1,2,3,4,5}
trap 'rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"' EXIT
[ -f $RUNS/$RUN/metrics.jsonl ] || { note "$RUN: no metrics.jsonl"; exit 1; }
TAG=sel2021_$(echo $CKPT | tr ',' '_')
note "val_lag of $RUN (checkpoints: $CKPT; lags $LAGS) started on $SLURMD_NODENAME"
T0=$(date +%s)
run1 "python -u val_lag.py --run $RUN --runs-root $RUNS --ckpt $CKPT --lags $LAGS --tag $TAG"
rc=$?
note "val_lag of $RUN ended rc=$rc after $(( $(date +%s) - T0 )) s -> runs/$RUN/val_lag/$TAG.json (.md: the tables)"
exit $rc
