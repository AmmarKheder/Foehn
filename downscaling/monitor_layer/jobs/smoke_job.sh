#!/bin/bash
#SBATCH --job-name=a11M_smoke
#SBATCH --account=<project>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:50:00
#SBATCH --output=logs/smoke_%j.out
# A11 experiment M: the tests and the smoke test of this step on ONE GCD (limit 50 min = 0.83 GCD-hour). No training of M.
#   T1  tests/test_m.py cpu    unit tests on the toy store and a synthetic, date-encoded assimilation set (no GPU)
#   T2  tests/test_m.py real   the loader on the real assimilation files 2017-2021 (test i; no GPU)
#   T3  tests/test_m.py ghap   test (ii) on the GHAP store: loss of one fixed batch, original package vs the copy, option off
#   S1  smoke training: 100 steps with the lag option (pmS_base settings; validation on 2 SEL days x 4,096 pixels), one
#       checkpoint
#   S2  val_lag.py on that checkpoint: 8 members at 4,096 pixels of 2021-01-08 and 2021-07-07, lags 1 and 5, with the
#       frozen pmS_final (lagged monitors) and pmG_final; tests/check_smoke.py: everything finite (test iii)
#   S3  val_lag.py on the frozen arms alone, 24 SEL days x 100,000 pixels, lags 0..5: the validation code must give back
#       the SEL numbers stored by A3 for pmG and pmS (lag 0); it also times the validation and gives the naive baseline
#   sbatch jobs/smoke_job.sh ["T1 T2 T3 S1 S2 S3"]
set -uo pipefail
source ${FOEHN_WORK}/global_model_forecasting_air_pollution/a11/stageM/jobs/m_lib.sh
export M_WHAT="smoke"
STEPS=${1:-"T1 T2 T3 S1 S2 S3"}
NAME=${M_SMOKE_RUN:-pmM_smoke_$SLURM_JOB_ID}      # M_SMOKE_RUN: an existing smoke run, to redo S2 alone
O=$SM/tests/out
mkdir -p $O
export STAGEM_TMP=/tmp/${USER}-stageM-$SLURM_JOB_ID
mkdir -p $STAGEM_TMP
trap 'rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID" "$STAGEM_TMP"' EXIT
TJ=$(date +%s)
declare -A RC
has() { case " $STEPS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
step_done() { RC[$1]=$2; note "step $1 rc=$2 at +$(( $(date +%s) - TJ )) s"; }
note "smoke job started on $SLURMD_NODENAME (partition $SLURM_JOB_PARTITION), steps: $STEPS"

if has T1; then
  run1 "export STAGEM_TMP=$STAGEM_TMP OMP_NUM_THREADS=7; python -u tests/test_m.py cpu" > $O/test_m_cpu_$SLURM_JOB_ID.log 2>&1
  step_done T1 $?
  grep -E "^\[(OK|FAIL)\]|^\[test_m" $O/test_m_cpu_$SLURM_JOB_ID.log | cut -c1-260
fi
if has T2; then
  run1 "python -u tests/test_m.py real"
  step_done T2 $?
fi
if has T3; then
  run1 "python -u tests/test_m.py ghap"
  step_done T3 $?
fi
if has S1; then
  SMOKE_COMMON=$(echo "$M_COMMON" | sed -e "s/--val-every 2000/--val-every 100/" -e "s/--val-days [0-9,]*/--val-days 0,180/" \
                 -e "s/--val-pixels 100000/--val-pixels 4096/" -e "s/--log-every 100/--log-every 20/")
  echo "smoke training: $SMOKE_COMMON $M_LAG"
  run1 "python -u -m cdffm_aq_m.train_aq --name $NAME --runs-root $RUNS $SMOKE_COMMON $M_LAG --stop-at-step 100 --time-budget-min 20"
  step_done S1 $?
  ls -la $RUNS/$NAME/ckpt
fi
if has S2; then
  run1 "python -u val_lag.py --run $NAME --runs-root $RUNS --ckpt 100 --days 0,180 --pixels 4096 --lags 1,5 --members 8 --ode-steps 10 --save-members --tag smoke" \
   && run1 "python -u tests/check_smoke.py smoke $RUNS/$NAME 100 $RUNS/$NAME/val_lag/smoke.json"
  step_done S2 $?
fi
if has S3; then
  run1 "python -u val_lag.py --runs-root $RUNS --lags 0,1,2,3,4,5 --tag frozen_sel2021" \
   && run1 "python -u tests/check_smoke.py a3 $RUNS/_frozen_arms/val_lag/frozen_sel2021.json"
  step_done S3 $?
fi
FAIL=""
for s in $STEPS; do [ "${RC[$s]:-1}" = 0 ] || FAIL="$FAIL $s"; done
if [ -z "$FAIL" ]; then
  [ "$STEPS" = "T1 T2 T3 S1 S2 S3" ] && printf '%s\n' "job $SLURM_JOB_ID: steps $STEPS passed $(date '+%F %H:%M:%S'); smoke run runs/$NAME" > $O/smoke_PASS_$SLURM_JOB_ID
  note "smoke job PASSED ($STEPS) in $(( $(date +%s) - TJ )) s; smoke run runs/$NAME"
  exit 0
fi
note "smoke job FAILED steps:$FAIL (of: $STEPS) after $(( $(date +%s) - TJ )) s"
exit 1
