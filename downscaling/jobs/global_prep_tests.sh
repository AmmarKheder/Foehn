#!/bin/bash
#SBATCH --job-name=gp_tests
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/global_prep_tests_%j.out
# Non-regression after the --store global wiring (CPU, 0 GPU-h):
#   test_x all; test_s0 shapes liif dateline train ddp store (full); test_s0 noise --quick (the full noise check
#   rewrites data/processed/smoke/noise_fit_smoke.json, which running jobs read: not touched);
#   scripts/test_global_wiring.py (store, firewall, batch, cli [, hcf]).
#   sbatch jobs/global_prep_tests.sh [all|wiring|hcf]
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/GLOBAL_PREP_STATUS.md
note() { printf '%s\n' "- $(date '+%F %H:%M') [global-prep] job $SLURM_JOB_ID: $1" >> "$S"; }
source $W/env.sh
export PYTHONDONTWRITEBYTECODE=1
cd $W
WHAT=${1:-all}
TMPD=/tmp/${USER}-gp-tests-$SLURM_JOB_ID
mkdir -p $TMPD
rc=0; msg=""
if [ $WHAT = all ]; then
  cdffm_run python -u -m cdffm_aq_x.test_x all --tmp $TMPD/x && msg="$msg test_x PASS;" || { rc=1; msg="$msg test_x FAIL;"; }
  cdffm_run python -u -m cdffm_aq.test_s0 shapes liif dateline train ddp store --tmp $TMPD/s0 \
     && msg="$msg test_s0 PASS;" || { rc=1; msg="$msg test_s0 FAIL;"; }
  cdffm_run python -u -m cdffm_aq.test_s0 noise --quick --tmp $TMPD/s0n \
     && msg="$msg test_s0 noise(quick) PASS;" || { rc=1; msg="$msg test_s0 noise FAIL;"; }
fi
if [ $WHAT = all ] || [ $WHAT = wiring ]; then
  cdffm_run python -u scripts/test_global_wiring.py store firewall batch cli --tmp $TMPD/g \
     && msg="$msg global wiring PASS;" || { rc=1; msg="$msg global wiring FAIL;"; }
fi
if [ $WHAT = hcf ]; then
  cdffm_run python -u scripts/test_global_wiring.py hcf --tmp $TMPD/h \
     && msg="$msg global hcf CLI PASS;" || { rc=1; msg="$msg global hcf CLI FAIL;"; }
fi
rm -rf $TMPD
note "tests ($WHAT):$msg log logs/global_prep_tests_$SLURM_JOB_ID.out"
exit $rc
