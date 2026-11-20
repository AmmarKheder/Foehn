#!/bin/bash
#SBATCH --job-name=a7_score
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/test2022_official/logs/score_%j.out
# Amendment A7, scoring side (CPU, world venv; no model code). Steps (comma list, default all in this order):
#   check     predictions against the frozen test and each other (no truth)          code/check_preds.py
#   truth     GHAP 2022 at the positions (logged opening; refuses if predictions are incomplete)   code/truth_official.py
#   score     observations + GHAP + predictions -> results/scores_official_2022.json    code/score_official.py
#   tables    results/tables_official_2022.md                                           code/report_tables.py
#   selftest  score mechanics on preds_smoke with synthetic observations (no real score)
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
O=$W/test2022_official
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache
note() { printf '%s\n' "- $(date '+%F %H:%M') job $SLURM_JOB_ID (scoring side): $1" >> $O/STATUS.md; }
STEPS=${1:-check,truth,score,tables}
for s in ${STEPS//,/ }; do
  T0=$(date +%s)
  case $s in
    check) $PY -I $O/code/check_preds.py $O/preds --tag full --against $O/preds_smoke ;;
    truth) OMP_NUM_THREADS=1 $PY -I $O/code/truth_official.py --procs 32 ;;
    score) $PY -I $O/code/score_official.py ;;
    tables) $PY -I $O/code/report_tables.py ;;
    selftest) $PY -I $O/code/score_official.py --selftest --preds $O/preds_smoke --min-days 3 && $PY -I $O/code/report_tables.py --selftest ;;
    papercheck) $PY -I $O/code/score_official.py --only-paper-checks ;;
    *) echo "unknown step $s"; exit 2 ;;
  esac
  rc=$?
  note "step $s rc=$rc ($(( $(date +%s) - T0 )) s)"
  [ $rc = 0 ] || exit $rc
done
