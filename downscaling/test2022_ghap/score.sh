#!/bin/bash
#SBATCH --job-name=ghap_sc22
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/test2022_ghap/logs/sc22_%j.out
# PROTOCOL.md A2(d) step (b): GHAP 2022 truth (logged opening), GHAP-subsample scores, station scores, figures
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
O=$W/test2022_ghap
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=$O/results/.mpl
note() { printf '%s\n' "- $(date '+%F %H:%M') [ghap] job $SLURM_JOB_ID (2022 scoring): $1" >> $W/logs/GLOBAL_PREP_STATUS.md; }
STEPS=${1:-truth,score,split,pinning,stscore,figs}
for s in ${STEPS//,/ }; do
  case $s in
    truth|score|split|pinning) $PY -I $O/score_2022.py $s --procs 32 ;;
    stscore) GHAP_ST_WHERE="monitor pixel" $PY -I $O/score_st_2022.py 2022 --preds $O/preds_st_mon --out $O/results/st_monitor && \
             for d in 1 3 10; do GHAP_ST_WHERE="ring of ${d} km around the monitor pixel (Chebyshev distance $d, monitor pixel excluded, A2(e))" \
               GHAP_ST_TAG="_ring${d}km" $PY -I $O/score_st_2022.py 2022 --preds $O/preds_ring$d --out $O/results/st_ring${d}km || break; done ;;
    figs) $PY -I $O/fig_world_ghap.py && $PY -I $O/fig_day_ghap.py ${FIGDAY:-2022-11-18} ;;
  esac
  rc=$?
  note "step $s rc=$rc"
  [ $rc = 0 ] || exit $rc
done
