#!/bin/bash
#SBATCH --job-name=cal_sc22
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/test2022_cal/logs/%x_%j.out
# Scoring of the calibrated system's 2022 pass with the frozen system's scorers (test2022_ghap/score.sh: steps score, split,
# stscore; "truth" and "pinning" are not repeated: the extracted GHAP 2022 truth and the pinning do not depend on the system),
# + the two extra tables (stpair, compare). Real pass: started by submit_2022_cal.sh after the predictions (the lead only).
#   sbatch [--dependency=afterok:<predict job>] score_cal.sh [steps]
# Rehearsal on selection-year data (no 2022, no station; outputs of an earlier rehearsal are overwritten):
#   sbatch --export=ALL,MECH=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/test2022_cal/mech2021 score_cal.sh
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
O=$W/test2022_cal
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
MECH=${MECH:-}
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
STEPS=${1:-score,split,stscore,stpair,compare}
note() { printf '%s\n' "- $(date '+%F %H:%M') job ${SLURM_JOB_ID:-nojob} (scoring${MECH:+, rehearsal}): $1" >> $O/STATUS.md; }
run_system() {                      # $1 = extra arguments of score_cal.py ("" for the real pass)
  local X="$1" s rc wh
  for s in ${STEPS//,/ }; do
    rc=0
    case $s in
      score|split) $PY -I $O/score_cal.py $s --procs 32 $X || rc=$? ;;
      stscore) for wh in mon ring1 ring3 ring10; do $PY -I $O/score_cal.py stscore --where $wh $X || { rc=$?; break; }; done ;;
      *) continue ;;
    esac
    note "step $s $X rc=$rc"
    [ $rc = 0 ] || return $rc
  done
}
extras() {                          # the two extra tables never stop the job
  local X="$1" s rc
  for s in stpair compare; do
    case ",$STEPS," in
      *",$s,"*) rc=0; $PY -I $O/score_cal.py $s $X || rc=$?; note "extra step $s $X rc=$rc" ;;
    esac
  done
}
if [ -n "$MECH" ]; then
  export MPLCONFIGDIR=$MECH/.mpl
  $PY -I $O/mech_setup.py truth || exit 1
  for sy in frozen cal; do
    ln -sfn ../st_ring_2022.npz "$MECH/$sy/st_ring_2022.npz"
    printf '{"system": "%s (rehearsal on 2021 selection days)", "system_dir": "%s"}\n' "$sy" "$MECH/$sy" > "$MECH/$sy/SYSTEM.json"
    run_system "--mech $MECH --system $sy" || exit 1
  done
  extras "--mech $MECH --system cal"
  note "rehearsal scoring done: $MECH/{frozen,cal}/results"
  exit 0
fi
[ -f $O/GO_2022 ] || { note "no GO_2022 (the lead starts the pass with submit_2022_cal.sh): not started"; exit 1; }
export MPLCONFIGDIR=$O/results/.mpl
[ -e $O/st_ring_2022.npz ] || ln -s ../test2022_ghap/st_ring_2022.npz $O/st_ring_2022.npz
run_system "" || exit 1
extras ""
note "scoring done: results/scores_ghap_2022.md, results/st_monitor/scores_st_2022.md, results/st_ring*km, results/compare_cal_vs_frozen_2022.md"
