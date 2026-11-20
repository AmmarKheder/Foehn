#!/bin/bash
# WP1b: the whole scoring in ONE re-runnable command (evaluation only; nothing is fitted; nothing is published).
#   bash run_all.sh [POINTS_DIR] [OUT_DIR] [ARMS] [FOCUS]
#   sbatch run_all_job.sh [same arguments]                     (the same on a CPU node of partition small)
# POINTS_DIR  points files of the reforecasts (default: ../points)
# OUT_DIR     where tables/, numbers.json and REPORT.md are written (default: this directory)
# ARMS        arms read by build_pairs.py; an arm of another points directory is NAME=DIR:KEY, for example
#             "km,km_ring1,km_ring3,km_ring10,km_cell_lin,rt04,cams,clim,bcsd,bcsd_m,an,kmcal=/.../replay/points_cal:km"
# FOCUS       the 1 km arms to score against every baseline, first = reference (default: km; with a calibrated ensemble: km,kmcal)
# Every arm of a table is scored on identical station-days (those where all its arms are finite).
set -euo pipefail
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=$(dirname "$S")
POINTS=${1:-$R/points}
OUT=${2:-$S}
ARMS=${3:-km,km_ring1,km_ring3,km_ring10,km_cell_lin,rt04,cams,clim,bcsd,bcsd_m,an}
FOCUS=${4:-km}
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT/work" "$OUT/tables" "$OUT/logs"
echo "[run_all] $(date +%FT%T) points=$POINTS out=$OUT focus=$FOCUS"
$PY -I -B "$S/build_pairs.py" --points "$POINTS" --days "$R/days_all.txt" --out "$OUT/work" --arms "$ARMS" | tail -14
[ -f "$OUT/work/refgrade.npz" ] || $PY -I -B "$S/refgrade.py" --out "$OUT/work" > "$OUT/logs/refgrade.log"
$PY -I -B -W ignore "$S/score.py" --pairs "$OUT/work/pairs.npz" --out "$OUT" --focus "$FOCUS" --B "${B:-2000}" --Bt "${BT:-400}"
$PY -I -B "$S/make_report.py" --out "$OUT"
echo "[run_all] $(date +%FT%T) done"
