#!/bin/bash
# fig_rt_skill on a login node (CPU only, about 1 min in all); usage: ./render.sh [--data [options of rt_skill_data.py]] [tag]
#   ./render.sh                       draws fig_rt_skill.pdf / .png from numbers.json and data/rt_skill.npz (15 s)
#   ./render.sh --data                first rebuilds numbers.json and data/ from the scorer's tables (30-45 s, about 3 GB)
#   ./render.sh --data --score DIR --arm2 kmcal --name2 "Calibrated" --tag cal
#                                     a second ensemble in panel f -> numbers_cal.json, data/rt_skill_cal.npz, fig_rt_skill_cal.*
cd "$(dirname "$0")"
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl-$USER-rtskill
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$MPLCONFIGDIR" data
PY=/projappl/project_462001804/envs/world/bin/python
TAG=""
if [ "$1" = "--data" ]; then
  shift
  ARGS=("$@")
  while [ $# -gt 0 ]; do [ "$1" = "--tag" ] && TAG="$2"; shift; done
  set -o pipefail
  $PY -B -W ignore rt_skill_data.py "${ARGS[@]}" 2>&1 | grep -v -i telemetry | tee "data/log_data${TAG:+_$TAG}.txt" || exit 1
else
  TAG="$1"
fi
$PY -B fig_rt_skill.py $TAG 2>&1 | grep -v -e findfont -e telemetry
