#!/bin/bash
# renders the figure on a login node (light: the world map is drawn from every 5th 1 km pixel); usage: ./render.sh [variant]
cd "$(dirname "$0")"
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl-$USER-rtday
mkdir -p "$MPLCONFIGDIR"
PY=/projappl/project_462001804/envs/world/bin/python
[ "$1" = "--data" ] && { shift; $PY -B rt_day_data.py "$@" || exit 1; }
$PY -B fig_rt_day.py "$@" 2>&1 | grep -v findfont
