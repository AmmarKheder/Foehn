#!/bin/bash
#SBATCH --job-name=sc_obsgap
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay/score/logs/obs_gap_%j.out
# Evaluation only (WP1b): fetch the OpenAQ archive days of the gap 2026-10-22..10-27 for the scoring of the reforecasts.
# ONLY 'verif.py obs' is run: no score, no publish. '--today 2026-10-29' fetches the local-day files 2026-10-19..10-26
# (10-27 and 10-28 are on disk and closed) and writes the UTC daily means of 2026-10-20..10-27: no existing file of
# verif/obs (2026-10-28 ..) is rewritten; 10-20 and 10-21 are two extra days that overlap the station-set truth file.
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution; V=$W/realtime_1km/verif
unset PYTHONPATH; export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache PYTHONDONTWRITEBYTECODE=1
PY=/projappl/project_462001804/envs/world/bin/python
cd $V
exec 9>$V/.lock; flock -w 600 9 || { echo "lock busy"; exit 1; }
echo "start $(date -u +%FT%TZ)"
ls obs | head -3
$PY -u verif.py obs --today 2026-10-29 || echo "obs 10-29 FAILED"
$PY - <<'PYEOF'
import numpy as np, glob, os
for f in sorted(glob.glob('obs/*.npz'))[:12]:
    z = np.load(f); print(os.path.basename(f), z['id'].size, bool(z['final']))
PYEOF
echo "end $(date -u +%FT%TZ)"
