#!/bin/bash
#SBATCH --job-name=bf_flexgreen
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/bf_flexgreen_%j.out
# Backfill: FLEXGREEN ERA5 lead 1 for the start days 2025-01-01 .. (last era5_B day - 8, and <= last fires day), as
# jobs/global_flexgreen.sh (8 ranks share the days); fires of 2025+ from the injh store (--fires-2025). Resumable (done markers).
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/GLOBAL_PREP_STATUS.md
source $W/env.sh
O=$W/data/processed/global/flexgreen
PY=/projappl/project_462001804/envs/world/bin/python
note() { printf '%s\n' "- $(date '+%F %H:%M') [backfill] job $SLURM_JOB_ID: $1" >> "$S"; }
# eligible start days per year (python: last era5_B day in wb2, last complete fires day)
eval $($PY -I - <<'PY'
import datetime as dt, json, glob, os
W = "/scratch/project_462001804/ammar/global_model_forecasting_air_pollution"
last_b = None
for f in sorted(glob.glob(W + "/data/processed/global/wb2/era5_B_20*_*.npy")):
    y, m = int(os.path.basename(f)[7:11]), int(os.path.basename(f)[12:14])
    import numpy as np
    n = np.load(f, mmap_mode="r").shape[0]
    d = dt.date(y, m, 1) + dt.timedelta(n - 1)
    if last_b is None or d > last_b: last_b = d
st = json.load(open(W + "/data/processed/global/backfill/state_fires.json"))
last_f = dt.date.fromisoformat(st["last_complete_day"])
last_s = min(last_b - dt.timedelta(8), last_f)
days = {}
for y in (2025, 2026):
    nd = 366 if y % 4 == 0 else 365
    days[y] = [k for k in range(nd) if dt.date(y, 1, 1) + dt.timedelta(k) <= last_s]
print(f'LAST_B={last_b}; LAST_F={last_f}; LAST_S={last_s}; D2025="{" ".join(map(str, days[2025]))}"; D2026="{" ".join(map(str, days[2026]))}"; N2025={len(days[2025])}; N2026={len(days[2026])}')
PY
)
note "FLEXGREEN 2025-2026 started (standard-g, 8 GCDs): start days to $LAST_S (era5_B to $LAST_B, fires to $LAST_F): 2025 $N2025 days, 2026 $N2026 days"
rc=0
for Y in 2025 2026; do
  if [ $Y = 2025 ]; then DAYS="$D2025"; else DAYS="$D2026"; fi
  [ -n "$DAYS" ] || continue
  echo "=== $Y $(date +%T)"
  srun --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B $CDFFM_BIND $CDFFM_SIF bash -c "source $CDFFM_ENVSH; cd $W/transport; \
    python -u flexgreen_pm.py run --src era5 --year $Y --days $DAYS --max-lead 1 --out-root $O \
      --wb-extra $W/data/processed/global/wb2 --fires-new $W/data/processed/global/coarse/fires_2023_2024.npy \
      --fires-2025 $W/data/processed/global/coarse/fires_injh_2025_2026.npy" || rc=1
  n=$(ls $O/era5/done_$Y 2>/dev/null | wc -l)
  echo "=== $Y done markers: $n $(date +%T)"
done
if [ $rc = 0 ]; then
  for Y in 2025 2026; do        # markers folded into done_<y>.txt (inode budget); build_global_t6 reads either
    [ -d $O/era5/done_$Y ] || continue
    ( ls $O/era5/done_$Y | sed 's/^0*//; s/^$/0/'; cat $O/era5/done_$Y.txt 2>/dev/null; true ) | sort -n | uniq > $O/era5/done_$Y.tmp && mv $O/era5/done_$Y.tmp $O/era5/done_$Y.txt && rm -r $O/era5/done_$Y
  done
  note "FLEXGREEN DONE: start days 2025-01-01 .. $LAST_S (2025: $(wc -l < $O/era5/done_2025.txt), 2026: $(wc -l < $O/era5/done_2026.txt) days), origins_2025/2026.npy in data/processed/global/flexgreen/era5/"
else
  note "FLEXGREEN incomplete (rc=$rc), logs/bf_flexgreen_$SLURM_JOB_ID.out; resubmit resumes from the done markers"
fi
exit $rc
