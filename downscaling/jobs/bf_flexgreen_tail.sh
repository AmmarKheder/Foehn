#!/bin/bash
#SBATCH --job-name=bf_fg_tail
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/bf_fg_tail_%j.out
# Backfill: the last start days (era5_B day - 8 < s <= era5_B day - 2) with --era5-days 3 (lead 1 needs days d .. d+2 only).
# Gate: 2024-12-20 recomputed with --era5-days 3 must match the stored origins (numerically equivalent); otherwise the
# tail is skipped (those days then belong to the real-time chain). Writes flexgreen/era5/tail_2026.json.
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/GLOBAL_PREP_STATUS.md
source $W/env.sh
O=$W/data/processed/global/flexgreen
T=$W/data/processed/global/flexgreen_tailcheck
PY=/projappl/project_462001804/envs/world/bin/python
note() { printf '%s\n' "- $(date '+%F %H:%M') [backfill] job $SLURM_JOB_ID: $1" >> "$S"; }
rm -rf $T; mkdir -p $T
# gate: one rank, one day (2024 doy 354 = 2024-12-20; its era5_B days are in wb2)
srun -n 1 --ntasks-per-node=1 --cpus-per-task=7 --gpus-per-node=8 singularity exec -B $CDFFM_BIND $CDFFM_SIF bash -c "source $CDFFM_ENVSH; cd $W/transport; \
  python -u flexgreen_pm.py run --src era5 --year 2024 --days 354 --max-lead 1 --out-root $T --era5-days 3 \
    --wb-extra $W/data/processed/global/wb2 --fires-new $W/data/processed/global/coarse/fires_2023_2024.npy"
GATE=$(singularity exec -B $CDFFM_BIND $CDFFM_SIF bash -c "source $CDFFM_ENVSH; python - <<'PY'
import numpy as np, json
W = '/scratch/project_462001804/ammar/global_model_forecasting_air_pollution'
def blk(path, i, shape, dt):
    with open(path, 'rb') as f:
        np.lib.format.read_magic(f); sh, _, _ = np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    n = int(np.prod(sh[1:])); return np.fromfile(path, dtype=dt, count=n, offset=hdr + i * n * np.dtype(dt).itemsize).reshape(sh[1:])
a = blk(W + '/data/processed/global/flexgreen/era5/origins_2024.npy', 354, None, np.float16).astype(np.float32)[0]
b = blk(W + '/data/processed/global/flexgreen_tailcheck/era5/origins_2024.npy', 354, None, np.float16).astype(np.float32)[0]
fa = blk(W + '/data/processed/global/flexgreen/era5/fire_2024.npy', 354, None, np.float32)[0]
fb = blk(W + '/data/processed/global/flexgreen_tailcheck/era5/fire_2024.npy', 354, None, np.float32)[0]
d = np.abs(a - b); d[..., 0] = np.minimum(d[..., 0], 360 - d[..., 0])
q = {k: float(np.nanpercentile(d[..., i], 99)) for i, k in enumerate(('lon', 'lat', 'zh', 'surv'))}
bit = float((a.view(np.uint32) == b.view(np.uint32)).mean())
rel = float(np.abs(fa - fb).sum() / max(np.abs(fa).sum(), 1e-30))
ok = q['lon'] <= 0.25 and q['lat'] <= 0.25 and q['zh'] <= 0.05 and q['surv'] <= 0.02 and rel <= 1e-2
r = {'day': '2024-12-20', 'p99_abs_diff': q, 'frac_bit_identical': bit, 'fire_terms_rel_l1_diff': rel, 'ok': bool(ok)}
json.dump(r, open(W + '/data/processed/global/backfill/tail_gate.json', 'w'), indent=1); print(json.dumps(r))
PY")
echo "gate: $GATE"
rm -rf $T
if ! echo "$GATE" | grep -q '"ok": true'; then
  note "FLEXGREEN tail SKIPPED: --era5-days 3 gate failed on 2024-12-20 ($GATE); the last days stay with the real-time chain"
  exit 0
fi
eval $($PY -I - <<'PY'
import datetime as dt, json, glob, os, numpy as np
W = "/scratch/project_462001804/ammar/global_model_forecasting_air_pollution"
last_b = None
for f in sorted(glob.glob(W + "/data/processed/global/wb2/era5_B_20*_*.npy")):
    y, m = int(os.path.basename(f)[7:11]), int(os.path.basename(f)[12:14])
    d = dt.date(y, m, 1) + dt.timedelta(np.load(f, mmap_mode="r").shape[0] - 1)
    if last_b is None or d > last_b: last_b = d
last_f = dt.date.fromisoformat(json.load(open(W + "/data/processed/global/backfill/state_fires.json"))["last_complete_day"])
lo, hi = last_b - dt.timedelta(8), min(last_b - dt.timedelta(2), last_f)
days = [k for k in range(366) if lo < dt.date(2026, 1, 1) + dt.timedelta(k) <= hi]
print(f'LO={lo}; HI={hi}; DAYS="{" ".join(map(str, days))}"')
PY
)
echo "tail start days $LO < s <= $HI: $DAYS"
[ -n "$DAYS" ] || { note "FLEXGREEN tail: no day to add"; exit 0; }
srun --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B $CDFFM_BIND $CDFFM_SIF bash -c "source $CDFFM_ENVSH; cd $W/transport; \
  python -u flexgreen_pm.py run --src era5 --year 2026 --days $DAYS --max-lead 1 --out-root $O --era5-days 3 \
    --wb-extra $W/data/processed/global/wb2 --fires-new $W/data/processed/global/coarse/fires_2023_2024.npy \
    --fires-2025 $W/data/processed/global/coarse/fires_injh_2025_2026.npy" || { note "FLEXGREEN tail FAILED"; exit 1; }
( ls $O/era5/done_2026 2>/dev/null | sed 's/^0*//; s/^$/0/'; cat $O/era5/done_2026.txt 2>/dev/null; true ) | sort -n | uniq > $O/era5/done_2026.tmp && mv $O/era5/done_2026.tmp $O/era5/done_2026.txt && rm -rf $O/era5/done_2026
$PY -I -c "import json; json.dump({'start_days_doy_2026': [int(x) for x in '$DAYS'.split()], 'first': '$LO', 'last': '$HI', 'gate': json.load(open('$W/data/processed/global/backfill/tail_gate.json'))}, open('$O/era5/tail_2026.json', 'w'), indent=1)"
note "FLEXGREEN tail DONE: start days $LO < s <= $HI run with --era5-days 3 (gate on 2024-12-20 passed: $GATE)"
