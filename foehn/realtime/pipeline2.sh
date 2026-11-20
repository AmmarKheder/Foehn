#!/bin/bash
# RT 1.2 data pipeline for a date range: waits for the downloads (bulk.py cams/gfas/ifs) of every day of the range,
# builds the day directories (bulk.py build), runs FLEXGREEN in batches of one node (flexgreen_batch.sh), and reports.
#   setsid nohup bash pipeline.sh 2026-06-01 2026-07-05 > logs/pipeline_jun.out 2>&1 &
cd /scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime
FIRST=$1; LAST=$2; TAG=${3:-$(echo $FIRST | tr -d -)}
unset PYTHONPATH; export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache
PY=/projappl/project_462001804/envs/world/bin/python
missing() { $PY - "$FIRST" "$LAST" <<'PYX'
import sys, datetime as dt, pathlib
f, l = (dt.date.fromisoformat(x) for x in sys.argv[1:3]); d = f; n = 0
while d <= l:
    D = pathlib.Path("data") / d.isoformat()
    ok = (D / "cams.npy").exists() and (D / "hres_B.npy").exists() and (D / "fires.npy").exists()
    ok = ok or ((D / "cams_00.zip").exists() and (D / "cams_12.zip").exists() and (D / "gfas.nc").exists() and all((D / "ifs" / f"{s:03d}.grib2").exists() for s in range(0, 193, 6)))
    n += 0 if ok else 1; d += dt.timedelta(days=1)
print(n)
PYX
}
echo "pipeline $FIRST..$LAST start $(date -u +%FT%TZ)"
while [ "$(missing)" != "0" ]; do echo "$(date -u +%FT%TZ) waiting: $(missing) days without complete inputs"; sleep 600; done
echo "inputs complete $(date -u +%FT%TZ); building"
$PY -u bulk.py build $FIRST $LAST --parallel 3 2>&1 | tail -3
# FLEXGREEN: NCHUNK nodes in parallel, each on a contiguous sub-range of the days (one GCD per day, 8 days at a time)
NCHUNK=${NCHUNK:-1}; JOBS=""
for c in $(seq 0 $((NCHUNK - 1))); do
  R=$($PY - "$FIRST" "$LAST" $c $NCHUNK <<'PYX'
import sys, datetime as dt
f, l = (dt.date.fromisoformat(x) for x in sys.argv[1:3]); c, n = int(sys.argv[3]), int(sys.argv[4])
N = (l - f).days + 1; a = f + dt.timedelta(days=c * N // n); b = f + dt.timedelta(days=(c + 1) * N // n - 1)
print(a, b)
PYX
)
  J=$(sbatch --parsable --job-name green_${TAG}_$c flexgreen_batch.sh $R); JOBS="$JOBS $J"; echo "flexgreen batch job $J ($R) submitted $(date -u +%FT%TZ)"
done
for J in $JOBS; do
  while squeue -h -j $J -o %T 2>/dev/null | grep -qE "PENDING|RUNNING|CONFIGURING|COMPLETING"; do sleep 300; done
  echo "flexgreen batch $J $(sacct -j $J --format=State -n -P | head -1) $(date -u +%FT%TZ)"
done
$PY -u bulk.py status $FIRST $LAST | awk '{g[$NF]++} END {for (k in g) print k, g[k]}'
echo "pipeline $FIRST..$LAST end $(date -u +%FT%TZ)"
