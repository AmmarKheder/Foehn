#!/bin/bash
#SBATCH --job-name=ftM_dry
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/dry_%j.out
# CPU DRY RUN of infer_m.py (draft Addendum 8 of A11), 0 GCD-hours, the job container on a CPU node:
#   R  the full-run guard: 'infer_m.py --days all' with a non-frozen configuration must be refused before anything loads;
#   D  one issue day (2026-10-20), 2 members, lead days 1 and 5, every arm (free, free0, m1, m2) on the CPU -> node /tmp;
#   C  test_final.py check-dry -> out/dry_cpu_<job>.json (free0 runs, is finite, shares the base noise of m1 / m2).
# No observation and no held-out value is read (audit-hook firewalls of infer_m.py and test_final.py).
#   sbatch jobs/dry_cpu_job.sh
set -uo pipefail
W=${FOEHN_WORK}/global_model_forecasting_air_pollution
FT=$W/a11/stageM/final_test
J=$SLURM_JOB_ID
source $W/env.sh
export PYTHONDONTWRITEBYTECODE=1
TMPD=/tmp/${USER}-cdffm-$J
mkdir -p $TMPD
trap 'rm -rf "$TMPD"' EXIT
run1() { singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" \
           bash -c "source $CDFFM_ENVSH; export OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1; cd $FT; $*"; }
T0=$(date +%s)
echo "[dry] start $(date -u +%FT%TZ) node $SLURMD_NODENAME"
run1 "python -u infer_m.py --days all --arms m1,m2 --out $TMPD/refused.npz" > $TMPD/refused.log 2>&1; RR=$?
cat $TMPD/refused.log
RROK=1; grep -q "REFUSED" $TMPD/refused.log && [ $RR != 0 ] && [ ! -e $TMPD/refused.npz ] && RROK=0
echo "[dry] R rc=$RR refused_ok=$RROK at +$(( $(date +%s) - T0 )) s"
run1 "python -u infer_m.py --days 2026-10-20 --arms free,free0,m1,m2 --keep-free --members 2 --leads 1,5 --out $TMPD/dry.npz --trace $TMPD/trace.json"; RD=$?
echo "[dry] D rc=$RD at +$(( $(date +%s) - T0 )) s"
run1 "python -u test_final.py check-dry --dry $TMPD/dry.npz --refused-ok $RROK --out $FT/out/dry_cpu_$J.json"; RC=$?
echo "[dry] C rc=$RC at +$(( $(date +%s) - T0 )) s"
echo "[dry] end $(date -u +%FT%TZ) rc R=$RROK D=$RD C=$RC wall $(( $(date +%s) - T0 )) s"
[ $RROK = 0 ] && [ $RD = 0 ] && [ $RC = 0 ]
