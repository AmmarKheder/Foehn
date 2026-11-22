#!/bin/bash
#SBATCH --job-name=ftM_smoke
#SBATCH --account=<project>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH --output=logs/smoke_%j.out
# THE ONE GPU smoke of the final test of the monitor-fed layer (draft Addendum 8 of A11): 1 GCD, limit 40 min (<= 0.67 GCD-h).
# No observation and no held-out value is read by any step (audit-hook firewall in every process).
#   G  test_final.py gpu      test (i) on 2021 (M with stations off = its 'nostations' validation; features; Sampler;
#                             the stored 2021 table of the selected checkpoint reproduced on the 24 SEL days)
#   S  infer_m.py, one issue day end to end at the 12,136 monitor pixels, every arm (free, m1, m2) -> out/smoke_<job>.npz
#   L  infer_m.py, the same day with the inputs built from copies whose held-out rows are noise (out/inputs_leak.npz,
#      made by 'test_final.py cpu --keep-leak' on the login node) -> forecast leak test
#   C  test_final.py check-smoke: identity of the station-free arm with the record (ii), leak test of the forecasts (iii-b),
#      guard, distances, and the PRICE of the full run -> out/smoke_check.json
#   sbatch jobs/smoke_job.sh [ISSUE_DAY (default 2026-10-20)]
set -uo pipefail
W=${FOEHN_WORK}/global_model_forecasting_air_pollution
FT=$W/a11/stageM/final_test
DAY=${1:-2026-10-20}
J=$SLURM_JOB_ID
source $W/env.sh
export PYTHONDONTWRITEBYTECODE=1
trap 'rm -rf "/tmp/${USER}-cdffm-$J"' EXIT
run1() { srun --ntasks=1 --cpus-per-task=7 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" \
           bash -c "source $CDFFM_ENVSH; export HIP_VISIBLE_DEVICES=\$SLURM_LOCALID OMP_NUM_THREADS=7 PYTHONDONTWRITEBYTECODE=1; cd $FT; $*"; }
T0=$(date +%s)
echo "[smoke] start $(date -u +%FT%TZ) node $SLURMD_NODENAME day $DAY"
[ -f $FT/out/inputs_leak.npz ] || { echo "[smoke] out/inputs_leak.npz missing: run 'test_final.py cpu --keep-leak out/inputs_leak.npz' first"; exit 2; }
run1 "python -u test_final.py gpu"; RG=$?
echo "[smoke] G rc=$RG at +$(( $(date +%s) - T0 )) s"
run1 "python -u infer_m.py --days $DAY --arms free,m1,m2 --keep-free --out $FT/out/smoke_$J.npz --trace $FT/out/trace_smoke_$J.json"; RS=$?
echo "[smoke] S rc=$RS at +$(( $(date +%s) - T0 )) s"
run1 "python -u infer_m.py --days $DAY --arms m1,m2 --inputs $FT/out/inputs_leak.npz --out /tmp/${USER}-cdffm-$J/smoke_leak.npz"; RL=$?
echo "[smoke] L rc=$RL at +$(( $(date +%s) - T0 )) s"
run1 "python -u test_final.py check-smoke --smoke $FT/out/smoke_$J.npz --leak /tmp/${USER}-cdffm-$J/smoke_leak.npz --out $FT/out/smoke_check_$J.json"; RC=$?
echo "[smoke] C rc=$RC at +$(( $(date +%s) - T0 )) s"
[ $RL = 0 ] && [ $RC = 0 ] && rm -f $FT/out/inputs_leak.npz
echo "[smoke] end $(date -u +%FT%TZ) rc G=$RG S=$RS L=$RL C=$RC wall $(( $(date +%s) - T0 )) s"
[ $RG = 0 ] && [ $RS = 0 ] && [ $RL = 0 ] && [ $RC = 0 ]
