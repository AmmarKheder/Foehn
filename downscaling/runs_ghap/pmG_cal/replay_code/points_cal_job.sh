#!/bin/bash
#SBATCH --job-name=cal_points
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_cal/logs/%x_%j.out
# PROTOCOL.md A6(b): the reforecasts in real-time mode at the monitor pixels with the calibrated sampler.
# Reads realtime_1km/replay/{work,positions*.npz,points,km,raw,code} (never written), writes runs_ghap/pmG_cal/replay_out/ and,
# in mode run, realtime_1km/replay/points_cal/<d>.npz (the directory of the calibrated reforecasts). No observation is opened.
#   sbatch points_cal_job.sh proof <list>     the copy with the frozen settings (tau 1) on the days of the list ->
#                                             replay_out/proof, compared bit for bit with replay/km, replay/raw and
#                                             replay/points; then ONE day with TAU/KNEE as a smoke test (replay_out/smoke_$TAG)
#   sbatch points_cal_job.sh pair <list>      per day and GCD: the ORIGINAL replay/code/infer_points.py (outputs redirected
#                                             to replay_out/pair/orig), then the copy with tau 1 (replay_out/pair/copy) in
#                                             the same MIOpen cache state; copy vs original and original vs existing files
#   sbatch --export=ALL,TAU=1.60,KNEE=1.0,TAG=TT points_cal_job.sh run <list>
#                                             the calibrated members of every day of the list -> replay_out/km_$TAG, then
#                                             assemble_cal.py -> realtime_1km/replay/points_cal/
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
C=$W/runs_ghap/pmG_cal
RC=$C/replay_code
S=$C/STATUS.md
PYW=/projappl/project_462001804/envs/world/bin/python
MODE=$1; LIST=$2
TAU=${TAU:-1.60}; KNEE=${KNEE:-1.0}; TAG=${TAG:-TT}
source $W/env.sh
note() { printf '%s\n' "- $(date '+%F %H:%M') job $SLURM_JOB_ID ($SLURM_JOB_NAME, $SLURM_JOB_PARTITION): $1" >> "$S"; }
T0=$(date +%s)
el() { echo $(( $(date +%s) - T0 )); }
run8() { srun --ntasks=8 --cpus-per-task=7 --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash $RC/points_rank.sh "$@"; }
ND=$(grep -cv '^[[:space:]]*$' $LIST)
echo "start $(date) node=$SLURMD_NODENAME mode=$MODE list=$LIST ($ND days) tau=$TAU knee=$KNEE tag=$TAG"
rc=0
if [ "$MODE" = proof ]; then
  O=$C/replay_out/proof
  rm -rf "${O:?}"; mkdir -p $O/km $O/raw
  run8 $LIST $O/km $O/raw 1.0 none || rc=1
  echo "proof inference done after $(el) s rc=$rc"
  $PYW -I $RC/compare_points.py --mine $O --list $LIST --out $C/results/replay_proof.json || rc=1
  note "replay proof (copy, frozen settings, $ND days): $(tail -1 $C/results/replay_proof.txt 2>/dev/null) [$(el) s]"
  rm -rf "${O:?}/raw"                                               # compared; 130 MB per day not kept
  D1=$(grep -v '^[[:space:]]*$' $LIST | sort | head -1)
  SM=$C/replay_out/smoke_$TAG; rm -rf "${SM:?}"; mkdir -p $SM
  echo $D1 > $SM/list.txt
  t1=$(el); rs=0
  srun --ntasks=1 --cpus-per-task=7 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash $RC/points_rank.sh $SM/list.txt $SM/km none $TAU $KNEE || rs=1
  note "smoke of the calibrated path (tau $TAU knee $KNEE) on $D1: rc=$rs, $(( $(el) - t1 )) s on one GCD -> replay_out/smoke_$TAG/km"
  [ $rs = 0 ] || rc=1
elif [ "$MODE" = pair ]; then
  O=$C/replay_out/pair
  rm -rf "${O:?}"; mkdir -p $O/orig/km $O/orig/raw $O/copy/km $O/copy/raw
  run8 $LIST $O 0 1.0 none pair || rc=1
  echo "pair inference done after $(el) s rc=$rc"
  r1=0; r2=0
  $PYW -I $RC/compare_points.py --mine $O/copy --ref $O/orig --list $LIST --out $C/results/replay_pair_copy_vs_original.json || r1=1
  $PYW -I $RC/compare_points.py --mine $O/orig --list $LIST --out $C/results/replay_pair_original_vs_existing.json || r2=1
  note "pair proof ($ND days, per-GCD MIOpen database): COPY (tau 1) vs ORIGINAL code rerun in the same cache state: $(tail -1 $C/results/replay_pair_copy_vs_original.txt) | ORIGINAL code rerun vs the existing replay files: $(tail -1 $C/results/replay_pair_original_vs_existing.txt) [$(el) s]"
  rm -rf "${O:?}/orig/raw" "${O:?}/copy/raw"
  [ $r1 = 0 ] || rc=1
else
  O=$C/replay_out/km_$TAG
  mkdir -p $O
  run8 $LIST $O none $TAU $KNEE || rc=1
  echo "inference done after $(el) s rc=$rc; files: $(ls $O/*.npz 2>/dev/null | wc -l)"
  $PYW -I $RC/assemble_cal.py --km $O --list $LIST --tag $TAG || rc=1
  note "calibrated reforecasts (tau $TAU knee $KNEE, tag $TAG): $(ls $O/*.npz 2>/dev/null | wc -l) km files, $(ls $W/realtime_1km/replay/points_cal/*.npz 2>/dev/null | wc -l) files in realtime_1km/replay/points_cal, rc=$rc, $(el) s of one node"
fi
rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
echo "end $(date) rc=$rc wall $(el) s"
exit $rc
