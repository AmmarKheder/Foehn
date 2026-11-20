#!/bin/bash
#SBATCH --job-name=a8_post
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=03:15:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_seed1/a8/logs/post_%j.out
# protocol_amendments/A8_seed_replicates.md (c): selection, prior refit, SEL evaluation and gates of the seed replicates:
# the five steps of runs_ghap/p1r_G.sh for each run of $RUNS, on ONE node, interleaved so that the GCDs are not idle:
#   latents(run 1) | latents(run 2) + [fit, test](run 1) | eval(run 1) + [fit, test](run 2) | eval(run 2)
# (the latents are CPU-bound, the prior fit and test are one process on one GCD started from the batch step, the
# evaluation uses the eight GCDs). Every command, option, seed and day list is p1r_G.sh's; only the timing differs.
#   sbatch [--dependency=afterok:<train1>:<train2>] --export=ALL,RUNS="pmG_seed1 pmG_seed2" post_job.sh
set -uo pipefail
source /scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_seed1/a8/a8_lib.sh
RUNS=${RUNS:-"pmG_seed1 pmG_seed2"}
export A8_WHAT="post"
export A8_NOTES=$A8/logs/notes_post.md
export A8_MEMLOG=$A8/logs/mem_post_$SLURM_JOB_ID.txt
FTGCD=${A8_FT_GCD:-7}
TJ=$(date +%s)
memmon_start "$A8_MEMLOG"
trap 'memmon_stop; rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"' EXIT
read -r -a ARR <<< "$RUNS"
N=${#ARR[@]}
note "job started on $SLURMD_NODENAME: runs ${ARR[*]} (fit/test mode ${A8_FT_MODE:-batch}, sequential latents ${A8_SEQ_LATENTS:-0})"
declare -A BAD FTPID
# reference checks on the FROZEN system (a8_lib.sh ref_day, val_noise; A8 Addendum 1 and 2): background processes on GCDs
# 5 and 6 while the latents run; they never block the replicates and their failure does not fail the job
REFPIDS=""
if [ -n "${A8_REF_DAY:-}" ]; then
  ( ref_day 6 $A8_REF_DAY 0 ) > $A8/logs/post_${SLURM_JOB_ID}_refday_es0.out 2>&1 &
  REFPIDS="$REFPIDS $!"
  ( ref_day 5 $A8_REF_DAY 1; val_noise 5 18000,20000,22000,24000 1,2,3 ) > $A8/logs/post_${SLURM_JOB_ID}_refday_es1.out 2>&1 &
  REFPIDS="$REFPIDS $!"
  note "reference checks on the frozen system started in the background (GCD 6: day $A8_REF_DAY, evaluation seed 0; GCD 5: day $A8_REF_DAY, evaluation seed 1, then validation-noise seeds)"
fi
for R in "${ARR[@]}"; do
  case "$R" in pmG_seed1|pmG_seed2) ;; *) note "refusing run '$R' (A8 handles pmG_seed1 and pmG_seed2 only)"; exit 2 ;; esac
  BAD[$R]=0
  [ "$(last_step $RGH/$R)" -ge 24000 ] || { note "$R: training not complete (last validated step $(last_step $RGH/$R))"; BAD[$R]=1; continue; }
  pc_select $R $RGH || BAD[$R]=1
done

finish_ft() {   # wait for the background fit + test of run $1, print its log
  local R=$1
  [ -n "${FTPID[$R]:-}" ] || return 0
  wait "${FTPID[$R]}" || BAD[$R]=1
  echo "----- fit + test of $R (background log) -----"; cat $A8/logs/post_${SLURM_JOB_ID}_fit_test_$R.out; echo "-----"
}
PREV=""
for R in "${ARR[@]}"; do
  if [ "${BAD[$R]}" = 0 ]; then pc_latents $R $RGH 0 || BAD[$R]=1; fi            # while PREV's fit + test run
  note "$R: latents phase over at +$(( $(date +%s) - TJ )) s (bad=${BAD[$R]})"
  [ -n "$PREV" ] && finish_ft $PREV
  if [ "${BAD[$R]}" = 0 ]; then
    ( pc_fit_test $R $RGH 0 $FTGCD ) > $A8/logs/post_${SLURM_JOB_ID}_fit_test_$R.out 2>&1 &
    FTPID[$R]=$!
  fi
  if [ -n "$PREV" ] && [ "${BAD[$PREV]}" = 0 ]; then                              # while R's fit + test run
    pc_eval_gates $PREV $RGH $RGH/$PREV/final 0 || BAD[$PREV]=1
    note "$PREV: evaluation phase over at +$(( $(date +%s) - TJ )) s (bad=${BAD[$PREV]})"
  fi
  PREV=$R
done
finish_ft $PREV
if [ "${BAD[$PREV]}" = 0 ]; then
  pc_eval_gates $PREV $RGH $RGH/$PREV/final 0 || BAD[$PREV]=1
  note "$PREV: evaluation phase over at +$(( $(date +%s) - TJ )) s (bad=${BAD[$PREV]})"
fi
if [ -n "$REFPIDS" ]; then
  for P in $REFPIDS; do wait $P; done
  for F in $A8/logs/post_${SLURM_JOB_ID}_refday_es0.out $A8/logs/post_${SLURM_JOB_ID}_refday_es1.out; do
    echo "----- $F (last lines) -----"; tail -n 22 $F | cut -c1-600; echo "-----"
  done
  python3 $A8/refday_compare.py 2>&1 | tee $A8/logs/refday_compare_$SLURM_JOB_ID.out
fi
rc=0
for R in "${ARR[@]}"; do [ "${BAD[$R]}" = 0 ] || rc=1; done
note "post job ended after $(( $(date +%s) - TJ )) s: $(for R in "${ARR[@]}"; do printf '%s bad=%s ' $R ${BAD[$R]}; done)"
# side-by-side report of seeds 0, 1, 2 (CPU, small json / npz files only); also run by hand at the end
if [ -f $A8/compare_seeds.py ]; then
  $PYW -I $A8/compare_seeds.py > $A8/logs/compare_seeds_$SLURM_JOB_ID.out 2>&1 && note "compare_seeds.py written to a8/results" \
    || note "compare_seeds.py failed (see a8/logs/compare_seeds_$SLURM_JOB_ID.out)"
fi
sacct -j $SLURM_JOB_ID --format=JobID%22,JobName%14,Elapsed,MaxRSS,AveRSS,State -P 2>/dev/null | tail -n 30
echo "end $(date) rc=$rc"
exit $rc
