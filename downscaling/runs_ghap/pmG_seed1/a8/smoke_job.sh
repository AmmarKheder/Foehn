#!/bin/bash
#SBATCH --job-name=a8_smoke
#SBATCH --account=project_462001804
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_seed1/a8/logs/smoke_%j.out
# protocol_amendments/A8_seed_replicates.md (e): the checks before the full runs, one node, about 12 min.
#   S1  200 steps of the recipe with seed 0 through the A8 launcher (train_seg) -> a8/smoke/s0_200; compared with the
#       first lines of the frozen log by a8/compare_smoke.py
#   S2  reference reproduction of the FROZEN seed 0 with today's code (read-only on pmG_base):
#       valrepro.py (in-training validation of step 22,000) on rank 0, psdrepro.py (spectra boxes, 8 days) on 8 ranks
#   S3  the selection chain on the 200-step checkpoint with reduced sizes (launcher test only; its numbers mean
#       nothing): select, both latent passes at once (memory of that layout is measured), prior fit + test started
#       from the batch step WHILE S2's 8-task step runs (the overlap the full job uses), evaluation, gates
#   sbatch -p dev-g smoke_job.sh        (or -p standard-g when the single dev-g slot of the account is taken)
set -uo pipefail
source /scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_seed1/a8/a8_lib.sh
export A8_WHAT="smoke"
export A8_NOTES=$A8/logs/notes_smoke.md
export A8_MEMLOG=$A8/logs/mem_smoke_$SLURM_JOB_ID.txt
SM=$A8/smoke
RES=$A8/results
mkdir -p $SM $RES
TJ=$(date +%s)
memmon_start "$A8_MEMLOG"
trap 'memmon_stop; rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"' EXIT
note "smoke job started on $SLURMD_NODENAME (partition $SLURM_JOB_PARTITION)"
echo "batch step: SLURM_LOCALID=${SLURM_LOCALID:-unset} SLURM_PROCID=${SLURM_PROCID:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset} cpus=$(nproc) $(srun --version)"

# ---- S1: 200 steps, seed 0
if [ "$(last_step $SM/s0_200)" -lt 200 ]; then
  train_seg s0_200 $SM 0 200 22 || { note "S1 FAILED"; exit 1; }
fi
note "S1 done at +$(( $(date +%s) - TJ )) s"
sync; sleep 5
python3 $A8/compare_smoke.py --smoke $A8/logs/smoke_$SLURM_JOB_ID.out --frozen $RGH/logs/train_ghap_train_23246364.out \
        --out $RES/smoke_seed0.json 2>&1 | tee -a $A8/logs/notes_smoke_compare.txt

# ---- S3a: selection + latents (both passes at once) on the 200-step checkpoint
T3=$(date +%s)
rc3=0
pc_select s0_200 $SM && pc_latents s0_200 $SM 1 || rc3=1
note "S3a (select + latents) rc=$rc3 at +$(( $(date +%s) - TJ )) s"

# ---- S3b in the background (prior fit + test, batch step, GCD 7) while S2 runs as an 8-task step
FTLOG=$A8/logs/smoke_fit_test_$SLURM_JOB_ID.out
if [ $rc3 = 0 ]; then
  ( A8_FT_MODE=batch pc_fit_test s0_200 $SM 1 7 ) > $FTLOG 2>&1 &
  FTPID=$!
fi
T2=$(date +%s)
run8s $A8/rank_ref.sh $RES
rc2=$?
note "S2 (valrepro + psdrepro) rc=$rc2 in $(( $(date +%s) - T2 )) s; node memory used, max $(memmon_max "$A8_MEMLOG" $T2) GB"
[ $rc2 = 0 ] && $PYW -I $A8/psdrepro_compare.py --dir $RES/psdrepro 2>&1 | tee -a $A8/logs/notes_smoke_compare.txt
if [ $rc3 = 0 ]; then
  wait $FTPID; rcb=$?
  cat $FTLOG
  note "S3b (fit + test from the batch step, in the background of S2) rc=$rcb"
  if [ $rcb != 0 ]; then                      # the overlap form does not work here: test p1r_G.sh's own srun form
    A8_FT_MODE=srun pc_fit_test s0_200 $SM 1 0 || rc3=1
    note "S3b retried with srun --ntasks=1: rc=$rc3"
  fi
fi

# ---- S3c: evaluation (8 days, every 512th pixel) + gates
if [ $rc3 = 0 ]; then
  pc_eval_gates s0_200 $SM $SM/s0_200/final 1 || rc3=1
fi
note "S3 (chain smoke) rc=$rc3; chain time $(( $(date +%s) - T3 )) s including S2"
note "smoke job ended after $(( $(date +%s) - TJ )) s (S2 rc=$rc2, S3 rc=$rc3)"
sacct -j $SLURM_JOB_ID --format=JobID%22,JobName%14,Elapsed,MaxRSS,AveRSS,State -P 2>/dev/null | tail -n 20
echo "end $(date) rc2=$rc2 rc3=$rc3"
[ $rc2 = 0 ] && [ $rc3 = 0 ]
