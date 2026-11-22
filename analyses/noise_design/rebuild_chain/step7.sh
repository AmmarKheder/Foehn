#!/bin/bash
#SBATCH --job-name=s1_step7
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Step 7 of REBUILD_PLAN 4c, official stations (STab. 8, T09, T16 'Foehn vs chain'): the 'score' and 'tables' steps of
# test2022_official/code/score.sh, called directly (score.sh appends to STATUS.md). Only the Foehn arm changes
# (eval_final_crps_e01_<D>); the downscaler and chain arms (preds/, class c) and the 0.4 deg input (class d) are read as stored.
# --no-paper-check: the A7(g)-1 check recomputes the paper's compare_fair file with the pre-keyfix coordinate key and has
# failed by design since the keyfix of 2026-11-20 (review/keyfix/README.md); the other checks run.
# The reading of the observations is logged in test2022_official/UNSEALING_A7_<D>.md (not in the A7 log).
#   sbatch --export=ALL,FOEHN_DESIGN=ar step7.sh          (independent of steps 2-6)
: "${FOEHN_WORK:?set FOEHN_WORK (see the README of this folder)}"
source ${FOEHN_WORK}/eccv/cran-pm/paper/natcomms/rebuild/s1_chain/common.sh
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
need $RD/eval_final_crps_e01_$D/part_0.npz $OFF/truth/ghap_official_2022.npy $OFF/preds/done_chain.npy
step $OFF/code/score_official.py --no-paper-check
step $OFF/code/report_tables.py
echo "done $(date -u +%FT%TZ)"
