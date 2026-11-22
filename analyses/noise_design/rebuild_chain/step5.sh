#!/bin/bash
#SBATCH --job-name=s1_step5
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Step 5 of REBUILD_PLAN 4c, reliability (SFig. 9, T10): s1_assemble (final + the four 16-member cases; 'before' = e02 is NOT
# re-run: its files stay the stored ones) -> s2_station final -> s2b final -> s3_pop -> s34_estimates (2021 = eval2021_sel_crps_e01_<D>)
# -> s5_members16_noise -> s6_digest -> fig_reliability    -> reliability/data_<D>/, results_<D>/, rebuild/figs_<D>/
#   sbatch --export=ALL,FOEHN_DESIGN=ar step5.sh          (independent of steps 2-4)
: "${FOEHN_WORK:?set FOEHN_WORK (see the README of this folder)}"
source ${FOEHN_WORK}/eccv/cran-pm/paper/natcomms/rebuild/s1_chain/common.sh
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
need $RD/eval_final_crps_e01_$D $RD/eval_pop_crps_e01_$D $RD/eval2021_sel_crps_e01_$D $RD/eval_case_eastasia_crps_e01_$D \
     $RD/eval_case_gironde_crps_e01_$D $RD/eval_case_gironde_long_crps_e01_$D $RD/eval_case_kano_crps_e01_$D \
     $N/analyses/reliability/results/station_scores_before.json
mkdir -p $N/analyses/reliability/data_$D
step $REL/s1_assemble.py final ea16 gir16 girlong16 kano16
step $REL/s2_station.py final
step $REL/s2b_station_scores.py final
step $REL/s3_pop.py 12
step $REL/s34_estimates.py
step $REL/s5_members16_noise.py
step $REL/s6_digest.py
step $REL/fig_reliability.py
echo "done $(date -u +%FT%TZ)"
