#!/bin/bash
#SBATCH --job-name=s1_step6
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out
# Step 6 of REBUILD_PLAN 4c, warnings (Fig. 2f, SFig. 5, STabs. 3-5, T04): the commands of hidden/job_sys.sh, v2/job_bc.sh,
# v3/job_v3.sh (same workers and threads) and their table scripts, under FOEHN_DESIGN=<D>:
#   v1 figdata_hidden_systems 32 -> tables -> make_si -> main_numbers                 -> hidden/data_<D>/, results_<D>/
#   v2 figdata_hidden_bc 32 -> check_identity -> tables_v2 -> make_si_v2 -> main_numbers_v2 -> check_text_numbers_v2
#      (the last one compares the stored hand-written v2 texts with the new numbers_v2.json and only prints)  -> v2/data_<D>/, results_<D>/
#   v3 figdata_hidden_v3 32 (start-day estimates = truth: the stored v3/data/start_est_v3.npz is read) -> tables_v3 -> make_si_v3
#      -> fig_hidden_warned_v3 (SFig. 5)                                                -> v3/data_<D>/, results_<D>/, rebuild/figs_<D>/
#   sbatch --export=ALL,FOEHN_DESIGN=ar step6.sh          (independent of steps 2-5)
: "${FOEHN_WORK:?set FOEHN_WORK (see the README of this folder)}"
source ${FOEHN_WORK}/eccv/cran-pm/paper/natcomms/rebuild/s1_chain/common.sh
need $RD/eval_pop_crps_e01_$D/pop_idx.npy $RD/eval_final_crps_e01_$D/part_0.npz $HID/v3/data/start_est_v3.npz $HID/data/exposure_sens_2022.npz
mkdir -p $HID/data_$D $HID/v2/data_$D $HID/v3/data_$D
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
step $HID/figdata_hidden_systems.py 32
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
step $HID/tables.py
step $HID/make_si.py
step $HID/main_numbers.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
step $HID/v2/figdata_hidden_bc.py 32
step $HID/v2/check_identity.py data_$D/hidden_bc_pop.npz
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
step $HID/v2/tables_v2.py
step $HID/v2/make_si_v2.py
step $HID/v2/main_numbers_v2.py
( step $HID/v2/check_text_numbers_v2.py ) || echo "check_text_numbers_v2: the stored v2 texts disagree with numbers_v2.json of $D (expected for a new design: list above)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
step $HID/v3/figdata_hidden_v3.py 32
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
step $HID/v3/tables_v3.py
step $HID/v3/make_si_v3.py
step $HID/v3/fig_hidden_warned_v3.py
echo "done $(date -u +%FT%TZ)"
