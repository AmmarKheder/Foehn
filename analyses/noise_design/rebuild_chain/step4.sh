#!/bin/bash
#SBATCH --job-name=s1_step4
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
# Step 4 of REBUILD_PLAN 4c: circularity rings (reads RUN/eval_pop_crps_e01_<D>) -> analyse 10000 (also reads the station-day
# file of step 3 and, as stored, the Foehn-CAMS file) -> make_outputs  -> circularity/data_<D>/, results_<D>/
#   sbatch --export=ALL,FOEHN_DESIGN=ar step4.sh          (needs steps 2 and 3)
: "${FOEHN_WORK:?set FOEHN_WORK (see the README of this folder)}"
source ${FOEHN_WORK}/eccv/cran-pm/paper/natcomms/rebuild/s1_chain/common.sh
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
need $RD/eval_pop_crps_e01_$D/pop_idx.npy $FIG/data/compare_fair_$L.npz $N/analyses/circularity/data_$D/station_day.npz
step $CIRC/extract_rings.py
step $CIRC/analyse.py 10000
step $CIRC/make_outputs.py
echo "done $(date -u +%FT%TZ)"
