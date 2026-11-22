#!/bin/bash
#SBATCH --job-name=s1_step2
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Step 2 of REBUILD_PLAN 4c: the commands of FIG/postproc_final.sh <L> RUN eval_final_crps_e01_<D> eval_case_eastasia_crps_e01_<D>,
# in parallel as there (A and C read as stored), + FIG/compare_ghap_global.py (needed by issuance/summarise.py and T05).
#   sbatch --export=ALL,FOEHN_DESIGN=ar step2.sh
: "${FOEHN_WORK:?set FOEHN_WORK (see the README of this folder)}"
source ${FOEHN_WORK}/eccv/cran-pm/paper/natcomms/rebuild/s1_chain/common.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
EV=eval_final_crps_e01_$D; CASE=eval_case_eastasia_crps_e01_$D
need $RD/$EV/part_0.npz $RD/$CASE/part_0.npz
G=$S1/logs/step2_$D
bg ${G}_compare_fair.log $FIG/compare_fair.py "$L=$RUN:$EV"
bg ${G}_extremes_stations.log $FIG/extremes_stations.py "$L=$RUN:$EV" "A=$A"
bg ${G}_tail_events.log $FIG/tail_events.py evaluate "$L=$RUN:$EV" "A=$A" "C=$C"
bg ${G}_figdata_case.log $FIG/figdata_case.py $RUN $CASE
bg ${G}_compare_ghap_global.log $FIG/compare_ghap_global.py "$L=$RUN:$EV"
waitall
for f in ${G}_*.log; do echo "#### $f"; cat $f; done; rm -f ${G}_*.log
ls -la $FIG/data/{compare_fair,extremes_stations,tail_events,compare_ghap_global}_$L.npz $FIG/data/case_eastasia_${RUN}_$D.npz
