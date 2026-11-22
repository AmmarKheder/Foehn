#!/bin/bash
#SBATCH --job-name=s1_step3
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Step 3 of REBUILD_PLAN 4c, under FOEHN_DESIGN=<D>:
#  (a) issuance land chain (not in rerun_keyfix.sh): tail_events_eqiss ISSUANCE=d1 DUMP=1 and ISSUANCE=d (A, C as stored),
#      compare_ghap_global_eqiss d1 and d, then land_freqmatch leads 1-7 and merge  -> issuance/data_<D>/
#  (b) rerun_keyfix.sh 'stations' without compare_fair (done in step 2) and without analyse/make_outputs (they also need the
#      rings: step 4): circularity extract_station_day -> figED_fair (SFig. 6) -> paper_numbers (stdout -> logs/paper_numbers_<D>.txt)
#  (c) rerun_keyfix.sh 'extremes' without extremes_stations (step 2): extremes_stations_mean, _own -> issuance own_eqiss,
#      matched -> summarise -> derive -> make_table -> fig_sedi_definitions P99/P95/P90 -> fig_exposure_eqiss (+ row)
# No figure is renamed or restored (rerun_keyfix's swapfig): the renders of a design go to rebuild/figs_<D>/.
#   sbatch --export=ALL,FOEHN_DESIGN=ar step3.sh          (needs step 2)
: "${FOEHN_WORK:?set FOEHN_WORK (see the README of this folder)}"
source ${FOEHN_WORK}/eccv/cran-pm/paper/natcomms/rebuild/s1_chain/common.sh
EV=eval_final_crps_e01_$D
need $RD/$EV/part_0.npz $RD/eval2021_sel_crps_e01_$D/part_0.npz $FIG/data/compare_fair_$L.npz $FIG/data/extremes_stations_$L.npz \
     $FIG/data/tail_events_$L.npz $FIG/data/compare_ghap_global_$L.npz $FIG/data/case_eastasia_${RUN}_$D.npz
mkdir -p $N/analyses/issuance/data_$D $N/analyses/circularity/data_$D
G=$S1/logs/step3_$D
# ---- (a) land chain: the four scorers in parallel (about 26 min, 23 GB), then the seven leads of land_freqmatch
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
( export ISSUANCE=d1 DUMP=1; bg ${G}_te_d1.log $ISS/tail_events_eqiss.py "$L=$RUN:$EV" "A=$A" "C=$C"
  export ISSUANCE=d DUMP=0;  bg ${G}_te_d.log $ISS/tail_events_eqiss.py "$L=$RUN:$EV" "A=$A" "C=$C"
  export ISSUANCE=d1;        bg ${G}_cg_d1.log $ISS/compare_ghap_global_eqiss.py "$L=$RUN:$EV"
  export ISSUANCE=d;         bg ${G}_cg_d.log $ISS/compare_ghap_global_eqiss.py "$L=$RUN:$EV"
  waitall ) || exit 4
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for h in 1 2 3 4 5 6 7; do bg ${G}_fm_h$h.log $ISS/land_freqmatch.py $L $h; done
waitall
step $ISS/land_freqmatch.py $L merge
for f in ${G}_*.log; do echo "#### $f"; cat $f; done; rm -f ${G}_*.log
# ---- (b) stations
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
step $CIRC/extract_station_day.py
step $FIG/figED_fair.py
step $FIG/paper_numbers.py > $S1/logs/paper_numbers_$D.txt; cat $S1/logs/paper_numbers_$D.txt
# ---- (c) extremes
step $FIG/extremes_stations_mean.py "$L=$RUN:$EV" "A=$A"
step $FIG/extremes_stations_own.py "$L=$RUN:$EV"
step $ISS/extremes_stations_own_eqiss.py "$L=$RUN:$EV"
step $ISS/extremes_stations_matched.py "$L=$RUN:$EV"
step $ISS/summarise.py $L
step $ISS/derive.py $L
step $ISS/make_table.py
for T in P99 P95 P90; do step $ISS/fig_sedi_definitions.py $T; done
step $ISS/fig_exposure_eqiss.py
step $ISS/fig_exposure_eqiss.py row
echo "done $(date -u +%FT%TZ)"
