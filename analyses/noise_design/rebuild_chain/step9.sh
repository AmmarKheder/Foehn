#!/bin/bash
#SBATCH --job-name=s1_step9
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
# Step 9 of REBUILD_PLAN 4c: figure data and renders not already made in steps 3, 5, 6 (figED_fair, fig_sedi_definitions,
# fig_exposure_eqiss: step 3; fig_reliability: step 5; fig_hidden_warned_v3: step 6). Every render goes to
# natcomms/rebuild/figs_<D>/ (nature_style.OUT, fig_style.OUT, render.py and the own savefig paths follow FOEHN_DESIGN);
# nothing is copied into work/figs (by hand, after the figures have been checked; REBUILD_PLAN 5c and 6).
#   Fig. 1  figdata_arch_fields.py leads (-> FIG/data/arch_fields_<D>.npz) -> figures_new/fig1_v2/fig_gc1_schematic_rt2.py
#   Fig. 2  figures_new/fig2_v3/fig_hidden_v4.py (reads hidden/v2/results_<D>/numbers_v2.json: step 6)
#   Fig. 3  NATURE=1 fig_overview.py (cache FIG/data/overview_cache_crps_e01_<D>.npz, made on the first run)
#   Fig. 5  NATURE=1 fig_gc2_realism.py (FIG/data/case_eastasia_<RUN>_<D>.npz: step 2)
#   Fig. 6  review/figfix/render.py fig_transport.py      Fig. 7  NATURE=1 fig_smoke.py      SFig. 7  figED_regions.py
#   sbatch --export=ALL,FOEHN_DESIGN=ar step9.sh          (needs steps 2 and 6)
: "${FOEHN_WORK:?set FOEHN_WORK (see the README of this folder)}"
source ${FOEHN_WORK}/eccv/cran-pm/paper/natcomms/rebuild/s1_chain/common.sh
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NATURE=1
need $RD/eval_jan21_leads_crps_e01_$D/map1km_d19_h1_member0.npy $RD/eval_jan21_crps_e01_$D/map1km_d19_h1.npy $RD/eval_final_crps_e01_$D/part_0.npz \
     $RD/eval_case_eastasia_crps_e01_$D $RD/eval_case_kano_crps_e01_$D $RD/eval_case_gironde_crps_e01_$D $RD/eval_case_gironde_long_crps_e01_$D \
     $FIG/data/case_eastasia_${RUN}_$D.npz $HID/v2/results_$D/numbers_v2.json
step $FIG/figdata_arch_fields.py leads
step $N/figures_new/fig1_v2/fig_gc1_schematic_rt2.py
step $N/figures_new/fig2_v3/fig_hidden_v4.py
step $FIG/fig_overview.py
step $FIG/fig_gc2_realism.py
step $N/review/figfix/render.py fig_transport.py
step $FIG/fig_smoke.py
step $FIG/figED_regions.py
ls -la $FIGS_D
echo "done $(date -u +%FT%TZ)"
