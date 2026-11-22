#!/bin/bash
# Step 10 of REBUILD_PLAN 4c (login node, seconds): the full Supplementary Tables 3-6 are regenerated in steps 4 and 6
# (circularity make_outputs, hidden make_si, make_si_v3: results_<D>/); here they are frozen, with the label, into
# source_data/supp_tables_full_<D>/ by refreeze.py (warned, matched, regions, sens, away_b; NOT away_a: see refreeze.py).
# The portrait tables of the manuscript (tools/si_tables_portrait.py main) are not touched: manuscript = step 11.
#   FOEHN_DESIGN=ar bash step10.sh          (needs steps 4 and 6)
: "${FOEHN_WORK:?set FOEHN_WORK (see the README of this folder)}"
source ${FOEHN_WORK}/eccv/cran-pm/paper/natcomms/rebuild/s1_chain/common.sh
need $HID/results_$D/si_table_regions.tex $HID/results_$D/si_table_sensitivity.tex $HID/v3/results_$D/warned_v3.json \
     $HID/v3/results_$D/matched_v3.json $N/analyses/circularity/results_$D/si_table.tex
step $S1/refreeze.py
ls -la $N/source_data/supp_tables_full_$D
