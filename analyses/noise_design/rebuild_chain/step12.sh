#!/bin/bash
#SBATCH --job-name=s1_step12
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
# Step 12 of REBUILD_PLAN 4c, Source Data only: python -B source_data/build_source_data.py under FOEHN_DESIGN=<D> reads the
# final model's folders *_<D> and every derived file of the design, and writes source_data/csv_<D>/, Source_Data_<D>.xlsx,
# MANIFEST_<D>.md, MANIFEST_internal_<D>.md. The figure list comes from the manuscript as it is (work/*.tex, read only).
# NOT run here: tools/build.sh (compiles work/main.tex into work/main.pdf: the manuscript, step 11) and tools/package.py
# (packages work/ and the v2 tables): both need the manuscript edits and the figures copied by hand first.
#   sbatch --export=ALL,FOEHN_DESIGN=ar step12.sh          (needs steps 2-10)
: "${FOEHN_WORK:?set FOEHN_WORK (see the README of this folder)}"
source ${FOEHN_WORK}/eccv/cran-pm/paper/natcomms/rebuild/s1_chain/common.sh
need $FIG/data/overview_cache_$L.npz $FIG/data/compare_fair_$L.npz $N/analyses/issuance/data_$D/derived.json \
     $N/analyses/reliability/results_$D/station_scores_final.json $HID/v3/results_$D/numbers_v3.json $N/source_data/supp_tables_full_$D
step $N/source_data/build_source_data.py
ls -la $N/source_data/Source_Data_$D.xlsx $N/source_data/MANIFEST_$D.md; ls $N/source_data/csv_$D | wc -l
