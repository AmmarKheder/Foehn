#!/bin/bash
# Validation helper of the 'check' label: RUN/<stored eval>_check -> <stored eval> (relative symlinks), so that the chain run
# with FOEHN_DESIGN=check reads the STORED runs through exactly the code path of a new design (folder + _<D>).
#   bash check_links.sh make | remove
set -euo pipefail
RD=${FOEHN_WORK}/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4
EVS="eval_final_crps_e01 eval_pop_crps_e01 eval_jan21_crps_e01 eval_jan21_leads_crps_e01 eval_case_eastasia_crps_e01 eval_case_kano_crps_e01
     eval_case_gironde_crps_e01 eval_case_gironde_long_crps_e01 eval2021_sel_crps_e01"
cd $RD
for e in $EVS; do
  case ${1:-} in
    make) [ -d $e ] && [ ! -e ${e}_check ] && ln -s $e ${e}_check; ls -ld ${e}_check ;;
    remove) [ -L ${e}_check ] && rm ${e}_check; echo "removed ${e}_check" ;;
    *) echo "usage: check_links.sh make|remove"; exit 2 ;;
  esac
done
