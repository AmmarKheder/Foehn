#!/bin/bash
# Case study kept for every model: the clean-air outbreak of 25 January 2022 over Beijing. Haze 107 -> 127 -> 149 ug/m3
# (22-24 January, mean of the 12 CNEMC stations), then 69 on the 25th; FLEXGREEN (256 particles): 86-91 % of the air
# arriving on the 25th came from aloft over Mongolia at leads 2-4 (50 % at lead 1). Runs one model over the Beijing box
# only (crop-only: the windows that overlap it, values identical to the global run), 16 members, start days 18-24 January
# (doy 17-23, i.e. every lead 1-7 for the 25th), members saved at 1 km.  Summary: python case_beijing_summary.py
#     ./case_beijing.sh runs/<run> [checkpoint, default final.ckpt] [afterok job ids] [extra eval args, e.g. "--psub 128"]
set -euo pipefail
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km
R=$1; CK=${2:-$R/final.ckpt}; DEP=${3:-}; EXTRA=${4:-}
sbatch --parsable --nodes 1 --job-name case_bj_$(basename $R | tail -c 12) ${DEP:+--dependency afterok:$DEP} eval_green_job.sh $R \
  --ckpt $CK --crop 41.5 38.5 114.5 118.5 --crop-only --crop-days 17 18 19 20 21 22 23 --doys 17 18 19 20 21 22 23 \
  --members 16 --steps 8 --refine-tail 3 --workers 2 --out case_beijing $EXTRA
