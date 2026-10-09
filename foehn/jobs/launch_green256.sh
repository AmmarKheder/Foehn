#!/bin/bash
# FLEXGREEN v2 (256 particles per receptor + anthropogenic emission term) for the HRES years 2017-2022, then the
# fine-tuning of Foehn-Green A from epoch 21 (256-particle store, 128 per sample, emission channels from zero), then its
# test on every second start day of 2022 (as the evaluation of epoch 21).  Run after tests/check_green_store.py passed.
#     ./launch_green256.sh
set -euo pipefail
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km
X="--particles 256 --store green256 --emis --chunk 40000"
ids=()
for y in 2017 2018 2019 2020 2021 2022; do
  last=$(( (y % 4 == 0) ? 365 : 364 ))
  j=$(sbatch --parsable --nodes 4 --job-name fg256_$y flexgreen_job.sh hres $y 0 $last "$X")
  ids+=($j); echo "FLEXGREEN 256 $y: $j"
done
dep=$(IFS=:; echo "${ids[*]}")
A=runs/green_flow_shift25_global_L7_noant_whres_ghres
ft=$(sbatch --parsable --nodes 4 --job-name green_A_ft256e --dependency afterok:$dep train_green_job.sh \
     --weather hres --green hres --store green256 --emis --psub 128 --init $A/ep21.ckpt --epochs 20 --lr 1e-4 --workers 6 --tag _ft256e)
echo "fine-tuning: $ft (after $dep)"
ev=$(sbatch --parsable --nodes 16 --job-name g1km_green_eval_ft --dependency afterok:$ft eval_green_job.sh \
     ${A}_ft256e --members 4 --steps 8 --refine-tail 3 --every 2 --psub 128 --out eval_ft)
echo "evaluation: $ev (after $ft)"
