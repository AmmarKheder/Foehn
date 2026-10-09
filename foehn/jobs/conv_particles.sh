#!/bin/bash
# Particle-number convergence of the FORECAST: one checkpoint run with N particles per member (N = 16 .. 256, drawn from
# the 256-particle store green256), validation year 2021 scored against GHAP only (stations never used), leads 1, 3, 7,
# one start day in seven (52 days). Summary: python conv_summary.py runs/<run>
#     ./conv_particles.sh runs/<run> [checkpoint, default final.ckpt] [afterok job ids, colon-separated]
set -euo pipefail
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km
R=$1; CK=${2:-$R/final.ckpt}; DEP=${3:-}
for N in 16 32 64 128 256; do
  j=$(sbatch --parsable --nodes 4 --job-name conv_p$N ${DEP:+--dependency afterok:$DEP} eval_green_job.sh $R --ckpt $CK \
      --year 2021 --store green256 --psub $N --leads 1 3 7 --every 7 --members 4 --steps 8 --refine-tail 3 --out eval2021_conv_p$N)
  echo "N=$N: $j"
done
