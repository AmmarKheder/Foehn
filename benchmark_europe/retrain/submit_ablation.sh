#!/bin/bash
# Wind-prior ablation: every variant x 3 seeds, 100 epochs, one node each.
# usage: bash submit_ablation.sh   or   VARIANTS="V1x V2x" bash submit_ablation.sh
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain
for seed in 42 43 44; do
  for v in ${VARIANTS:-V1 V1r V1s V2 V2s}; do
    sbatch --time=10:00:00 --job-name=cran_${v}_s${seed} -o logs/cran_${v}_s${seed}_%j.out \
      train_job.sh $v --epochs 100 --warmup 5 --seed $seed
  done
done
