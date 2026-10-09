#!/bin/bash
#SBATCH --job-name=smoke_base
#SBATCH --account=project_462001140
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain/logs/%x_%j.out
# one short epoch of every baseline on the corrected inputs
for m in simvp convlstm climax earthformer topoflow; do
  bash train_baselines_job.sh $m --epochs 1 --limit-train 30 --limit-val 5 --tag _smoke || echo "FAILED $m"
done
