#!/bin/bash
#SBATCH --job-name=smoke_var
#SBATCH --account=project_462001140
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain/logs/%x_%j.out
# a few steps of every wind-ablation variant, then the full-grid evaluation path on 2 days
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain
for v in V1 V1r V1s V2 V2s; do
  rm -rf runs/${v}_smoke
  bash train_job.sh $v --epochs 1 --limit-train 20 --limit-val 3 --tag _smoke || echo "FAILED train $v"
done
for v in V1 V2; do
  bash eval_job.sh $v runs/${v}_smoke/ckpt/last.ckpt runs/${v}_smoke/eval_wshift --day-stride 180 --wind shift || echo "FAILED eval $v"
done
