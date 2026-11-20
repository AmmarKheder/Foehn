#!/bin/bash
#SBATCH --job-name=cdffm_envcheck
#SBATCH --account=project_462001804
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/%x_%j.out
# Checks the cdffm environment on 2 GCDs of a dev-g node (<= 0.33 GPU-h; random data, writes nothing but this log):
#   1. one GCD: versions, GPU seen, CD-FFM model forward/backward with its own loss code, CPU/GPU agreement
#   2. torch.distributed over RCCL on 2 GCDs + 3 DDP steps (weights identical across ranks)
#   3. Lightning 2.4 DDP via SLURMEnvironment, 3 steps of CD-FFM's FastFFM
set -uo pipefail
source /scratch/project_462001804/ammar/global_model_forecasting_air_pollution/env.sh
PY=$CDFFM_W/scripts/env_check.py
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export PYTHONDONTWRITEBYTECODE=1
echo "start $(date) node=$SLURMD_NODENAME job=$SLURM_JOB_ID"
rc=0
for step in ${*:-gpu ddp lightning}; do  # usage: sbatch jobs/env_check_job.sh [gpu] [ddp] [lightning]
    echo "=== $step"
    case $step in
        gpu) n=1; extra=() ;;
        ddp) n=2; extra=(MASTER_PORT=29511 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,NET) ;;
        lightning) n=2; extra=(MASTER_PORT=29522) ;;
        *) echo "unknown step $step"; rc=1; continue ;;
    esac
    env "${extra[@]}" srun --ntasks=$n --cpus-per-task=7 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" \
        bash -c "source $CDFFM_ENVSH; python -u $PY $step" || rc=1
done
echo "end $(date) rc=$rc"
exit $rc
