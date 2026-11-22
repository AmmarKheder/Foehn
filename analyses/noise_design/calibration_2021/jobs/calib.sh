#!/bin/bash
#SBATCH --job-name=a15_calib
#SBATCH --account=<project>
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=logs/a15.out
#SBATCH --open-mode=append
# A15 run C (calibration, 60 forecasts: Cal lead 1 at rho 0.70 + Cal lead 2 at rho 0.50, 0.70, 0.90) -> data/calib/part_0..7.npz
# (a15/A15_draft.md section 5; eval_ar.py --plan calib, 8 ranks, items sorted by (lead, start, rho), todo[rank::8]).
# DataLoader workers: the 'workers' field of results/smoke_check.json (4, or 6 if the smoke measured io wait > 15 % of GPU time).
set -uo pipefail
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"
H=${FOEHN_WORK}/global_model_forecasting_air_pollution/a15
cd $H
grep -q '"all_passed": true' results/check_cpu.json 2>/dev/null && grep -q '"all_passed": true' results/smoke_check.json 2>/dev/null || { echo "A15 calib: check_cpu.json or smoke_check.json missing or failed"; exit 3; }
W=$(grep -o '"workers": [0-9]*' results/smoke_check.json | head -1 | grep -o '[0-9]*$'); W=${W:-4}
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate;
  export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_PROCID; export PYTHONDONTWRITEBYTECODE=1"
echo "=== A15 calib job ${SLURM_JOB_ID} start $(date) workers=$W"
srun --cpu-bind=$CPU_BIND singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; cd $H; python -u eval_ar.py --plan calib --workers $W"
rc=$?
echo "=== A15 calib job ${SLURM_JOB_ID} end $(date) rc=$rc"
exit $rc
