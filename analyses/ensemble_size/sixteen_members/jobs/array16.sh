#!/bin/bash
#SBATCH --job-name=a13w16_run16
#SBATCH --account=<project>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%A_%a.out
# usage: sbatch --array=0-14 --time=... jobs/array16.sh
# Array task k (0..14) samples, with 16 members, the start days 24k and 24k + 12 (two of the 30 start days 0, 12, ..., 348 =
# every sixth start day of analyses/hidden/v3), leads 1 and 3: 4 forecasts, one GCD, files data/run16/{pop16,keys16,done16}_k<k>.*
# Environment as jobs/gpu_job.sh (template eccv/cran-pm/experiments/global1km/eval_green_job.sh). Restartable (done file).
set -uo pipefail
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
CKPT=${FOEHN_WORK}/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4/final.ckpt
K=${SLURM_ARRAY_TASK_ID}
D1=$((24 * K)); D2=$((24 * K + 12))
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate;
  export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_JOB_ID; export PYTHONDONTWRITEBYTECODE=1"
cd ${FOEHN_WORK}/global_model_forecasting_air_pollution/a13/warn16
mkdir -p data/run16
echo "start $(date) task $K start days $D1 $D2"
srun singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python -u sample16.py --ckpt $CKPT --members 16 --doys $D1 $D2 --leads 1 3 --out data/run16 --tag k$(printf %02d $K)"
rc=$?
echo "end $(date) rc=$rc"
exit $rc
