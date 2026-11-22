#!/bin/bash
#SBATCH --job-name=a13w16_all
#SBATCH --account=<project>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%A_%a.out
# usage: sbatch --array=0-81 --time=03:30:00 jobs/array16all.sh
# The 16-member run on all 179 start days of analyses/hidden/v3 (plan16all.py): array task k samples the 8 forecasts of line k+1
# of jobs/run16all_pairs.txt ('start day:lead' items) with the frozen final 1 km model, 16 members, at the 8.1 M populated
# cells -> data/run16all/{pop16,keys16,done16}_k<kk>.* . Same sampler, container and environment as jobs/array16.sh
# (sample16.py, eval_global_green.py imported unchanged); only the forecast list differs (--pairs). Restartable (done file).
# CHECK_PAIRS="d:h ..." (smoke only): after its 8 forecasts, the task samples these into data/run16all_check (tag k<kk>chk), to
# compare a fresh process with data/run16 (all 16 members) and with the stored 4 members.
set -uo pipefail
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
CKPT=${FOEHN_WORK}/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4/final.ckpt
K=${SLURM_ARRAY_TASK_ID}
KK=$(printf %02d $K)
cd ${FOEHN_WORK}/global_model_forecasting_air_pollution/a13/warn16
PAIRS=$(sed -n "$((K + 1))p" jobs/run16all_pairs.txt)
if [ -z "$PAIRS" ]; then echo "no line $((K + 1)) in jobs/run16all_pairs.txt"; exit 3; fi
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate;
  export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_JOB_ID; export PYTHONDONTWRITEBYTECODE=1"
rc=0
mkdir -p data/run16all
echo "start $(date) task $K pairs $PAIRS"
srun singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python -u sample16.py --ckpt $CKPT --members 16 --pairs $PAIRS --out data/run16all --tag k$KK" || rc=$?
echo "end $(date) rc=$rc"
if [ -n "${CHECK_PAIRS:-}" ]; then
  mkdir -p data/run16all_check
  echo "start $(date) task $K CHECK pairs $CHECK_PAIRS"
  srun singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python -u sample16.py --ckpt $CKPT --members 16 --pairs $CHECK_PAIRS --out data/run16all_check --tag k${KK}chk" || rc=$?
  echo "end check $(date) rc=$rc"
fi
exit $rc
