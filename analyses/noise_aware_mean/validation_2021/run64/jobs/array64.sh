#!/bin/bash
#SBATCH --job-name=a14_r2
#SBATCH --account=<project>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --output=../logs/%x_%A_%a.out
# usage (from a14/r2): sbatch --array=0-14 --time=03:15:00 --dependency=afterok:<smoke> jobs/array64.sh
# A14 run R2 (protocol_amendments/A14_infinite_ensemble.md, section 5): array task k samples the 2 forecasts of line k+1 of
# jobs/pairs2021.txt (start day d of 2021, leads 1 and 3; days fixed by select2021.py) with the frozen final 1 km model,
# 64 members (noise_tiles64.npy), at the 8.1 M populated pixels -> a14/r2/data/run64/{pop64,keys64,done64}_k<kk>.* ;
# sampler a14/r2/sample_m.py (= a13/warn16/sample16.py + --year/--noise). Restartable (done file). Decides nothing.
set -uo pipefail
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
CKPT=${FOEHN_WORK}/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4/final.ckpt
K=${SLURM_ARRAY_TASK_ID}
KK=$(printf %02d $K)
cd ${FOEHN_WORK}/global_model_forecasting_air_pollution/a14/r2
PAIRS=$(sed -n "$((K + 1))p" jobs/pairs2021.txt)
if [ -z "$PAIRS" ]; then echo "no line $((K + 1)) in jobs/pairs2021.txt"; exit 3; fi
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate;
  export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_JOB_ID; export PYTHONDONTWRITEBYTECODE=1"
rc=0
mkdir -p data/run64
echo "start $(date) task $K pairs $PAIRS"
srun singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python -u sample_m.py --ckpt $CKPT --members 64 --year 2021 --noise noise_tiles64.npy --pairs $PAIRS --out data/run64 --tag k$KK" || rc=$?
echo "end $(date) rc=$rc"
exit $rc
