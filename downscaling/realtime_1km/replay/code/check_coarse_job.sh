#!/bin/bash
#SBATCH --job-name=rp_ck_coarse
#SBATCH --account=project_462001804
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay/logs/%x_%j.out
# WP1a check of step A on one archived day: task 0 = the sampler alone on the ARCHIVED FLEXGREEN store (symlink),
# task 1 = FLEXGREEN recomputed + sampler. Both are then compared with RT/out/<day>/pred.npy by check_coarse.py.
#   sbatch check_coarse_job.sh 2026-11-05
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1   # no .pyc next to the imported production modules
R=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay
DAY=$1
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source /projappl/project_462001804/envs/torchenv/bin/activate;
  export PYTHONPATH=/scratch/project_462001804/ammar/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_PROCID; export HF_HOME=/tmp/hf-\$USER"
echo "start $(date -u +%FT%TZ) coarse check $DAY (job $SLURM_JOB_ID)"
T0=$(date +%s)
srun singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; cd $R/code;
  if [ \$SLURM_PROCID = 0 ]; then SLURM_NTASKS=1 python -u coarse_replay.py --root $R/check/coarse_archgreen --link-green $DAY;
  else SLURM_NTASKS=1 SLURM_PROCID=0 python -u coarse_replay.py --root $R/check/coarse_recomp $DAY; fi"
rc=$?
echo "end $(date -u +%FT%TZ) rc=$rc wall $(( $(date +%s) - T0 )) s"
exit $rc
