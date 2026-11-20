#!/bin/bash
#SBATCH --job-name=rp_coarse
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay/logs/%x_%j.out
# WP1a step A (replay of the 0.4 degree real-time forecast): FLEXGREEN + forecast, one GCD per day stream.
#   sbatch coarse_job.sh <root under replay/> [--link-green] <day> [<day> ...]
# Environment = realtime/rt_job.sh and flexgreen_batch.sh (container + torchenv). Writes only under replay/.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1   # no .pyc next to the imported production modules
R=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay
ROOT=$1; shift
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source /projappl/project_462001804/envs/torchenv/bin/activate;
  export PYTHONPATH=/scratch/project_462001804/ammar/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_PROCID; export HF_HOME=/tmp/hf-\$USER"
echo "start $(date -u +%FT%TZ) coarse replay root=$ROOT args=$* (job $SLURM_JOB_ID, $SLURM_NTASKS tasks)"
T0=$(date +%s)
srun singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; cd $R/code; python -u coarse_replay.py --root $R/$ROOT $*"
rc=$?
echo "end $(date -u +%FT%TZ) rc=$rc wall $(( $(date +%s) - T0 )) s"
exit $rc
