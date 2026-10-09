#!/bin/bash
#SBATCH --job-name=flexgreen
#SBATCH --account=project_462001140
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/logs/%x_%j.out
# usage: sbatch [--nodes N] flexgreen_job.sh <hres|era5> <year> [first_doy last_doy]   (ranks share the days)
set -uo pipefail
SRC=$1; YEAR=$2; D0=${3:-0}; D1=${4:-$(( (YEAR % 4 == 0) ? 365 : 364 ))}; EXTRA=${5:-}
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"
ENV="mkdir -p /tmp/\${USER}-miopen-\$SLURM_JOB_ID-\$SLURM_LOCALID; export MIOPEN_USER_DB_PATH=/tmp/\${USER}-miopen-\$SLURM_JOB_ID-\$SLURM_LOCALID;
  export MIOPEN_CUSTOM_CACHE_DIR=\$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source /projappl/project_462001140/envs/torchenv/bin/activate"
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km
echo "start $(date) $SRC $YEAR days $D0..$D1"
srun --cpu-bind=$CPU_BIND singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python -u flexgreen.py run --src $SRC --year $YEAR --days \$(seq $D0 $D1) $EXTRA"
rc=$?
echo "end $(date)"
exit $rc
