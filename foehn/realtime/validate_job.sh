#!/bin/bash
#SBATCH --job-name=rt12_val
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime/logs/%x_%j.out
# Validation of a fine-tuned checkpoint on the September 2026 day directories: predict.py on every day (8 days at a
# time, one GCD each) -> out/<d>/pred_<tag>.npy, then hindcast.py score with the tag.
#   sbatch validate_job.sh <ckpt> <tag> [first] [last]
set -uo pipefail
CKPT=$1; TAG=$2; FIRST=${3:-2026-09-01}; LAST=${4:-2026-09-20}
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source /projappl/project_462001804/envs/torchenv/bin/activate;
  export PYTHONPATH=/scratch/project_462001804/ammar/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_PROCID; export HF_HOME=/tmp/hf-\$USER"
cd /scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime
echo "start $(date -u +%FT%TZ) validate $CKPT tag $TAG $FIRST..$LAST"
srun singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python -u - <<'PY'
import os, subprocess, datetime as dt, time
from pathlib import Path
rank, world = int(os.environ.get('SLURM_PROCID', 0)), int(os.environ.get('SLURM_NTASKS', 1))
first = dt.date.fromisoformat('$FIRST'); last = dt.date.fromisoformat('$LAST'); days = []; d = first
while d <= last:
    if (Path('data') / d.isoformat() / 'green').exists() and not (Path('out') / d.isoformat() / 'pred_$TAG.npy').exists():
        days.append(d)
    d += dt.timedelta(days=1)
for d in days[rank::world]:
    env = dict(os.environ, SLURM_PROCID='0', SLURM_NTASKS='1')
    t = time.time(); r = subprocess.run(['python', '-u', 'predict.py', d.isoformat(), '--ckpt', '$CKPT', '--out', 'pred_$TAG.npy'], capture_output=True, text=True, env=env)
    print(f'rank {rank} {d}: rc {r.returncode} in {time.time() - t:.0f}s' + ('' if r.returncode == 0 else ' | ' + r.stdout[-300:] + r.stderr[-400:]), flush=True)
PY"
echo "predictions done $(date -u +%FT%TZ)"
unset PYTHONPATH; export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache
/projappl/project_462001804/envs/world/bin/python hindcast.py score $FIRST $LAST $TAG 2>&1 | grep -v findfont | tee logs/validate_$TAG.out
echo "end $(date -u +%FT%TZ)"
