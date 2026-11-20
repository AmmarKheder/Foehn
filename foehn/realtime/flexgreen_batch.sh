#!/bin/bash
#SBATCH --job-name=green_batch
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime/logs/%x_%j.out
# FLEXGREEN for a range of built day directories, 8 days at a time on one node (one GCD per day, ~14 min each):
#   sbatch flexgreen_batch.sh 2026-06-01 2026-07-05
# Days already holding green/origins_<year>.npy are skipped; each rank takes the days i = rank (mod 8).
set -uo pipefail
FIRST=$1; LAST=$2
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
G1=/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source /projappl/project_462001804/envs/torchenv/bin/activate;
  export PYTHONPATH=/scratch/project_462001804/ammar/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_PROCID; export HF_HOME=/tmp/hf-\$USER"
cd $G1
echo "start $(date -u +%FT%TZ) flexgreen batch $FIRST..$LAST"
srun singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python -u - <<'PY'
import os, subprocess, datetime as dt, json, time
from pathlib import Path
rank, world = int(os.environ.get('SLURM_PROCID', 0)), int(os.environ.get('SLURM_NTASKS', 1))
RT = Path('realtime/data'); first = dt.date.fromisoformat('$FIRST'); last = dt.date.fromisoformat('$LAST')
days = []; d = first
while d <= last:
    D = RT / d.isoformat()
    if (D / 'hres_B.npy').exists() and (D / 'cams.npy').exists() and (D / 'fires.npy').exists() and not (D / 'green' / f'origins_{d.year}.npy').exists():
        days.append(d)
    d += dt.timedelta(days=1)
mine = days[rank::world]
print(f'rank {rank}: {len(mine)} of {len(days)} days', flush=True)
for d in mine:
    t = time.time()
    env = dict(os.environ, SLURM_PROCID='0', SLURM_NTASKS='1')                  # each day is a one-rank FLEXGREEN run (it would wait for 8 ranks otherwise); SLURM_LOCALID keeps the GPU
    r = subprocess.run(['python', '-u', 'flexgreen.py', 'run', '--src', 'hres', '--rt-dir', f'realtime/data/{d}', '--particles', '32', '--emis', '--emis-fallback', '--instant', '--all-cells'], capture_output=True, text=True, env=env)
    print(f'rank {rank} {d}: rc {r.returncode} in {time.time() - t:.0f}s' + ('' if r.returncode == 0 else ' | ' + r.stdout[-300:] + r.stderr[-300:]), flush=True)
PY"
echo "end $(date -u +%FT%TZ) rc=$?"
