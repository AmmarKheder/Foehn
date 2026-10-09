#!/bin/bash
#SBATCH --job-name=full_smoke
#SBATCH --account=project_462001140
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/fullimage/logs/%x_%j.out
# measure s/step and peak memory of whole-image training, then score 2 days end to end (1 km, stations, 0.25 deg)
set -uo pipefail
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source /projappl/project_462001140/envs/torchenv/bin/activate;
  export PYTHONPATH=/scratch/project_462001140/ammar/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_PROCID"
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/fullimage
echo "start $(date)"
rm -rf runs/flow_smoke
srun --cpu-bind=$CPU_BIND singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV;
  python -u fullflow.py train --epochs 1 --limit 0.03 --tag _smoke" || echo "FAILED train"
echo "train $(date)"
srun --cpu-bind=$CPU_BIND singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV;
  cd ../retrain; python -u evaluate.py --full ../fullimage/runs/flow_smoke --out ../fullimage/runs/flow_smoke/eval_hres --weather hres \
  --day-stride 45 --horizons 1 4 --members 2 --steps2 4 --save-days 45" || echo "FAILED eval"
singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; cd ../retrain; python evaluate.py --merge ../fullimage/runs/flow_smoke/eval_hres"
echo "end $(date)"
