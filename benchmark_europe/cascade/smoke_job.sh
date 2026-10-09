#!/bin/bash
#SBATCH --job-name=cascade_smoke
#SBATCH --account=project_462001140
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/cascade/logs/%x_%j.out
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
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/cascade
echo "start $(date)"
if [ ! -f data/pm5km.npy ]; then
  singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python build_pm5km.py init"
  for y in 2017 2018 2019 2020 2021 2022; do
    singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python build_pm5km.py $y" &
  done
  wait
fi
echo "pm5km $(date)"
rm -rf runs/flow_smoke
srun --cpu-bind=$CPU_BIND singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV;
  python -u cascade.py train --epochs 1 --limit 0.03 --crops 1 --tag _smoke --no-warm" || echo "FAILED train"
echo "train $(date)"
singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV;
  python -u tiling_test.py runs/flow_smoke/final.ckpt ../dejavu/runs/noise_t31_fpstab/final.ckpt 40" || echo "FAILED tiling"
echo "tiling $(date)"
srun --cpu-bind=$CPU_BIND singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV;
  cd ../retrain; python -u evaluate.py --cascade ../cascade/runs/flow_smoke --out ../cascade/runs/flow_smoke/eval_hres --weather hres --day-stride 45 --members 2 --horizons 1 4" || echo "FAILED eval"
singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; cd ../retrain; python evaluate.py --merge ../cascade/runs/flow_smoke/eval_hres"
echo "end $(date)"
