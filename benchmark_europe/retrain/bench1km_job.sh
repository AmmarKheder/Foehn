#!/bin/bash
#SBATCH --job-name=bench1km
#SBATCH --account=project_462001140
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain/logs/%x_%j.out
# Fair 1 km table: the 0.25 deg models (baselines + our 0.25 deg flow) brought to 1 km (bilinear, and + today's 1 km
# pattern), real HRES forecast weather, 2022, scored by evaluate.py (1 km grid, EEA stations, 0.25 deg block means).
# usage: sbatch bench1km_job.sh [extra evaluate.py args, e.g. --day-stride 90]; OUTSFX=_smoke for test output dirs
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
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain
SFX=${OUTSFX:-}
MODELS=${MODELS:-"flow convlstm earthformer simvp topoflow climax"}
run() {  # $1 out dir, rest: evaluate.py args
  local out=$1; shift
  echo "=== $out $(date)"
  srun --cpu-bind=$CPU_BIND singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python -u evaluate.py --out $out --weather hres $*" \
    && singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python evaluate.py --merge $out" || echo "FAILED $out"
}
for m in $MODELS; do
  for pat in none add; do
    if [ $m = flow ]; then
      run ../dejavu/runs/noise_t31_fpstab/eval1km_hres_${pat}$SFX --dejavu ../dejavu/runs/noise_t31_fpstab --members 8 --steps1 8 --fine-pattern $pat "$@"
    else
      ck=$(ls runs/B_${m}_mettarget31/ckpt/best-*.ckpt | head -1)
      run runs/B_${m}_mettarget31/eval1km_hres_${pat}$SFX --baseline --ckpt $ck --fine-pattern $pat "$@"
    fi
  done
done
echo "end $(date)"
