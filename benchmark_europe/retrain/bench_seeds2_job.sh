#!/bin/bash
#SBATCH --job-name=bench_seeds2
#SBATCH --account=project_462001140
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain/logs/%x_%j.out
# the uniform main table: every model through evaluate.py (1 km, EEA stations, 0.25 deg block means, scale-aware), HRES,
# 2022; seeds 43/44 of the baselines and of FOEHN (+ its no-footprint ablation, 3 seeds), CAMS and Aurora
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}; export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source /projappl/project_462001140/envs/torchenv/bin/activate;
  export PYTHONPATH=/scratch/project_462001140/ammar/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_PROCID"
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain
run() { local out=$1; shift; echo "=== $out $(date)"; [ -f $out/metrics.json ] && { echo "exists"; return; }
  srun --cpu-bind=$CPU_BIND singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python -u evaluate.py --out $out --weather hres $*" \
    && singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python evaluate.py --merge $out" || echo "FAILED $out"; }
rm -rf runs/B_topoflow_mettarget31_s44/eval1km_hres_none   # partial output of the timed-out job
for ms in topoflow_s44 climax_s43 climax_s44; do
  run runs/B_${ms%_*}_mettarget31_${ms#*_}/eval1km_hres_none --baseline --ckpt runs/B_${ms%_*}_mettarget31_${ms#*_}/ckpt
done
echo "end $(date)"
