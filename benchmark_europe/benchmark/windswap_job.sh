#!/bin/bash
#SBATCH --job-name=windswap
#SBATCH --account=project_462001140
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/benchmark/logs/%x_%j.out
# wind blindness test: real HRES weather, but the forecast wind (10 m + 5 levels, and FOEHN's footprint winds) of the
# forecast issued 15 days later; one GPU per model (seed 42), 2022
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
EXP=/scratch/project_462001140/ammar/eccv/cran-pm/experiments
ENVB="unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch; source /projappl/project_462001140/envs/torchenv/bin/activate;
  export PYTHONPATH=/scratch/project_462001140/ammar/eccv/cran-pm/src; export WIND_SWAP=15"
k=0
for m in convlstm earthformer simvp topoflow climax; do
  DB=/tmp/${USER}-miopen-${SLURM_JOB_ID}-$k
  HIP_VISIBLE_DEVICES=$k singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENVB; export MPLCONFIGDIR=/tmp/mpl-$k; mkdir -p $DB;
    export MIOPEN_USER_DB_PATH=$DB MIOPEN_CUSTOM_CACHE_DIR=$DB; cd $EXP/retrain; python -u eval_coarse.py runs/B_${m}_mettarget31 --year 2022 --met-source hres" \
    > $EXP/benchmark/logs/windswap_$m.log 2>&1 &
  k=$((k + 1))
done
for r in noise_t31_fpstab noise_t31; do
  DB=/tmp/${USER}-miopen-${SLURM_JOB_ID}-$k
  HIP_VISIBLE_DEVICES=$k singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENVB; export EVAL_TAG=windswap15 MPLCONFIGDIR=/tmp/mpl-$k; mkdir -p $DB;
    export MIOPEN_USER_DB_PATH=$DB MIOPEN_CUSTOM_CACHE_DIR=$DB; cd $EXP/dejavu; python -u dejavu.py eval runs/$r --year 2022 --weather hres --steps 8 --members 8" \
    > $EXP/benchmark/logs/windswap_$r.log 2>&1 &
  k=$((k + 1))
done
wait
echo done
