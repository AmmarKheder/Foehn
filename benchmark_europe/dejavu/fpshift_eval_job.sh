#!/bin/bash
#SBATCH --job-name=dv_fpshift_ev
#SBATCH --account=project_462001140
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=21
#SBATCH --mem=180G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/dejavu/logs/%x_%j.out
# ablation "footprint trained and tested on the forecast wind of a day 15 days away": HRES weather, 2022, same protocol and per-day records as FOEHN (rec8)
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/dejavu
k=0
for run in noise_t31_fpshift15stab noise_t31_fpshift15stab_s43 noise_t31_fpshift15stab_s44; do
  DB=/tmp/${USER}-miopen-${SLURM_JOB_ID}-$k
  HIP_VISIBLE_DEVICES=$k singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
    source /projappl/project_462001140/envs/torchenv/bin/activate; export MPLCONFIGDIR=/tmp/mpl-$k; mkdir -p $DB;
    export MIOPEN_USER_DB_PATH=$DB; export MIOPEN_CUSTOM_CACHE_DIR=$DB; export SAVE_RECORDS=1 RECORD_S=8 EVAL_TAG=rec8 FP_WIND_SWAP=15;
    python -u dejavu.py eval runs/$run --year 2022 --weather hres --steps 8 --members 8" > logs/fpshift_${run}.log 2>&1 &
  k=$((k + 1))
done
wait
ls -la runs/noise_t31_fpshift15stab*/eval_2022_hres_rec8.json
