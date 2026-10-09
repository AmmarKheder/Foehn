#!/bin/bash
#SBATCH --job-name=records025
#SBATCH --account=project_462001140
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain/logs/%x_%j.out
# per-day 0.25 deg scores (HRES weather, 2022) of the 5 baselines x 3 seeds, for the extreme-day figure; 8 at a time
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain
k=0
for m in convlstm earthformer simvp topoflow climax; do for s in "" _s43 _s44; do
  g=$((k % 8)); DB=/tmp/${USER}-miopen-${SLURM_JOB_ID}-$k
  HIP_VISIBLE_DEVICES=$g singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
    source /projappl/project_462001140/envs/torchenv/bin/activate; export PYTHONPATH=/scratch/project_462001140/ammar/eccv/cran-pm/src;
    export MPLCONFIGDIR=/tmp/mpl-$k; mkdir -p $DB; export MIOPEN_USER_DB_PATH=$DB MIOPEN_CUSTOM_CACHE_DIR=$DB;
    python -u eval_coarse.py runs/B_${m}_mettarget31$s --year 2022 --met-source hres" > logs/records_${m}$s.log 2>&1 &
  k=$((k + 1)); [ $((k % 8)) -eq 0 ] && wait
done; done
wait
ls runs/B_*_mettarget31*/records_coarse_2022_hres.npy | wc -l
