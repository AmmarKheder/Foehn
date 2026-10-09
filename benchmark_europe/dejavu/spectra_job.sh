#!/bin/bash
#SBATCH --job-name=dv_spectra
#SBATCH --account=project_462001140
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/dejavu/logs/%x_%j.out
# usage: sbatch eval_job.sh runs/dejavu [runs/noise ...]   (2021 and 2022)
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}; export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH; mkdir -p $MIOPEN_USER_DB_PATH
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/dejavu
for run in once; do for yr in once; do
  singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
    source /projappl/project_462001140/envs/torchenv/bin/activate; export MPLCONFIGDIR=/tmp/mpl-\$USER;
    export MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH; export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH;
    python -u spectra.py" || echo FAILED
done; done
