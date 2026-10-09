#!/bin/bash
#SBATCH --job-name=tiledseams
#SBATCH --account=project_462001140
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/regional/logs/%x_%j.out
# one container call per model, so that a GPU crash of one model does not stop the others
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}; export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source /projappl/project_462001140/envs/torchenv/bin/activate; export PYTHONPATH=/scratch/project_462001140/ammar/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-x; export PYTHONPATH=\$PYTHONPATH:/scratch/project_462001140/ammar/eccv/cran-pm/experiments/dejavu"
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/regional
for m in earthformer simvp; do
  timeout 600 singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python -u tiled_seams.py $m" || echo "$m: CRASH (exit $?)"
done
