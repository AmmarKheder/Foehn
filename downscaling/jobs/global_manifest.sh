#!/bin/bash
#SBATCH --job-name=cdffm_g_manifest
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/global_manifest_%j.out
# manifest (no hash) -> CPU loader test (GlobalStore + unchanged data_aq.BatchStream) -> manifest with SHA-256.
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/BUILD_STATUS.md
PY=/projappl/project_462001804/envs/world/bin/python
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=8
note() { printf '%s\n' "- $(date '+%F %H:%M') [global-build] job $SLURM_JOB_ID: $1" >> "$S"; }
trap 'note "manifest/loader test FAILED, see logs/global_manifest_$SLURM_JOB_ID.out"' ERR
$PY -I $W/scripts/build_global_manifest.py --no-hash
singularity exec -B /scratch,/pfs,/projappl $SIF bash -c "unset PYTHONPATH; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=16; source /opt/miniconda3/bin/activate pytorch; python -u $W/scripts/test_global_loader.py --batches 20"
note "CPU loader test PASSED (manifest_parts/loader_test.json)"
$PY -I $W/scripts/build_global_manifest.py --procs 32
note "manifest.json written with SHA-256"
