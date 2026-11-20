#!/bin/bash
#SBATCH --job-name=hid_bc
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/hidden/v2/logs/%x_%j.out
# people warned by the bias-corrected CAMS forecast (quantile map 2017-2020 of the downscaling project; cell value and
# climatological 1 km pattern) at the populated 1 km pixels, 358 forecasts of 2022 (v2/figdata_hidden_bc.py, 32 forked
# workers) -> v2/data/hidden_bc_pop.npz, then the identity check against ../data/hidden_systems_pop.npz
cd /scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/hidden/v2
unset PYTHONPATH; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/projappl/project_462001804/envs/world/bin/python
echo "start $(date -u +%FT%TZ)"
/usr/bin/time -v $PY -u -B figdata_hidden_bc.py 32 2>&1 | grep -v "^\s*\(Page\|Swaps\|File\|Socket\|Signals\|Voluntary\|Involuntary\|Average\|Major\|Minor\|Percent\)"
$PY -u -B check_identity.py data/hidden_bc_pop.npz
echo "end $(date -u +%FT%TZ)"
