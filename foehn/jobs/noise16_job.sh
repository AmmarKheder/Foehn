#!/bin/bash
#SBATCH --job-name=g1km_noise16
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/logs/%x_%j.out
source /scratch/project_462001140/ammar/eccv/world/env.sh
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km
python -u prep_test.py noise 16
python - <<'PY'
import numpy as np
a = np.load("/scratch/project_462001140/ammar/eccv/data/world/global1km/noise_tiles.npy", mmap_mode="r")
b = np.load("/scratch/project_462001140/ammar/eccv/data/world/global1km/noise_tiles16.npy", mmap_mode="r")
print("members 0-3 identical to noise_tiles.npy:", all(np.array_equal(a[m, 10, 20], b[m, 10, 20]) for m in range(4)), "| shape", b.shape)
PY
