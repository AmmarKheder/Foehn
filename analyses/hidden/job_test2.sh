#!/bin/bash
#SBATCH --job-name=hid_test2
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/hidden/logs/%x_%j.out
# check that figdata_hidden_systems.py with the EVP_EXTRA option (unset) still gives the file of job 23864988 on 4 forecasts
cd /scratch/project_462001804/ammar/eccv/cran-pm/paper/natcomms/analyses/hidden
unset PYTHONPATH; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/projappl/project_462001804/envs/world/bin/python
$PY -u -B figdata_hidden_systems.py 4 0,64
$PY -u -B - <<'PYEOF'
import numpy as np
a = np.load("data/hidden_systems_pop_test2.npz"); b = np.load("data/hidden_systems_pop.npz")
K = b["keys"]; worst = 0.0
for i, (d, h, t) in enumerate(a["keys"]):
    j = int(np.flatnonzero((K[:, 0] == d) & (K[:, 1] == h))[0])
    worst = max(worst, float(np.abs(a["T1"][i] - b["T1"][j]).max()))
t2 = [int(np.flatnonzero(b["targets2"] == t)[0]) for t in a["targets2"]]
w2 = float(np.nanmax(np.abs(a["T2"] - b["T2"][t2])))
print("CHECK max abs deviation of T1 (residents):", worst, "| T2:", w2, "| identical:", worst == 0.0 and w2 == 0.0)
PYEOF
