#!/bin/bash
#SBATCH --job-name=ghap_st22
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/test2022_ghap/logs/st22_%j.out
# PROTOCOL.md A2(d), after the freeze: SYSTEM.json of the 2022 pass + 2022 station positions/truth (logged opening)
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
O=$W/test2022_ghap
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1
note() { printf '%s\n' "- $(date '+%F %H:%M') [ghap] job $SLURM_JOB_ID: $1" >> $W/logs/GLOBAL_PREP_STATUS.md; }
trap 'note "2022 stations step FAILED (test2022_ghap/logs/st22_$SLURM_JOB_ID.out)"' ERR
$PY -I - <<'EOP'
import json
from pathlib import Path
W = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
fd = W / "runs_ghap/pmG_final"
g = json.loads((fd / "VALIDATION_GATES.json").read_text())
sysd = {"system": "pmG_final", "system_dir": str(fd), "frozen_utc": g.get("frozen_utc"), "gates_passed": g["passed"],
        "failed_gates": [k for k, v in g["gates"].items() if v.get("pass") is False],
        "label": "gate-passing system" if g["passed"] else "gate-FAILING system (A2(c): tested once, labelled)",
        "rule": "PROTOCOL.md A2(c)-(d)"}
(W / "test2022_ghap/SYSTEM.json").write_text(json.dumps(sysd, indent=1))
print(sysd)
EOP
$PY -I $O/stations_2022.py
note "2022 stations opened (logged) -> test2022_ghap/st_pix_2022.npz, st_truth_2022.npz, st_qc_2022.json; SYSTEM.json written"
