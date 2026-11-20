#!/bin/bash
# A8(e)-2: one srun task (inside the container): this rank's spectra day of the FROZEN system (psdrepro.py), and on
# rank 0 also the in-training validation of the frozen step 22,000 (valrepro.py). Read-only on runs_ghap/pmG_base.
#   bash rank_ref.sh RESULTS_DIR
set -uo pipefail
RES=$1
A8=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_seed1/a8
source "$CDFFM_ENVSH"
cd "$CDFFM_W"
export PYTHONDONTWRITEBYTECODE=1
python -u $A8/psdrepro.py --out $RES/psdrepro & p1=$!
r2=0
if [ "$SLURM_PROCID" = 0 ]; then
  python -u $A8/valrepro.py --run pmG_base --steps 22000 --out $RES/valrepro_seed0.json & p2=$!
  wait $p2; r2=$?
fi
wait $p1; r1=$?
[ $r1 = 0 ] && [ $r2 = 0 ] || { echo "[rank_ref r$SLURM_PROCID] psdrepro rc=$r1 valrepro rc=$r2"; exit 1; }
