#!/bin/bash
#SBATCH --job-name=a7_pred
#SBATCH --account=project_462001804
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/test2022_official/logs/pred_%x_%j.out
# Amendment A7, model process (frozen runs_ghap/pmG_final under the GhapStore firewall; no truth is read).
# Environment and srun line of test2022_ghap/predict_2022.sh.
#   sbatch -p standard-g -N 1 -t 0:30:00 predict.sh smoke     (8 same-day days + 8 issue days -> preds_smoke/)
#   sbatch -p standard-g -N 1 -t 2:30:00 predict.sh full      (365 days + 214 issue days -> preds/)
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
O=$W/test2022_official
SYSD=$W/runs_ghap/pmG_final
MODE=${1:-full}
note() { printf '%s\n' "- $(date '+%F %H:%M') job $SLURM_JOB_ID (predict $MODE, $SLURM_JOB_PARTITION, $SLURM_NNODES node): $1" >> $O/STATUS.md; }
source $W/env.sh
cd $W
export PYTHONDONTWRITEBYTECODE=1
NT=$(( SLURM_NNODES * 8 ))
srunN() { srun --ntasks=$NT --ntasks-per-node=8 --cpus-per-task=7 --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "source $CDFFM_ENVSH; cd $CDFFM_W; $*"; }
echo "start $(date) nodes=$SLURM_JOB_NODELIST mode=$MODE"
T0=$(date +%s)
[ -f $SYSD/VALIDATION_GATES.json ] || { note "pmG_final not frozen"; exit 1; }
[ -f $O/pos/st_ring_official.npz ] && [ -f $O/inputs/inputs.json ] || { note "positions or inputs missing: not started"; exit 1; }
if [ "$MODE" = smoke ]; then
  OUTD=$O/preds_smoke
  ARGS="--sameday 0,120,121,150,200,250,300,333 --issues 120,121,150,180,200,250,300,333 --tag smoke"
else
  OUTD=$O/preds
  ARGS="--tag full"
fi
env -u PYTHONPATH PYTHONNOUSERSITE=1 /projappl/project_462001804/envs/world/bin/python -I $O/code/init_preds.py $OUTD || { note "init failed"; exit 1; }
note "started on $SLURM_JOB_NODELIST ($NT GCDs) -> $OUTD"
srunN "python -u test2022_official/code/predict_official.py --out $OUTD $ARGS"
rc=$?
rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
DONE=$(env -u PYTHONPATH PYTHONNOUSERSITE=1 /projappl/project_462001804/envs/world/bin/python -I -c "
import numpy as np, sys
d='$OUTD'
print(int(np.load(d + '/done_sameday.npy').sum()), 'same-day days,', int(np.load(d + '/done_chain.npy').sum()), 'issue days')")
note "ended rc=$rc after $(( ($(date +%s) - T0) / 60 )) min ($(( $(date +%s) - T0 )) s) on $SLURM_NNODES node(s): done = $DONE"
exit $rc
