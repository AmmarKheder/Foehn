#!/bin/bash
#SBATCH --job-name=a7_recheck
#SBATCH --account=project_462001804
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/test2022_official/logs/pred_%x_%j.out
# Amendment A7, diagnosis of the failed check 3 (2026-11-19): the same-day members of the 17 days that ONE process of the
# frozen test's job 23246562 computed (its rank 30) are recomputed a second time, on another node, with the unchanged
# predict_official.py (same code hash as preds/), plus 7 control days. Same-day mode only; no truth is read.
#   sbatch -p dev-g -N 1 -t 0:12:00 predict_recheck.sh        -> preds_recheck/
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
O=$W/test2022_official
SYSD=$W/runs_ghap/pmG_final
note() { printf '%s\n' "- $(date '+%F %H:%M') job $SLURM_JOB_ID (predict recheck, $SLURM_JOB_PARTITION, $SLURM_NNODES node): $1" >> $O/STATUS.md; }
source $W/env.sh
cd $W
export PYTHONDONTWRITEBYTECODE=1
NT=$(( SLURM_NNODES * 8 ))
srunN() { srun --ntasks=$NT --ntasks-per-node=8 --cpus-per-task=7 --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "source $CDFFM_ENVSH; cd $CDFFM_W; $*"; }
echo "start $(date) nodes=$SLURM_JOB_NODELIST"
T0=$(date +%s)
[ -f $SYSD/VALIDATION_GATES.json ] || { note "pmG_final not frozen"; exit 1; }
OUTD=$O/preds_recheck
DAYS=7,16,25,34,43,52,61,70,90,118,166,186,214,262,282,310,358,0,100,101,200,201,300,301
env -u PYTHONPATH PYTHONNOUSERSITE=1 /projappl/project_462001804/envs/world/bin/python -I $O/code/init_preds.py $OUTD || { note "init failed"; exit 1; }
note "started on $SLURM_JOB_NODELIST ($NT GCDs) -> $OUTD (same-day days $DAYS)"
srunN "python -u test2022_official/code/predict_official.py --out $OUTD --sameday $DAYS --issues none --tag recheck"
rc=$?
rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
DONE=$(env -u PYTHONPATH PYTHONNOUSERSITE=1 /projappl/project_462001804/envs/world/bin/python -I -c "
import numpy as np
print(int(np.load('$OUTD/done_sameday.npy').sum()), 'same-day days')")
note "ended rc=$rc after $(( $(date +%s) - T0 )) s on $SLURM_NNODES node(s): done = $DONE"
exit $rc
