#!/bin/bash
#SBATCH --job-name=ftM_full
#SBATCH --account=<project>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=logs/full_%j.out
# THE FULL RUN of the final test (draft Addendum 8 of A11), submitted separately once FILED exists: the 48 issue days, arms free
# (identity with the record), free0 (diagnostic of check 4: pmG_final without the fitted prior, the base noise of
# m1/m2), m1 (primary: monitors of d - 1) and m2 (sensitivity: monitors of d - 2), 1 GCD. infer_m.py refuses '--days all'
# with any other configuration.
# Time limit = the cap of the addendum (smoke price x 2, out/smoke_check_<job>.json). RESUMABLE: out/forecasts.npz is
# rewritten every 6 issue days (temporary file + fsync + rename); a re-submission continues with the days not done.
# Submit only after FILED exists (the addendum is filed) -- the order is in ADDENDUM_draft.md.
set -uo pipefail
W=${FOEHN_WORK}/global_model_forecasting_air_pollution
FT=$W/a11/stageM/final_test
J=$SLURM_JOB_ID
[ -f $FT/FILED ] || { echo "[full] FILED missing: the addendum is not filed"; exit 2; }
source $W/env.sh
export PYTHONDONTWRITEBYTECODE=1
trap 'rm -rf "/tmp/${USER}-cdffm-$J"' EXIT
T0=$(date +%s)
echo "[full] start $(date -u +%FT%TZ) node $SLURMD_NODENAME"
srun --ntasks=1 --cpus-per-task=7 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" \
  bash -c "source $CDFFM_ENVSH; export HIP_VISIBLE_DEVICES=\$SLURM_LOCALID OMP_NUM_THREADS=7 PYTHONDONTWRITEBYTECODE=1; cd $FT; \
           python -u infer_m.py --days all --arms free,free0,m1,m2 --out $FT/out/forecasts.npz --save-every 6"
rc=$?
echo "[full] end $(date -u +%FT%TZ) rc=$rc wall $(( $(date +%s) - T0 )) s"
exit $rc
