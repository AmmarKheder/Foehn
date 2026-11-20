#!/bin/bash
#SBATCH --job-name=rp_prep
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay/logs/%x_%j.out
# WP1a step B1 (CPU, internet): prep_replay.py for a list of issue days, 8 at a time. One log for the whole job.
#   sbatch prep_job.sh <met|pm|all> <file with one day per line>
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1   # no .pyc next to the imported production modules
R=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay
ONLY=$1; LIST=$2
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/projappl/project_462001804/envs/world/bin/python
cd $R/code
echo "start $(date -u +%FT%TZ) prep $ONLY $(wc -l < $LIST) days (job $SLURM_JOB_ID)"
T0=$(date +%s)
one() {
  d=$1
  if [ "$2" != pm ] && [ -f $R/work/$d/met_ifs.npy ] && grep -q seconds_met $R/work/$d/prep.json 2>/dev/null; then
    [ "$2" = met ] && { echo "$d: met_ifs.npy exists, skipped"; return 0; }
  fi
  out=$($PY -I -u prep_replay.py $d --only $2 2>&1); rc=$?
  echo "$out" | tail -n 2 | sed "s/^/$d: /"
  [ $rc = 0 ] || { echo "$d: FAILED rc=$rc"; echo "$out" | tail -n 12 | sed "s/^/$d ! /"; }
  return $rc
}
export -f one; export R PY
xargs -a $LIST -P 8 -I{} bash -c 'one {} '"$ONLY"
rc=$?
echo "end $(date -u +%FT%TZ) rc=$rc wall $(( $(date +%s) - T0 )) s; met files: $(ls $R/work/*/met_ifs.npy 2>/dev/null | wc -l)"
exit $rc
