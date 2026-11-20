#!/bin/bash
#SBATCH --job-name=cdffm_g_flexgreen
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/global_flexgreen_%j.out
# FLEXGREEN ERA5 lead 1, start days 2022-01-01 .. 2024-12-30 (PLAN.md 2.7 item 5) -> data/processed/global/flexgreen/era5/
# 8 ranks share the days of each year (flexgreen.py convention); done markers are folded into done_<y>.txt at the end.
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
S=$W/logs/BUILD_STATUS.md
source $W/env.sh
O=$W/data/processed/global/flexgreen
note() { printf '%s\n' "- $(date '+%F %H:%M') [global-build] job $SLURM_JOB_ID: $1" >> "$S"; }
note "FLEXGREEN lead-1 run started (standard-g, 8 GCDs)"
rc=0
for Y in 2022 2023 2024; do
  echo "=== $Y $(date +%T)"
  srun --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B $CDFFM_BIND $CDFFM_SIF bash -c "source $CDFFM_ENVSH; cd $W/transport; \
    python -u flexgreen_pm.py run --src era5 --year $Y --days \$(seq 0 364) --max-lead 1 --out-root $O \
      --wb-extra $W/data/processed/global/wb2 --fires-new $W/data/processed/global/coarse/fires_2023_2024.npy" || rc=1
  n=$(ls $O/era5/done_$Y 2>/dev/null | wc -l)
  echo "=== $Y done markers: $n $(date +%T)"
  [ "$n" = 365 ] || rc=1
done
if [ $rc = 0 ]; then
  for Y in 2022 2023 2024; do
    ls $O/era5/done_$Y | sed 's/^0*//; s/^$/0/' | sort -n > $O/era5/done_$Y.txt && [ $(wc -l < $O/era5/done_$Y.txt) = 365 ] && rm -r $O/era5/done_$Y
  done
  note "FLEXGREEN DONE: 1,095 start days (2022-01-01 .. 2024-12-30), markers folded into done_<y>.txt"
else
  note "FLEXGREEN incomplete (rc=$rc), see logs/global_flexgreen_$SLURM_JOB_ID.out; rerun resumes from the done markers"
fi
exit $rc
