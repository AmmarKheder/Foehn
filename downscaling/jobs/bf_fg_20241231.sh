#!/bin/bash
#SBATCH --job-name=bf_fg_1231
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/logs/bf_fg_1231_%j.out
# Backfill: the FLEXGREEN start day 2024-12-31 (doy index 365 of 2024, lead 1), absent from the 2024 store because its ERA5
# days 2025-01-01.. did not exist when 2024 was built. T6 2025-01-01 (row 0 of t6_2025.npy) needs it, and store_global.Backfill2025
# serves NO 2025-2026 day while t6_2025 has no complete-day entry (COVERAGE_2025_2026.json). One rank, as the tail gate.
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
O=$W/data/processed/global/flexgreen
S=$W/logs/GLOBAL_PREP_STATUS.md
note() { printf '%s\n' "- $(date '+%F %H:%M') [backfill] job $SLURM_JOB_ID: $1" >> "$S"; }
source $W/env.sh
srun -n 1 --ntasks-per-node=1 --cpus-per-task=7 --gpus-per-node=8 singularity exec -B $CDFFM_BIND $CDFFM_SIF bash -c "source $CDFFM_ENVSH; cd $W/transport; \
  python -u flexgreen_pm.py run --src era5 --year 2024 --days 365 --max-lead 1 --out-root $O \
    --wb-extra $W/data/processed/global/wb2 --fires-new $W/data/processed/global/coarse/fires_2023_2024.npy" \
  || { note "FLEXGREEN 2024-12-31 FAILED (logs/bf_fg_1231_$SLURM_JOB_ID.out)"; exit 1; }
if [ -d $O/era5/done_2024 ]; then          # marker folded into done_2024.txt (inode budget); build_global_t6 reads either
  ( ls $O/era5/done_2024 | sed 's/^0*//; s/^$/0/'; cat $O/era5/done_2024.txt 2>/dev/null; true ) | sort -n | uniq > $O/era5/done_2024.tmp \
    && mv $O/era5/done_2024.tmp $O/era5/done_2024.txt && rm -r $O/era5/done_2024
fi
grep -qx 365 $O/era5/done_2024.txt || { note "FLEXGREEN 2024-12-31: no done marker after the run"; exit 1; }
note "FLEXGREEN start day 2024-12-31 built (origins_2024.npy row 365; done_2024.txt now $(wc -l < $O/era5/done_2024.txt) days) -> T6 2025 row 0 can be built"
