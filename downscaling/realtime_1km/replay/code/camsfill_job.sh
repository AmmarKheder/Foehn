#!/bin/bash
#SBATCH --job-name=rp_camsfill
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay/logs/%x_%j.out
# WP1a, after the CAMS forecast fetch (rp_camsfc): 'cams' at the monitors for every issue day that has the
# ten leads (camsfc/points/<d>.npz), then the safety net: fill 'cams' in the FINISHED points/<d>.npz files where a lead
# day is all-NaN and can now be computed (atomic rewrite, other arrays untouched, provenance updated, report in
# camsfc/fill_report.json). Points files assembled after the fetch already hold 'cams' (assemble.py reads camsfc/ first).
#   sbatch --dependency=afterany:<rp_camsfc job> camsfill_job.sh
set -uo pipefail
R=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/replay
unset PYTHONPATH
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache OMP_NUM_THREADS=1
PY=/projappl/project_462001804/envs/world/bin/python
cd $R/code
echo "start $(date -u +%FT%TZ) camsfill (job $SLURM_JOB_ID)"
cat $R/camsfc/summary.json 2>/dev/null
$PY -I -u cams_points.py points; rc1=$?
$PY -I -u cams_points.py fill; rc2=$?
echo "end $(date -u +%FT%TZ) rc points=$rc1 fill=$rc2; points files: $(ls $R/points/*.npz 2>/dev/null | wc -l)"
exit $(( rc1 + rc2 ))
