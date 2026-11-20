#!/bin/bash
#SBATCH --job-name=a7_screen
#SBATCH --account=project_462001804
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/test2022_official/logs/score_%j.out
# Amendment A7, sensitivity (v): the scoring with the implausible EEA values removed (code/score_screened.py).
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
O=$W/test2022_official
PY=/projappl/project_462001804/envs/world/bin/python
unset PYTHONPATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/projappl/project_462001804/envs/.cache
note() { printf '%s\n' "- $(date '+%F %H:%M') job $SLURM_JOB_ID (scoring side, screened sensitivity): $1" >> $O/STATUS.md; }
T0=$(date +%s)
printf '%s\n' "- $(date '+%F %T') (Helsinki), job $SLURM_JOB_ID: [A7] score_screened.py reads the 2022 observations of the six official networks and test2022_official/truth/ghap_official_2022.npy a second time for the sensitivity (v) of the protocol note (EEA values <= 0 or > 300 ug m-3 removed) -> test2022_official/screened/results/. Evaluation only." >> $O/UNSEALING_A7.md
$PY -I $O/code/score_screened.py score; rc=$?
note "step score rc=$rc ($(( $(date +%s) - T0 )) s)"
[ $rc = 0 ] || exit $rc
$PY -I $O/code/score_screened.py tables; rc=$?
note "step tables rc=$rc"
exit $rc
