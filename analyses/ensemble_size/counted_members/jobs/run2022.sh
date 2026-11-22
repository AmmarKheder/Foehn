#!/bin/bash
#SBATCH --job-name=a14E_2022
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# A14 class E of 2022 on all 179 start days (classE.py run2022, version 2): samples conf149 (confirmatory), seen30,
# seen30_paperblocks, all179; World, Foehn 3 days against the references 1 day, regions at 4/8/16 members.
# Submit from a14/classE after check13 (job 23900798), choosing the gate of R1 (classE.gate_check13; exit 2 if R1 is not usable):
#   sbatch --dependency=afterany:23900798 jobs/run2022.sh             gate 'check13' (the original gate: check13.json "passed": true
#                                                                     and the A13 rounding level)
#   sbatch --dependency=afterany:23900798 --export=ALL,GATE=check13b jobs/run2022.sh
#                                                                     gate 'check13b' (amendment A14a: check13b "passed": true and
#                                                                     the A13 rounding level of check13.json)
# The pass then checks members 0-3 of every forecast it reads against the stored members again (exit 2 beyond the tolerance).
# From jobs/cpu_job.sh (copy of a14/jobs/cpu_job.sh). CPU only; no station or monitor file.
set -uo pipefail
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${FOEHN_ENVS}/world/bin/python
GATE=${GATE:-check13}
cd ${FOEHN_WORK}/global_model_forecasting_air_pollution/a14/classE
echo "start $(date) host $(hostname) gate=$GATE"
"$PY" -u -B classE.py run2022 32 "$GATE"
rc=$?
echo "end $(date) rc=$rc"
exit $rc
