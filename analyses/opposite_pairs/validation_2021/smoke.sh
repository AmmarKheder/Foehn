#!/bin/bash
#SBATCH --job-name=a17_smoke
#SBATCH --account=<project>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=../data/a17_gpu.out
#SBATCH --open-mode=append
# A17 smoke gate (a17/A17_draft.md section 6.2), one GCD, about 0.6 GCD-hour; main.sh refuses to start unless
# data/smoke_check.json says all_passed with the same code. Starts from empty data/smoke_alone and data/smoke_eight.
#  (1) eval17.py --plan smoke_alone: member ids 4..7 alone, Conf lead-2 forecasts (9, 2) and (17, 2) -> data/smoke_alone
#  (2) eval17.py --plan smoke_eight: member ids 0..7 together, (9, 2)                                 -> data/smoke_eight
#  (3) smoke17.py on this node -> data/smoke_check.json (exit 1 if a gating check fails). No skill statistic is computed.
# Lead 2 is not a scored lead of A17 (leads 1 and 3), so the smoke spends nothing on a scored forecast.
set -uo pipefail
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
A17C=${FOEHN_WORK}/global_model_forecasting_air_pollution/a17/code
D=${FOEHN_WORK}/global_model_forecasting_air_pollution/a17/data
cd $A17C
grep -q '"all_passed": true' out/check17.json 2>/dev/null || { echo "A17 smoke: code/out/check17.json missing or not passed"; exit 3; }
[ -z "$(ls -A $D/smoke_alone 2>/dev/null)" ] && [ -z "$(ls -A $D/smoke_eight 2>/dev/null)" ] || { echo "A17 smoke: data/smoke_alone or data/smoke_eight is not empty"; exit 3; }
RHO=$(grep -o '"rho_star": [0-9.]*' ${FOEHN_WORK}/global_model_forecasting_air_pollution/a15/results/rho.json | grep -o '[0-9.]*$')
[ "$RHO" = "0.59" ] || { echo "A17 smoke: rho* of a15/results/rho.json is '$RHO', A17 uses 0.59"; exit 3; }
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate;
  export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_JOB_ID; export PYTHONDONTWRITEBYTECODE=1"
X="singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c"
echo "=== A17 smoke job ${SLURM_JOB_ID} start $(date)"
rc=0
srun $X "$ENV; cd $A17C; python -u eval17.py --plan smoke_alone" || rc=$?
echo "--- A17 smoke step alone end $(date) rc=$rc"
[ $rc -eq 0 ] || exit $rc
srun $X "$ENV; cd $A17C; python -u eval17.py --plan smoke_eight" || rc=$?
echo "--- A17 smoke step eight end $(date) rc=$rc"
[ $rc -eq 0 ] || exit $rc
srun $X "$ENV; cd $A17C; python -u smoke17.py ${SLURM_JOB_ID}" || rc=$?
echo "=== A17 smoke job ${SLURM_JOB_ID} end $(date) rc=$rc"
exit $rc
