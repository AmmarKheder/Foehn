#!/bin/bash
#SBATCH --job-name=a16_g11
#SBATCH --account=<project>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=04:05:00
#SBATCH --array=0-44
#SBATCH --output=logs/g11_%A.out
#SBATCH --open-mode=append
# A16 run G11 (protocol_amendments/A16_test_year_evolving_noise.md section 1): replaces F11 of natcomms/rebuild/REBUILD_PLAN.md
# (16 members at the 8.1 M populated pixels, 2022, the 179 start days x leads 1 and 3 = 358 forecasts) as ONE array of 45
# one-GCD tasks: task k samples the 8 forecasts of line k+1 of jobs/pairs_g11.txt (6 for the last) with sample16_ar.py
# (a13/warn16/sample16.py with only the noise changed, audit hook against station files kept, final.ckpt as the stored
# command) -> a16/data/run16/{pop16,keys16,done16}_k<kk>.* (format of a14/data/run16_13) + a16_run.json.
# Layout and environment of a14/jobs/array16_13.sh; limit = longest stored task (02:40:39) x 1.5. Resumable per task.
# Refuses to run unless results/check_cpu.json and results/smoke_check.json say all_passed and rho* = 0.59.
set -uo pipefail
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
A16=${FOEHN_WORK}/global_model_forecasting_air_pollution/a16
RUN=${FOEHN_WORK}/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4
CK=$RUN/ckpts/epoch=01.ckpt
FINAL=$RUN/final.ckpt
cd $A16
grep -q '"all_passed": true' results/check_cpu.json 2>/dev/null && grep -q '"all_passed": true' results/smoke_check.json 2>/dev/null || { echo "A16 G11: results/check_cpu.json or results/smoke_check.json missing or not passed"; exit 3; }
RHO=$(grep -o '"rho_star": [0-9.]*' ${FOEHN_WORK}/global_model_forecasting_air_pollution/a15/results/rho.json | grep -o '[0-9.]*$')
[ "$RHO" = "0.59" ] || { echo "A16 G11: rho* of a15/results/rho.json is '$RHO', A16 section 1 says 0.59"; exit 3; }
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate;
  export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_JOB_ID; export PYTHONDONTWRITEBYTECODE=1"
K=${SLURM_ARRAY_TASK_ID}
KK=$(printf %02d $K)
PAIRS=$(sed -n "$((K + 1))p" $A16/jobs/pairs_g11.txt)
if [ -z "$PAIRS" ]; then echo "A16 G11: no line $((K + 1)) in jobs/pairs_g11.txt"; exit 3; fi
ARGS="--plan G11 --ckpt $FINAL --members 16 --pairs $PAIRS --out $A16/data/run16 --tag k$KK --rho $RHO --base philox"
echo "=== A16 G11 task $K job ${SLURM_JOB_ID} start $(date) args=$ARGS"
srun singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; cd $A16; python -u sample16_ar.py $ARGS"
rc=$?
echo "=== A16 G11 task $K job ${SLURM_JOB_ID} end $(date) rc=$rc"
exit $rc
