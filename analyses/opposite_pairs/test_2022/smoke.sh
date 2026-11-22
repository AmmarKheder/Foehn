#!/bin/bash
#SBATCH --job-name=a19_smoke
#SBATCH --account=<project>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=../data/a19_gpu.out
#SBATCH --open-mode=append
# A19 smoke gate (a19/A19_draft.md section 6.2), one GCD, about 0.7 GCD-hour (limit 1:00 = at most 1.0); main.sh refuses to
# start unless data/smoke_check.json says all_passed with the same code. Starts from empty data/smoke_alone, data/smoke_eight.
#  (1) eval19.py --plan smoke_alone: member ids 4..7 alone, the G2 forecasts (2, 1) (first, cost) and (2, 3) -> data/smoke_alone
#  (2) eval19.py --plan smoke_eight: member ids 0..7 together, (2, 3)                                  -> data/smoke_eight
#  (3) smoke19.py on this node -> data/smoke_check.json (exit 1 if a gating check fails). No skill statistic is computed.
set -uo pipefail
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
A19C=${FOEHN_WORK}/global_model_forecasting_air_pollution/a19/code
D=${FOEHN_WORK}/global_model_forecasting_air_pollution/a19/data
RUN=${FOEHN_WORK}/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4
CK=$RUN/ckpts/epoch=01.ckpt
cd $A19C
grep -q '"all_passed": true' out/check19.json 2>/dev/null || { echo "A19 smoke: code/out/check19.json missing or not passed"; exit 3; }
[ -z "$(ls -A $D/smoke_alone 2>/dev/null)" ] && [ -z "$(ls -A $D/smoke_eight 2>/dev/null)" ] || { echo "A19 smoke: data/smoke_alone or data/smoke_eight is not empty"; exit 3; }
RHO=$(grep -o '"rho_star": [0-9.]*' ${FOEHN_WORK}/global_model_forecasting_air_pollution/a15/results/rho.json | grep -o '[0-9.]*$')
[ "$RHO" = "0.59" ] || { echo "A19 smoke: rho* of a15/results/rho.json is '$RHO', A19 uses 0.59"; exit 3; }
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate;
  export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_JOB_ID; export PYTHONDONTWRITEBYTECODE=1"
ARGS_A="--plan smoke_alone $RUN --ckpt $CK --psub 16 --steps 4 --refine-tail 1 --members 4 --pop --doys 2 --leads 1 3 --out $D/smoke_alone --rho $RHO --base philox"
ARGS_E="--plan smoke_eight $RUN --ckpt $CK --psub 16 --steps 4 --refine-tail 1 --members 8 --pop --doys 2 --leads 3 --out $D/smoke_eight --rho $RHO --base philox"
X="singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c"
echo "=== A19 smoke job ${SLURM_JOB_ID} start $(date) epoch $(date +%s)"
rc=0
echo "--- A19 smoke step alone $(date)"
srun $X "$ENV; cd $A19C; python -u -B eval19.py $ARGS_A" || rc=$?
echo "--- A19 smoke step alone end $(date) rc=$rc"
[ $rc -eq 0 ] || exit $rc
echo "--- A19 smoke step eight $(date)"
srun $X "$ENV; cd $A19C; python -u -B eval19.py $ARGS_E" || rc=$?
echo "--- A19 smoke step eight end $(date) rc=$rc"
[ $rc -eq 0 ] || exit $rc
echo "--- A19 smoke step check $(date)"
srun $X "$ENV; cd $A19C; python -u -B smoke19.py ${SLURM_JOB_ID}" || rc=$?
echo "=== A19 smoke job ${SLURM_JOB_ID} end $(date) rc=$rc"
exit $rc
