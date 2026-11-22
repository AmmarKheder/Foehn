#!/bin/bash
#SBATCH --job-name=a16_smoke
#SBATCH --account=<project>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=01:40:00
#SBATCH --output=logs/a16.out
#SBATCH --open-mode=append
# A16 check C2 (protocol_amendments/A16_test_year_evolving_noise.md section 3), one GCD, about 1.0 GCD-hour; the runs G1-G9
# and G11 refuse to start unless results/smoke_check.json says all_passed. Starts from an empty a16/data/smoke.
#  (a)   eval_ar22.py base stored, rho 1: F1's forecasts (0, 1) and (0, 2) -> data/smoke/a (reproduction of eval_final_crps_e01)
#  (b4)  eval_ar22.py rho*, 4 members, --pop: (2, 3) as in G2                     -> data/smoke/b4
#  (b16) sample16_ar.py rho*, 16 members: (2, 3) as in G11                         -> data/smoke/b16
#  (c)   eval_ar22.py rho*, 2021, Conf start 9, lead 2 (reproduction of a15/data/main) -> data/smoke/c
#  then smoke_check.py on this node -> results/smoke_check.json (exit 1 if a gating check fails).
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
grep -q '"all_passed": true' results/check_cpu.json 2>/dev/null || { echo "A16 smoke: results/check_cpu.json missing or not passed"; exit 3; }
[ -z "$(ls -A $A16/data/smoke 2>/dev/null)" ] || { echo "A16 smoke: a16/data/smoke is not empty"; exit 3; }
RHO=$(grep -o '"rho_star": [0-9.]*' ${FOEHN_WORK}/global_model_forecasting_air_pollution/a15/results/rho.json | grep -o '[0-9.]*$')
[ "$RHO" = "0.59" ] || { echo "A16 smoke: rho* of a15/results/rho.json is '$RHO', A16 section 1 says 0.59"; exit 3; }
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate;
  export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_JOB_ID; export PYTHONDONTWRITEBYTECODE=1"
ARGS_A="--plan S_a $RUN --ckpt $CK --psub 16 --steps 4 --refine-tail 1 --members 4 --doys 0 --leads 1 2 --out $A16/data/smoke/a --rho 1 --base stored"
ARGS_B4="--plan S_b4 $RUN --ckpt $CK --psub 16 --steps 4 --refine-tail 1 --members 4 --pop --doys 2 --leads 3 --out $A16/data/smoke/b4 --rho $RHO --base philox"
ARGS_B16="--plan S_b16 --ckpt $FINAL --members 16 --pairs 2:3 --out $A16/data/smoke/b16 --tag s --rho $RHO --base philox"
ARGS_C="--plan S_c $RUN --ckpt $CK --year 2021 --psub 16 --steps 4 --refine-tail 1 --members 4 --doys 9 --leads 2 --out $A16/data/smoke/c --rho $RHO --base philox"
X="singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c"
echo "=== A16 smoke job ${SLURM_JOB_ID} start $(date)"
rc=0
echo "--- A16 smoke step a $(date)"
srun $X "$ENV; cd $A16; python -u eval_ar22.py $ARGS_A" || rc=$?
echo "--- A16 smoke step a end $(date) rc=$rc"
[ $rc -eq 0 ] || exit $rc
echo "--- A16 smoke step b4 $(date)"
srun $X "$ENV; cd $A16; python -u eval_ar22.py $ARGS_B4" || rc=$?
echo "--- A16 smoke step b4 end $(date) rc=$rc"
[ $rc -eq 0 ] || exit $rc
echo "--- A16 smoke step b16 $(date)"
srun $X "$ENV; cd $A16; python -u sample16_ar.py $ARGS_B16" || rc=$?
echo "--- A16 smoke step b16 end $(date) rc=$rc"
[ $rc -eq 0 ] || exit $rc
echo "--- A16 smoke step c $(date)"
srun $X "$ENV; cd $A16; python -u eval_ar22.py $ARGS_C" || rc=$?
echo "--- A16 smoke step c end $(date) rc=$rc"
[ $rc -eq 0 ] || exit $rc
echo "--- A16 smoke step check $(date)"
srun $X "$ENV; cd $A16; python -u smoke_check.py ${SLURM_JOB_ID}" || rc=$?
echo "=== A16 smoke job ${SLURM_JOB_ID} end $(date) rc=$rc"
exit $rc
