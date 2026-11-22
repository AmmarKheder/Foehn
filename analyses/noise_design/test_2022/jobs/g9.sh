#!/bin/bash
#SBATCH --job-name=a16_g9
#SBATCH --account=<project>
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=02:15:00
#SBATCH --output=logs/g9_%j.out
# A16 run G9 (protocol_amendments/A16_test_year_evolving_noise.md section 1): completes F9 of natcomms/rebuild/REBUILD_PLAN.md
# section 1 (2021, the 15 Cal start days x leads 2-7, 4 members; format of a15/data/main). The stored command with ONLY the noise changed: eval_ar22.py (eval_global_green.run, marked hunks),
# base philox, rho* of a15/results/rho.json. Layout of the stored job (1 node as A15 run M; 12 forecasts per rank at about 420 s: 1:29 expected); limit = stored wall clock x 1.5.
# Output: a16/data/cal2021 (new folder; refused if it holds a file A16 does not write). Resumable: re-submit the same script.
# Refuses to run unless results/check_cpu.json and results/smoke_check.json say all_passed and rho* = 0.59.
set -uo pipefail
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"
A16=${FOEHN_WORK}/global_model_forecasting_air_pollution/a16
RUN=${FOEHN_WORK}/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4
CK=$RUN/ckpts/epoch=01.ckpt
cd $A16
grep -q '"all_passed": true' results/check_cpu.json 2>/dev/null && grep -q '"all_passed": true' results/smoke_check.json 2>/dev/null || { echo "A16 G9: results/check_cpu.json or results/smoke_check.json missing or not passed"; exit 3; }
RHO=$(grep -o '"rho_star": [0-9.]*' ${FOEHN_WORK}/global_model_forecasting_air_pollution/a15/results/rho.json | grep -o '[0-9.]*$')
[ "$RHO" = "0.59" ] || { echo "A16 G9: rho* of a15/results/rho.json is '$RHO', A16 section 1 says 0.59"; exit 3; }
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate;
  export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_PROCID; export PYTHONDONTWRITEBYTECODE=1"
ARGS="--plan G9 $RUN --ckpt $CK --psub 16 --steps 4 --refine-tail 1 --year 2021 --members 4 --doys 1 25 49 73 97 121 145 169 193 217 241 265 289 313 337 --leads 2 3 4 5 6 7 --out $A16/data/cal2021 --rho $RHO --base philox"
echo "=== A16 G9 job ${SLURM_JOB_ID} start $(date) nodes=${SLURM_NNODES} args=$ARGS"
srun --cpu-bind=$CPU_BIND singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; cd $A16; python -u eval_ar22.py $ARGS"
rc=$?
echo "=== A16 G9 job ${SLURM_JOB_ID} end $(date) rc=$rc"
exit $rc
