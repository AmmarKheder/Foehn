#!/bin/bash
#SBATCH --job-name=a19_main
#SBATCH --account=<project>
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=00:55:00
#SBATCH --output=../data/a19_gpu.out
#SBATCH --open-mode=append
# A19 main run (a19/A19_draft.md section 5): the 4 negated members (ids 4..7) of A16's G2 forecasts (2022, 179 start days x
# leads 1 and 3 = 358, populated pixels) -> data/pop_neg (format of RUN/eval_pop_crps_e01_ar, part_<r> = the forecasts of G2's
# part_<r>). Layout of a16/jobs/g2.sh (8 nodes, 64 GCDs); limit 0:55 (G2 ran 0:44:42 = 47.7 GCD-hours), so that the worst
# case 64 x 0:55 = 58.7 GCD-hours plus the smoke's at most 1.0 stays inside the cap of 60. No re-submission: this script
# refuses to start when a main job has already started (a main start line in data/a19_gpu.out); if main stops before its
# 358 forecasts, A19 stops and reports. Refuses to run unless code/out/check19.json and data/smoke_check.json say all_passed
# (eval19.py also checks the code and the smoke's projection).
set -uo pipefail
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"
A19C=${FOEHN_WORK}/global_model_forecasting_air_pollution/a19/code
D=${FOEHN_WORK}/global_model_forecasting_air_pollution/a19/data
RUN=${FOEHN_WORK}/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4
CK=$RUN/ckpts/epoch=01.ckpt
cd $A19C
grep -q '"all_passed": true' out/check19.json 2>/dev/null && grep -q '"all_passed": true' $D/smoke_check.json 2>/dev/null || { echo "A19 main: code/out/check19.json or data/smoke_check.json missing or not passed"; exit 3; }
RHO=$(grep -o '"rho_star": [0-9.]*' ${FOEHN_WORK}/global_model_forecasting_air_pollution/a15/results/rho.json | grep -o '[0-9.]*$')
[ "$RHO" = "0.59" ] || { echo "A19 main: rho* of a15/results/rho.json is '$RHO', A19 uses 0.59"; exit 3; }
NPREV=$(grep -c '^=== A19 main job .* start' $D/a19_gpu.out 2>/dev/null); NPREV=${NPREV:-0}
[ "$NPREV" -eq 0 ] || { echo "A19 main: $NPREV main job(s) already started; A19 allows no re-submission: A19 stops and reports"; exit 3; }
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate;
  export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_PROCID; export PYTHONDONTWRITEBYTECODE=1"
ARGS="--plan main $RUN --ckpt $CK --psub 16 --steps 4 --refine-tail 1 --members 4 --pop --every 2 --leads 1 3 --out $D/pop_neg --rho $RHO --base philox"
echo "=== A19 main job ${SLURM_JOB_ID} start $(date) nodes=${SLURM_NNODES}"
srun --cpu-bind=$CPU_BIND singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; cd $A19C; python -u -B eval19.py $ARGS"
rc=$?
echo "=== A19 main job ${SLURM_JOB_ID} end $(date) rc=$rc"
exit $rc
