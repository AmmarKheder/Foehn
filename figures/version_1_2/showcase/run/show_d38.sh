#!/bin/bash
#SBATCH --job-name=show_d38  # SHOW
#SBATCH --account=<project>
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=show_d38_%j.out  # SHOW
# A16 run G3 (protocol_amendments/A16_test_year_evolving_noise.md section 1): replaces F3 of natcomms/rebuild/REBUILD_PLAN.md
# section 1 (20 January 2022, lead 1, 1 km maps, 4 members). The stored command with ONLY the noise changed: eval_ar22.py (eval_global_green.run, marked hunks),
# base philox, rho* of a15/results/rho.json. Layout of the stored job (1 node, 00:09:01); limit = stored wall clock x 1.5.
# Output: RUN/eval_jan21_crps_e01_ar (new folder; refused if it holds a file A16 does not write). Resumable: re-submit the same script.
# Refuses to run unless results/check_cpu.json and results/smoke_check.json say all_passed and rho* = 0.59.
# SHOW (natcomms/rebuild/figs_show, 2026-11-21): a marked copy of a16/jobs/g3.sh (changed lines end with '# SHOW'): the forecast of
# run G1 for start day 38 (8 Feb 2022), lead 1, re-run with its 1 km map saved, through figs_show/run/eval_show.py (plan S38);
# every other argument, the checkpoint, the sampler, the noise (rho*, philox) and the layout (1 node, 8 GCDs) are those of G3.
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
SHOW=${FOEHN_WORK}/eccv/cran-pm/paper/natcomms/rebuild/figs_show/run  # SHOW
cd $A16
grep -q '"all_passed": true' results/check_cpu.json 2>/dev/null && grep -q '"all_passed": true' results/smoke_check.json 2>/dev/null || { echo "A16 G3: results/check_cpu.json or results/smoke_check.json missing or not passed"; exit 3; }
RHO=$(grep -o '"rho_star": [0-9.]*' ${FOEHN_WORK}/global_model_forecasting_air_pollution/a15/results/rho.json | grep -o '[0-9.]*$')
[ "$RHO" = "0.59" ] || { echo "A16 G3: rho* of a15/results/rho.json is '$RHO', A16 section 1 says 0.59"; exit 3; }
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate;
  export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_PROCID; export PYTHONDONTWRITEBYTECODE=1"
ARGS="--plan S38 $RUN --ckpt $CK --psub 16 --steps 4 --refine-tail 1 --members 4 --doys 38 --leads 1 --save-1km 38 --out $SHOW/eval_show_d38_crps_e01_ar --rho $RHO --base philox"  # SHOW
echo "=== SHOW d38 job ${SLURM_JOB_ID} start $(date) nodes=${SLURM_NNODES} args=$ARGS"
srun --cpu-bind=$CPU_BIND singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; cd $SHOW; python -u eval_show.py $ARGS"  # SHOW
rc=$?
echo "=== SHOW d38 job ${SLURM_JOB_ID} end $(date) rc=$rc"
exit $rc
