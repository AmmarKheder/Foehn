#!/bin/bash
#SBATCH --job-name=a17_main
#SBATCH --account=<project>
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=01:20:00
#SBATCH --output=../data/a17_gpu.out
#SBATCH --open-mode=append
# A17 main run (a17/A17_draft.md section 5): the 4 negated members (ids 4..7) of the 30 Conf starts x leads 1 and 3 of 2021
# (60 forecasts) at rho* -> data/main/part_0..7.npz (format of a15/data/main + member_ids). 8 ranks, items sorted by (lead,
# start), todo[rank::8]: at most 8 forecasts per rank, about 1:02 at 428 s per forecast; limit 1:20 (8 x 1:20 + the smoke's
# 1:00 = 11.7 GCD-hours, cap 12). Resumable, but at most ONE re-submission, with --time=T such that 8 T plus the GCD-hours
# already used by A17's GPU jobs (sacct) stays within 12; this script refuses a third start (A17_draft.md section 5).
# Refuses to run unless code/out/check17.json and data/smoke_check.json say all_passed (eval17.py also checks the code).
set -uo pipefail
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"
A17C=${FOEHN_WORK}/global_model_forecasting_air_pollution/a17/code
D=${FOEHN_WORK}/global_model_forecasting_air_pollution/a17/data
cd $A17C
grep -q '"all_passed": true' out/check17.json 2>/dev/null && grep -q '"all_passed": true' $D/smoke_check.json 2>/dev/null || { echo "A17 main: code/out/check17.json or data/smoke_check.json missing or not passed"; exit 3; }
NPREV=$(grep -c '^=== A17 main job .* start' $D/a17_gpu.out 2>/dev/null); NPREV=${NPREV:-0}
[ "$NPREV" -le 1 ] || { echo "A17 main: $NPREV main jobs already started; A17 allows one re-submission only"; exit 3; }
W=$(grep -o '"workers": [0-9]*' $D/smoke_check.json | tail -1 | grep -o '[0-9]*$'); W=${W:-4}
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate;
  export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_PROCID; export PYTHONDONTWRITEBYTECODE=1"
echo "=== A17 main job ${SLURM_JOB_ID} start $(date) workers=$W"
srun --cpu-bind=$CPU_BIND singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; cd $A17C; python -u eval17.py --plan main --workers $W"
rc=$?
echo "=== A17 main job ${SLURM_JOB_ID} end $(date) rc=$rc"
exit $rc
