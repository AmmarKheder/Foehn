#!/bin/bash
#SBATCH --job-name=a15_smoke
#SBATCH --account=<project>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:15:00
#SBATCH --output=logs/a15.out
#SBATCH --open-mode=append
# A15 run S (a15/A15_draft.md sections 5 and 6.2), one GCD, checkpoint ckpts/epoch=01.ckpt (that of eval2021_sel_crps_e01):
#  (a) eval_ar.py --plan smoke_stored: base stored, rho 1 (= the stored design), start day 9, leads 1 and 2 -> data/smoke/part_stored.npz
#  (b) eval_ar.py --plan smoke_cost: philox, rho 0.70, Cal start 1, lead 7 (cost and finite pattern, never scored) -> part_cost.npz
#  then smoke_check.py on this node (exit 1 on failure: run C depends on this job with afterok).
set -uo pipefail
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
H=${FOEHN_WORK}/global_model_forecasting_air_pollution/a15
cd $H
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate;
  export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_JOB_ID; export PYTHONDONTWRITEBYTECODE=1"
grep -q '"all_passed": true' results/check_cpu.json 2>/dev/null || { echo "A15 smoke: results/check_cpu.json missing or failed"; exit 3; }
echo "=== A15 smoke job ${SLURM_JOB_ID} start $(date)"
rc=0
srun singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; cd $H; python -u eval_ar.py --plan smoke_stored --tag stored" || rc=$?
echo "end (a) $(date) rc=$rc"
[ $rc -eq 0 ] || exit $rc
srun singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; cd $H; python -u eval_ar.py --plan smoke_cost --tag cost" || rc=$?
echo "end (b) $(date) rc=$rc"
[ $rc -eq 0 ] || exit $rc
srun singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; cd $H; python -u smoke_check.py" || rc=$?
echo "=== A15 smoke job ${SLURM_JOB_ID} end $(date) rc=$rc"
exit $rc
