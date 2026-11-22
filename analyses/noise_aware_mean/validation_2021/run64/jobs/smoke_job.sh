#!/bin/bash
#SBATCH --job-name=a14_r2smoke
#SBATCH --account=<project>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=../logs/%x_%j.out
# A14 run R2, smoke test (protocol_amendments/A14_infinite_ensemble.md, section 5), one GCD:
#  (a) eval_global_green.py UNCHANGED, --year 2021 --pop, 4 members, 4 steps, start day 9 lead 1, ckpts/epoch=01.ckpt (the
#      checkpoint of eval2021_sel_crps_e01), into a14/r2/smoke_run/<run name> (same name: same flags parsed from it);
#  (b) sample_m.py --year 2021 --noise noise_tiles64.npy, 4 members, final.ckpt, 9:1 -> a14/r2/smoke (tag b9);
#  (b2) the same on the first R2 forecast (first item of jobs/pairs2021.txt) -> tag b2, for test (c) after the run;
#  then smoke_check.py (exit 1 on failure: the 64-member array depends on this job with afterok).
set -uo pipefail
unset PYTHONPATH
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export OMP_NUM_THREADS=4
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
RUN=${FOEHN_WORK}/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4
H=${FOEHN_WORK}/global_model_forecasting_air_pollution/a14/r2
cd $H
FIRST=$(head -1 jobs/pairs2021.txt | awk '{print $1}')
[ -n "$FIRST" ] || { echo "no jobs/pairs2021.txt"; exit 3; }
mkdir -p smoke_run/$(basename $RUN) smoke
ENV="mkdir -p \$MIOPEN_USER_DB_PATH; unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
  source ${FOEHN_ENVS}/torchenv/bin/activate;
  export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src; export MPLCONFIGDIR=/tmp/mpl-\$SLURM_JOB_ID; export PYTHONDONTWRITEBYTECODE=1"
rc=0
echo "start $(date) (a)"
srun singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; cd ${FOEHN_WORK}/eccv/cran-pm/experiments/global1km; python -u eval_global_green.py $H/smoke_run/$(basename $RUN) --ckpt $RUN/ckpts/epoch=01.ckpt --members 4 --steps 4 --year 2021 --doys 9 --leads 1 --pop --out smoke2021 --workers 4" || rc=$?
echo "end (a) $(date) rc=$rc"
[ $rc -eq 0 ] || exit $rc
echo "start $(date) (b) 9:1 and (b2) $FIRST"
srun singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python -u sample_m.py --ckpt $RUN/final.ckpt --members 4 --year 2021 --noise noise_tiles64.npy --pairs 9:1 --out smoke --tag b9" || rc=$?
[ $rc -eq 0 ] || exit $rc
srun singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python -u sample_m.py --ckpt $RUN/final.ckpt --members 4 --year 2021 --noise noise_tiles64.npy --pairs $FIRST --out smoke --tag b2" || rc=$?
echo "end (b) $(date) rc=$rc"
[ $rc -eq 0 ] || exit $rc
srun singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "$ENV; python -u smoke_check.py" || rc=$?
echo "end smoke_check $(date) rc=$rc"
exit $rc
