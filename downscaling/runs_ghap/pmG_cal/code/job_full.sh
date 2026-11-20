#!/bin/bash
#SBATCH --job-name=cal_full
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_cal/logs/%x_%j.out
#SBATCH --time=01:40:00
# PROTOCOL.md A6(f) stage 3: the full selection evaluation of A2(c) (eval_x, all mask pixels x 24 SEL days, 8 members x 10
# Euler, P3, guard T3, regime-box spectra) + validation_gates.py.   sbatch --export=ALL,RUN=V2,STEP=1500 job_full.sh
source /scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_cal/code/job_head.sh
RUN=${RUN:?}; STEP=${STEP:?}; TAU=${TAU:-1.0}; KNEE=${KNEE:-none}
SUF=""; [ "$TAU" != "1.0" ] && SUF="_tau$TAU"; [ "$KNEE" != "none" ] && SUF="${SUF}_k$KNEE"
TAG=sel2021_M8_s10_P3_guard_T3_s${STEP}$SUF
E=$C/runs/$RUN/eval_$TAG
echo "start $(date) node=$SLURMD_NODENAME run=$RUN step=$STEP tau=$TAU"
note "FULL selection evaluation of $RUN step $STEP tau $TAU knee $KNEE started on $SLURMD_NODENAME"
[ -f $E/metrics.json ] || run8 "export CAL_TAU=$TAU CAL_KNEE=$KNEE; python -u -m calib.eval_cal --run $RUN --runs-root $C/runs --ckpt $STEP $EVAL_COMMON --psd-centres $CENTRES --tag $TAG"
rc=$?
rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
[ $rc = 0 ] || { note "full evaluation FAILED rc=$rc"; exit $rc; }
$PYW -I $W/runs_global/validation_gates.py --eval $E --out $E/VALIDATION_GATES.json
$PYW -I $C/code/calib/gates_cal.py full $E | tee $E/GATES.txt
note "FULL evaluation of $RUN step $STEP tau $TAU knee $KNEE ended after $(elapsed) node-min -> runs/$RUN/eval_$TAG: $(tail -1 $E/GATES.txt)"
echo "end $(date) node-min $(elapsed)"
