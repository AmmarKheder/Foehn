#!/bin/bash
#SBATCH --job-name=cal_screen
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_cal/logs/%x_%j.out
#SBATCH --time=00:50:00
# PROTOCOL.md A6(f) stage 2: eval_x at every 8th mask pixel (24 SEL days) + the regime-box spectra, then the gates.
#   sbatch --export=ALL,RUN=V2,STEP=1500 job_screen.sh        (temperature system: RUN=pmG_ref,STEP=22000,TAU=1.30)
source /scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/pmG_cal/code/job_head.sh
RUN=${RUN:?}; STEP=${STEP:?}; TAU=${TAU:-1.0}; KNEE=${KNEE:-none}
SUF=""; [ "$TAU" != "1.0" ] && SUF="_tau$TAU"; [ "$KNEE" != "none" ] && SUF="${SUF}_k$KNEE"
echo "start $(date) node=$SLURMD_NODENAME run=$RUN step=$STEP tau=$TAU"
note "screening of $RUN step $STEP tau $TAU knee $KNEE started on $SLURMD_NODENAME"
E1=$C/runs/$RUN/eval_scr8_s${STEP}$SUF; E2=$C/runs/$RUN/eval_scrpsd_s${STEP}$SUF
[ -f $E1/metrics.json ] || run8 "export CAL_TAU=$TAU CAL_KNEE=$KNEE; python -u -m calib.eval_cal --run $RUN --runs-root $C/runs --ckpt $STEP $EVAL_COMMON --pixel-stride 8 --tag scr8_s${STEP}$SUF" || { note "screening eval FAILED"; exit 1; }
echo "stride-8 evaluation done after $(elapsed) node-min"
[ -f $E2/metrics.json ] || run8 "export CAL_TAU=$TAU CAL_KNEE=$KNEE CAL_NOHOLES=1; python -u -m calib.eval_cal --run $RUN --runs-root $C/runs --ckpt $STEP $EVAL_COMMON --psd-centres $CENTRES --psd-boxes-only --tag scrpsd_s${STEP}$SUF" || { note "screening spectra FAILED"; exit 1; }
rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
$PYW -I $W/runs_global/validation_gates.py --eval $E1 --out $E1/VALIDATION_GATES.json
$PYW -I $W/runs_global/validation_gates.py --eval $E2 --out $E2/VALIDATION_GATES.json
$PYW -I $C/code/calib/gates_cal.py screen $E1 $E2 | tee $E1/SCREEN.txt
note "screening of $RUN step $STEP tau $TAU knee $KNEE ended after $(elapsed) node-min: $(tail -1 $E1/SCREEN.txt)"
echo "end $(date) node-min $(elapsed)"
