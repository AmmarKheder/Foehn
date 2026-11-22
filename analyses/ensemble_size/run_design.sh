#!/bin/bash
#SBATCH --job-name=a14_design
#SBATCH --account=<project>
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=220G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
# REBUILD_PLAN 4c step 8 (A14) under a noise design, in order, CPU only (a14/DESIGN_SWITCH.md):
#   sbatch --export=ALL,FOEHN_DESIGN=ar run_design.sh        the runs of A16 (noise drawn anew at every start day)
#   sbatch --export=ALL,FOEHN_DESIGN=check run_design.sh     validation: the stored runs, labelled outputs, fixed-design intervals
#   sbatch --export=ALL,FOEHN_DESIGN=check0 run_design.sh    validation: as 'check' with V_draw = 0 and factor 1
# FOEHN_DESIGN unset is refused: the stored A14 outputs are never rebuilt by this driver. Every python step runs under the write
# guard of the S1 chain (natcomms/rebuild/s1_chain/run.py: no write to a path without the '_<D>' label); thread counts and
# worker counts are those of the stored jobs (reproduction under 'check'). Order:
#   1 check13.py, check13b.py (training container, as a14/jobs/cpu_job.sh)            -> a14/results_<D>/
#   2 classU passes, in parallel: st_cf_pass, rel_pass, pop_pass stored, land25_pass  -> classU/data_<D>/, results_<D>/
#   3 'check'/'check0' only: pop_pass run16 on a13/warn16/data/run16 (the 30 seen days, as job 23901270)
#   4 r1_pipeline_b.py (gate check13b; pop_pass of the sixteen members, st_cf_score, rel_score, chain_classU, make_table, *_179)
#   5 classE.py run2022 32 check13b                                                   -> classE/data_<D>/, results_<D>/
# The scripts that read only other systems (cams12h_classU, rt_classU, c25_pass, emis_pair, ds2022_classU, r2score) are not run:
# their stored records are read. 0 GCD-hours; nothing here writes outside a14.
set -uo pipefail
: "${FOEHN_DESIGN:?set FOEHN_DESIGN=ar (or check / check0)}"
case "$FOEHN_DESIGN" in ar|check|check0) ;; *) echo "FOEHN_DESIGN=$FOEHN_DESIGN: only ar, check, check0"; exit 2 ;; esac
export FOEHN_DESIGN; D=$FOEHN_DESIGN
A14=${FOEHN_WORK}/global_model_forecasting_air_pollution/a14
A16=$A14/../a16
RD=${FOEHN_WORK}/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4
N=${FOEHN_WORK}/eccv/cran-pm/paper/natcomms
FIG=${FOEHN_WORK}/eccv/cran-pm/experiments/figures
GUARD=$N/rebuild/s1_chain/run.py
PYW=${FOEHN_ENVS}/world/bin/python
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1.sif
LOGD=$A14/logs/design_${D}_${SLURM_JOB_ID:-$$}
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=${FOEHN_ENVS}/.cache
export MPLCONFIGDIR=/tmp/mpl-$USER-${SLURM_JOB_ID:-$$}
need() { for p in "$@"; do [ -e "$p" ] || { echo "missing input: $p"; exit 3; }; done; }
need "$GUARD" "$PYW" "$SIF"
if [ "$D" = ar ]; then
  grep -q '"all_passed": true' $A16/results/check_runs.json 2>/dev/null || { echo "a16/results/check_runs.json: A16 C3/C4 not passed"; exit 3; }
  need $RD/eval_final_crps_e01_ar/part_0.npz $RD/eval_pop_crps_e01_ar/pop_idx.npy $RD/eval2021_sel_crps_e01_ar/part_0.npz \
       $A16/data/run16/a16_run.json $A16/jobs/pairs_g11.txt \
       $FIG/data/compare_fair_crps_e01_ar.npz $FIG/data/compare_ghap_global_crps_e01_ar.npz \
       $N/analyses/reliability/data_ar/st_final.npz $N/analyses/reliability/data_ar/sums_final.npz $N/analyses/reliability/data_ar/sums_pop.npz \
       $N/analyses/reliability/results_ar/station_scores_final.json $N/analyses/reliability/results_ar/estimates_scores.json \
       $A14/../test2022_official/results_ar/scores_official_2022.json \
       $N/analyses/hidden/v3/data_ar/hidden_v3_pop.npz $N/analyses/hidden/v2/data_ar/hidden_bc_pop.npz
fi
echo "start $(date -u +%FT%TZ) FOEHN_DESIGN=$D host $(hostname)"
# step <threads> <label> <command ...>: one step with the stored thread count; stops the driver on the first failure
step() {
  local thr=$1 lab=$2; shift 2; local t0; t0=$(date +%s)
  echo "===== $(date -u +%FT%TZ) $lab (threads $thr)"
  OMP_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr MKL_NUM_THREADS=$thr "$@"; local rc=$?
  echo "===== rc=$rc $(( $(date +%s) - t0 )) s $lab"; return $rc; }
cont() {   # the training container of a14/jobs/cpu_job.sh (check13 and check13b ran there)
  singularity exec -B /projappl,/scratch,/pfs,/appl,/tmp $SIF bash -c "unset PYTHONPATH; source /opt/miniconda3/bin/activate pytorch;
    source ${FOEHN_ENVS}/torchenv/bin/activate; export PYTHONPATH=${FOEHN_WORK}/eccv/cran-pm/src;
    export MPLCONFIGDIR=/tmp/mpl-\$USER; export OMP_NUM_THREADS=8; export PYTHONDONTWRITEBYTECODE=1; export FOEHN_DESIGN=$D;
    cd $A14; python -u -B $GUARD $*"; }

# 1. the check13b-type gate on the sixteen-member forecasts of the design
step 8 check13 cont check13.py 8 || exit 1
step 8 check13b cont check13b.py || exit 1
grep -q '"passed": true' $A14/results_$D/check13b.txt || { echo "check13b of $D did not pass: nothing else run"; exit 4; }

# 2. classU passes (independent; the stored jobs ran them on separate nodes)
cd $A14/classU/scripts || exit 1
mkdir -p "$LOGD"
( step 16 st_cf_pass $PYW -u -B $GUARD st_cf_pass.py ) > $LOGD/st_cf_pass.log 2>&1 & p1=$!
( step 16 rel_pass $PYW -u -B $GUARD rel_pass.py ) > $LOGD/rel_pass.log 2>&1 & p2=$!
( step 32 pop_pass_stored $PYW -u -B $GUARD pop_pass.py stored 12 ) > $LOGD/pop_pass_stored.log 2>&1 & p3=$!
( step 16 land25_pass $PYW -u -B $GUARD land25_pass.py ) > $LOGD/land25_pass.log 2>&1 & p4=$!
bad=0; for p in $p1 $p2 $p3 $p4; do wait $p || bad=1; done
for f in $LOGD/st_cf_pass.log $LOGD/rel_pass.log $LOGD/pop_pass_stored.log $LOGD/land25_pass.log; do echo "#### $f"; cat $f; done
rm -f $LOGD/*.log; rmdir $LOGD
[ $bad -eq 0 ] || { echo "a classU pass failed"; exit 5; }

# 3. the 30 seen start days of the stored design (under 'ar' they are inside a16/data/run16, read by step 4)
if [ "$D" != ar ]; then
  step 32 pop_pass_run16 $PYW -u -B $GUARD pop_pass.py run16 run16 $A14/../a13/warn16/data/run16 --workers 8 || exit 6
fi

# 4. sixteen members on all 179 start days and the class-U table (r1_pipeline_b runs its steps under the same guard)
step 64 r1_pipeline_b $PYW -u -B $GUARD r1_pipeline_b.py || exit 7

# 5. class E (count rules) on all 179 start days, gate check13b of the design
cd $A14/classE || exit 1
step 1 classE_run2022 $PYW -u -B $GUARD classE.py run2022 32 check13b || exit 8
echo "end $(date -u +%FT%TZ) FOEHN_DESIGN=$D: a14/results_$D, classU/{data,results}_$D, classE/{data,results}_$D"
