#!/bin/bash
# Queue the 2022 evaluations of every run behind its training job (afterok):
#   eval_test2022        normal inputs
#   eval_test2022_wshift wind prior (order / upwind bias) from a day 15 days away, other inputs true
#   eval_test2022_shift  whole meteorology (ERA5 + CAMS) from a day 15 days away
# plus the CAMS EAC4 reference. usage: bash submit_evals.sh
cd /scratch/project_462001140/ammar/eccv/cran-pm/experiments/retrain
jobid() { squeue -u $USER -h -n "$1" -o %i | head -1; }
ev() {  # name variant run dep extra...
  local name=$1 v=$2 run=$3 dep=$4 sub=$5; shift 5
  [ -f runs/$run/$sub/metrics.json ] && return
  squeue -u $USER -h -n ev_$name | grep -q . && return  # already queued
  sbatch ${dep:+--dependency=afterok:$dep} --job-name=ev_$name -o logs/ev_${name}_%j.out \
    eval_job.sh $v runs/$run/ckpt runs/$run/$sub "$@"
}
for seed in 42 43 44; do
  for v in V1 V1r V1s V2 V2s V1x V2x; do
    run=$v; [ $seed != 42 ] && run=${v}_s$seed
    dep=$(jobid cran_${v}_s$seed)
    ev ${run} $v $run "$dep" eval_test2022
    ev ${run}_met $v $run "$dep" eval_test2022_shift --met shift
    case $v in V1|V2) ev ${run}_wind $v $run "$dep" eval_test2022_wshift --wind shift ;; esac
  done
done
ev V0 V0 V0 "$(jobid cran_V0)" eval_test2022
ev V0_met V0 V0 "$(jobid cran_V0)" eval_test2022_shift --met shift
for m in simvp convlstm climax earthformer topoflow; do
  ev B_$m V1 B_$m "$(jobid base_$m)" eval_test2022 --baseline
  ev B_${m}_met V1 B_$m "$(jobid base_$m)" eval_test2022_shift --baseline --met shift
done
mkdir -p runs/cams
ev cams V1 cams "" eval_test2022 --cams
