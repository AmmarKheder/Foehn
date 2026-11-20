# a8_lib.sh -- sourced by the A8 job scripts (host side of a Slurm job on ONE LUMI-G node, 8 GCDs).
# protocol_amendments/A8_seed_replicates.md. Copies of the commands of runs_ghap/train_G.sh and runs_ghap/p1r_G.sh
# (PROTOCOL.md A2(c)) with: the run name / seed as parameters, outputs under runs_ghap/pmG_seed<k> only, notes to an A8
# file (never logs/GLOBAL_PREP_STATUS.md), nothing written to runs_ghap/pmG_final or pmG_base.
# The five steps of p1r_G.sh are split into four functions so that a job can interleave two runs on one node (the
# latents are CPU-bound, the prior fit uses one GCD, the evaluation uses eight):
#   pc_select RUN ROOT            step 1  select_best.py
#   pc_latents RUN ROOT [SMOKE]   step 2  latents of the fit days and of the check days (8 srun tasks) + merge
#   pc_fit_test RUN ROOT [SMOKE] [GCD]   step 3  copula_x fit, then test (one process on one GCD; may run in the background)
#   pc_eval_gates RUN ROOT FINAL_DIR [SMOKE]   steps 4-5  prior rule, eval_x on the SEL days, validation_gates.py
# Every command line, option, seed and day list is that of p1r_G.sh. SMOKE=1 = launcher test with reduced sizes.
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
RGH=$W/runs_ghap
A8=$RGH/pmG_seed1/a8
PYW=/projappl/project_462001804/envs/world/bin/python
source $W/env.sh
cd $W
export PYTHONDONTWRITEBYTECODE=1
A8_CENTRES=45.0:10.0,36.5:116.0,26.5:81.0,13.0:2.0,-11.0:-62.0,39.5:-85.0
A8_ZB=10; A8_T=3

note() { printf '%s\n' "- $(date '+%F %H:%M:%S') job ${SLURM_JOB_ID:-?} ($A8_WHAT): $1" | tee -a "$A8_NOTES"; }
run1() { srun --ntasks=1 --cpus-per-task=7 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "source $CDFFM_ENVSH; cd $CDFFM_W; $*"; }
run8() { srun --ntasks=8 --cpus-per-task=7 --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "source $CDFFM_ENVSH; cd $CDFFM_W; $*"; }
# one bash script per rank inside the container (the script sources env.sh itself)
run8s() { srun --ntasks=8 --cpus-per-task=7 --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash "$@"; }
# one process started from the batch step (no srun: it can run while an 8-task step is running) on GCD $1
runb() { local G=$1; shift; SLURM_LOCALID=$G SLURM_PROCID=0 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "source $CDFFM_ENVSH; cd $CDFFM_W; $*"; }

# node memory monitor (the node is exclusive: "used" of free = this job's processes): epoch used buff/cache available, GB
memmon_start() { ( while true; do printf '%s %s\n' "$(date +%s)" "$(free -g | awk '/^Mem:/{print $3" "$6" "$7}')"; sleep 10; done ) >> "$1" & MEMMON_PID=$!; }
memmon_stop() { kill "$MEMMON_PID" 2>/dev/null; wait "$MEMMON_PID" 2>/dev/null; return 0; }
memmon_max() { awk -v t0="$2" '$1 >= t0 { if ($2 > m) m = $2 } END { print m + 0 }' "$1"; }   # max used GB since epoch t0

last_step() { python3 -c "import json;print(max(json.loads(l)['step'] for l in open('$1/metrics.jsonl') if l.strip()))" 2>/dev/null || echo 0; }
sel_step() { python3 -c "import json;print(json.load(open('$1/SELECTED.json'))['selected_step'])"; }

# --------------------------------------------------------------------------------------------------------------
# training segment: train_G.sh's srun line, with the seed and the run directory as parameters
#   train_seg NAME RUNS_ROOT SEED STOP BUDGET_MIN
# --------------------------------------------------------------------------------------------------------------
train_seg() {
  local NAME=$1 RG=$2 SEED=$3 STOP=$4 BUDGET=$5 RES="" rc
  local COMMON="--store ghap --fires injh --split G --grid 025 --transport t6 --batch-size 2 --n-points 16384 --seed $SEED \
 --max-steps 60000 --val-every 2000 --val-days $(seq -s, 0 15 345) --val-pixels 100000 --val-members 8 \
 --val-ode-steps 10 --log-every 100 --sigma-k 0"
  [ -f $RG/$NAME/ckpt/last.ckpt ] && RES="--resume"
  echo "train_seg start $(date) node=$SLURMD_NODENAME name=$NAME seed=$SEED stop=$STOP budget=$BUDGET min resume='$RES'"
  note "training $NAME (seed $SEED) to step $STOP started on $SLURMD_NODENAME (budget $BUDGET min, '$RES')"
  local T0; T0=$(date +%s)
  srun --ntasks=8 --cpus-per-task=7 --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" \
       bash -c "source $CDFFM_ENVSH; cd $CDFFM_W; python -u -m cdffm_aq.train_aq --name $NAME --runs-root $RG $COMMON $RES \
                --stop-at-step $STOP --time-budget-min $BUDGET"
  rc=$?
  local LAST VAL
  LAST=$(last_step $RG/$NAME)
  VAL=$(python3 -c "
import json;r=[json.loads(l) for l in open('$RG/$NAME/metrics.jsonl') if l.strip()][-1]['model']['ug']
print('SEL crps %.4f rmseM %.3f ssF %.3f' % (r['crps_fair'], r['rmse_ensmean'], r.get('spread_skill_fortin', float('nan'))))" 2>/dev/null || echo "?")
  note "training segment of $NAME ended rc=$rc after $(( $(date +%s) - T0 )) s; last validated step $LAST: $VAL"
  echo "train_seg end $(date) rc=$rc last_step=$LAST"
  return $rc
}

# --------------------------------------------------------------------------------------------------------------
# p1r_G.sh step 1: selection (lowest SEL fair CRPS among the validated checkpoints)
# --------------------------------------------------------------------------------------------------------------
pc_select() {
  local RUN=$1 RG=$2 LASTV STEP
  LASTV=$(python3 -c "import json;print(max(json.loads(l)['step'] for l in open('$RG/$RUN/metrics.jsonl') if l.strip()))") || { note "$RUN: no metrics.jsonl"; return 1; }
  singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "source $CDFFM_ENVSH; cd $CDFFM_W; python -u runs_global/select_best.py --run-dir $RG/$RUN --max-steps $LASTV" \
    || { note "$RUN: selection FAILED"; return 1; }
  STEP=$(sel_step $RG/$RUN)
  note "$RUN selected step $STEP ($(python3 -c "import json;d=json.load(open('$RG/$RUN/SELECTED.json'));print('SEL CRPS',round(d['model']['ug']['crps_fair'],4),'rmseM',round(d['model']['ug']['rmse_ensmean'],3),'of steps',sorted(int(k) for k in d['all_steps']))"))"
}

# --------------------------------------------------------------------------------------------------------------
# p1r_G.sh step 2: latents of the selected checkpoint on the 2021 V3 days minus SEL (odd k -> fit, even k -> check).
# Both passes run at the same time when neither exists (A8(c): layout only); if that step fails (e.g. memory) or with
# A8_SEQ_LATENTS=1, each missing pass is done alone, as p1r_G.sh.
# --------------------------------------------------------------------------------------------------------------
pc_latents() {
  local RUN=$1 RG=$2 SMOKE=${3:-0} NPR=32 NDAY=0 STEP DIR SP tl conc_ok=0
  if [ "$SMOKE" = 1 ]; then NPR=4; NDAY=1; fi
  STEP=$(sel_step $RG/$RUN) || return 1
  DIR=$RG/$RUN/copula_s$STEP
  mkdir -p $DIR
  if [ ! -f $DIR/latents_v3odd.npz ] && [ ! -f $DIR/latents_v3even.npz ] && [ "${A8_SEQ_LATENTS:-0}" != 1 ]; then
    tl=$(date +%s)
    if run8s $A8/rank_latents.sh $RUN $RG $STEP $DIR $NPR both $NDAY; then
      conc_ok=1
      note "$RUN: latents of both passes computed concurrently in $(( $(date +%s) - tl )) s; node memory used, max $(memmon_max "$A8_MEMLOG" $tl) GB"
    else
      note "$RUN: concurrent latents step FAILED after $(( $(date +%s) - tl )) s (node memory used, max $(memmon_max "$A8_MEMLOG" $tl) GB): redoing each pass alone"
    fi
  fi
  for SP in v3odd v3even; do
    [ -f $DIR/latents_$SP.npz ] && continue
    if [ $conc_ok != 1 ]; then
      rm -f $DIR/latents_${SP}_r?.npz
      tl=$(date +%s)
      run8s $A8/rank_latents.sh $RUN $RG $STEP $DIR $NPR $SP $NDAY || { note "$RUN: latents $SP FAILED"; return 1; }
      note "$RUN: latents $SP computed alone in $(( $(date +%s) - tl )) s; node memory used, max $(memmon_max "$A8_MEMLOG" $tl) GB"
    fi
    $PYW -I - "$DIR" "$SP" <<'EOF' || { note "$RUN: latents merge FAILED"; return 1; }
import json, sys
import numpy as np
from pathlib import Path
d, sp = Path(sys.argv[1]), sys.argv[2]
parts = [np.load(d / f"latents_{sp}_r{r}.npz") for r in range(8)]
out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files if k != "meta"}
meta = json.loads(str(parts[0]["meta"]))
meta["split"], meta["ranks"] = sp, 8
out["meta"] = np.array(json.dumps(meta))
np.savez(d / f"latents_{sp}.npz", **out)
print(sp, {k: v.shape for k, v in out.items() if k != "meta"}, "days", len(set(out["day"].tolist())))
for r in range(8):
    (d / f"latents_{sp}_r{r}.npz").unlink()
EOF
  done
  return 0
}

# --------------------------------------------------------------------------------------------------------------
# p1r_G.sh step 3: prior fit on the fit days (30 epochs), test on the check days. One process on one GCD.
# A8_FT_MODE=batch (default): started from the batch step on GCD $4, so it can run while an 8-task step runs;
# A8_FT_MODE=srun: p1r_G.sh's own "srun --ntasks=1" form.
# --------------------------------------------------------------------------------------------------------------
pc_fit_test() {
  local RUN=$1 RG=$2 SMOKE=${3:-0} GCD=${4:-0} EPOCHS=30 STEP DIR tl
  [ "$SMOKE" = 1 ] && EPOCHS=2
  STEP=$(sel_step $RG/$RUN) || return 1
  DIR=$RG/$RUN/copula_s$STEP
  local CX="python -u -m cdffm_aq_x.copula_x"
  tl=$(date +%s)
  if [ "${A8_FT_MODE:-batch}" = srun ]; then
    [ -f $DIR/priors.pt ] || run1 "$CX fit --dir $DIR --epochs $EPOCHS --fit-file latents_v3odd.npz" || { note "$RUN: prior fit FAILED"; return 1; }
    [ -f $DIR/test.json ] || run1 "$CX test --dir $DIR --test-file latents_v3even.npz" || { note "$RUN: prior test FAILED"; return 1; }
  else
    [ -f $DIR/priors.pt ] || runb $GCD "$CX fit --dir $DIR --epochs $EPOCHS --fit-file latents_v3odd.npz" || { note "$RUN: prior fit FAILED"; return 1; }
    [ -f $DIR/test.json ] || runb $GCD "$CX test --dir $DIR --test-file latents_v3even.npz" || { note "$RUN: prior test FAILED"; return 1; }
  fi
  note "$RUN: prior fit + test done in $(( $(date +%s) - tl )) s (mode ${A8_FT_MODE:-batch}, GCD $GCD)"
  return 0
}

# --------------------------------------------------------------------------------------------------------------
# p1r_G.sh steps 3 (rule), 4 and 5: P3 only if the interval of P3 - P1r is above 0, else P1r; eval_x on the SEL days,
# all mask pixels, regime-box spectra, guard |z| <= 10, cap P_max, |z0| <= 3; validation_gates.py --eval.
# The gate file goes to the replicate's own directory; nothing is frozen.
# --------------------------------------------------------------------------------------------------------------
pc_eval_gates() {
  local RUN=$1 RG=$2 FD=$3 SMOKE=${4:-0} EVAL_EXTRA="" ZB=$A8_ZB T=$A8_T
  local PMAX SEL_DAYS STEP DIR NAME SUMMARY TAG rc grc GP tl
  PMAX=$(python3 -c "import json;print(json.load(open('$W/data/processed/ghap/manifest.json'))['P_max_ug'])")
  SEL_DAYS=$(seq -s, 0 15 345)
  if [ "$SMOKE" = 1 ]; then SEL_DAYS=$(seq -s, 0 45 315); EVAL_EXTRA="--pixel-stride 512"; fi
  STEP=$(sel_step $RG/$RUN) || return 1
  DIR=$RG/$RUN/copula_s$STEP
  [ -f $DIR/test.json ] || { note "$RUN: no test.json (prior not fitted)"; return 1; }
  NAME=$(python3 -c "import json;d=json.load(open('$DIR/test.json'))['delta_P3_vs_P1r'];print('P3' if d['ci95'][0] > 0 else 'P1r')")
  SUMMARY=$(python3 -c "
import json;d=json.load(open('$DIR/test.json'));v=d['delta_vs_P1']
f=lambda x:f\"{x['mean']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]\"
print('v3even', d['days'], 'days: P1r-P1', f(v['P1r']), '| P3-P1', f(v['P3']), '| P3-P1r', f(d['delta_P3_vs_P1r']), 'nats/pt; latent std', round(d['latent_std'],3))")
  note "$RUN step $STEP: prior fitted on v3odd, checked on $SUMMARY -> prior $NAME"
  TAG=sel2021_M8_s10_${NAME}_guard_T3
  tl=$(date +%s)
  [ -f $RG/$RUN/eval_$TAG/metrics.json ] || run8 "python -u -m cdffm_aq_x.eval_x --run $RUN --runs-root $RG --ckpt $STEP --members 8 --ode-steps 10 \
      --days $SEL_DAYS --psd-centres $A8_CENTRES --prior-file $DIR/priors.pt:$NAME --z-bound $ZB --pmax-ug $PMAX --z0-trunc $T --tag $TAG $EVAL_EXTRA"
  rc=$?
  [ $rc = 0 ] || { note "$RUN: eval + $NAME + guard T$T FAILED rc=$rc"; return $rc; }
  note "$RUN: evaluation of step $STEP + $NAME + guard T$T done in $(( $(date +%s) - tl )) s; node memory used, max $(memmon_max "$A8_MEMLOG" $tl) GB"
  mkdir -p $FD
  $PYW -I $W/runs_global/validation_gates.py --eval $RG/$RUN/eval_$TAG --out $FD/VALIDATION_GATES.json
  grc=$?
  $PYW -I - "$RG" "$RUN" "$STEP" "$DIR" "$NAME" "$TAG" "$ZB" "$PMAX" "$T" "$grc" "$FD" "$W" <<'EOF' || { note "$RUN: gate assembly FAILED"; return 1; }
import datetime as dt, json, sys
from pathlib import Path
rg, run, step, d, name, tag = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), sys.argv[5], sys.argv[6]
zb, pmax, T, grc, fd, w = float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9]), int(sys.argv[10]), Path(sys.argv[11]), Path(sys.argv[12])
sel = json.loads((rg / run / "SELECTED.json").read_text())
assert sel["selected_step"] == step
label = ("A8 seed replicate of the frozen system pmG_final (protocol_amendments/A8_seed_replicates.md): measures training "
         "noise only; NOT a system of record, nothing is frozen or deployed from it")
(fd / "PRIOR.json").write_text(json.dumps({"file": str(d / "priors.pt"), "name": name, "run": run, "step": step,
                                           "fit": str(d / "fit.json"), "test": str(d / "test.json"),
                                           "eval": f"{run}/eval_{tag}", "checkpoint": sel["checkpoint"],
                                           "sha256": sel["sha256"], "protocol": "A8 (recipe of A2(c))", "label": label}, indent=1))
man = json.loads((w / "data/processed/ghap/manifest.json").read_text())
(fd / "SAMPLER.json").write_text(json.dumps({"z_bound": zb, "pmax_ug": pmax, "z0_trunc": T, "eval": f"{run}/eval_{tag}",
    "run": run, "step": step, "G0a_threshold_T_G_ug": man["T_G_train_max_points"],
    "protocol": "A8 (recipe of A2(c): A1(e) guard + A1(g) truncated base measure)", "label": label}, indent=1))
g = json.loads((fd / "VALIDATION_GATES.json").read_text())
m = json.loads((rg / run / f"eval_{tag}" / "metrics.json").read_text())
g["G0a_threshold_ug"] = m["sanity"].get("gt_440_means_gt_ug")
g["G0a_note"] = "A2(c): the gate key keeps its LGHAP name; for GHAP it counts members above T_G (G0a_threshold_ug)"
ev = m["guard"]
g["gates"]["G0_guard_matches_SAMPLER.json"] = {"evaluated": ev, "pass": ev.get("z_bound") == zb and ev.get("pmax_ug") == pmax and ev.get("z0_trunc") == T}
g["passed"] = all(v["pass"] for v in g["gates"].values() if v["pass"] is not None)
g["label"] = label
g["prior"] = name
g["assembled_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
(fd / "VALIDATION_GATES.json").write_text(json.dumps(g, indent=1))
print(f"{run}: gates passed: {g['passed']}", [k for k, v in g["gates"].items() if v["pass"] is False])
EOF
  GP=$(python3 -c "import json;g=json.load(open('$FD/VALIDATION_GATES.json'));print('PASS' if g['passed'] else 'FAIL '+','.join(k for k,v in g['gates'].items() if v['pass'] is False))")
  note "$RUN step $STEP + $NAME + guard T$T evaluated on the SEL days -> $RUN/eval_$TAG; gates: $GP (replicate: nothing frozen)"
  echo "pc_eval_gates end $(date) run=$RUN gates=$GP"
  return 0
}

# --------------------------------------------------------------------------------------------------------------
# Reference checks on the FROZEN system (A8 Addendum 1 and 2; they read pmG_base and never write in it). Each is one
# process started from the batch step on GCD $1, so that it runs while the latents (CPU-bound) occupy the node.
#   ref_day GCD HOLD_INDEX [EVAL_SEED]   one full selection day re-evaluated with the unmodified cdffm_aq_x.eval_x and the
#       options of p1r_G.sh, through a8/refday_eval.py, which also records where the member points at the state bound
#       are; about 20 min. EVAL_SEED 0 (default) = the evaluation noise of the frozen evaluation (a reproduction of its
#       stored per-day row); another value = the same model and day under another evaluation noise draw.
#       The run is seen through a view made of symbolic links (a8/ref/pmG_base); eval_x writes in
#       a8/ref/pmG_base/eval_ref_day<t>[_es<seed>]/.
#   val_noise GCD STEPS SEEDS            train_aq.validate of frozen checkpoints with other validation-noise seeds
#       (a8/valrepro.py --val-seeds): sampling noise of the selection metric; about 15 s per (step, seed).
# --------------------------------------------------------------------------------------------------------------
ref_day() {
  local G=$1 TD=$2 ES=${3:-0} V=$A8/ref/pmG_base PMAX tl TAG
  TAG=ref_day$TD; [ "$ES" != 0 ] && TAG=ref_day${TD}_es$ES
  mkdir -p $V
  [ -e $V/args.json ] || ln -s $RGH/pmG_base/args.json $V/args.json
  [ -e $V/ckpt ] || ln -s $RGH/pmG_base/ckpt $V/ckpt
  [ -f $V/eval_$TAG/metrics.json ] && { note "reference day $TD (evaluation seed $ES) already evaluated"; return 0; }
  PMAX=$(python3 -c "import json;print(json.load(open('$W/data/processed/ghap/manifest.json'))['P_max_ug'])")
  tl=$(date +%s)
  WORLD_SIZE=1 RANK=0 LOCAL_RANK=$G SLURM_LOCALID=$G SLURM_PROCID=0 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "source $CDFFM_ENVSH; cd $CDFFM_W; \
      python -u $A8/refday_eval.py --hits-out $A8/results/refday_hits_${TAG#ref_}.npz -- --run pmG_base --runs-root $A8/ref --ckpt 22000 --members 8 --ode-steps 10 \
      --days $TD --psd-centres $A8_CENTRES --prior-file $RGH/pmG_base/copula_s22000/priors.pt:P3 --z-bound $A8_ZB --pmax-ug $PMAX --z0-trunc $A8_T --seed $ES --tag $TAG" \
    || { note "reference day $TD (evaluation seed $ES) of the frozen system FAILED (the replicates are not affected)"; return 1; }
  note "reference day $TD (evaluation seed $ES) of the frozen system re-evaluated on GCD $G in $(( $(date +%s) - tl )) s -> a8/ref/pmG_base/eval_$TAG"
}
val_noise() {
  local G=$1 STEPS=$2 SEEDS=$3 tl
  [ -f $A8/results/valnoise_seed0.json ] && { note "validation-noise check already done"; return 0; }
  tl=$(date +%s)
  runb $G "python -u $A8/valrepro.py --run pmG_base --steps $STEPS --val-seeds $SEEDS --out $A8/results/valnoise_seed0.json" \
    || { note "validation-noise check FAILED (the replicates are not affected)"; return 1; }
  note "validation-noise check of the frozen checkpoints $STEPS with noise seeds $SEEDS done on GCD $G in $(( $(date +%s) - tl )) s -> a8/results/valnoise_seed0.json"
}
