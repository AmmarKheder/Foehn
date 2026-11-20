#!/bin/bash
#SBATCH --job-name=ghap_p1r
#SBATCH --account=project_462001804
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/runs_ghap/logs/p1r_%x_%j.out
# PROTOCOL.md A2(c): selection, prior refit, SEL evaluation, gates and freeze of the GHAP system (as runs_global/p1r_K0.sh +
# p1r_final23.sh, A1(c'''), A1(e), A1(g)):
# 1. select_best.py on pmG_base (lowest SEL fair CRPS among the validated checkpoints);
# 2. latents of the selected checkpoint on the 2021 V3 days minus SEL (odd k -> fit, even k -> check; 8 GCDs);
# 3. copula_x fit (30 epochs) on v3odd, test on v3even; P3 replaces P1r only if P3 - P1r has a 95 % interval above 0;
# 4. eval_x on the 24 SEL days, all mask pixels, regime-box spectra, guard |z| <= 10, cap P_max (store manifest), |z0| <= 3
#    -> pmG_base/eval_sel2021_M8_s10_<prior>_guard_T3/;
# 5. validation_gates.py --eval (G0a threshold = T_G, written by eval_x in the sanity block) -> runs_ghap/pmG_final/
#    {ckpt/best.ckpt, args.json, prior.pt, PRIOR.json, SAMPLER.json, VALIDATION_GATES.json}. Exit 0 also when a gate fails
#    (A2(c): the failing system is still frozen and tested once, labelled); non-zero only if a step crashed.
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
RG=$W/runs_ghap
S=$W/logs/GLOBAL_PREP_STATUS.md
RUN=pmG_base
ZB=10; T=3
PMAX=$(python3 -c "import json;print(json.load(open('$W/data/processed/ghap/manifest.json'))['P_max_ug'])")
NPR=32; EPOCHS=30
note() { printf '%s\n' "- $(date '+%F %H:%M') [ghap] job $SLURM_JOB_ID (select+prior+eval+gates): $1" >> "$S"; }
CENTRES=45.0:10.0,36.5:116.0,26.5:81.0,13.0:2.0,-11.0:-62.0,39.5:-85.0
SEL_DAYS=$(seq -s, 0 15 345)
PYW=/projappl/project_462001804/envs/world/bin/python
source $W/env.sh
cd $W
export PYTHONDONTWRITEBYTECODE=1
run1() { srun --ntasks=1 --cpus-per-task=7 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "source $CDFFM_ENVSH; cd $CDFFM_W; $*"; }
run8() { srun --ntasks=8 --cpus-per-task=7 --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "source $CDFFM_ENVSH; cd $CDFFM_W; $*"; }
echo "start $(date) node=$SLURMD_NODENAME run=$RUN pmax=$PMAX"
LASTV=$(python3 -c "import json;print(max(json.loads(l)['step'] for l in open('$RG/$RUN/metrics.jsonl') if l.strip()))") || { note "no metrics.jsonl"; exit 1; }
singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "source $CDFFM_ENVSH; cd $CDFFM_W; python -u runs_global/select_best.py --run-dir $RG/$RUN --max-steps $LASTV" \
  || { note "selection FAILED"; exit 1; }
STEP=$(python3 -c "import json;print(json.load(open('$RG/$RUN/SELECTED.json'))['selected_step'])")
note "$RUN selected step $STEP ($(python3 -c "import json;d=json.load(open('$RG/$RUN/SELECTED.json'));print('SEL CRPS',round(d['model']['ug']['crps_fair'],4),'rmseM',round(d['model']['ug']['rmse_ensmean'],3),'of steps',sorted(int(k) for k in d['all_steps']))"))"
DIR=$RG/$RUN/copula_s$STEP
mkdir -p $DIR
CX="python -u -m cdffm_aq_x.copula_x"
for SP in v3odd v3even; do
  [ -f $DIR/latents_$SP.npz ] && continue
  PAR=$([ $SP = v3odd ] && echo 1 || echo 0)
  run8 "DAYS=\$(python3 -c \"import os;r=int(os.environ['SLURM_PROCID']);ks=[3*k for k in range(120) if k % 2 == $PAR and (3*k) % 15];print(','.join(str(d) for d in ks[r::8]))\"); \
        $CX latents --run $RUN --runs-root $RG --ckpt $STEP --dir $DIR --splits ${SP}_r\$SLURM_PROCID --day-idx \$DAYS --n-per-radius $NPR" \
    || { note "latents $SP FAILED"; exit 1; }
  $PYW -I - "$DIR" "$SP" <<'EOF' || { note "latents merge FAILED"; exit 1; }
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
[ -f $DIR/priors.pt ] || run1 "$CX fit --dir $DIR --epochs $EPOCHS --fit-file latents_v3odd.npz" || { note "prior fit FAILED"; exit 1; }
[ -f $DIR/test.json ] || run1 "$CX test --dir $DIR --test-file latents_v3even.npz" || { note "prior test FAILED"; exit 1; }
NAME=$(python3 -c "import json;d=json.load(open('$DIR/test.json'))['delta_P3_vs_P1r'];print('P3' if d['ci95'][0] > 0 else 'P1r')")
SUMMARY=$(python3 -c "
import json;d=json.load(open('$DIR/test.json'));v=d['delta_vs_P1']
f=lambda x:f\"{x['mean']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]\"
print('v3even', d['days'], 'days: P1r-P1', f(v['P1r']), '| P3-P1', f(v['P3']), '| P3-P1r', f(d['delta_P3_vs_P1r']), 'nats/pt; latent std', round(d['latent_std'],3))")
note "$RUN step $STEP: prior fitted on v3odd, checked on $SUMMARY -> prior $NAME"
TAG=sel2021_M8_s10_${NAME}_guard_T3
[ -f $RG/$RUN/eval_$TAG/metrics.json ] || run8 "python -u -m cdffm_aq_x.eval_x --run $RUN --runs-root $RG --ckpt $STEP --members 8 --ode-steps 10 \
      --days $SEL_DAYS --psd-centres $CENTRES --prior-file $DIR/priors.pt:$NAME --z-bound $ZB --pmax-ug $PMAX --z0-trunc $T --tag $TAG"
rc=$?
rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
[ $rc = 0 ] || { note "eval of $RUN + $NAME + guard T$T FAILED rc=$rc"; exit $rc; }
FD=$RG/pmG_final
mkdir -p $FD/ckpt
$PYW -I $W/runs_global/validation_gates.py --eval $RG/$RUN/eval_$TAG --out $FD/VALIDATION_GATES.json
grc=$?
$PYW -I - "$RG" "$RUN" "$STEP" "$DIR" "$NAME" "$TAG" "$ZB" "$PMAX" "$T" "$grc" <<'EOF' || { note "pmG_final assembly FAILED"; exit 1; }
import datetime as dt, json, os, shutil, sys
from pathlib import Path
rg, run, step, d, name, tag = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), sys.argv[5], sys.argv[6]
zb, pmax, T, grc = float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9]), int(sys.argv[10])
fd = rg / "pmG_final"
sel = json.loads((rg / run / "SELECTED.json").read_text())
assert sel["selected_step"] == step
for src, dst in ((Path(sel["checkpoint"]), fd / "ckpt" / "best.ckpt"), (rg / run / "args.json", fd / "args.json"),
                 (d / "priors.pt", fd / "prior.pt")):
    shutil.copy2(src, str(dst) + ".tmp")
    os.replace(str(dst) + ".tmp", dst)
(fd / "PRIOR.json").write_text(json.dumps({"file": "prior.pt", "name": name, "run": run, "step": step,
                                           "fit": str(d / "fit.json"), "test": str(d / "test.json"),
                                           "eval": f"{run}/eval_{tag}", "protocol": "A2(c) (A1(c'''), A1(e), A1(g))"}, indent=1))
man = json.loads((rg.parent / "data/processed/ghap/manifest.json").read_text())
(fd / "SAMPLER.json").write_text(json.dumps({"z_bound": zb, "pmax_ug": pmax, "z0_trunc": T, "eval": f"{run}/eval_{tag}",
    "run": run, "step": step, "G0a_threshold_T_G_ug": man["T_G_train_max_points"],
    "protocol": "A2(c): A1(e) guard + A1(g) truncated base measure; P_max and T_G from the GHAP store manifest"}, indent=1))
g = json.loads((fd / "VALIDATION_GATES.json").read_text())
m = json.loads((rg / run / f"eval_{tag}" / "metrics.json").read_text())
g["G0a_threshold_ug"] = m["sanity"].get("gt_440_means_gt_ug")
g["G0a_note"] = "A2(c): the gate key keeps its LGHAP name; for GHAP it counts members above T_G (G0a_threshold_ug)"
ev = m["guard"]
g["gates"]["G0_guard_matches_SAMPLER.json"] = {"evaluated": ev, "pass": ev.get("z_bound") == zb and ev.get("pmax_ug") == pmax and ev.get("z0_trunc") == T}
g["passed"] = all(v["pass"] for v in g["gates"].values() if v["pass"] is not None)
g["frozen_utc"] = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
(fd / "VALIDATION_GATES.json").write_text(json.dumps(g, indent=1))
print("pmG_final frozen; gates passed:", g["passed"], [k for k, v in g["gates"].items() if v["pass"] is False])
EOF
GP=$(python3 -c "import json;g=json.load(open('$FD/VALIDATION_GATES.json'));print('PASS' if g['passed'] else 'FAIL '+','.join(k for k,v in g['gates'].items() if v['pass'] is False))")
note "$RUN step $STEP + $NAME + guard T$T evaluated on the 24 SEL days -> $RUN/eval_$TAG; gates: $GP; FROZEN runs_ghap/pmG_final"
echo "end $(date) rc=0 gates=$GP"
exit 0
