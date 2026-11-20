#!/bin/bash
#SBATCH --job-name=cal_pred22
#SBATCH --account=project_462001804
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/test2022_cal/logs/%x_%j.out
# The 2022 protocol of the frozen system (test2022_ghap/predict_2022.sh, PROTOCOL.md A2(d) step (a)) with the calibrated
# sampler of runs_ghap/pmG_cal/final: test2022_cal/predict_cal.py = test2022_ghap/predict_2022.py unmodified + the temperature
# on the prior field (model process under the GhapStore firewall; no truth is read). Everything is written under test2022_cal/.
#   THE SINGLE 2022 PASS (the lead only, through submit_2022_cal.sh, which freezes the candidate and writes SYSTEM.json):
#     sbatch -p standard-g -N 4 -t 2:30:00 predict_cal.sh 2022 [FULL_EVERY=3]
#   Tests that touch neither 2022 nor any station file:
#     sbatch -p standard-g -N 1 -t 0:25:00 predict_cal.sh check <2021 SEL date> [out dir]   eval_x reproduction on one 2021 day
#     sbatch -p standard-g -N 1 -t 0:30:00 predict_cal.sh mech cal|frozen                   the prediction loop on 12 SEL days of
#                                              2021 at pseudo-monitor pixels (mech2021/st_ring_2022.npz made by mech_setup.py)
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
O=$W/test2022_cal
SYSD=$W/runs_ghap/pmG_cal/final
PYW=/projappl/project_462001804/envs/world/bin/python
MODE=$1
S=$O/STATUS.md
note() { printf '%s\n' "- $(date '+%F %H:%M') job $SLURM_JOB_ID ($SLURM_JOB_NAME, $MODE): $1" >> "$S"; }
source $W/env.sh
cd $W
export PYTHONDONTWRITEBYTECODE=1
NT=$(( SLURM_NNODES * 8 ))
srunN() { srun --ntasks=$NT --ntasks-per-node=8 --cpus-per-task=7 --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "source $CDFFM_ENVSH; cd $CDFFM_W; export PYTHONDONTWRITEBYTECODE=1; $*"; }
echo "start $(date) nodes=$SLURM_JOB_NODELIST mode=$MODE"
T0=$(date +%s)
# the candidate must be the one that was evaluated: every file against its SHA-256 in MANIFEST.json
$PYW -I - "$SYSD" <<'EOP' || { note "runs_ghap/pmG_cal/final does not match its MANIFEST.json: not started"; exit 1; }
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1])
man = json.loads((d / "MANIFEST.json").read_text())["files"]
bad = [k for k, v in man.items() if hashlib.sha256((d / k).read_bytes()).hexdigest() != v]
print("candidate files", "OK" if not bad else f"CHANGED: {bad}")
sys.exit(1 if bad else 0)
EOP
if [ "$MODE" = check ]; then
  DATE=$2; OUTD=${3:-$O/check2021}
  srunN "python -u test2022_cal/predict_cal.py --system $SYSD --check $DATE --out $OUTD"
  rc=$?
  $PYW -I - "$OUTD" "$DATE" "$SYSD" <<'EOP' || rc=1
import json, math, sys
from pathlib import Path
d, date, sysd = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = [json.loads(p.read_text()) for p in sorted(d.glob(f"check_{date}_r*.json"))]
s = [sum(r["sums"][i] for r in rows) for i in range(5)]
ev = json.loads((Path(sysd) / "SAMPLER.json").read_text())["eval"]
m = json.loads((Path(sysd).parent / ev / "metrics.json").read_text())
ref = next(r for r in m["daily"] if r["date"] == date)
out = {"date": date, "ranks": len(rows), "crps": s[0] / s[2], "rmse_ensmean": math.sqrt(s[1] / s[2]),
       "eval_x_crps": ref["model"]["crps_fair"], "eval_x_rmse": ref["model"]["rmse_ensmean"], "eval_x_dir": str(Path(sysd).parent / ev),
       "n_member_at_bound": s[3], "n_member_z0_trunc": s[4],
       "full_day_gcd_min": sum(r["full_pass_s_for_1_over_world"] for r in rows) / 60,
       "sub_day_gcd_min": max(r["sub_pass_s"] for r in rows) / 60,
       "max_abs_diff_sub_vs_full": max(r["max_abs_diff_sub_vs_full"] for r in rows),
       "max_rel_diff_sub_vs_full": max(r["max_rel_diff"] for r in rows)}
out["rel_diff_crps"] = out["crps"] / out["eval_x_crps"] - 1
out["rel_diff_rmse"] = out["rmse_ensmean"] / out["eval_x_rmse"] - 1
out["reproduces_eval_x"] = abs(out["rel_diff_crps"]) < 1e-5 and abs(out["rel_diff_rmse"]) < 1e-5
(d / f"check_{date}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
sys.exit(0 if out["reproduces_eval_x"] else 1)
EOP
  note "check (2021 SEL day $DATE, calibrated sampler, no 2022) rc=$rc -> $OUTD/check_$DATE.json after $(( ($(date +%s) - T0) / 60 )) min"
  rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
  exit $rc
fi
if [ "$MODE" = mech ]; then
  WHICH=$2; D=$O/mech2021
  DAYS=${MECH_DAYS:-7,37,67,97,127,157,187,217,247,277,307,337}
  [ -f $D/st_ring_2022.npz ] || { note "mech: $D/st_ring_2022.npz missing (mech_setup.py positions)"; exit 1; }
  if [ "$WHICH" = frozen ]; then      # the frozen system through its own, unmodified script (outputs redirected)
    srunN "python -u test2022_ghap/predict_2022.py --system $W/runs_ghap/pmG_final --year 2021 --days $DAYS --full-every 0 --out $D/frozen --st-pos $D/st_ring_2022.npz"
  else                                # the calibrated system; the last day also as a full-field day
    srunN "python -u test2022_cal/predict_cal.py --system $SYSD --year 2021 --days $DAYS --full-every ${MECH_FULL:-337} --out $D/cal --st-pos $D/st_ring_2022.npz"
  fi
  rc=$?
  note "mech $WHICH (2021 days $DAYS, pseudo-monitor pixels, no 2022, no station) rc=$rc after $(( ($(date +%s) - T0) / 60 )) min: $(ls $D/$WHICH/preds/cdffm 2>/dev/null | wc -l) day files, $(ls $D/$WHICH/fields/cdffm 2>/dev/null | wc -l) full fields"
  rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
  exit $rc
fi
# ------------------------------------------------------------------------------------------- the single 2022 pass
[ "$MODE" = 2022 ] || { echo "unknown mode $MODE"; exit 2; }
[ -f $O/GO_2022 ] || { note "no GO_2022 (the lead starts the pass with submit_2022_cal.sh): not started"; exit 1; }
[ -f $O/SYSTEM.json ] || { note "SYSTEM.json missing: not started"; exit 1; }
[ -f $W/test2022_ghap/st_ring_2022.npz ] || { note "test2022_ghap/st_ring_2022.npz missing: not started"; exit 1; }
[ -f $O/results/scores_ghap_2022.json ] && { note "already scored: the 2022 pass is single"; exit 1; }
FULL=${2:-3}
note "started on $SLURM_JOB_NODELIST ($NT GCDs): system $SYSD, all 365 days of 2022, full fields every ${FULL}rd day"
srunN "python -u test2022_cal/predict_cal.py --system $SYSD --out $O --full-every $FULL"
rc=$?
rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
N=$(for a in cdffm cams_bilinear bcsd clim; do ls $O/preds/$a 2>/dev/null | grep -c "^2022-..-..\.npz$"; done | tr '\n' ' ')
NF=$(ls $O/fields/cdffm 2>/dev/null | grep -c "^2022-..-..\.npz$")
note "ended rc=$rc after $(( ($(date +%s) - T0) / 60 )) min on $SLURM_NNODES nodes; day files preds [cdffm cams_bilinear bcsd clim] = $N; full fields = $NF"
exit $rc
