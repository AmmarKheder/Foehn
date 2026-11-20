#!/bin/bash
#SBATCH --job-name=ghap_pred22
#SBATCH --account=project_462001804
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH --output=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/test2022_ghap/logs/pred22_%x_%j.out
# PROTOCOL.md A2(d) step (a): test2022_ghap/predict_2022.py with the frozen runs_ghap/pmG_final (model process under the
# GhapStore firewall; no truth is read).
#   sbatch -p standard-g -N 4 -t 2:30:00 predict_2022.sh 2022 [FULL_EVERY=3]
#   sbatch -p dev-g -N 1 -t 0:25:00 predict_2022.sh check <2021 SEL date> <out dir>     (eval_x reproduction, no 2022)
set -uo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
O=$W/test2022_ghap
SYSD=$W/runs_ghap/pmG_final
S=$W/logs/GLOBAL_PREP_STATUS.md
MODE=$1
note() { printf '%s\n' "- $(date '+%F %H:%M') [ghap] job $SLURM_JOB_ID (2022 predictions): $1" >> "$S"; }
source $W/env.sh
cd $W
export PYTHONDONTWRITEBYTECODE=1
NT=$(( SLURM_NNODES * 8 ))
srunN() { srun --ntasks=$NT --ntasks-per-node=8 --cpus-per-task=7 --cpu-bind=$CDFFM_CPU_BIND_8 singularity exec -B "$CDFFM_BIND" "$CDFFM_SIF" bash -c "source $CDFFM_ENVSH; cd $CDFFM_W; $*"; }
echo "start $(date) nodes=$SLURM_JOB_NODELIST mode=$MODE"
T0=$(date +%s)
[ -f $SYSD/VALIDATION_GATES.json ] || { note "pmG_final not frozen (no VALIDATION_GATES.json)"; exit 1; }
if [ "$MODE" = check ]; then
  DATE=$2; OUTD=$3
  srunN "python -u test2022_ghap/predict_2022.py --system $SYSD --check $DATE --out $OUTD"
  rc=$?
  /projappl/project_462001804/envs/world/bin/python -I - "$OUTD" "$DATE" "$SYSD" <<'EOP'
import json, math, sys
from pathlib import Path
d, date, sysd = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = [json.loads(p.read_text()) for p in sorted(d.glob(f"check_{date}_r*.json"))]
s = [sum(r["sums"][i] for r in rows) for i in range(5)]
ev = json.loads((Path(sysd) / "SAMPLER.json").read_text())["eval"]
ref = next(r for r in json.loads((Path(sysd).parent / ev / "metrics.json").read_text())["daily"] if r["date"] == date)
out = {"date": date, "ranks": len(rows), "crps": s[0] / s[2], "rmse_ensmean": math.sqrt(s[1] / s[2]),
       "eval_x_crps": ref["model"]["crps_fair"], "eval_x_rmse": ref["model"]["rmse_ensmean"],
       "full_day_gcd_min": sum(r["full_pass_s_for_1_over_world"] for r in rows) / 60,
       "sub_day_gcd_min": max(r["sub_pass_s"] for r in rows) / 60,
       "max_abs_diff_sub_vs_full": max(r["max_abs_diff_sub_vs_full"] for r in rows)}
(d / f"check_{date}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
EOP
  note "check mode (2021 SEL day $DATE, no 2022) rc=$rc -> $OUTD/check_$DATE.json"
  exit $rc
fi
[ -f $O/st_pix_2022.npz ] || { note "test2022_ghap/st_pix_2022.npz missing: not started"; exit 1; }
FULL=${2:-3}
note "started on $SLURM_JOB_NODELIST ($NT GCDs): system $SYSD, all 365 days of 2022, full fields every ${FULL}rd day"
srunN "python -u test2022_ghap/predict_2022.py --system $SYSD --full-every $FULL"
rc=$?
rm -rf "/tmp/${USER}-cdffm-$SLURM_JOB_ID"
N=$(for a in cdffm cams_bilinear bcsd clim; do ls $O/preds/$a 2>/dev/null | grep -c "^2022-..-..\.npz$"; done | tr '\n' ' ')
NF=$(ls $O/fields/cdffm 2>/dev/null | grep -c "^2022-..-..\.npz$")
note "ended rc=$rc after $(( ($(date +%s) - T0) / 60 )) min on $SLURM_NNODES nodes; day files preds [cdffm cams_bilinear bcsd clim] = $N; full fields = $NF"
exit $rc
