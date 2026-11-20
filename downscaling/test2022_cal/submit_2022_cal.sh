#!/bin/bash
# THE SINGLE 2022 PASS OF THE CALIBRATED DOWNSCALER (runs_ghap/pmG_cal/final). ONLY THE LEAD RUNS THIS, ONCE,
# after reading runs_ghap/pmG_cal/results/DECISION.md and adopting protocol_amendments/A9_calibrated_2022.md.
#
#   bash submit_2022_cal.sh                 dry run: checks and prints what would be submitted; submits and writes nothing
#   bash submit_2022_cal.sh GO [FULL_EVERY] freezes the candidate for the pass (GO_2022, SYSTEM.json), then submits
#                                           1. predict_cal.sh 2022 FULL_EVERY   (standard-g, 4 nodes, limit 2:30; about 80 min
#                                              with FULL_EVERY=3, the frozen pass's option; 0 = no full fields, about a
#                                              third of the cost, same scores up to rounding)
#                                           2. score_cal.sh                     (small, 32 cores; after 1. succeeded)
# Nothing here reads 2022 or a station file: the predictions run under the GhapStore firewall, the scoring job opens the
# truth and logs it in logs/UNSEALING.md.
set -euo pipefail
W=/scratch/project_462001804/ammar/global_model_forecasting_air_pollution
O=$W/test2022_cal
SYSD=$W/runs_ghap/pmG_cal/final
PY=/projappl/project_462001804/envs/world/bin/python
MODE=${1:-dry}
FULL=${2:-3}
ok=1
chk() { if eval "$2"; then echo "  ok   $1"; else echo "  FAIL $1"; ok=0; fi; }
echo "checks:"
chk "candidate files match runs_ghap/pmG_cal/final/MANIFEST.json" "$PY -I -c \"
import hashlib, json, sys
from pathlib import Path
d = Path('$SYSD'); man = json.loads((d / 'MANIFEST.json').read_text())['files']
sys.exit(0 if all(hashlib.sha256((d / k).read_bytes()).hexdigest() == v for k, v in man.items()) else 1)\""
chk "calibration present in SAMPLER.json (z0_tau, z0_knee)" "grep -q z0_tau $SYSD/SAMPLER.json && grep -q z0_knee $SYSD/SAMPLER.json"
chk "protocol note protocol_amendments/A9_calibrated_2022.md exists" "[ -f $W/protocol_amendments/A9_calibrated_2022.md ]"
chk "no earlier pass (no GO_2022, no 2022 prediction, no score)" "[ ! -e $O/GO_2022 ] && [ ! -e $O/preds/cdffm ] && [ ! -e $O/results/scores_ghap_2022.json ]"
chk "station positions of the frozen pass exist (test2022_ghap/st_ring_2022.npz; not opened here)" "[ -f $W/test2022_ghap/st_ring_2022.npz ]"
chk "extracted GHAP 2022 truth of the frozen pass exists (not opened here)" "[ -f $W/data/processed/ghap/sealed/ghap_sub_2022.npy ]"
chk "frozen pass results exist for the paired table (test2022_ghap/results/accum_2022.npz; not opened here)" "[ -f $W/test2022_ghap/results/accum_2022.npz ]"
chk "rehearsal on 2021 passed (mech2021/REHEARSAL_OK)" "[ -f $O/mech2021/REHEARSAL_OK ]"
echo "would submit:"
echo "  J=\$(sbatch --parsable -p standard-g -N 4 -t 2:30:00 $O/predict_cal.sh 2022 $FULL)"
echo "  sbatch --dependency=afterok:\$J $O/score_cal.sh"
if [ "$MODE" != GO ]; then
  echo "dry run (nothing written, nothing submitted). To start the single pass: bash $O/submit_2022_cal.sh GO [FULL_EVERY]"
  exit 0
fi
[ $ok = 1 ] || { echo "a check failed: nothing submitted"; exit 1; }
NOW=$(date -u +%FT%TZ)
$PY -I - "$SYSD" "$O" "$NOW" "$FULL" <<'EOP'
import json, sys
from pathlib import Path
sysd, o, now, full = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
g = json.loads((sysd / "VALIDATION_GATES.json").read_text())
s = json.loads((sysd / "SAMPLER.json").read_text())
man = json.loads((sysd / "MANIFEST.json").read_text())
sysj = {"system": "pmG_cal", "system_dir": str(sysd), "base_system": "pmG_final (same checkpoint, prior and guard)",
        "calibration": {"z0_tau": s["z0_tau"], "z0_knee": s["z0_knee"], "map": s["z0_map"]},
        "frozen_utc": now, "gates_passed": g["passed"], "failed_gates": [k for k, v in g["gates"].items() if v.get("pass") is False],
        "label": "gate-FAILING system (G0b, as the frozen system); calibration accepted by the POST HOC addendum of A6",
        "A6_verdict": g.get("A6_verdict"), "sha256": man["files"], "full_every": int(full),
        "rule": "protocol of PROTOCOL.md A2(d)-(e); system of A6; single pass of amendment A9"}
(o / "SYSTEM.json").write_text(json.dumps(sysj, indent=1))
(o / "GO_2022").write_text(json.dumps({"started_utc": now, "by": "lead (submit_2022_cal.sh GO)", "full_every": int(full)}, indent=1))
print(json.dumps(sysj, indent=1))
EOP
J=$(sbatch --parsable -p standard-g -N 4 -t 2:30:00 $O/predict_cal.sh 2022 $FULL)
J2=$(sbatch --parsable --dependency=afterok:$J $O/score_cal.sh)
printf '%s\n' "- $(date '+%F %H:%M') SINGLE 2022 PASS STARTED by submit_2022_cal.sh GO: candidate frozen at $NOW (SYSTEM.json), predictions job $J (4 nodes, full fields every $FULL), scoring job $J2 (after $J)" >> $O/STATUS.md
echo "submitted: predictions $J, scoring $J2 (afterok). Log: $O/STATUS.md; results: $O/results/"
