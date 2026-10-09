#!/usr/bin/env python3
"""Selection of the final model after the CRPS fine-tuning (rule fixed before the results: lowest CRPS against GHAP on the
validation year 2021, mean over the seven lead times, identical forecasts for every candidate; stations never used), then the
2022 evaluations of the winner if it is a fine-tuned checkpoint:
  * eval_final_<tag>: every second start day, leads 1-7, four members, stations (as eval_final_e02);
  * eval_jan21_<tag>: the global 1 km forecast of 21 January 2022 (Fig. 2, Extended Data regions);
  * eval_case_eastasia_<tag>: the January 2022 East Asian haze, 16 members (Fig. 3).
    python select_crps_and_launch.py [--dry]"""
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
R0 = "runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12"
RC = "runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"
CAND = {"e02": (R0, f"{R0}/ckpts/epoch=02.ckpt", f"{R0}/eval2021_sel_e02", ["--steps", "8", "--refine-tail", "3"]),
        "crps_e00": (RC, f"{RC}/ckpts/epoch=00.ckpt", f"{RC}/eval2021_sel_crps_e00", ["--steps", "4", "--refine-tail", "1"]),
        "crps_e01": (RC, f"{RC}/ckpts/epoch=01.ckpt", f"{RC}/eval2021_sel_crps_e01", ["--steps", "4", "--refine-tail", "1"])}


def load(ev):
    parts = sorted(glob.glob(str(ROOT / ev / "part_*.npz")))
    if not parts:
        return None
    z0 = np.load(parts[0]); C = list(z0["rec_cols"])
    R = np.concatenate([np.load(p)["recs"] for p in parts])
    return {(int(r[0]), int(r[1]), int(r[2])): (r[C.index("n")], r[C.index("crps")]) for r in R}


def main():
    D = {k: load(v[2]) for k, v in CAND.items()}
    D = {k: v for k, v in D.items() if v}
    common = set.intersection(*[set(v) for v in D.values()])
    score = {}
    for k, d in D.items():
        per_lead = []
        for h in range(1, 8):
            n = sum(d[x][0] for x in common if x[1] == h); c = sum(d[x][1] for x in common if x[1] == h)
            per_lead.append(c / n)
        score[k] = float(np.mean(per_lead))
        print(f"{k:9s} mean CRPS over leads 1-7 (2021, {len(common)} common records): {score[k]:.4f}  by lead "
              + " ".join(f"{v:.3f}" for v in per_lead), flush=True)
    win = min(score, key=score.get)
    print("selected:", win, flush=True)
    if win == "e02" or "--dry" in sys.argv:
        return
    run, ck, _, samp = CAND[win]
    base = [run, "--ckpt", ck, "--psub", "16"] + samp
    jobs = [("final_" + win, 8, base + ["--members", "4", "--every", "2", "--out", f"eval_final_{win}"]),
            ("jan21_" + win, 1, base + ["--members", "4", "--doys", "19", "--leads", "1", "--save-1km", "19", "--out", f"eval_jan21_{win}"]),
            ("case_ea_" + win, 4, base + ["--members", "16", "--workers", "2", "--crop", "42", "33", "112", "131", "--crop-only",
                                          "--crop-days"] + [str(d) for d in range(28)] + ["--doys"] + [str(d) for d in range(28)]
             + ["--out", f"eval_case_eastasia_{win}"])]
    for name, nodes, args in jobs:
        out = subprocess.run(["sbatch", "--parsable", f"--job-name={name}", f"--nodes={nodes}", "eval_green_job.sh"] + args,
                             cwd=ROOT, capture_output=True, text=True)
        print(name, "->", out.stdout.strip() or out.stderr.strip(), flush=True)


if __name__ == "__main__":
    main()
