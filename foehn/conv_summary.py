#!/usr/bin/env python3
"""Particle-number convergence of the forecast (conv_particles.sh): fair CRPS and RMSE against GHAP 2021 (1 km land
pixels of the evaluation regions), by lead and number of particles per member, for one checkpoint; and the change
relative to the largest N. Identical start days and leads for every N.
    python conv_summary.py runs/<run>"""
import glob
import sys
from pathlib import Path

import numpy as np

R = Path(sys.argv[1])
res = {}
for d in sorted(R.glob("eval2021_conv_p*")):
    N = int(d.name.split("_p")[-1])
    P = [np.load(p) for p in sorted(glob.glob(str(d / "part_*.npz")))]
    if not P:
        continue
    cols = list(P[0]["rec_cols"])
    rec = np.concatenate([p["recs"] for p in P])
    res[N] = (rec, cols, [str(r) for r in P[0]["regions"]])
Ns = sorted(res)
common = set.intersection(*[{(int(a), int(b)) for a, b in res[N][0][:, :2]} for N in Ns])
print(f"{R.name}: N = {Ns}, common forecasts {len(common)}")
for scope in ["all regions", "East Asia", "South Asia", "Europe", "North America"]:
    print(f"\n{scope}: CRPS / RMSE (ug m-3) against GHAP 2021")
    print("lead " + "".join(f"{'N=%d' % N:>16s}" for N in Ns))
    for h in sorted({int(l) for l in res[Ns[0]][0][:, 1]}):
        row = []
        for N in Ns:
            rec, cols, regs = res[N]
            c = {k: i for i, k in enumerate(cols)}
            m = (rec[:, 1] == h) & np.array([(int(a), int(b)) in common for a, b in rec[:, :2]])
            if scope != "all regions":
                m &= rec[:, 2] == regs.index(scope)
            n = rec[m, c["n"]].sum()
            row.append((rec[m, c["crps"]].sum() / n, np.sqrt(rec[m, c["sse"]].sum() / n)))
        ref = row[-1]
        print(f"J+{h:<3d}" + "".join(f"{cr:7.3f} /{rm:6.2f}" for cr, rm in row)
              + "   | CRPS vs N=%d: " % Ns[-1] + " ".join(f"{100 * (cr / ref[0] - 1):+.1f}%" for cr, _ in row))
