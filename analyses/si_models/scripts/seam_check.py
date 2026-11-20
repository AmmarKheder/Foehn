#!/usr/bin/env python3
"""Recompute the seam ratios of Supplementary Fig. 8d that are hard-coded in experiments/figures/fig_km.py for the
tile-free 1 km European network (1.01) and for the same network trained by regression (1.19), with the definition of
experiments/regional/tiled_seams.py (mean absolute jump across the edges of the 768 x 1536 tiling divided by the mean
jump elsewhere; land pixels; start days 40, 100, 200, 300 of 2022; lead 1 day). Read-only on the project.
    python3 seam_check.py
"""
import glob
import json
from pathlib import Path

import numpy as np

EXP = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments")
OUT = Path(__file__).resolve().parents[1] / "out"
FH, FW, NR, NC = 4192, 6992, 768, 1536
rows = list(range(0, FH - NR, NR)) + [FH - NR]
cols = list(range(0, FW - NC, NC)) + [FW - NC]
rlines, clines = rows[1:], cols[1:]
land = np.load(EXP / "fullimage/data/static1km.npz")["land"].astype(np.float32)


def ratio(a):
    a = np.where(land > 0, a, np.nan)
    dc, dr = np.abs(np.diff(a, axis=1)), np.abs(np.diff(a, axis=0))
    return float(0.5 * (np.nanmean(dc[:, [c - 1 for c in clines]]) / np.nanmean(dc) + np.nanmean(dr[[r - 1 for r in rlines]]) / np.nanmean(dr)))


res = {"tile_edges_rows": rlines, "tile_edges_cols": clines}
for name, d in (("tile_free_flow_shift25", "fullimage/runs/flow_shift25/eval_hres"), ("same_network_regression", "fullimage/runs/reg/eval_hres"),
                ("tile_free_flow_noshift", "fullimage/runs/flow/eval_hres")):
    vals = {}
    for p in sorted(glob.glob(str(EXP / d / "part_*.npz"))):
        with np.load(p) as z:
            for k in z.files:
                if k.startswith("pred_") and k.endswith("_h1"):
                    vals[k] = ratio(z[k].astype(np.float32))
    res[name] = dict(per_map=vals, mean=float(np.mean(list(vals.values()))) if vals else None)
    print(name, {k: round(v, 3) for k, v in sorted(vals.items())}, "mean", None if not vals else round(float(np.mean(list(vals.values()))), 3), flush=True)
for name in ("earthformer", "simvp"):
    vals = {}
    for d in (40, 100, 200, 300):
        f = EXP / ("regional/runs/%s/tiled_europe_d%d_h1.npy" % (name, d))
        if f.exists():
            vals["d%d" % d] = ratio(np.load(f).astype(np.float32))
    res[name + "_tiled"] = dict(per_map=vals, mean=float(np.mean(list(vals.values()))) if vals else None)
    print(name, "tiled", {k: round(v, 3) for k, v in vals.items()}, "mean", round(float(np.mean(list(vals.values()))), 3), flush=True)
(OUT / "seam_check.json").write_text(json.dumps(res, indent=1))
print("written", OUT / "seam_check.json")
