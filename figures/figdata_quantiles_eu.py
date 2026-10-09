#!/usr/bin/env python3
"""Data of the Extended Data figure 'distribution of the forecasts versus the AI baselines' (Europe, 0.25 deg, 2022):
quantiles of the forecast PM2.5 over all valid land cells and start days, against the same quantiles of the target
(GHAP 0.25 deg cell means), at leads 1 and 3 days. Systems: Foehn (global model, 1 km -> 0.25 deg cell means of the
ensemble mean and of one member; the global evaluation), and the Europe-trained AI baselines on the same days
(figdata.py: ConvLSTM, Earthformer, SimVP, TopoFlow, and the same network as the Europe flow model trained by
regression). All systems on the same cells and days.
    python figdata_quantiles_eu.py <global run>   -> data/quantiles_eu_<run>.npz"""
import glob
import sys
from pathlib import Path

import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else "fp_flow_shift25_global_L7_fix"
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
E = EXP / "global1km" / "runs" / RUN / "eval_global"
R0, C0 = 48, 620                                   # Europe window (72 N, 25 W) in the global 0.25 deg grid (84 N, 180 W)
P = np.array([0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.8, 99.9])
BASE = {"convlstm": "ConvLSTM", "earthformer": "Earthformer", "simvp": "SimVP", "topoflow": "TopoFlow", "reg": "Regression (same network)"}

meta = np.load(EXP / "dejavu/data/meta.npz")
land = meta["land"].astype(bool)
i0 = int(np.where(meta["year"] == 2022)[0][0])
pm = np.load(EXP / "dejavu/data/pm.npy", mmap_mode="r")
parts = [np.load(p) for p in sorted(glob.glob(str(E / "part_*.npz")))]
keys = np.concatenate([p["keys"] for p in parts])
kidx = {(int(d), int(h)): (k, j) for k, p in enumerate(parts) for j, (d, h, t) in enumerate(p["keys"])}
gp = np.load("/scratch/project_462001140/ammar/eccv/data/world/global1km/pm.npy", mmap_mode="r")
HAS_M0 = "map25_m0" in parts[0].files                  # one member (evaluations after 2026-10-03)
if not HAS_M0:
    print("no member maps in this evaluation: the one-member curve is omitted")

ref = np.load(HERE / "data" / "earthformer.npz")
days, leads = [int(d) for d in ref["days"]], [int(h) for h in ref["leads"]]
out = {"p": P, "leads": np.array(leads)}
for li, h in enumerate(leads):
    vals = {"target": [], "foehn_mean": [], "foehn_member": []} | {k: [] for k in BASE}
    chk = []
    for di, d in enumerate(days):
        if (d, h) not in kidx or d + h >= 365:
            continue
        k, j = kidx[(d, h)]
        g_mean = parts[k]["map25"][j][R0:R0 + 168, C0:C0 + 280].astype(np.float32)
        g_mem = (parts[k]["map25_m0"][j][R0:R0 + 168, C0:C0 + 280].astype(np.float32) if HAS_M0 else g_mean)
        t = np.asarray(pm[i0 + d + h], np.float32)
        chk.append(np.nanmax(np.abs(t - np.asarray(gp[1826 + d + h], np.float32)[R0:R0 + 168, C0:C0 + 280])))
        base = {}
        for bk in BASE:
            z = np.load(HERE / "data" / f"{bk}.npz")
            bd = {int(x): n for n, x in enumerate(z["days"])}
            base[bk] = z["pred"][bd[d], li, 0].astype(np.float32) if d in bd else None
        m = land & np.isfinite(t) & np.isfinite(g_mean) & np.isfinite(g_mem) & np.all([np.isfinite(b) for b in base.values() if b is not None], 0)
        if any(b is None for b in base.values()) or m.sum() == 0:
            continue
        vals["target"].append(t[m]); vals["foehn_mean"].append(g_mean[m]); vals["foehn_member"].append(g_mem[m])
        for bk, b in base.items():
            vals[bk].append(b[m])
    print(f"lead {h}: {len(vals['target'])} days, {sum(len(v) for v in vals['target'])} cell-days | Europe vs global target max |diff| {max(chk):.3g}")
    if not HAS_M0:
        vals.pop("foehn_member")
    for name, v in vals.items():
        x = np.concatenate(v)
        out[f"q_{name}_{h}"] = np.percentile(x, P)
        out[f"mean_{name}_{h}"] = x.mean()
    tq = out[f"q_target_{h}"]
    print("   quantile ratio forecast/target at 95 / 99 / 99.9 %:")
    for name in vals:
        q = out[f"q_{name}_{h}"]
        print(f"   {name:14s} {q[9] / tq[9]:.2f} {q[11] / tq[11]:.2f} {q[14] / tq[14]:.2f}")
np.savez(HERE / "data" / f"quantiles_eu_{RUN}.npz", **out, names=np.array(["target", "foehn_mean", "foehn_member"] + list(BASE)),
         labels=np.array(["GHAP (target)", "Foehn, ensemble mean", "Foehn, one member"] + list(BASE.values())))
print("saved")
