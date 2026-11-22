#!/usr/bin/env python3
"""A14 class U, fix 15: the emission gains on 0.25 deg cell means (main_text.tex:275, supplementary.tex:510: North China
Plain 5.6, Indo-Gangetic Plain 3.0 over the week; East Asia 2.6, South Asia 2.5). analyses/emissions/recompute.py c25(): the
same four models, common forecasts, masks (land cells with GHAP > 0 and every model finite), boxes and sums, replayed with its
arithmetic (identity of n and sse asserted against its out/sums_c25.npz), plus for each model its own member-0 term of the
stored 0.25 deg means (map25_m0): q = sum (m0 - xbar)^2 / 3, so MSE_inf = (sse - q)/n and MSE_8 = (sse - q/2)/n (A14 4a,
member 0 only). No station is read.
    python c25_pass.py   -> data/c25_terms.npz"""
import os  # FOEHN_WORK, FOEHN_ENVS: roots of the working trees and of the environments
import glob
import sys
import time

import numpy as np

from u_common import DATA, REL, safe_npz

E_DIR = REL.parent / "emissions"
RUNS = (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres")
EV = {"A": "_ghres/eval_final", "C": "_notransport/eval_final", "e02": "_ghres_ftemis12/eval_final_e02", "ctrl": "_ghres_ftemis12ctrl0/eval_final_e02"}
MODELS = list(EV)
PLAINS = {"North China Plain": (41.0, 34.0, 112.0, 120.0), "Indo-Gangetic Plain": (31.0, 24.0, 74.0, 88.0)}
REGIONS = {"Europe": (72, 35, -25, 45), "North America": (72, 15, -170, -50), "South America": (15, -56, -95, -30),
           "Africa": (37, -35, -20, 52), "Middle East": (42, 12, 35, 63), "South Asia": (37, 5, 60, 97),
           "East Asia": (54, 18, 97, 146), "Southeast Asia": (18, -11, 92, 142), "Oceania": (-10, -48, 110, 180)}
ref = np.load(E_DIR / "out" / "sums_c25.npz")
assert [str(x) for x in ref["names"]] == list(REGIONS) + ["All regions", "All land"] + list(PLAINS) and [str(x) for x in ref["models"]] == MODELS
t0 = time.time()
NR, NC = 576, 1440
lat = 84.0 - 0.25 * np.arange(NR); lon = -180.0 + 0.25 * np.arange(NC)
LA, LO = np.meshgrid(lat, lon, indexing="ij")
boxes = dict(REGIONS); boxes.update(PLAINS)
masks = {k: (LA <= b[0]) & (LA > b[1]) & (LO >= b[2]) & (LO < b[3]) for k, b in boxes.items()}
names = list(REGIONS) + ["All regions", "All land"] + list(PLAINS)
index = {}
for k, ev in EV.items():
    index[k] = {}
    for p in sorted(glob.glob(f"{RUNS}{ev}/part_*.npz")):
        for n, (d, h, _) in enumerate(np.load(p)["keys"]):
            index[k][(int(d), int(h))] = (p, n)
common = sorted(set.intersection(*[set(v) for v in index.values()]))
days = sorted({d for d, _ in common}); di = {d: i for i, d in enumerate(days)}
assert days == [int(x) for x in ref["days"]]
T = np.zeros((len(MODELS), len(names), 7, len(days), 3))               # n, sse, q (member-0 term)
cache = {}


def arrays(path):
    if path not in cache:
        if len(cache) > 8:
            cache.pop(next(iter(cache)))
        z = np.load(path)
        cache[path] = (z["map25"], z["map25_m0"], z["map25_ghap"])
    return cache[path]


by_part = {}
for (d, h) in common:
    by_part.setdefault(index["A"][(d, h)][0], []).append((d, h))
for ip, (pA, fcs) in enumerate(by_part.items()):
    for (d, h) in fcs:
        F, Q = [], []
        y = None
        for k in MODELS:
            p, n = index[k][(d, h)]
            m25, m0, g25 = arrays(p)
            F.append(m25[n].astype(np.float32)); Q.append(m0[n].astype(np.float32))
            if y is None:
                y = g25[n].astype(np.float32)
        ok = np.isfinite(y) & (y > 0)
        for f in F:
            ok &= np.isfinite(f)
        for b, nm in enumerate(names):
            if nm == "All regions":
                continue
            mk = ok if nm == "All land" else (ok & masks[nm])
            if not mk.any():
                continue
            yy = y[mk]
            for m in range(len(MODELS)):
                e = F[m][mk] - yy
                u = (Q[m][mk] - F[m][mk]).astype(np.float64)
                row = [mk.sum(), float((e.astype(np.float64) ** 2).sum()), float(np.nansum(u * u) / 3.0)]
                T[m, b, h - 1, di[d]] += row
                if nm in REGIONS:
                    T[m, names.index("All regions"), h - 1, di[d]] += row
    if ip % 16 == 0:
        print(f"  c25: part {ip + 1}/{len(by_part)} {time.time() - t0:.0f}s", flush=True)
same_n = bool(np.array_equal(T[..., 0], ref["T"][..., 0])); dse = float(np.max(np.abs(T[..., 1] - ref["T"][..., 1]) / np.maximum(ref["T"][..., 1], 1e-9)))
print("identity with sums_c25.npz: n equal", same_n, "| largest relative difference of sse", dse, flush=True)
assert same_n and dse < 1e-12
safe_npz(DATA / "c25_terms.npz", T=T, names=np.array(names), models=np.array(MODELS), days=np.array(days),
         identity=np.array(f"n equal {same_n}; sse max rel diff {dse:.2e}"))
print(f"done {time.time() - t0:.0f} s", flush=True)
