#!/usr/bin/env python3
"""A14 class U over land on 0.25 deg cell means (experiments/figures/compare_ghap_global.py: the paper's correlation with the
1 km estimates 0.89 / 0.80 / 0.76 at 1 / 4 / 7 days, RMSE a third below the climatology at 1 day and 3 % at 7 days).
1. compare_ghap_global.py executed from its unchanged source (paper label crps_e01 = final run:eval_final_crps_e01),
   np.savez redirected into a14/classU/data; identity of S and U75 with experiments/figures/data/compare_ghap_global_crps_e01.npz.
2. Its loop replayed (same expressions; the S replica must equal the executed S bit for bit) adding, for Foehn, the member-0
   terms of the stored 0.25 deg means (map25_m0; four members, A14 4a): u = m0 - xbar_4 per cell-day, sums of u and u^2.
   MSE_inf = MSE - mean(u^2)/3; MSE_8 = MSE_inf + mean(u^2)/6; pooled correlation (one centering):
   var_inf(xbar) = var(xbar) - var_c(u)/3, var_8(xbar) = var_inf + var_c(u)/6, cov(xbar, y) unchanged.
No station is read. No interval is printed in the paper for these numbers; point values only.
    python land25_pass.py   -> results/land25_classU.json"""
import time

import numpy as np

from u_common import DATA, FIGS, ISFX, RES, exec_script, safe_json

t0 = time.time()
SRC = FIGS / "compare_ghap_global.py"
ARGV = [f"crps_e01{ISFX}=green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4:eval_final_crps_e01{ISFX}"]   # design: label and folder _ar
ns, out, sha, written = exec_script(SRC, ARGV, DATA)
ref = np.load(FIGS / "data" / f"compare_ghap_global_crps_e01{ISFX}.npz")      # design: the S1 step-2 output of the same design
ident = {k: bool(np.array_equal(np.asarray(ns[k]), ref[k])) for k in ("S", "U75")}
print("[a14] executed unchanged, identity with the paper's npz:", ident, f"{time.time() - t0:.0f} s", flush=True)
runs, common, arrays, cams, aur, pers, clim, bil = (ns[k] for k in ("runs", "common", "arrays", "cams", "aur", "pers", "clim", "bil"))
RMASK, REG, SYS, alat, alon = ns["RMASK"], ns["REG"], ns["SYS"], ns["alat"], ns["alon"]
S2 = np.zeros_like(ns["S"]); U2 = np.zeros_like(ns["U75"])
EX = np.zeros((len(REG), 7, 4))                                   # n, sum u, sum u^2, sum u * ... (spare)
m0c = {}


def m0_of(p, n):
    if p not in m0c:
        if len(m0c) > 12:
            m0c.pop(next(iter(m0c)))
        m0c[p] = np.load(p)["map25_m0"]
    return m0c[p][n].astype(np.float32)


nbad = 0
for c_, (d, h) in enumerate(common):
    t = d + h
    if h <= 4 and d not in aur:
        continue
    X = []
    for label, idx in runs:
        p, n = idx[(d, h)]; m25, g25 = arrays(p); X.append(m25[n].astype(np.float32))
    p, n = runs[0][1][(d, h)]; y = arrays(p)[1][n].astype(np.float32)
    nr = len(runs)
    X += [bil(np.asarray(cams[d, 0, h], np.float32), 90.0, 0.4, 0.0, 0.4) if h <= 4 else np.full_like(y, np.nan),
          bil(aur[d][h - 1].astype(np.float32), float(alat[0]), float(alat[0] - alat[1]), float(alon[0]), float(alon[1] - alon[0]))
          if h <= 4 else np.full_like(y, np.nan)]
    if d in pers:
        pp, pn = pers[d]; X.append(arrays(pp)[1][pn].astype(np.float32))
    else:
        X.append(np.full_like(y, np.nan))
    X.append(clim[t])
    use = list(range(nr)) + ([nr, nr + 1] if h <= 4 else []) + [nr + 2, nr + 3]
    ok = np.isfinite(y) & (y > 0) & np.all([np.isfinite(X[k]) for k in use], 0)
    m0 = m0_of(p, n)
    u = (m0 - X[0]).astype(np.float64)
    bad = ok & ~np.isfinite(u)
    nbad += int(bad.sum())
    for g, (rn, rm) in enumerate(RMASK.items()):
        m = ok & rm
        if not m.any():
            continue
        yy = y[m]
        for k in use:
            x = X[k][m]; e = x - yy
            S2[g, h - 1, k] += [m.sum(), e.sum(), (e * e).sum(), np.abs(e).sum(), x.sum(), yy.sum(), (x * x).sum(), (yy * yy).sum(),
                                (x * yy).sum(), ((x > 35) & (yy > 35)).sum(), ((x > 35) | (yy > 35)).sum(), ((x > 75) & (yy > 75)).sum()]
            U2[g, h - 1, k] += ((x > 75) | (yy > 75)).sum()
        um = np.nan_to_num(u[m])
        EX[g, h - 1] += [m.sum(), um.sum(), (um * um).sum(), 0.0]
    if c_ % 200 == 0:
        print(c_, "/", len(common), f"{time.time() - t0:.0f} s", flush=True)
rep = {"S": bool(np.array_equal(S2, ns["S"])), "U75": bool(np.array_equal(U2, ns["U75"]))}
print("[a14] replay identity:", rep, "member 0 missing on", nbad, "cell-days", flush=True)
assert rep["S"] and rep["U75"], rep

RESU = {"script": str(SRC), "sha256": sha, "argv": ARGV, "identity_vs_paper_npz": ident, "replay_identity": rep,
        "member0_missing_cell_days": nbad, "systems": [str(s) for s in SYS], "regions": list(REG), "leads": {}}
for g, rn in enumerate(REG):
    for h in range(1, 8):
        v = ns["S"][g, h - 1]; ex = EX[g, h - 1]
        n = v[0, 0]
        if n == 0:
            continue
        mx, my = v[0, 4] / n, v[0, 5] / n
        vx, vy, cov = v[0, 6] / n - mx * mx, v[0, 7] / n - my * my, v[0, 8] / n - mx * my
        mu2 = ex[2] / n; vu = ex[2] / n - (ex[1] / n) ** 2
        mse = v[0, 2] / n
        e = {"n": float(n), "rmse": {"as_run": float(np.sqrt(mse)), "M8": float(np.sqrt(mse - mu2 / 3 + mu2 / 6)), "inf": float(np.sqrt(mse - mu2 / 3))},
             "r": {"as_run": float(cov / np.sqrt(vx * vy)), "M8": float(cov / np.sqrt((vx - vu / 3 + vu / 6) * vy)),
                   "inf": float(cov / np.sqrt((vx - vu / 3) * vy))},
             "amplitude_sd_ratio": {"as_run": float(np.sqrt(vx / vy)), "inf": float(np.sqrt((vx - vu / 3) / vy))}}
        refs = {}
        for k, nm in enumerate(SYS):
            w = ns["S"][g, h - 1, k]
            if w[0] == 0 or k == 0:
                continue
            mxr, myr = w[4] / w[0], w[5] / w[0]
            rr = (w[8] / w[0] - mxr * myr) / np.sqrt((w[6] / w[0] - mxr * mxr) * (w[7] / w[0] - myr * myr))
            rm = np.sqrt(w[2] / w[0])
            refs[str(nm)] = {"rmse": float(rm), "r": float(rr), "gain_pct": {f: 100 * (e["rmse"][f] / rm - 1) for f in e["rmse"]}}
        e["references"] = refs
        RESU["leads"].setdefault(rn, {})[h] = e
safe_json(RES / "land25_classU.json", RESU)
print(f"[a14] done {time.time() - t0:.0f} s", flush=True)
