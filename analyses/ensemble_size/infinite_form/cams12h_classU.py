#!/usr/bin/env python3
"""A14 class U for Foehn-CAMS in Aurora's protocol (blind record, D4: its first score, eight members, as the Methods state).
Scorer experiments/global1km/eval_cams12h.py: its functions (to_domain, truth_days) and constants are imported unchanged and
its loop is replayed with the same arithmetic on runs/cams04_flow_shift2.6_emis/cams12h/pred.npy (214 initialisations, 8
members); identity: the replayed per-initialisation RMSEs, pooled RMSE, CRPS and spread/skill equal
experiments/figures/data/cams12h_scores_foehn_emis.npz (the paper's file). Added per initialisation, lead and region: the
latitude-weighted member variance (ddof 1) and pair term, hence
  RMSE_inf of an initialisation = sqrt(MSE - mean s2 / 8), RMSE_8 = RMSE as run (eight members), and the scores of the paper:
  mean over initialisations of the per-initialisation RMSE, its change against Aurora and the CAMS forecast with the paper's
  paired moving-block bootstrap over initialisations (default_rng(0), blocks of 7, 10,000 resamples);
  spread/skill sqrt(mean s2) / RMSE_inf (pooled), CRPS fair (= infinite) and eight-member (= as run plus nothing: M = 8 issued).
The noise of Foehn-CAMS is drawn afresh for every initialisation and lead (foehn_cams.py l. 376: seed gi * 1000 + k), so the
draw of the members is resampled with the initialisations: V_draw = 0 (NOTES.md item 4).
    python cams12h_classU.py   -> results/cams12h_classU.json"""
import glob
import sys
import time

import numpy as np

from u_common import CRAN, FIGS, RES, safe_json

sys.path.insert(0, str(CRAN / "experiments" / "global1km"))
import eval_cams12h as EC  # noqa: E402  (unchanged; main() not run)

t0 = time.time()
NY, NX, NL, LAT, LON, REGIONS = EC.NY, EC.NX, EC.NL, EC.LAT, EC.LON, EC.REGIONS
w = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, NX))
LA2, LO2 = np.meshgrid(LAT, LON, indexing="ij")
RM = {r: (np.ones((NY, NX), bool) if b is None else (LA2 <= b[0]) & (LA2 > b[1]) & (LO2 >= b[2]) & (LO2 < b[3])) for r, b in REGIONS.items()}
aur = {}
for p in sorted(glob.glob(str(EC.AUR / "part_*.npz"))):
    z = np.load(p)
    for d, pr in zip(z["days"], z["pred"]):
        aur[int(d)] = EC.to_domain(pr, z["lat"], z["lon"])
cfc_meta = {int(p.name[5:7]): p for p in sorted(EC.CFC.glob("2022_*.npz"))}
OUTD = sys.argv[1] if len(sys.argv) > 1 else "cams12h"                  # cams12h (first score) | cams12h_lag1 | cams12h_lag1b
REFN = sys.argv[2] if len(sys.argv) > 2 else "cams12h_scores_foehn_emis.npz"
fdir = EC.HERE / "runs" / "cams04_flow_shift2.6_emis" / OUTD
fdoys = list(np.load(fdir / "doys.npy"))
done = {int(p.name[5:8]) for p in fdir.glob("done_*")}
foehn = (np.load(fdir / "pred.npy", mmap_mode="r"), {d: n for n, d in enumerate(fdoys) if d in done})
use_cfc = bool(cfc_meta)
doys = [d for d in range(120, 334) if d in aur and (not use_cfc or int(str(np.datetime64("2022-01-01") + np.timedelta64(d, "D"))[5:7]) in cfc_meta)
        and d in foehn[1]]
SYS = ["Foehn-CAMS (mean)", "Foehn-CAMS (member)", "Aurora", "CAMS forecast", "persistence"]
nS, nR = len(SYS), len(REGIONS)
wsum = np.array([(w * m).sum() for m in RM.values()])
per_init = np.full((nS, len(doys), NL, nR), np.nan, np.float32)
mse_i = np.zeros((nS, len(doys), NL, nR)); var_i = np.zeros((len(doys), NL, nR)); crps_i = np.zeros((len(doys), NL, nR)); pair_i = np.zeros((len(doys), NL, nR))
T = EC.truth_days(range(min(doys), max(doys) + NL // 2 + 1))
cfc_cache = {}
for c_, d in enumerate(doys):
    A = aur[d].astype(np.float32)
    date = np.datetime64("2022-01-01") + np.timedelta64(d, "D")
    mo, dom = int(str(date)[5:7]), int(str(date)[8:10]) - 1
    if mo not in cfc_cache:
        zc = np.load(cfc_meta[mo]); cfc_cache = {mo: (zc["pm25"], zc["lat"], zc["lon"])}
    pc, clat, clon = cfc_cache[mo]
    Cf = EC.to_domain(pc[dom].astype(np.float32), clat, clon)
    P = np.broadcast_to(T[d][0], (NL, NY, NX))
    E = np.asarray(foehn[0][foehn[1][d]], np.float32)
    fc = [E.mean(0), E[0], A, Cf, P]
    for k in range(1, NL + 1):
        vt = 2 * d + k
        y = T[vt // 2][vt % 2]
        for s, F_ in enumerate(fc):
            e = F_[k - 1] - y
            se = w * e ** 2
            for r, m in enumerate(RM.values()):
                mse_r = (se * m).sum() / wsum[r]
                per_init[s, c_, k - 1, r] = np.sqrt(mse_r); mse_i[s, c_, k - 1, r] = mse_r
        Ek = np.sort(E[:, k - 1], 0); M = Ek.shape[0]
        pair = (2 * (2 * np.arange(M) - M + 1)[:, None, None] * Ek).sum(0) / (M * (M - 1))
        cr = np.abs(Ek - y).mean(0) - 0.5 * pair
        vr = Ek.var(0, ddof=1)
        for r, m in enumerate(RM.values()):
            crps_i[c_, k - 1, r] = (w * cr * m).sum() / wsum[r]; var_i[c_, k - 1, r] = (w * vr * m).sum() / wsum[r]
            pair_i[c_, k - 1, r] = (w * pair * m).sum() / wsum[r]
    if c_ % 20 == 0:
        print(f"{c_}/{len(doys)} {time.time() - t0:.0f} s", flush=True)
ref = np.load(FIGS / "data" / REFN)
n = len(doys)
ident = {"doys": bool(np.array_equal(np.array(doys), ref["doys"])), "systems": [str(s) for s in ref["systems"]] == SYS,
         "rmse_per_init": bool(np.array_equal(per_init, ref["rmse_per_init"], equal_nan=True)),
         "rmse_per_init_maxabs": float(np.nanmax(np.abs(per_init - ref["rmse_per_init"]))),
         "crps_maxabs": float(np.max(np.abs(crps_i.sum(0) / n - ref["crps"]))),
         "spread_skill_maxabs": float(np.max(np.abs(np.sqrt(var_i.sum(0) / n) * np.sqrt(9 / 8) / np.sqrt(mse_i[0].sum(0) / n) - ref["spread_skill"])))}
print("identity with cams12h_scores_foehn_emis.npz:", ident, flush=True)
M = 8
rm_inf = np.sqrt(np.maximum(mse_i[0] - var_i / M, 0))                                 # (inits, leads, regions)
rng = np.random.default_rng(0)
nd, B = n, 7
starts = rng.integers(0, max(nd - B + 1, 1), size=(10000, -(-nd // B)))
idx = (starts[:, :, None] + np.arange(B)).reshape(10000, -1)[:, :nd]
OUT = {"identity": ident, "n_init": n, "members": M, "leads_h": list(range(12, 121, 12)), "regions": list(REGIONS), "by_region": {},
       "v_draw": "0: the members are drawn afresh for every initialisation and lead (resampled with the initialisations)"}
for r, rn in enumerate(REGIONS):
    o = {"rmse_as_run": per_init[0, :, :, r].astype(np.float64).mean(0).tolist(), "rmse_inf": rm_inf[:, :, r].mean(0).tolist(),
         "rmse_aurora": per_init[2, :, :, r].astype(np.float64).mean(0).tolist(), "rmse_cams": per_init[3, :, :, r].astype(np.float64).mean(0).tolist()}
    for j, nm in ((2, "Aurora"), (3, "CAMS forecast")):
        b = per_init[j, :, :, r].astype(np.float64)
        for form, a in (("as_run", per_init[0, :, :, r].astype(np.float64)), ("inf", rm_inf[:, :, r])):
            pt = 100 * (a.mean(0) / b.mean(0) - 1)
            bs = 100 * (a[idx].mean(1) / b[idx].mean(1) - 1)
            lo, hi = np.percentile(bs, [2.5, 97.5], 0)
            o[f"change_pct_vs_{nm}_{form}"] = {"v": pt.tolist(), "ci": [lo.tolist(), hi.tolist()]}
    mp = mse_i[0, :, :, r].mean(0); vp = var_i[:, :, r].mean(0)
    o["spread_skill_as_run"] = (np.sqrt(vp) * np.sqrt((M + 1) / M) / np.sqrt(mp)).tolist()
    o["spread_skill_inf"] = (np.sqrt(vp) / np.sqrt(mp - vp / M)).tolist()
    o["crps_fair_as_run_equals_inf"] = crps_i[:, :, r].mean(0).tolist()
    o["crps_8_equals_as_run_note"] = "M = 8 members as run: the eight-member CRPS is fair + pair/8 = " + str(np.round((crps_i[:, :, r] + 0.5 * pair_i[:, :, r] / 8).mean(0), 4).tolist())
    OUT["by_region"][rn] = o
OUT["run"] = OUTD; OUT["reference_file"] = REFN
safe_json(RES / ("cams12h_classU.json" if OUTD == "cams12h" else f"cams12h_classU_{OUTD}.json"), OUT)
print(f"done {time.time() - t0:.0f} s", flush=True)
