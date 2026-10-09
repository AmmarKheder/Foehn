#!/usr/bin/env python3
"""'Foehn trained as Aurora': PM2.5 forecasts scored against the CAMS analysis, Aurora's protocol. Initialisations 00 UTC
1 May - 30 Nov 2022, leads 12, 24, ..., 120 h, instantaneous fields, on the native 0.4 deg CAMS / Aurora grid restricted
to rows 84 N .. 60 S (the domain common to all systems). Every system on the same initialisations, valid times and cells:
  Foehn-CAMS   foehn_cams.py predict (M members): ensemble mean, and member 0 (a single sample, like a deterministic run)
  Aurora       ../benchmark/aurora/out_12h (Aurora's 12-hour model, run from the same CAMS analyses, 10 steps)
  CAMS         operational forecasts, 00 UTC runs (cams_forecast_12h, dl_cams_forecast12h.py)
  persistence  the analysis at t0
Truth: the CAMS analysis at the valid time (../benchmark/aurora/raw/<doy>.npz, float32, Aurora's initial conditions).
Per lead and region: latitude-weighted RMSE (per initialisation, then averaged over initialisations, as Aurora reports
it), RMSE of the pooled squared errors, bias; for Foehn the fair CRPS (lat-weighted) and the spread / skill ratio.
    python eval_cams12h.py [--run <foehn_cams run>] [--out cams12h] [--tag x]   -> ../figures/data/cams12h_scores<tag>.npz"""
import argparse
import glob
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "benchmark" / "aurora" / "raw"
AUR = HERE.parent / "benchmark" / "aurora" / "out_12h"
CFC = Path("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_12h")
NY, NX, NL = 361, 900, 10
LAT = 84.0 - 0.4 * np.arange(NY)
LON = -180.0 + 0.4 * np.arange(NX)
REGIONS = {"Global": None, "Europe": (72, 35, -25, 45), "North America": (72, 15, -170, -50), "South America": (15, -56, -95, -30),
           "Africa": (37, -35, -20, 52), "Middle East": (42, 12, 35, 63), "South Asia": (37, 5, 60, 97),
           "East Asia": (54, 18, 97, 146), "Southeast Asia": (18, -11, 92, 142), "Oceania": (-10, -48, 110, 180)}


def to_domain(a, lat, lon):
    """(..., H, W) on a 0.4 deg grid with decreasing lat and lon from 0 or -180 -> (..., 361, 900) at LAT x LON (exact)"""
    i0 = int(np.argmin(np.abs(np.asarray(lat) - 84.0)))
    assert abs(lat[i0] - 84.0) < 1e-3 and abs(lat[i0 + NY - 1] + 60.0) < 1e-3, (lat[:3], lat[-3:])
    a = a[..., i0:i0 + NY, :]
    j0 = int(np.argmin(np.abs(((np.asarray(lon) + 180.0) % 360.0))))           # column of lon -180 (= 180)
    assert abs(((lon[j0] + 180.0) % 360.0)) < 1e-3 and abs((lon[1] - lon[0]) - 0.4) < 1e-4
    return np.roll(a, -j0, axis=-1)


def truth_days(doys):
    """CAMS analysis PM2.5 (ug m-3) at 00 and 12 UTC of the needed days -> {doy: (2, NY, NX)}"""
    out = {}
    for d in sorted(set(doys)):
        z = np.load(RAW / f"{d:03d}.npz")
        out[d] = to_domain(z["surf"][:, 5] * 1e9, z["lat"], z["lon"]).astype(np.float32)
    return out


def main(a):
    w = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, NX))
    LA2, LO2 = np.meshgrid(LAT, LON, indexing="ij")
    RM = {r: (np.ones((NY, NX), bool) if b is None else (LA2 <= b[0]) & (LA2 > b[1]) & (LO2 >= b[2]) & (LO2 < b[3]))
          for r, b in REGIONS.items()}
    # --- systems
    aur = {}
    for p in sorted(glob.glob(str(AUR / "part_*.npz"))):
        z = np.load(p)
        for d, pr in zip(z["days"], z["pred"]):
            aur[int(d)] = to_domain(pr, z["lat"], z["lon"])                                       # (10, NY, NX) float16
    cfc_meta = {}
    for p in sorted(CFC.glob("2022_*.npz")):
        cfc_meta[int(p.name[5:7])] = p
    foehn, fdoys = None, []
    if a.run:
        fdir = HERE / "runs" / a.run / a.out
        fdoys = list(np.load(fdir / "doys.npy"))
        done = {int(p.name[5:8]) for p in fdir.glob("done_*")}
        foehn = (np.load(fdir / "pred.npy", mmap_mode="r"), {d: n for n, d in enumerate(fdoys) if d in done})
    use_cfc = bool(cfc_meta)                                 # CAMS forecasts not downloaded yet: score the others
    doys = [d for d in range(120, 334) if d in aur and (not use_cfc or int(str(np.datetime64("2022-01-01") + np.timedelta64(d, "D"))[5:7]) in cfc_meta)
            and (foehn is None or d in foehn[1])]
    print(f"{len(doys)} common initialisations (Aurora {len(aur)}, CAMS months {sorted(cfc_meta)}, "
          f"Foehn {len(foehn[1]) if foehn else '-'})", flush=True)
    SYS = (["Foehn-CAMS (mean)", "Foehn-CAMS (member)"] if foehn else []) + ["Aurora"] + (["CAMS forecast"] if use_cfc else []) + ["persistence"]
    nS, nR = len(SYS), len(REGIONS)
    rm_init = np.zeros((nS, NL, nR)); mse_pool = np.zeros((nS, NL, nR)); bias = np.zeros((nS, NL, nR)); wsum = np.zeros(nR)
    crps = np.zeros((NL, nR)); var = np.zeros((NL, nR)); n_init = np.zeros(NL)
    msemap = np.zeros((nS, NL, NY, NX), np.float32)
    per_init = np.full((nS, len(doys), NL, nR), np.nan, np.float32)            # RMSE of each forecast (bootstrap)
    cache, cfc_cache = {}, {}
    for r, m in enumerate(RM.values()):
        wsum[r] = (w * m).sum()
    T = truth_days(range(min(doys), max(doys) + NL // 2 + 1))
    for c_, d in enumerate(doys):
        A = aur[d].astype(np.float32)
        date = np.datetime64("2022-01-01") + np.timedelta64(d, "D")
        mo, dom = int(str(date)[5:7]), int(str(date)[8:10]) - 1
        if use_cfc and mo not in cfc_cache:
            zc = np.load(cfc_meta[mo]); cfc_cache = {mo: (zc["pm25"], zc["lat"], zc["lon"])}
        if use_cfc:
            pc, clat, clon = cfc_cache[mo]
            Cf = to_domain(pc[dom].astype(np.float32), clat, clon)
        P = np.broadcast_to(T[d][0], (NL, NY, NX))
        fc = []
        if foehn:
            E = np.asarray(foehn[0][foehn[1][d]], np.float32)                                         # (M, 10, NY, NX)
            fc += [E.mean(0), E[0]]
        fc += [A] + ([Cf] if use_cfc else []) + [P]
        for k in range(1, NL + 1):
            vt = 2 * d + k
            y = T[vt // 2][vt % 2]
            for s, F_ in enumerate(fc):
                e = F_[k - 1] - y
                se = w * e ** 2
                msemap[s, k - 1] += e ** 2
                for r, m in enumerate(RM.values()):
                    mse_r = (se * m).sum() / wsum[r]
                    rm_init[s, k - 1, r] += np.sqrt(mse_r); mse_pool[s, k - 1, r] += mse_r
                    per_init[s, c_, k - 1, r] = np.sqrt(mse_r)
                    bias[s, k - 1, r] += (w * e * m).sum() / wsum[r]
            if foehn:
                Ek = np.sort(E[:, k - 1], 0); M = Ek.shape[0]
                pair = (2 * (2 * np.arange(M) - M + 1)[:, None, None] * Ek).sum(0) / (M * (M - 1))      # sum |x_a - x_b| / M(M-1)
                cr = np.abs(Ek - y).mean(0) - 0.5 * pair
                vr = Ek.var(0, ddof=1)
                for r, m in enumerate(RM.values()):
                    crps[k - 1, r] += (w * cr * m).sum() / wsum[r]; var[k - 1, r] += (w * vr * m).sum() / wsum[r]
            n_init[k - 1] += 1
        if c_ % 20 == 0:
            print(f"{c_}/{len(doys)}", flush=True)
    n = n_init[:, None]
    res = dict(systems=np.array(SYS), regions=np.array(list(REGIONS)), leads_h=np.arange(12, 121, 12), n_init=n_init, doys=np.array(doys),
               rmse=rm_init / n, rmse_pooled=np.sqrt(mse_pool / n), bias=bias / n, mse_map=msemap / n_init[:, None, None],
               rmse_per_init=per_init)
    if foehn:
        M = foehn[0].shape[1]
        res.update(crps=crps / n, spread_skill=np.sqrt(var / n) * np.sqrt((M + 1) / M) / res["rmse_pooled"][0], members=M)
    np.savez(HERE.parent / "figures" / "data" / f"cams12h_scores{a.tag}.npz", **res)
    for r, rn in enumerate(REGIONS):
        if rn != "Global" and not a.regions:
            continue
        print(f"\n=== {rn}: lat-weighted RMSE (ug m-3) vs the CAMS analysis, mean over {int(n_init[0])} initialisations")
        print(f"{'system':22s}" + "".join(f"{h:>7d}h" for h in range(12, 121, 12)))
        for s, sn in enumerate(SYS):
            print(f"{sn:22s}" + "".join(f"{v:8.2f}" for v in res["rmse"][s, :, r]))
        if foehn:
            print(f"{'Foehn CRPS':22s}" + "".join(f"{v:8.2f}" for v in res["crps"][:, r]))
            print(f"{'Foehn spread/skill':22s}" + "".join(f"{v:8.2f}" for v in res["spread_skill"][:, r]))
            for b in [x for x in ("Aurora", "CAMS forecast") if x in SYS]:
                j = SYS.index(b)
                print(f"{'Foehn vs ' + b:22s}" + "".join(f"{100 * (v / u - 1):+7.0f}%" for v, u in zip(res['rmse'][0, :, r], res['rmse'][j, :, r])))
        print(f"{'bias ' + SYS[0]:22s}" + "".join(f"{v:8.2f}" for v in res["bias"][0, :, r]))
    # paired moving-block bootstrap over initial days (blocks of 7 days, 10,000 resamples): relative RMSE difference of
    # the first system against each other, global, 95 % percentile intervals
    rng = np.random.default_rng(0)
    nd, B = len(doys), 7
    starts = rng.integers(0, max(nd - B + 1, 1), size=(10000, -(-nd // B)))
    idx = (starts[:, :, None] + np.arange(B)).reshape(10000, -1)[:, :nd]
    g = list(REGIONS).index("Global")
    for j in range(1, nS):
        a, b = per_init[0, :, :, g], per_init[j, :, :, g]                        # (days, leads)
        d = a[idx].mean(1) / b[idx].mean(1) - 1                                   # (resamples, leads)
        lo, hi = np.percentile(d, [2.5, 97.5], 0)
        print(f"{SYS[0]} vs {SYS[j]} (global RMSE, 95% CI): " + " ".join(
            f"{12 * (k + 1)}h {100 * (a[:, k].mean() / b[:, k].mean() - 1):+.0f}% [{100 * lo[k]:+.0f},{100 * hi[k]:+.0f}]" for k in range(NL)))
    print("saved")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--out", default="cams12h")
    ap.add_argument("--tag", default="")
    ap.add_argument("--regions", action="store_true", help="print every region (default: global only)")
    main(ap.parse_args())
