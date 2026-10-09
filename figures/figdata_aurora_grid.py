#!/usr/bin/env python3
"""Scores of Foehn-CAMS, Aurora, the CAMS forecast and climatology in Aurora's setup, for the 3 x 5 grid (RMSE, CRPS, spread-skill
ratio x Global, Europe, Americas, Africa, Asia). Initialisations 00 UTC 1 May - 30 Nov 2022 (those of eval_cams12h.py), leads
12-120 h, against the CAMS analysis, on the 0.4 deg domain 84 N - 60 S, every score latitude-weighted:
  RMSE  per initialisation, then averaged over initialisations (Aurora's convention; Foehn: ensemble mean);
  CRPS  fair CRPS of Foehn's eight members; absolute error for the deterministic Aurora and CAMS forecasts; for climatology, the
        CRPS of the climatological ensemble (CAMS analyses of 2018-2021 at the same hour within +-15 days: 124 members);
  SSR   Foehn's spread-skill ratio sqrt((M+1)/M var) / RMSE.
    python figdata_aurora_grid.py   -> data/aurora_grid.npz"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "global1km"))
from eval_cams12h import AUR, CFC, LAT, NX, NY, to_domain, truth_days  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "global1km" / "runs" / "cams04_flow_shift2.6_emis" / "cams12h"
SERIES = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km/cams_series.npy")
FIRST = np.datetime64("2017-10-01")
YEARS, WIN = (2018, 2019, 2020, 2021), 15
BOX = {"Europe": [(72, 35, -25, 45)], "Americas": [(72, 15, -170, -50), (15, -56, -95, -30)], "Africa": [(37, -35, -20, 52)],
       "Asia": [(54, -11, 60, 146)]}
SYS = ["Foehn-CAMS", "Aurora", "CAMS forecast", "Climatology"]


def masks():
    LA, LO = np.meshgrid(LAT, -180.0 + 0.4 * np.arange(NX), indexing="ij")
    out = {"Global": np.ones((NY, NX), bool)}
    for r, boxes in BOX.items():
        m = np.zeros((NY, NX), bool)
        for n, s, w, e in boxes:
            m |= (LA <= n) & (LA > s) & (LO >= w) & (LO < e)
        out[r] = m
    return out


def crps_ens(E, y):
    """fair CRPS per pixel of the ensemble E (M, ...) against y (sorted-ensemble formula)"""
    M = E.shape[0]
    Es = np.sort(E, 0)
    k = np.arange(1, M + 1, dtype=np.float32).reshape((M,) + (1,) * (E.ndim - 1))
    pair = (Es * (2 * k - M - 1)).sum(0) * 2 / (M * (M - 1))                       # mean |xi - xj| over i != j
    return np.abs(E - y).mean(0) - 0.5 * pair


def main():
    cz = np.load(HERE / "data" / "cams12h_scores_foehn_emis.npz")
    doys = [int(d) for d in cz["doys"]]; NL = len(cz["leads_h"])
    R = masks(); RN = list(R)
    w = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, NX))
    W = {r: w * m for r, m in R.items()}; WS = {r: W[r].sum() for r in RN}
    # climatological ensemble: the analyses of 2018-2021 at the same hour within +-15 days
    s = np.load(SERIES, mmap_mode="r")
    need = sorted({(2 * d + k) // 2 for d in doys for k in range(1, NL + 1)})
    daily = {}
    for doy in range(min(need) - WIN, max(need) + WIN + 1):
        daily[doy] = np.stack([np.asarray(s[2 * int((np.datetime64(f"{y}-01-01") + np.timedelta64(doy, "D") - FIRST).astype(int)) + np.arange(2), 1],
                                          np.float32) for y in YEARS])               # (4 years, 2 hours, NY, NX)
    print("climatology members loaded", flush=True)
    T = truth_days(range(min(doys), max(need) + 1))
    clim_crps, clim_mean = {}, {}
    for doy in need:
        for hr in (0, 1):
            ens = np.concatenate([daily[x][:, hr] for x in range(doy - WIN, doy + WIN + 1)])          # (124, NY, NX)
            clim_mean[(doy, hr)] = ens.mean(0)
            clim_crps[(doy, hr)] = crps_ens(ens, T[doy][hr])
    print("climatology CRPS done", flush=True)
    aur = {}
    for p in sorted(AUR.glob("part_*.npz")):
        z = np.load(p)
        for d, pr in zip(z["days"], z["pred"]):
            aur[int(d)] = (p, int(np.where(z["days"] == d)[0][0]))
    fd = list(np.load(RUN / "doys.npy")); pred = np.load(RUN / "pred.npy", mmap_mode="r")
    rmse = np.zeros((len(SYS), NL, len(RN))); crps = np.zeros((len(SYS), NL, len(RN))); ssr_v = np.zeros((NL, len(RN)))
    ssr_e = np.zeros((NL, len(RN))); n = 0; cfc = {}
    for c, d in enumerate(doys):
        p, k_ = aur[d]; z = np.load(p); A = to_domain(z["pred"][k_].astype(np.float32), z["lat"], z["lon"])
        mo = int(str(np.datetime64("2022-01-01") + np.timedelta64(d, "D"))[5:7]); dom = int(str(np.datetime64("2022-01-01") + np.timedelta64(d, "D"))[8:10]) - 1
        if mo not in cfc:
            zc = np.load(CFC / f"2022_{mo:02d}.npz"); cfc = {mo: (zc["pm25"], zc["lat"], zc["lon"])}
        pc, la, lo = cfc[mo]; Cf = to_domain(pc[dom].astype(np.float32), la, lo)
        E = np.asarray(pred[fd.index(d)], np.float32)                                  # (M, NL, NY, NX)
        M = E.shape[0]
        for k in range(NL):
            vt = 2 * d + k + 1; y = T[vt // 2][vt % 2]
            Em = E[:, k].mean(0)
            fields = {"Foehn-CAMS": (Em, crps_ens(E[:, k], y)), "Aurora": (A[k], np.abs(A[k] - y)), "CAMS forecast": (Cf[k], np.abs(Cf[k] - y)),
                      "Climatology": (clim_mean[(vt // 2, vt % 2)], clim_crps[(vt // 2, vt % 2)])}
            var = E[:, k].var(0, ddof=1)
            for si, sn in enumerate(SYS):
                f, cr = fields[sn]; e2 = (f - y) ** 2
                for ri, r in enumerate(RN):
                    rmse[si, k, ri] += np.sqrt((W[r] * e2).sum() / WS[r]); crps[si, k, ri] += (W[r] * cr).sum() / WS[r]
            e2 = (Em - y) ** 2
            for ri, r in enumerate(RN):
                ssr_v[k, ri] += (W[r] * var).sum() / WS[r]; ssr_e[k, ri] += (W[r] * e2).sum() / WS[r]
        n += 1
        if c % 40 == 0:
            print(c, "/", len(doys), flush=True)
    rmse /= n; crps /= n
    ssr = np.sqrt((M + 1) / M * ssr_v / n) / np.sqrt(ssr_e / n)
    np.savez(HERE / "data" / "aurora_grid.npz", rmse=rmse, crps=crps, ssr=ssr, systems=np.array(SYS), regions=np.array(RN),
             leads_h=cz["leads_h"], n_init=n)
    for ri, r in enumerate(RN):
        print(r, "RMSE 72h:", {sn: round(float(rmse[si, 5, ri]), 2) for si, sn in enumerate(SYS)},
              "CRPS 72h:", {sn: round(float(crps[si, 5, ri]), 2) for si, sn in enumerate(SYS)}, "SSR:", np.round(ssr[:, ri], 2))


if __name__ == "__main__":
    main()
