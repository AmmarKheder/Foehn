#!/usr/bin/env python3
"""Station scores of several global runs on identical station-days (forecasts evaluated for every run): fair CRPS,
RMSE of the ensemble mean, spread/skill, by lead. Stations: official networks only (EEA, CNEMC, EPA, AirKorea, NIES, CPCB).
    python compare_models.py run1 run2[:eval_dir] ... [--eval eval_global]   (run:eval_dir: another evaluation of the run)"""
import argparse, glob
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--eval", default="eval_global")
ap.add_argument("--clim", action="store_true", help="also the probabilistic climatology (2017-2020 GHAP at the station, +-7 days)")
a = ap.parse_args()
GW = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
st = np.load(GW / "stations_2022.npz")
obs, src = st["daily"], st["source"]
official = src != 2
key = lambda la, lo: np.round(la, 4) * 1e6 + np.round(lo, 4)
ref = {v: i for i, v in enumerate(key(st["lat"], st["lon"]))}
sets = {}
for f in sorted(GW.glob("stations_2022*.npz")):
    z = np.load(f); sets[len(z["lat"])] = np.array([ref.get(v, -1) for v in key(z["lat"], z["lon"])])
D = {}
for r in a.runs:
    rn, ev = (r.split(":") + [a.eval])[:2]
    P = [np.load(p) for p in sorted(glob.glob(f"runs/{rn}/{ev}/part_*.npz"))]
    k = np.concatenate([p["keys"] for p in P]); e = np.concatenate([p["st_ens"] for p in P])
    idx = sets[e.shape[1]]                                       # map this run's stations onto stations_2022.npz
    full = np.full((e.shape[0], len(st["lat"]), e.shape[2]), np.nan, np.float16)
    full[:, idx[idx >= 0]] = e[:, idx >= 0]
    D[r] = ({(int(d), int(h)): n for n, (d, h, t) in enumerate(k)}, full)
common = set.intersection(*[set(v[0]) for v in D.values()])
CRC_T = None
if a.clim:                                   # fair CRPS of the climatological ensemble (as figures/figdata_global_skill.py)
    HIST = np.load(GW / "clim" / "st_ghap_2017_2020.npy").astype(np.float32)          # (1461, stations)
    assert HIST.shape[1] == obs.shape[0], (HIST.shape, obs.shape)
    hdoy = np.load(GW / "meta.npz")["doy"][:1461]
    CRC_T = np.full((obs.shape[1], obs.shape[0]), np.nan, np.float32)
    for t in range(obs.shape[1]):
        dd = np.abs(hdoy - t); dd = np.minimum(dd, 365 - dd)
        Xs = np.sort(np.where(HIST[dd <= 7] > 0, HIST[dd <= 7], np.nan).T, 1)
        Mc = np.isfinite(Xs).sum(1); kk = np.arange(1, Xs.shape[1] + 1)[None, :]
        wgt = np.where(kk <= Mc[:, None], 2 * kk - Mc[:, None] - 1, 0)
        pair = np.sum(np.nan_to_num(Xs) * wgt, 1) / np.maximum(Mc * (Mc - 1), 1)
        with np.errstate(all="ignore"):
            CRC_T[t] = np.where(Mc >= 2, np.nanmean(np.abs(Xs - obs[:, t][:, None]), 1) - pair, np.nan)
print("forecasts per run:", {r: len(v[0]) for r, v in D.items()}, "| common:", len(common))
print(f"{'lead':4s} " + " | ".join(f"{r[:34]:>34s}" for r in a.runs))
for h in range(1, 8):
    S = np.zeros((len(a.runs), 4)); C = 0.0
    for (d, hh) in sorted(x for x in common if x[1] == h):
        y = obs[:, d + h]
        Es = [D[r][1][D[r][0][(d, hh)]].astype(np.float32) for r in a.runs]
        ok = np.isfinite(y) & official & np.all([np.isfinite(E).all(1) for E in Es], 0)
        if CRC_T is not None:                    # identical station-days for the climatology
            ok &= np.isfinite(CRC_T[d + h]); C += CRC_T[d + h][ok].sum()
        for i, E in enumerate(Es):
            Eo, yo = E[ok], y[ok]; M = Eo.shape[1]
            Es_ = np.sort(Eo, 1); pair = (Es_ * (2 * np.arange(1, M + 1) - M - 1)).sum(1) / (M * (M - 1))
            S[i] += [ok.sum(), (np.abs(Eo - yo[:, None]).mean(1) - pair).sum(), ((Eo.mean(1) - yo) ** 2).sum(), Eo.var(1, ddof=1).sum()]
    if S[0, 0] == 0:
        continue
    crps, rmse = S[:, 1] / S[:, 0], np.sqrt(S[:, 2] / S[:, 0])
    ss = np.sqrt(5 / 4 * S[:, 3] / S[:, 0]) / rmse
    print(f"{h:4d} " + " | ".join(f"CRPS {c:5.2f} RMSE {rm:5.2f} ss {s:4.2f}" for c, rm, s in zip(crps, rmse, ss))
          + (f" | climatology CRPS {C / S[0, 0]:5.2f}" if CRC_T is not None else "") + f"  (n={int(S[0, 0])})")
