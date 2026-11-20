#!/usr/bin/env python3
"""The model of the Aurora protocol (Foehn-CAMS: trained on the CAMS analyses at 0.4 deg; no 1 km estimate and no station
anywhere in its training, selection or inputs) at the official stations: station-day values.

This is the logic of experiments/global1km/eval_cams12h_stations.py (read-only source, not modified), keeping every value
instead of sums, so that the scores can be resampled by network and split by station class.
Forecasts initialised at 00 UTC, 1 May - 30 Nov 2022 (214 initialisations), 12-hour steps to 120 h, 0.4 deg, 8 members.
Daily mean of day d+n (n = 1..4) = 0.25 x (00 UTC d+n) + 0.5 x (12 UTC d+n) + 0.25 x (00 UTC d+n+1), read bilinearly at
the stations; the same for Aurora (run from the same CAMS analyses) and for the CAMS forecast of the same 00 UTC run.

For each initialisation d, n = 1..4 and official station:
  fc     Foehn-CAMS ensemble mean (8 members)        fc0   member 0 alone (a single sample)
  aur    Aurora                                      cams  CAMS forecast (00 UTC run of day d)
  obs    measurement of day d+n (UTC day)            obsm  measurement of day d-1 (the mask of the original script)
`paper` = the mask of the original script, which reproduces experiments/figures/data/cams12h_stations.npz (checked).

    python extract_foehn_cams.py     -> ../data/foehn_cams_station_day.npz
"""
import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data"
EXP = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments")
sys.path.insert(0, str(EXP / "global1km"))
from eval_cams12h import AUR, CFC, NX, NY, to_domain  # noqa: E402  (constants and the exact re-gridding; numpy only)

G = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
REF = EXP / "figures/data/cams12h_stations.npz"
RUNDIR = EXP / "global1km/runs/cams04_flow_shift2.6_emis/cams12h"
NETS = {0: "Europe (EEA)", 1: "China (CNEMC)", 3: "USA (EPA)", 4: "Korea (AirKorea)", 5: "Japan (NIES)", 6: "India (CPCB)"}
W3 = (0.25, 0.5, 0.25)

st = np.load(G / "stations_2022.npz")
keep = np.isin(st["source"], list(NETS)) & (st["lat"] < 84.0) & (st["lat"] > -60.0)
sid = np.flatnonzero(keep)                                                       # index in stations_2022.npz
slat, slon, src, obs = st["lat"][keep], st["lon"][keep], st["source"][keep], st["daily"][keep].astype(np.float32)
fi = (84.0 - slat) / 0.4; fj = ((slon + 180.0) % 360.0) / 0.4                   # domain grid: 84 - 0.4 i, -180 + 0.4 j
i0 = np.clip(np.floor(fi).astype(int), 0, NY - 2); wi = (fi - i0).astype(np.float32)
j0 = np.floor(fj).astype(int) % NX; j1 = (j0 + 1) % NX; wj = (fj - np.floor(fj)).astype(np.float32)
at_st = lambda f: ((1 - wi) * ((1 - wj) * f[..., i0, j0] + wj * f[..., i0, j1]) + wi * ((1 - wj) * f[..., i0 + 1, j0] + wj * f[..., i0 + 1, j1]))

fdoys = list(np.load(RUNDIR / "doys.npy")); E_all = np.load(RUNDIR / "pred.npy", mmap_mode="r")
print("Foehn-CAMS predictions", E_all.shape, E_all.dtype, flush=True)
aur = {}
for p in sorted(glob.glob(str(AUR / "part_*.npz"))):
    z = np.load(p)
    for d, pr in zip(z["days"], z["pred"]):
        aur[int(d)] = at_st(to_domain(pr.astype(np.float32), z["lat"], z["lon"]))           # (10, n_st)
cfc = {}
for p in sorted(CFC.glob("2022_*.npz")):
    z = np.load(p); m = int(p.name[5:7])
    f = to_domain(z["pm25"].astype(np.float32), z["lat"], z["lon"])                         # (days, 10, NY, NX)
    first = int((np.datetime64(f"2022-{m:02d}-01") - np.datetime64("2022-01-01")).astype(int))
    for k in range(f.shape[0]):
        cfc[first + k] = at_st(f[k])
    print("CAMS month", m, flush=True)
doys = [d for d in fdoys if d in aur and d in cfc]
NAMES = ["fc", "aur", "cams", "obsm", "fc0", "obs"]
A = {k: np.full((len(doys), 4, len(sid)), np.nan, np.float32) for k in NAMES}
paper = np.zeros((len(doys), 4, len(sid)), bool)
for c_, d in enumerate(doys):
    E = at_st(np.asarray(E_all[fdoys.index(d)], np.float32))                                # (M, 10, n_st)
    Au, C = aur[d], cfc[d]
    for n in range(1, 5):
        ks = (2 * n - 1, 2 * n, 2 * n + 1)                                                  # leads 24n, 24n+12, 24n+24 h
        if d + n >= 365:
            continue
        y = obs[:, d + n]
        em = sum(w * E[:, k] for w, k in zip(W3, ks))                                       # (M, n_st) daily-mean members
        fc = [em.mean(0), sum(w * Au[k] for w, k in zip(W3, ks)), sum(w * C[k] for w, k in zip(W3, ks)), obs[:, d - 1]]
        paper[c_, n - 1] = np.isfinite(y) & np.all([np.isfinite(f) for f in fc], 0)
        for k, v in zip(NAMES, fc + [em[0], y]):
            A[k][c_, n - 1] = v
    if c_ % 40 == 0:
        print(f"{c_}/{len(doys)}", flush=True)
OUT.mkdir(exist_ok=True)
np.savez_compressed(OUT / "foehn_cams_station_day.npz", doys=np.array(doys), sid=sid, src=src, lat=slat, lon=slon, paper=paper,
                    members=int(E_all.shape[1]), **A)

ref = np.load(REF)["acc"]                                                                   # system, lead, network, (sse, se, crps, n)
nets = list(NETS)
worst = 0.0
for g in range(7):
    m = paper & ((src == nets[g]) if g < 6 else True)
    for n in range(4):
        assert int(m[:, n].sum()) == int(ref[0, n, g, 3]), (g, n)
        for s, k in enumerate(["fc", "aur", "cams", "obsm"]):
            e = (A[k][:, n] - A["obs"][:, n])[m[:, n]].astype(np.float64)
            worst = max(worst, abs((e * e).sum() / ref[s, n, g, 0] - 1))
print(f"check against {REF.name}: identical station-day counts (7 groups x 4 leads); largest relative difference of a sum of "
      f"squared errors over 4 systems: {worst:.2e}")
print("saved", OUT / "foehn_cams_station_day.npz", "initialisations", len(doys), "stations", len(sid))
