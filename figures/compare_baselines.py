#!/usr/bin/env python3
"""Station scores of Foehn runs against the baselines, on identical station-days (official networks: EEA, CNEMC, EPA,
AirKorea, NIES, CPCB; stations only evaluate). Systems: the Foehn runs given (ensemble mean; fair CRPS of the ensemble),
CAMS (00 UTC run of day d, 0.4 deg, leads 1-4), Aurora (0.4 deg, from the CAMS analysis of day d, leads 1-4),
persistence (the station's observation of day d), climatology (2017-2020 GHAP at the station, same calendar days +-7:
mean for the deterministic scores, the ~60 values as an ensemble for the CRPS), GHAP of the target day (a product, not a
forecast). Leads 1-4: every system on the same station-days; leads 5-7: all but CAMS and Aurora. Europe (EEA stations,
start days of the Europe AI baselines, leads 1 and 3): also ConvLSTM, Earthformer, SimVP, TopoFlow and the regression
network (0.25 deg maps from figdata.py). CRPS of a deterministic forecast = its absolute error.
    python compare_baselines.py <label>=<run>[:<eval dir>] ...   -> data/compare_baselines_<first label>.npz"""
import glob
import sys
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
RUNS = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/runs")
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
AUR = HERE.parent / "benchmark" / "aurora" / "out_global"
NETS = {0: "Europe (EEA)", 1: "China (CNEMC)", 3: "USA (EPA)", 4: "Korea (AirKorea)", 5: "Japan (NIES)", 6: "India (CPCB)"}
AI = {"convlstm": "ConvLSTM", "earthformer": "Earthformer", "simvp": "SimVP", "topoflow": "TopoFlow", "reg": "Regression"}

st = np.load(G / "stations_2022.npz")
lat, lon, src, obs = st["lat"], st["lon"], st["source"], st["daily"].astype(np.float32)
NS = len(lat)
official = src != 2
key = lambda la, lo: np.round(la, 4) * 1e6 + np.round(lo, 4)
ref = {v: i for i, v in enumerate(key(lat, lon))}
sets = {}
for f in sorted(G.glob("stations_2022*.npz")):
    z = np.load(f); sets[len(z["lat"])] = np.array([ref.get(v, -1) for v in key(z["lat"], z["lon"])])

# ------------------------------------------------------------------ Foehn runs
runs = []
for arg in sys.argv[1:]:
    label, spec = arg.split("=")
    run, ev = (spec.split(":") + ["eval_global"])[:2]
    P = [np.load(p) for p in sorted(glob.glob(str(RUNS / run / ev / "part_*.npz")))]
    k = np.concatenate([p["keys"] for p in P]); e = np.concatenate([p["st_ens"] for p in P])
    g = np.concatenate([p["st_ghap"] for p in P])
    idx = sets[e.shape[1]]
    full = np.full((e.shape[0], NS, e.shape[2]), np.nan, np.float16); full[:, idx[idx >= 0]] = e[:, idx >= 0]
    gfull = np.full((e.shape[0], NS), np.nan, np.float32); gfull[:, idx[idx >= 0]] = g[:, idx >= 0]
    runs.append((label, {(int(d), int(h)): n for n, (d, h, t) in enumerate(k)}, full, gfull))
    print(f"{label}: {run}/{ev}, {len(k)} forecasts, {e.shape[2]} members", flush=True)
common = sorted(set.intersection(*[set(r[1]) for r in runs]))
print("common forecasts:", len(common))

# ------------------------------------------------------------------ baselines
cams = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
aur = {}
for p in glob.glob(str(AUR / "part_*.npz")):
    z = np.load(p); alat, alon = z["lat"], z["lon"]
    for d, pr in zip(z["days"], z["pred"]):
        aur[int(d)] = pr
CL = np.load(G / "clim" / "st_clim_2022.npy").astype(np.float32)
HIST = np.load(G / "clim" / "st_ghap_2017_2020.npy").astype(np.float32)
assert CL.shape[1] == NS and HIST.shape[1] == NS
hdoy = np.load(G / "meta.npz")["doy"][:1461]
CRC_T = np.full((obs.shape[1], NS), np.nan, np.float32)            # fair CRPS of the climatological ensemble
for t in range(obs.shape[1]):
    dd = np.abs(hdoy - t); dd = np.minimum(dd, 365 - dd)
    Xs = np.sort(np.where(HIST[dd <= 7] > 0, HIST[dd <= 7], np.nan).T, 1)
    Mc = np.isfinite(Xs).sum(1); kk = np.arange(1, Xs.shape[1] + 1)[None, :]
    wgt = np.where(kk <= Mc[:, None], 2 * kk - Mc[:, None] - 1, 0)
    pair = np.sum(np.nan_to_num(Xs) * wgt, 1) / np.maximum(Mc * (Mc - 1), 1)
    with np.errstate(all="ignore"):
        CRC_T[t] = np.where(Mc >= 2, np.nanmean(np.abs(Xs - obs[:, t][:, None]), 1) - pair, np.nan)
ai = {k: np.load(HERE / "data" / f"{k}.npz") for k in AI}
ai_days = {int(d): n for n, d in enumerate(ai["earthformer"]["days"])}
ai_leads = [int(h) for h in ai["earthformer"]["leads"]]
for k in AI:
    assert list(ai[k]["days"]) == list(ai["earthformer"]["days"]) and list(ai[k]["leads"]) == ai_leads


def bil(field, la, lo, lat0, dlat, lon0, dlon, wrap=True):
    nlat, nlon = field.shape
    fi = (lat0 - la) / dlat; fj = ((lo - lon0) % 360.0 if wrap else (lo - lon0)) / dlon
    i0 = np.clip(np.floor(fi).astype(int), 0, nlat - 2); j0 = np.clip(np.floor(fj).astype(int), 0, nlon - 1 if wrap else nlon - 2)
    j1 = (j0 + 1) % nlon
    wi, wj = np.clip(fi - i0, 0, 1), np.clip(fj - j0, 0, 1)
    v = (1 - wi) * ((1 - wj) * field[i0, j0] + wj * field[i0, j1]) + wi * ((1 - wj) * field[i0 + 1, j0] + wj * field[i0 + 1, j1])
    if not wrap:                                                    # outside a regional window: NaN
        v = np.where((fi < 0) | (fi > nlat - 1) | (fj < 0) | (fj > nlon - 1), np.nan, v)
    return v.astype(np.float32)


R0, C0 = 48, 620                                                    # Europe window of the AI baselines (72 N, 25 W)
EU_LAT0, EU_LON0 = 84.0 - 0.25 * R0, -180.0 + 0.25 * C0
SYS = [r[0] for r in runs] + ["CAMS", "Aurora", "persistence", "climatology", "GHAP (target day)"]
EU_SYS = SYS + list(AI.values())
NG = len(NETS) + 1                                                  # networks + all official
# per (group, lead, system): n, sum e, sum e2, sum x, sum y, sum x2, sum y2, sum xy, sum crps, csi35 hits, union, csi75 hits, union
S = np.zeros((NG, 7, len(SYS), 13))
EU = np.zeros((2, len(EU_SYS), 13))
gid = np.array([list(NETS).index(s) if s in NETS else -1 for s in src])


def add(acc, x, y, crps):
    e = x - y
    acc += np.array([len(e), e.sum(), (e * e).sum(), x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum(), crps.sum(),
                     ((x > 35) & (y > 35)).sum(), ((x > 35) | (y > 35)).sum(), ((x > 75) & (y > 75)).sum(), ((x > 75) | (y > 75)).sum()])


def fair_crps(E, y):
    M = E.shape[1]
    Es = np.sort(E, 1); pair = (Es * (2 * np.arange(1, M + 1) - M - 1)).sum(1) / (M * (M - 1))
    return np.abs(E - y[:, None]).mean(1) - pair


for c, (d, h) in enumerate(common):
    t = d + h
    if t >= obs.shape[1]:
        continue
    y = obs[:, t]
    X = np.full((len(SYS), NS), np.nan, np.float32)
    Ens = []
    for k, (label, kidx, full, gfull) in enumerate(runs):
        E = full[kidx[(d, h)]].astype(np.float32); Ens.append(E); X[k] = E.mean(1)
    nr = len(runs)
    use = list(range(nr)) + [nr + 2, nr + 3, nr + 4]
    if h <= 4:
        if d not in aur:
            continue                                                 # leads 1-4: every system on the same days
        X[nr] = bil(np.asarray(cams[d, 0, h], np.float32), lat, lon, 90.0, 0.4, 0.0, 0.4)
        X[nr + 1] = bil(aur[d][h - 1].astype(np.float32), lat, lon, float(alat[0]), float(alat[0] - alat[1]), float(alon[0]),
                        float(alon[1] - alon[0]))
        use += [nr, nr + 1]
    X[nr + 2] = obs[:, d]
    X[nr + 3] = CL[t]
    X[nr + 4] = runs[0][3][runs[0][1][(d, h)]]
    ok = official & np.isfinite(y) & np.all(np.isfinite(X[use]), 0) & np.all([np.isfinite(E).all(1) for E in Ens], 0) \
        & np.isfinite(CRC_T[t])
    for g in range(NG):
        m = ok & ((gid == g) if g < len(NETS) else True)
        if not m.any():
            continue
        for k in use:
            crps = fair_crps(Ens[k][m], y[m]) if k < nr else (CRC_T[t][m] if k == nr + 3 else np.abs(X[k, m] - y[m]))
            add(S[g, h - 1, k], X[k, m], y[m], crps)
    # Europe with the AI baselines (EEA stations, their start days and leads)
    if h in ai_leads and d in ai_days and h <= 4:
        XA = np.stack([bil(ai[k]["pred"][ai_days[d], ai_leads.index(h), 0].astype(np.float32), lat, lon, EU_LAT0, 0.25, EU_LON0,
                           0.25, wrap=False) for k in AI])
        m = ok & (src == 0) & np.all(np.isfinite(XA), 0)
        if m.any():
            li = ai_leads.index(h)
            for k in use:
                crps = fair_crps(Ens[k][m], y[m]) if k < nr else (CRC_T[t][m] if k == nr + 3 else np.abs(X[k, m] - y[m]))
                add(EU[li, k], X[k, m], y[m], crps)
            for j in range(len(AI)):
                add(EU[li, len(SYS) + j], XA[j, m], y[m], np.abs(XA[j, m] - y[m]))
    if c % 200 == 0:
        print(c, "/", len(common), flush=True)


def scores(a):
    n = a[0]
    if n == 0:
        return None
    mx, my = a[3] / n, a[4] / n
    r = (a[7] / n - mx * my) / np.sqrt(max(a[5] / n - mx * mx, 1e-9) * max(a[6] / n - my * my, 1e-9))
    return dict(n=int(n), rmse=np.sqrt(a[2] / n), bias=a[1] / n, r=r, crps=a[8] / n, csi35=a[9] / max(a[10], 1), csi75=a[11] / max(a[12], 1))


def table(title, A, names, leads):
    print(f"\n=== {title}")
    print(f"{'system':22s} " + " ".join(f"{'J+%d' % h:>38s}" for h in leads))
    print(f"{'':22s} " + " ".join(f"{'CRPS  RMSE  bias    r  CSI35 CSI75':>38s}" for h in leads))
    for k, nm in enumerate(names):
        row = []
        for li, h in enumerate(leads):
            s = scores(A[li][k])
            row.append(f"{s['crps']:5.2f} {s['rmse']:5.1f} {s['bias']:5.1f} {s['r']:5.2f} {s['csi35']:5.2f} {s['csi75']:5.2f}" if s else f"{'-':>37s}")
        print(f"{nm:22s} " + " ".join(f"{v:>38s}" for v in row))
    n = [scores(A[li][0])["n"] if scores(A[li][0]) else 0 for li in range(len(leads))]
    print(f"{'station-days':22s} " + " ".join(f"{v:>38d}" for v in n))


table("All official networks, leads 1-4 (every system on the same station-days)", [S[NG - 1, h - 1] for h in (1, 2, 3, 4)], SYS, [1, 2, 3, 4])
table("All official networks, leads 5-7 (without CAMS and Aurora)", [S[NG - 1, h - 1] for h in (5, 6, 7)], SYS, [5, 6, 7])
for g, nm in enumerate(NETS.values()):
    table(f"{nm}, leads 1 and 3", [S[g, 0], S[g, 2]], SYS, [1, 3])
table("Europe (EEA) with the Europe AI baselines (their start days), leads 1 and 3", [EU[0], EU[1]], EU_SYS, ai_leads)
out = HERE / "data" / f"compare_baselines_{runs[0][0]}.npz"
np.savez(out, S=S, EU=EU, systems=np.array(SYS), eu_systems=np.array(EU_SYS), nets=np.array(list(NETS.values()) + ["all official"]),
         n_common=len(common))
print("saved", out)
