#!/usr/bin/env python3
"""Independent check of items B and C (stations of the six official networks, 2022, final model under design ar). Read only.
Re-extracts every station-day from the inputs that compare_fair.py reads (run parts st_ens/st_ghap/map25, stations_2022,
CAMS daily zarr, Aurora out_global + out_12h_all + raw analyses, GHAP climatology) and rebuilds the paper mask
(official network, measurement finite, all ten systems of compare_fair finite). Writes one JSON."""
import glob
import json
import os
import sys

import numpy as np
import zarr

RUN = (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4")
EV = f"{RUN}/eval_final_crps_e01_ar"
G = (os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km")
AURD = (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/benchmark/aurora")
SD = (os.environ["FOEHN_WORK"] + "/eccv/cran-pm/paper/natcomms/analyses/circularity/data_ar/station_day.npz")
NETS = (0, 1, 3, 4, 5, 6)

st = np.load(f"{G}/stations_2022.npz")
lat, lon, src, obs = st["lat"].astype(np.float64), st["lon"].astype(np.float64), st["source"], st["daily"].astype(np.float32)
NS = len(lat)
official = np.isin(src, NETS)


def at_st(f, lat0, dlat, lon0, dlon):
    nlat, nlon = f.shape
    y = (lat0 - lat) / dlat; x = ((lon - lon0) % 360.0) / dlon
    a = np.clip(np.floor(y).astype(int), 0, nlat - 2); b = np.floor(x).astype(int) % nlon; b1 = (b + 1) % nlon
    u = np.clip(y - a, 0, 1); v = x - np.floor(x)
    return ((1 - u) * ((1 - v) * f[a, b] + v * f[a, b1]) + u * ((1 - v) * f[a + 1, b] + v * f[a + 1, b1])).astype(np.float32)


parts = sorted(glob.glob(f"{EV}/part_*.npz"))
loc, FO, GHS = {}, {}, {}
GH = {}
for p in parts:
    z = np.load(p)
    assert z["st_ens"].shape[1] == NS
    e = z["st_ens"].astype(np.float32).mean(-1); g = z["st_ghap"].astype(np.float32)
    for n, (d, h, t) in enumerate(z["keys"]):
        loc[(int(d), int(h))] = (p, n); FO[(int(d), int(h))] = e[n]; GHS[(int(d), int(h))] = g[n]
        GH.setdefault(int(t), g[n])
print(len(loc), "forecasts", flush=True)
CLIM = np.load(f"{G}/clim/st_clim_2022.npy").astype(np.float32)
cams = zarr.open((os.environ["FOEHN_WORK"] + "/eccv/data/world/cams_forecast_daily/2022.zarr"), mode="r")
AD, AE = {}, {}
for p in glob.glob(f"{AURD}/out_global/part_*.npz"):
    z = np.load(p); la_, lo_ = z["lat"], z["lon"]
    for d, pr in zip(z["days"], z["pred"]):
        AD[int(d)] = np.stack([at_st(pr[k].astype(np.float32), float(la_[0]), float(la_[0] - la_[1]), float(lo_[0]), float(lo_[1] - lo_[0])) for k in range(pr.shape[0])])
for p in glob.glob(f"{AURD}/out_12h_all/part_*.npz"):
    z = np.load(p); la_, lo_ = z["lat"], z["lon"]
    for d, pr in zip(z["days"], z["pred"]):
        AE[int(d)] = np.stack([at_st(pr[k].astype(np.float32), float(la_[0]), float(la_[0] - la_[1]), float(lo_[0]), float(lo_[1] - lo_[0])) for k in range(pr.shape[0])])
print("aurora loaded", flush=True)
_m25 = {}


def coarse(d, h, res):
    p, n = loc[(d, h)]
    if p not in _m25:
        _m25.clear(); _m25[p] = np.load(p)["map25"]
    m = _m25[p][n].astype(np.float32)
    if res == 0.5:
        with np.errstate(all="ignore"):
            m = np.nanmean(m.reshape(288, 2, 720, 2), (1, 3))
    lat0 = 84.0 - res / 2
    ok = np.isfinite(m).astype(np.float32)
    with np.errstate(all="ignore"):
        return at_st(np.nan_to_num(m) * ok, lat0, res, -180.0 + res / 2, res) / at_st(ok, lat0, res, -180.0 + res / 2, res)


keys = sorted(k for k in loc if k[1] <= 4 and k[0] + k[1] < 365 and k[0] + 1 <= 364 and k[0] in AD and (k[0] + 1) in AE and k[0] in GH)
days = sorted({d for d, h in keys})
ND = len(days); di = {d: i for i, d in enumerate(days)}
F = np.full((ND, 4, NS), np.nan, np.float32); C1 = F.copy(); A1 = F.copy(); Y = F.copy(); GHt = F.copy()
MASK = np.zeros((ND, 4, NS), bool); FIGM = np.zeros((ND, 4, NS), bool)
for c, (d, h) in sorted(enumerate(keys), key=lambda x: loc[x[1]][0]):
    t = d + h; y = obs[:, t]
    c0 = at_st(np.asarray(cams[d, 0, h], np.float32), 90.0, 0.4, 0.0, 0.4)
    c1 = at_st(np.asarray(cams[d + 1, 0, h - 1], np.float32), 90.0, 0.4, 0.0, 0.4)
    A = AE[d + 1]; k0 = 2 * (h - 1)
    if h == 1:
        z = np.load(f"{AURD}/raw/{d + 1:03d}.npz")
        first = at_st(z["surf"][0, 5] * 1e9, float(z["lat"][0]), float(z["lat"][0] - z["lat"][1]), float(z["lon"][0]), float(z["lon"][1] - z["lon"][0]))
    else:
        first = A[k0 - 1]
    a1 = 0.25 * first + 0.5 * A[k0] + 0.25 * A[k0 + 1]
    X = np.stack([FO[(d, h)], coarse(d, h, 0.25), coarse(d, h, 0.5), c0, c1, AD[d][h - 1], a1, obs[:, d], GH[d], CLIM[t]])
    i = di[d]
    MASK[i, h - 1] = official & np.isfinite(y) & np.all(np.isfinite(X), 0)
    FIGM[i, h - 1] = official & np.isfinite(y) & np.isfinite(FO[(d, h)]) & np.isfinite(c1) & np.isfinite(GHS[(d, h)])
    F[i, h - 1], C1[i, h - 1], A1[i, h - 1], Y[i, h - 1], GHt[i, h - 1] = FO[(d, h)], c1, a1, y, GHS[(d, h)]
    if c % 100 == 0:
        print(c, "/", len(keys), flush=True)
days = np.array(days)
R = {"n_start_days": int(ND), "keys": len(keys)}

# ---- comparison with the derived file station_day.npz
sd = np.load(SD)
same_days = np.array_equal(sd["days"], days)
R["station_day_check"] = dict(same_days=bool(same_days), mask_equal=bool(same_days and np.array_equal(sd["paper"], MASK)),
                              mask_n=[int(MASK.sum()), int(sd["paper"].sum())])
if same_days:
    for nm, arr in (("foehn", F), ("cams1", C1), ("aur1", A1), ("obs", Y)):
        m = MASK & sd["paper"]
        R["station_day_check"][nm + "_maxdiff"] = float(np.nanmax(np.abs(arr[m] - sd[nm][m])))

# ---- B: four basins (boxes of fig_overview.py), lead 1
CENTRES = {"Delhi": (28.7, 77.6, 700), "Po": (45.35, 10.0, 600), "LA": (34.05, -117.9, 300), "Seoul": (37.56, 127.0, 300)}
BOX = {}
for nm, (la, lo, wkm) in CENTRES.items():
    dla = wkm / 1.5 / 111.2 / 2; dlo = wkm / (111.2 * np.cos(np.deg2rad(la))) / 2
    BOX[nm] = (la + dla, la - dla, lo - dlo, lo + dlo)
date = lambda d: str(np.datetime64("2022-01-01") + np.timedelta64(int(d), "D"))


def basin_table(M):
    out = {}
    rm = {s: np.full((ND, 4), np.nan) for s in ("F", "C", "A")}
    nst = np.zeros((ND, 4), int)
    for nm, (n, s, w, e) in BOX.items():
        ib = (lat <= n) & (lat >= s) & (lon >= w) & (lon <= e)
        rf, rc, ra, ns = (np.full(ND, np.nan) for _ in range(4))
        for i in range(ND):
            m = M[i, 0] & ib
            if m.sum() == 0:
                continue
            ns[i] = m.sum()
            rf[i] = np.sqrt(np.mean((F[i, 0, m] - Y[i, 0, m]) ** 2)); rc[i] = np.sqrt(np.mean((C1[i, 0, m] - Y[i, 0, m]) ** 2))
            ra[i] = np.sqrt(np.mean((A1[i, 0, m] - Y[i, 0, m]) ** 2))
        out[nm] = dict(rf=rf, rc=rc, ra=ra, ns=ns)
    return out


def summarise(T):
    res = {}
    valid = np.all([np.isfinite(T[nm]["rf"]) for nm in T], 0)
    for nm, v in T.items():
        ok = np.isfinite(v["rf"])
        res[nm] = dict(n_days=int(ok.sum()), below_cams=int((v["rf"] < v["rc"])[ok].sum()), below_both=int(((v["rf"] < v["rc"]) & (v["rf"] < v["ra"]))[ok].sum()),
                       above_cams=int((v["rf"] > v["rc"])[ok].sum()))
    all4 = np.all([T[nm]["rf"] < T[nm]["rc"] for nm in T], 0) & valid
    all4b = np.all([(T[nm]["rf"] < T[nm]["rc"]) & (T[nm]["rf"] < T[nm]["ra"]) for nm in T], 0) & valid
    res["days_all_four_valid"] = int(valid.sum())
    res["below_cams_all_four"] = int(all4.sum()); res["below_both_all_four"] = int(all4b.sum())
    res["all_four_dates"] = [date(days[i] + 1) for i in np.flatnonzero(all4)]
    ratio = np.min([T[nm]["rc"] / T[nm]["rf"] for nm in T], 0); ratio[~valid] = np.nan
    marg = np.min([T[nm]["rc"] - T[nm]["rf"] for nm in T], 0); marg[~valid] = np.nan
    o = np.argsort(-np.nan_to_num(ratio, nan=-9)); om = np.argsort(-np.nan_to_num(marg, nan=-9))
    res["rank_min_ratio"] = [(date(days[i] + 1), round(float(ratio[i]), 3)) for i in o[:4]]
    res["rank_min_margin"] = [(date(days[i] + 1), round(float(marg[i]), 2)) for i in om[:4]]
    for tag in (38, 357):
        if tag in di:
            i = di[tag]
            res[f"start{tag}_{date(tag + 1)}"] = {nm: [round(float(T[nm][k][i]), 1) for k in ("rf", "rc", "ra")] + [int(T[nm]["ns"][i])] for nm in T}
    return res


R["B_paper_mask"] = summarise(basin_table(MASK))
R["B_figure_pairing"] = summarise(basin_table(FIGM))
# yearly RMSE per basin, lead 1 (figure pairing: as the overview cache) for the printed numbers
yr = {}
for nm, (n, s, w, e) in BOX.items():
    ib = (lat <= n) & (lat >= s) & (lon >= w) & (lon <= e)
    m = FIGM[:, 0] & ib[None]
    yr[nm] = [round(float(np.sqrt(np.mean((F[:, 0][m] - Y[:, 0][m]) ** 2))), 1), round(float(np.sqrt(np.mean((C1[:, 0][m] - Y[:, 0][m]) ** 2))), 1),
              int(m.any(0).sum())]
R["B_yearly_figure_pairing"] = yr

# ---- C: by lead, and Foehn 3 days vs references 1 day on identical station-target days
rng = np.random.default_rng(0)


def boot(sa, na, sb, nb, B=7, nres=10000):
    nd = len(sa); rng = np.random.default_rng(0)                                   # as compare_fair.py: one draw from seed 0
    idx = (rng.integers(0, nd - B + 1, size=(nres, -(-nd // B)))[:, :, None] + np.arange(B)).reshape(nres, -1)[:, :nd]
    r = np.sqrt(sa[idx].sum(1) / na[idx].sum(1)) / np.sqrt(sb[idx].sum(1) / nb[idx].sum(1)) - 1
    pt = np.sqrt(sa.sum() / na.sum()) / np.sqrt(sb.sum() / nb.sum()) - 1
    lo, hi = np.percentile(r, [2.5, 97.5])
    return [round(-100 * pt, 1), round(-100 * hi, 1), round(-100 * lo, 1)]          # % lower, [interval]


groups = {"all": official, "China": src == 1, "outside China": official & (src != 1)}
C = {}
for gname, gm in groups.items():
    byl = {}
    for h in range(4):
        m = MASK[:, h] & gm[None]
        byl[h + 1] = [round(float(np.sqrt(np.mean((X[:, h][m] - Y[:, h][m]) ** 2))), 2) for X in (F, C1, A1)] + [int(m.sum())]
    # Foehn lead 3 (start d) vs refs lead 1 (start d + 2): same target t = d + 3
    sF, sC, sA, nn, tt = [], [], [], [], []
    for i, d in enumerate(days):
        if (d + 2) not in di:
            continue
        j = di[d + 2]
        m = MASK[i, 2] & MASK[j, 0] & gm
        assert np.allclose(Y[i, 2][m], Y[j, 0][m], equal_nan=True)
        if m.sum() == 0:
            continue
        sF.append(((F[i, 2, m] - Y[i, 2, m]) ** 2).sum()); sC.append(((C1[j, 0, m] - Y[j, 0, m]) ** 2).sum())
        sA.append(((A1[j, 0, m] - Y[j, 0, m]) ** 2).sum()); nn.append(m.sum()); tt.append(d + 3)
    sF, sC, sA, nn = map(np.array, (sF, sC, sA, nn))
    C[gname] = dict(by_lead_F_C_A_n=byl, F3_vs_ref1=dict(n_station_days=int(nn.sum()), n_days=len(nn),
                    rmse_F3_C1_A1=[round(float(np.sqrt(s.sum() / nn.sum())), 2) for s in (sF, sC, sA)],
                    lower_than_cams=boot(sF, nn, sC, nn), lower_than_aurora=boot(sF, nn, sA, nn)))
    # same lead (1 day) for reference
    s1F, s1C, s1A, n1 = [], [], [], []
    for i in range(ND):
        m = MASK[i, 0] & gm
        if m.sum() == 0:
            continue
        s1F.append(((F[i, 0, m] - Y[i, 0, m]) ** 2).sum()); s1C.append(((C1[i, 0, m] - Y[i, 0, m]) ** 2).sum()); s1A.append(((A1[i, 0, m] - Y[i, 0, m]) ** 2).sum()); n1.append(m.sum())
    s1F, s1C, s1A, n1 = map(np.array, (s1F, s1C, s1A, n1))
    C[gname]["F1_vs_ref1"] = dict(lower_than_cams=boot(s1F, n1, s1C, n1), lower_than_aurora=boot(s1F, n1, s1A, n1))
R["C"] = C


def conv(o):
    if isinstance(o, dict):
        return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [conv(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return conv(o.tolist())
    return o


txt = json.dumps(conv(R), indent=1)
path = sys.argv[1]; tmp = path + ".tmp"
with open(tmp, "w") as fh:
    fh.write(txt); fh.flush(); os.fsync(fh.fileno())
os.replace(tmp, path)
print(txt)
