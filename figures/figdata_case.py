#!/usr/bin/env python3
"""Data of the case-study figure: the January 2022 East Asian haze (North China Plain -> Korea), from the case
evaluation (eval_global_fp.py --crop-only, 16 members, start days 0-27, leads 1-7, 1 km members over 42-33 N,
112-131 E). Stations are only used to evaluate.
  peak: target day of the highest Beijing-Tianjin-Hebei station mean (official monitors);
  maps of the peak day: GHAP 1 km; Foehn ensemble mean and P(PM2.5 > 75) issued 1, 3, 5, 7 days ahead; CAMS and Aurora
  issued 1 and 3 days ahead (0.4 deg);
  series (target days of January): station means of Beijing-Tianjin-Hebei and Seoul metro, observations, GHAP at the
  stations, Foehn members (16), CAMS, Aurora, at leads 1, 3 and 5;
  convergence: Foehn members of the peak-day Beijing mean for leads 1..7;
  warnings > 75 ug/m3 (1 km, population-weighted over the box, all target days with every system): hits, misses,
  false alarms of Foehn P >= 0.5 and of CAMS / Aurora (> 75), by lead.
    python figdata_case.py <run> [<eval dir name>]   -> data/case_eastasia_<run>.npz"""
import glob
import sys
from pathlib import Path

import numpy as np
import zarr

RUN = sys.argv[1] if len(sys.argv) > 1 else "fp_flow_shift25_global_L7_fix"
EV = sys.argv[2] if len(sys.argv) > 2 else "eval_case_eastasia_q8"
HERE = Path(__file__).resolve().parent
D = HERE.parent / "global1km" / "runs" / RUN / EV
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
AUR = HERE.parent / "benchmark/aurora/out_12h_all"               # Aurora initialised at 00 UTC of day d+1 (equal issuance)
AURD = HERE.parent / "benchmark/aurora"
TH = 75.0
I0 = 1826
st = np.load(G / "stations_2022.npz"); lat, lon, src, obs = st["lat"], st["lon"], st["source"], st["daily"]
ZONES = {"bth": ((36.0, 41.0, 114.0, 119.5), [1]), "seoul": ((37.0, 38.0, 126.5, 127.5), [4])}

# ---------------- members at the stations (16) for every (start, lead)
parts = [np.load(p) for p in sorted(glob.glob(str(D / "part_*.npz")))]
keys = np.concatenate([p["keys"] for p in parts]); ens = np.concatenate([p["st_ens"] for p in parts]).astype(np.float32)
gh = np.concatenate([p["st_ghap"] for p in parts]).astype(np.float32)
kidx = {(int(d), int(h)): n for n, (d, h, t) in enumerate(keys)}
cams = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
aur = {}
for p in glob.glob(str(AUR / "part_*.npz")):
    z = np.load(p); alat, alon = z["lat"], z["lon"]
    for d, pr in zip(z["days"], z["pred"]):
        aur[int(d)] = pr


def bil(field, la, lo, lat0, dlat, lon0, dlon):
    nlat, nlon = field.shape
    fi = (lat0 - la) / dlat; fj = ((lo - lon0) % 360.0) / dlon
    i0 = np.clip(np.floor(fi).astype(int), 0, nlat - 2); j0 = np.floor(fj).astype(int) % nlon; j1 = (j0 + 1) % nlon
    wi, wj = np.clip(fi - i0, 0, 1), fj - np.floor(fj)
    return (1 - wi) * ((1 - wj) * field[i0, j0] + wj * field[i0, j1]) + wi * ((1 - wj) * field[i0 + 1, j0] + wj * field[i0 + 1, j1])


an00 = {}


def aur_field(d, h):
    """Aurora issued with Foehn's forecast of day d (00 UTC of d+1), daily mean of day d+h: 0.25 x 00 UTC + 0.5 x 12 UTC + 0.25 x
    00 UTC of the next day, the first 00 UTC of lead 1 being the CAMS analysis (as compare_fair.py); None when unavailable"""
    if h > 4 or (d + 1) not in aur:
        return None
    A = aur[d + 1].astype(np.float32); k0 = 2 * (h - 1)
    if h == 1:
        if d + 1 not in an00:
            z = np.load(AURD / "raw" / f"{d + 1:03d}.npz"); an00[d + 1] = z["surf"][0, 5, :A.shape[1]].astype(np.float32) * 1e9
        first = an00[d + 1]
    else:
        first = A[k0 - 1]
    return 0.25 * first + 0.5 * A[k0] + 0.25 * A[k0 + 1]


def cams_field(d, h):
    """the CAMS run of 00 UTC of day d+1 (issued with Foehn's forecast of day d), its lead day h-1"""
    return np.asarray(cams[d + 1, 0, h - 1], np.float32) if (h <= 4 and d + 1 <= 364) else None


cams_at = lambda d, h, la, lo: (bil(cams_field(d, h), la, lo, 90.0, 0.4, 0.0, 0.4) if cams_field(d, h) is not None else np.full(la.shape, np.nan))
aur_at = lambda d, h, la, lo: (bil(aur_field(d, h), la, lo, float(alat[0]), float(alat[0] - alat[1]), float(alon[0]), float(alon[1] - alon[0]))
                               if aur_field(d, h) is not None else np.full(la.shape, np.nan))
out = {}
for zk, ((s, n_, w, e), nets) in ZONES.items():
    sel = (lat >= s) & (lat <= n_) & (lon >= w) & (lon <= e) & np.isin(src, nets)
    out[f"{zk}_nst"] = int(sel.sum())
    for h in (1, 3, 5):
        rows = []
        for t in range(h, 31):
            d = t - h
            if (d, h) not in kidx:
                continue
            o = obs[sel, t]; E = ens[kidx[(d, h)]][sel]; g = gh[kidx[(d, h)]][sel]
            ok = np.isfinite(o) & np.isfinite(E).all(1) & np.isfinite(g)
            if ok.sum() < 3:
                continue
            c = cams_at(d, h, lat[sel][ok], lon[sel][ok]); a_ = aur_at(d, h, lat[sel][ok], lon[sel][ok])
            rows.append(np.concatenate([[t, o[ok].mean(), g[ok].mean(), np.nanmean(c), np.nanmean(a_)], E[ok].mean(0)]))
        out[f"{zk}_series_{h}"] = np.array(rows, np.float32)     # t, obs, GHAP, CAMS, Aurora, 16 members
bth = out["bth_series_1"]
full = (bth[:, 0] >= 7) & (bth[:, 0] <= 28)                          # target days with every lead 1..7
peak = int(bth[full][np.nanargmax(bth[full, 1]), 0])
print("peak target day (doy, 0 = 1 January):", peak, "| BTH obs", float(np.nanmax(bth[:, 1])))
out["peak"] = peak
# convergence of the peak-day Beijing mean
sel = (lat >= 36.0) & (lat <= 41.0) & (lon >= 114.0) & (lon <= 119.5) & (src == 1)
conv = []
for h in range(1, 8):
    d = peak - h
    if (d, h) not in kidx:
        conv.append(np.full(16 + 3, np.nan)); continue
    o = obs[sel, peak]; E = ens[kidx[(d, h)]][sel]; ok = np.isfinite(o) & np.isfinite(E).all(1)
    conv.append(np.concatenate([[o[ok].mean(), np.nanmean(cams_at(d, h, lat[sel][ok], lon[sel][ok])),
                                 np.nanmean(aur_at(d, h, lat[sel][ok], lon[sel][ok]))], E[ok].mean(0)]))
out["convergence"] = np.array(conv, np.float32)                     # lead 1..7: obs, CAMS, Aurora, 16 members

# ---------------- 1 km maps of the peak day
crop = lambda d, h: np.load(D / f"crop0_d{d}_h{h}.npz")
c0 = crop(peak - 1, 1)
box, (r0, r1), (cc0, cc1) = c0["box"], c0["rows"], c0["cols"]
out["box"] = box
TILE, NTR, NTC = 512, 29, 71


def tiles_box(path, k, r0, r1, c0, c1):
    with open(path, "rb") as fh:
        np.lib.format.read_magic(fh); sh, _, dt = np.lib.format.read_array_header_1_0(fh); hdr = fh.tell()
    tr0, tr1, tc0, tc1 = r0 // TILE, (r1 - 1) // TILE, c0 // TILE, (c1 - 1) // TILE
    nb, B = tc1 - tc0 + 1, TILE * TILE
    blk = np.stack([np.fromfile(path, dtype=dt, count=nb * B, offset=hdr + (((k * NTR + tr) * NTC + tc0) * B) * 2).reshape(nb, TILE, TILE)
                    for tr in range(tr0, tr1 + 1)])
    blk = blk.transpose(0, 2, 1, 3).reshape((tr1 - tr0 + 1) * TILE, nb * TILE)
    return blk[r0 - tr0 * TILE:r1 - tr0 * TILE, c0 - tc0 * TILE:c1 - tc0 * TILE].astype(np.float32)


ghap = tiles_box(G / "ghap1km_tiles.npy", I0 + peak, r0, r1, cc0, cc1)
land = tiles_box(G / "static1km_tiles.npy", 1, r0, r1, cc0, cc1) > 0
out["ghap_peak"] = np.where(ghap > 0, ghap, np.nan).astype(np.float16)
for h in (1, 3, 5, 7):
    m = crop(peak - h, h)["members"].astype(np.float32)
    m[:, ~land] = np.nan
    out[f"foehn_mean_{h}"] = m.mean(0).astype(np.float16)
    out[f"foehn_p75_{h}"] = (m > TH).mean(0).astype(np.float16)
    out[f"foehn_m0_{h}"] = m[0].astype(np.float16)
for h in (1, 3):
    out[f"cams_{h}"] = cams_field(peak - h, h)
    af = aur_field(peak - h, h)
    out[f"aurora_{h}"] = af if af is not None else np.full((450, 900), np.nan, np.float32)
out["aurora_lat"], out["aurora_lon"] = alat, alon
out["st_lat"], out["st_lon"], out["st_obs_peak"] = lat, lon, obs[:, peak]

# ---------------- warnings > 75 ug/m3, population weighted (1 km), by lead
with open(G / "pop1km_global.npy", "rb") as fh:
    np.lib.format.read_magic(fh); sh, _, dt = np.lib.format.read_array_header_1_0(fh); hdr = fh.tell()
W = sh[1]
pop = np.stack([np.fromfile(G / "pop1km_global.npy", dtype=dt, count=cc1 - cc0, offset=hdr + ((r * W) + cc0) * np.dtype(dt).itemsize)
                for r in range(r0, r1)]).astype(np.float32)
LA = 84.0 - 0.01 * (np.arange(r0, r1) + 0.5); LO = -180.0 + 0.01 * (np.arange(cc0, cc1) + 0.5)
LAg, LOg = np.meshgrid(LA, LO, indexing="ij")
warn = np.zeros((7, 3, 3))                                           # lead, system (Foehn, CAMS, Aurora), (hit, miss, false alarm)
for h in range(1, 8):
    for t in range(max(h, 3), 31):
        d = t - h
        f = D / f"crop0_d{d}_h{h}.npz"
        if not f.exists() or (h <= 4 and aur_field(d, h) is None):       # leads 1-4: the three systems on the same days
            continue
        g = tiles_box(G / "ghap1km_tiles.npy", I0 + t, r0, r1, cc0, cc1)
        ok = (g > 0) & land
        y = g > TH
        fm = np.load(f)["members"].astype(np.float32)
        sysw = [(fm > TH).mean(0) >= 0.5]
        if h <= 4 and aur_field(d, h) is not None:
            sysw.append(bil(cams_field(d, h), LAg.ravel(), LOg.ravel(), 90.0, 0.4, 0.0, 0.4).reshape(LAg.shape) > TH)
            sysw.append(bil(aur_field(d, h), LAg.ravel(), LOg.ravel(), float(alat[0]), float(alat[0] - alat[1]),
                            float(alon[0]), float(alon[1] - alon[0])).reshape(LAg.shape) > TH)
        for k, wv in enumerate(sysw):
            warn[h - 1, k] += [(pop * (wv & y & ok)).sum(), (pop * (~wv & y & ok)).sum(), (pop * (wv & ~y & ok)).sum()]
    print("lead", h, "Foehn hits/misses/false (M people-days):", (warn[h - 1, 0] / 1e6).round(1), "| CAMS", (warn[h - 1, 1] / 1e6).round(1))
out["warn"] = warn
np.savez_compressed(HERE / "data" / f"case_eastasia_{RUN}.npz", **out)
print("saved")
