#!/usr/bin/env python3
"""Data of the global teaser (Fig. 1): 'Following the air'. For one forecast (start day D0, lead H):
(1) back-trajectories of the air reaching cities at 12 UTC of the target day, integrated through the HRES 3-hourly wind of
    the run of day D0 exactly as in the footprint layer (learned 10 m / 925 hPa mix of the checkpoint, 3 h steps, the
    receptor's latitude in the metric term);
(2) Foehn ensemble mean at 1 km (map1km_d{D0}_h{H}.npy of the global evaluation): 4 km block means for the globe, 1 km
    crops for the zones; GHAP of the target day for the zones; station observations of the target day;
(3) CAMS (00 UTC run of D0) and Aurora (initialised on D0) at the same lead over the hero zone;
(4) daily series at lead H over 2022 in every zone (mean over the zone's official monitors, same station-days for all):
    observations, GHAP at the station pixels, Foehn (ensemble mean, range of the members), CAMS, Aurora.
    python figdata_teaser.py <run> [D0] [H]          (needs torch: run in the container)  -> data/teaser_<run>_d<D0>_h<H>.npz"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

RUN = sys.argv[1] if len(sys.argv) > 1 else "fp_flow_shift25_global_L7_fix"
D0 = int(sys.argv[2]) if len(sys.argv) > 2 else 19
H = int(sys.argv[3]) if len(sys.argv) > 3 else 2
HERE = Path(__file__).resolve().parent
GK = HERE.parent / "global1km"
E = GK / "runs" / RUN / "eval_global"
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
AUR = HERE.parent / "benchmark/aurora/out_global"
GH, GW, TILE, NTR, NTC = 14400, 36000, 512, 29, 71
I0 = 1826                                                  # index of 1 January 2022 in the daily stores

ZONES = {   # key: (title, (N, S, W, E), networks of the series: 0 EEA, 1 CNEMC, 2 OpenAQ, 3 EPA, 4 AirKorea, 5 NIES, 6 CPCB)
    "po": ("Po Valley and Alps", (47.0, 43.8, 6.5, 13.5), [0]),
    "ncp": ("North China Plain", (41.5, 34.5, 112.0, 121.0), [1]),
    "igp": ("Indo-Gangetic Plain", (31.5, 24.5, 73.0, 84.0), [6]),
    "la": ("Los Angeles basin", (35.2, 33.0, -119.8, -116.2), [3]),
    "scl": ("Santiago de Chile", (-32.7, -34.3, -71.7, -69.9), [2]),
    "kor": ("Korean Peninsula", (38.6, 34.5, 125.8, 129.6), [4]),
}
HERO = "ncp"
CITIES = {"Los Angeles": (34.05, -118.25), "Mexico City": (19.43, -99.13), "Santiago": (-33.45, -70.67),
          "São Paulo": (-23.55, -46.63), "Lagos": (6.52, 3.38), "Cairo": (30.04, 31.24), "Paris": (48.86, 2.35),
          "Milan": (45.46, 9.19), "Tehran": (35.69, 51.39), "Lahore": (31.55, 74.34), "Delhi": (28.61, 77.21),
          "Dhaka": (23.81, 90.41), "Bangkok": (13.76, 100.50), "Beijing": (39.90, 116.40), "Shanghai": (31.23, 121.47),
          "Seoul": (37.57, 126.98), "Jakarta": (-6.21, 106.85), "Johannesburg": (-26.20, 28.05),
          "Turin": (45.07, 7.69), "Venice": (45.44, 12.32), "Bologna": (44.49, 11.34), "Tianjin": (39.13, 117.20),
          "Shijiazhuang": (38.04, 114.51), "Jinan": (36.65, 117.12), "Kanpur": (26.45, 80.33),
          "Patna": (25.59, 85.14), "Amritsar": (31.63, 74.87), "Riverside": (33.95, -117.40), "Long Beach": (33.77, -118.19),
          "Rancagua": (-34.17, -70.74), "Busan": (35.18, 129.08), "Daegu": (35.87, 128.60)}


class Npy:
    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            np.lib.format.read_magic(f); self.shape, _, self.dtype = np.lib.format.read_array_header_1_0(f); self.hdr = f.tell()
        self.fd = os.open(self.path, os.O_RDONLY)

    def read(self, offset, count):
        b = os.pread(self.fd, count * self.dtype.itemsize, self.hdr + offset * self.dtype.itemsize)
        return np.frombuffer(b, self.dtype, count)


def rows2d(npy, r0, r1, c0, c1):
    """rows r0:r1, columns c0:c1 of a 2-D (GH, GW) .npy"""
    W = npy.shape[-1]
    return npy.read(r0 * W, (r1 - r0) * W).reshape(r1 - r0, W)[:, c0:c1].astype(np.float32)


def tiles_box(npy, k, r0, r1, c0, c1):
    """window of map k of a (K, 29, 71, 512, 512) block store"""
    tr0, tr1, tc0, tc1 = r0 // TILE, (r1 - 1) // TILE, c0 // TILE, (c1 - 1) // TILE
    nb, B = tc1 - tc0 + 1, TILE * TILE
    blk = np.stack([npy.read(((k * NTR + tr) * NTC + tc0) * B, nb * B).reshape(nb, TILE, TILE) for tr in range(tr0, tr1 + 1)])
    blk = blk.transpose(0, 2, 1, 3).reshape((tr1 - tr0 + 1) * TILE, nb * TILE)
    return blk[r0 - tr0 * TILE:r1 - tr0 * TILE, c0 - tc0 * TILE:c1 - tc0 * TILE].astype(np.float32)


def box_px(n, s, w, e):
    return int(round((84 - n) * 100)), int(round((84 - s) * 100)), int(round((w + 180) * 100)), int(round((e + 180) * 100))


# ------------------------------------------------------------------ (1) trajectories, exactly as in FootprintG
sd = torch.load(GK / "runs" / RUN / "final.ckpt", map_location="cpu", weights_only=False)["state_dict"]
mw = torch.softmax(sd["net.fp.mix"].float(), 0).numpy() if "net.fp.mix" in sd else np.array([0.5, 0.5])
hw = Npy(G / "hres_wind.npy")                                    # (365, 64, 4, 288, 720): run of day d, 3-hourly
n1 = int(np.prod(hw.shape[1:]))
wind = hw.read(D0 * n1, n1).reshape(hw.shape[1:]).astype(np.float32)
DT, CELL = 3 * 3600.0, 55.6e3


def wind_at(st, la, lo):
    r = np.clip((84.0 - 0.125 - la) / 0.5, 0, 287 - 1e-3); c = ((lo + 180.0 - 0.125) / 0.5) % 720
    r0, c0 = int(r), int(c); fr, fc = r - r0, c - c0; c1 = (c0 + 1) % 720
    w = wind[st][:, [r0, r0, r0 + 1, r0 + 1], [c0, c1, c0, c1]] @ np.array([(1 - fr) * (1 - fc), (1 - fr) * fc, fr * (1 - fc), fr * fc])
    return mw[0] * w[0] + mw[1] * w[2], mw[0] * w[1] + mw[1] * w[3]


names = list(CITIES)
paths = np.zeros((len(names), 8 * H + 1, 2), np.float32)
for n, name in enumerate(names):
    la, lo = CITIES[name]
    cl = max(np.cos(np.deg2rad(la)), 0.05)
    for k in range(8 * H + 1):
        paths[n, k] = (lo, la)
        u, v = wind_at(8 * H + 4 - k, la, lo)
        la, lo = la - v * DT / CELL * 0.5, lo - u * DT / (CELL * cl) * 0.5
        lo = (lo + 180.0) % 360.0 - 180.0
print("wind mix (10 m, 925 hPa):", mw, "| path length (km) of Beijing:",
      float(np.sum(np.hypot(np.diff(paths[names.index("Beijing"), :, 0]) * 111 * np.cos(np.deg2rad(39.9)), np.diff(paths[names.index("Beijing"), :, 1]) * 111))))

# ------------------------------------------------------------------ (2) maps
fm = Npy(E / f"map1km_d{D0}_h{H}.npy")
stat = Npy(G / "static1km_tiles.npy")                             # map 1: land mask (as in the evaluation)
F4 = np.zeros((GH // 4, GW // 4), np.float32)
for r in range(0, GH, 400):                                       # 4 km block means over land (NaN = sea / not covered)
    a = rows2d(fm, r, r + 400, 0, GW)
    ok = np.isfinite(a) & (a > 0) & (tiles_box(stat, 1, r, r + 400, 0, GW) > 0)
    s = np.where(ok, a, 0).reshape(100, 4, GW // 4, 4).sum((1, 3)); c = ok.reshape(100, 4, GW // 4, 4).sum((1, 3))
    F4[r // 4:r // 4 + 100] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
fine = Npy(G / "ghap1km_tiles.npy")
st = np.load(G / "stations_2022.npz")
lat, lon, src, obs = st["lat"], st["lon"], st["source"], st["daily"]
t_doy = D0 + H
out = dict(names=np.array(names), paths=paths, mix=mw, F4=F4.astype(np.float16), d0=D0, h=H, hero=HERO,
           st_lat=lat, st_lon=lon, st_src=src, st_obs_day=obs[:, t_doy])
for key, (title, (n, s, w, e), nets) in ZONES.items():
    r0, r1, c0, c1 = box_px(n, s, w, e)
    out[f"z_{key}_box"] = np.array([n, s, w, e]); out[f"z_{key}_title"] = title; out[f"z_{key}_nets"] = np.array(nets)
    land = tiles_box(stat, 1, r0, r1, c0, c1) > 0
    out[f"z_{key}_foehn"] = np.where(land, rows2d(fm, r0, r1, c0, c1), np.nan).astype(np.float16)
    g = tiles_box(fine, I0 + t_doy, r0, r1, c0, c1)
    out[f"z_{key}_ghap"] = np.where(g > 0, g, np.nan).astype(np.float16)

# ------------------------------------------------------------------ (3) CAMS and Aurora over the hero zone
cams = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
aur = {}
for p in glob.glob(str(AUR / "part_*.npz")):
    z = np.load(p); alat, alon = z["lat"], z["lon"]
    for d, pr in zip(z["days"], z["pred"]):
        aur[int(d)] = pr
out["cams_day"] = np.asarray(cams[D0, 0, H], np.float32)           # (451, 900): 90 .. -90, 0 .. 359.6 (0.4 deg)
out["aurora_day"] = aur[D0][H - 1].astype(np.float32) if D0 in aur else np.full((450, 900), np.nan, np.float32)
out["aurora_lat"], out["aurora_lon"] = alat, alon


# ------------------------------------------------------------------ (4) daily series at lead H
def bil(field, la, lo, lat0, dlat, lon0, dlon):
    nlat, nlon = field.shape
    fi = (lat0 - la) / dlat; fj = ((lo - lon0) % 360.0) / dlon
    i0 = np.clip(np.floor(fi).astype(int), 0, nlat - 2); j0 = np.floor(fj).astype(int) % nlon; j1 = (j0 + 1) % nlon
    wi, wj = np.clip(fi - i0, 0, 1), fj - np.floor(fj)
    return (1 - wi) * ((1 - wj) * field[i0, j0] + wj * field[i0, j1]) + wi * ((1 - wj) * field[i0 + 1, j0] + wj * field[i0 + 1, j1])


P = [np.load(p) for p in sorted(glob.glob(str(E / "part_*.npz")))]
keys = np.concatenate([p["keys"] for p in P]); ens = np.concatenate([p["st_ens"] for p in P]); gh = np.concatenate([p["st_ghap"] for p in P])
kidx = {(int(d), int(h)): k for k, (d, h, t) in enumerate(keys)}
for key, (title, (n, s, w, e), nets) in ZONES.items():
    sel = (lat <= n) & (lat >= s) & (lon >= w) & (lon <= e) & np.isin(src, nets)
    rows = []
    for t in range(H, 365):
        d = t - H
        if (d, H) not in kidx:
            continue
        o = obs[sel, t]; g = gh[kidx[(d, H)]][sel].astype(np.float32); En = ens[kidx[(d, H)]][sel].astype(np.float32)
        ok = np.isfinite(o) & np.isfinite(g) & np.isfinite(En).all(1)
        if ok.sum() < max(1, 0.3 * sel.sum()):
            continue
        c = bil(np.asarray(cams[d, 0, H], np.float32), lat[sel][ok], lon[sel][ok], 90.0, 0.4, 0.0, 0.4).mean()
        a_ = (bil(aur[d][H - 1].astype(np.float32), lat[sel][ok], lon[sel][ok], float(alat[0]), float(alat[0] - alat[1]),
                  float(alon[0]), float(alon[1] - alon[0])).mean() if d in aur else np.nan)
        mem = En[ok].mean(0)
        rows.append((t, o[ok].mean(), g[ok].mean(), mem.mean(), mem.min(), mem.max(), c, a_, ok.sum()))
    out[f"s_{key}"] = np.array(rows, np.float32)          # target doy, obs, GHAP, Foehn mean, min, max member, CAMS, Aurora, n stations
    R = out[f"s_{key}"]
    r = lambda k: np.corrcoef(R[np.isfinite(R[:, k]), k], R[np.isfinite(R[:, k]), 1])[0, 1]
    print(f"{title:22s} {int(sel.sum()):4d} stations {len(R)} days | r with obs: Foehn {r(3):.2f} GHAP {r(2):.2f} CAMS {r(6):.2f} Aurora {r(7):.2f}")
np.savez_compressed(HERE / "data" / f"teaser_{RUN}_d{D0}_h{H}.npz", **out)
print("saved", HERE / "data" / f"teaser_{RUN}_d{D0}_h{H}.npz")
