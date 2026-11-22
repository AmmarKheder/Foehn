#!/usr/bin/env python3
"""Foehn read AWAY from the station pixel (test year 2022, final model crps_e01; the stations only score).

Source: the evaluation that stored every member at the 8.1 million pixels with at least 100 residents and at the station
pixels (eval_global_green.py --pop --every 2 --leads 1 3 -> runs/<run>/eval_pop_crps_e01: pop_idx.npy, pop_<rank>.npy
(forecasts, pixels, 4 members), pop_ghap_<rank>.npy (the 1 km estimate of the target day), part_<rank>.npz (keys)).
Nothing is recomputed by the network here: the stored forecasts are read at other pixels.

Around each station, the pixels at Chebyshev distance k = 1..12 pixels (0.01 deg, about 1.1 km north-south) from the
station pixel form the square ring k (8 k pixels). A ring pixel is ELIGIBLE when it
  * is one of the stored populated pixels (>= 100 residents, GHS-POP 2020),
  * holds no station of stations_2022.npz (any of the 8,110 stations, the seven sources), so that no ring value is the
    estimate at a pixel with a monitor in our files,
  * is land (land1km_global.npy) and inside the forecast windows.
For every forecast (start day d, lead h in {1, 3}), station and k the script stores the sum of the ensemble mean over
the eligible ring pixels (f_sum; the count n_el is fixed), the ensemble mean at ONE eligible pixel drawn once per station
and ring (f_one; seed 0), and the same for the 1 km estimate of the target day (g_sum, g_n: pixels with a valid estimate;
g_one). k = 0 is the station pixel itself.

    python extract_rings.py     -> ../data/rings.npz
"""
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SFX = f"_{os.environ['FOEHN_DESIGN']}" if os.environ.get("FOEHN_DESIGN") else ""   # A16 rebuild: design label (unset = unchanged)
OUT = HERE.parent / f"data{SFX}"
EXP = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments"))
RUN = EXP / f"global1km/runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4/eval_pop_crps_e01{SFX}"
G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
GH, GW, FH, FW = 14400, 36000, 4192, 6992
KMAX, NK = 12, 13
import os  # noqa: E402
NRANK = int(os.environ.get("NRANK", 64))            # 64 ranks of the evaluation; a smaller number only for a test
TEST = NRANK < 64

st = np.load(G / "stations_2022.npz")
lat, lon, src = st["lat"], st["lon"], st["source"]
NS = len(lat)
sr = np.clip(np.floor((84.0 - lat) / 0.01).astype(np.int64), 0, GH - 1)      # as eval_global_green.py
sc = np.clip(np.floor((lon + 180.0) / 0.01).astype(np.int64), 0, GW - 1)
inside = (lat < 84.0) & (lat > -60.0)

pp = np.load(G / "pop_pixels.npz")
pop_idx, pop_val, min_pop = pp["idx"], pp["pop"], float(pp["min_pop"])
assert np.array_equal(pop_idx, np.load(RUN / "pop_idx.npy")) and np.all(np.diff(pop_idx) > 0)
win = np.load(RUN / "part_0.npz")["windows"]                                   # (21, 2): window origins in 0.25 deg units
covered = np.zeros((GH, GW), bool)
for k0, j0 in win:
    covered[25 * k0:25 * k0 + FH, 25 * j0:25 * j0 + FW] = True
land = np.load(G / "land1km_global.npy")            # bool: the 1 km estimates present on more than half of the days 2017-2020
pr, pc = pop_idx // GW, pop_idx % GW
st_flat = sr * GW + sc
is_station_px = np.isin(pop_idx, np.unique(st_flat))
land_at = land[pr, pc]
del land
elig = (pop_val >= min_pop) & ~is_station_px & land_at & covered[pr, pc]
print(f"stored pixels {len(pop_idx):,}; station pixels {int(is_station_px.sum()):,}; eligible ring pixels {int(elig.sum()):,} "
      f"(populated {int((pop_val >= min_pop).sum()):,}, of which not land {int(((pop_val >= min_pop) & ~land_at).sum()):,}, "
      f"not covered {int(((pop_val >= min_pop) & ~covered[pr, pc]).sum()):,})", flush=True)
s_ok = inside & covered[sr, sc]

# ---- ring pixel lists: (position in pop_idx, station, k)
off = np.array([(dr, dc) for dr in range(-KMAX, KMAX + 1) for dc in range(-KMAX, KMAX + 1)])
kk = np.abs(off).max(1)
P_pos, P_st, P_k = [], [], []
for s in np.flatnonzero(s_ok):
    r, c = sr[s] + off[:, 0], sc[s] + off[:, 1]
    okb = (r >= 0) & (r < GH) & (c >= 0) & (c < GW)
    flat = r[okb] * GW + c[okb]
    pos = np.searchsorted(pop_idx, flat)
    pos[pos == len(pop_idx)] = 0
    hit = pop_idx[pos] == flat
    keep = hit & ((kk[okb] == 0) | elig[pos])                                  # k = 0: the station pixel (always stored)
    P_pos.append(pos[keep]); P_st.append(np.full(int(keep.sum()), s)); P_k.append(kk[okb][keep])
P_pos, P_st, P_k = np.concatenate(P_pos), np.concatenate(P_st), np.concatenate(P_k)
cell = P_st * NK + P_k
n_el = np.bincount(cell, minlength=NS * NK).reshape(NS, NK).astype(np.int32)
# the pixel list was built with an earlier station file: a station pixel is in it when it was a station then or is populated.
# The station pixel itself is therefore taken from st_ens of the same run (stored for every station); k = 0 is a check.
print(f"station pixel present in the pixel list for {int((n_el[s_ok, 0] == 1).sum())} of {int(s_ok.sum())} stations", flush=True)
upos, inv = np.unique(P_pos, return_inverse=True)
print(f"{int(s_ok.sum())} stations inside the forecast; ring entries {len(P_pos):,} on {len(upos):,} distinct pixels", flush=True)

# one eligible pixel per station and ring, drawn once (seed 0)
rng = np.random.default_rng(0)
order = np.lexsort((rng.random(len(cell)), cell))                              # random order inside each (station, ring)
first = order[np.r_[True, np.diff(cell[order]) != 0]]
one = np.full(NS * NK, -1, np.int64); one[cell[first]] = first                 # index into the ring entries
has_one = one >= 0

keys, f_sum, f_one, g_sum, g_n, g_one, st_mean, st_ghap = [], [], [], [], [], [], [], []
for rank in range(NRANK):
    z = np.load(RUN / f"part_{rank}.npz")
    P = np.load(RUN / f"pop_{rank}.npy", mmap_mode="r")
    Gt = np.load(RUN / f"pop_ghap_{rank}.npy", mmap_mode="r")
    assert P.shape[0] == len(z["keys"]) == Gt.shape[0] and P.shape[1] == len(pop_idx)
    for n, key in enumerate(z["keys"]):
        f = np.asarray(P[n], np.float32)[upos].mean(-1)[inv]                    # ensemble mean at the ring entries
        g = np.asarray(Gt[n], np.float32)[upos][inv]
        gv = np.isfinite(g)
        f_sum.append(np.bincount(cell, weights=f, minlength=NS * NK).reshape(NS, NK).astype(np.float32))
        g_sum.append(np.bincount(cell, weights=np.where(gv, g, 0.0), minlength=NS * NK).reshape(NS, NK).astype(np.float32))
        g_n.append(np.bincount(cell, weights=gv, minlength=NS * NK).reshape(NS, NK).astype(np.int16))
        o = np.full(NS * NK, np.nan, np.float32); o[has_one] = f[one[has_one]]; f_one.append(o.reshape(NS, NK))
        o = np.full(NS * NK, np.nan, np.float32); o[has_one] = g[one[has_one]]; g_one.append(o.reshape(NS, NK))
        keys.append(key)
        st_mean.append(z["st_ens"][n].astype(np.float32).mean(-1)); st_ghap.append(z["st_ghap"][n].astype(np.float32))
    print(f"rank {rank + 1}/{NRANK}", flush=True)

keys = np.array(keys, np.int32)
f_sum, f_one, g_sum, g_n, g_one = (np.stack(v) for v in (f_sum, f_one, g_sum, g_n, g_one))
st_mean, st_ghap = np.stack(st_mean), np.stack(st_ghap)
# the station pixel read from the pixel list must be the station value stored by the same run
h0 = s_ok & (n_el[:, 0] == 1)
d0 = np.abs(f_sum[:, h0, 0] - st_mean[:, h0])
print(f"station pixel, pixel list against st_ens of the same run ({int(h0.sum())} stations): max |difference| {np.nanmax(d0):.4f} ug m-3",
      flush=True)
OUT.mkdir(exist_ok=True)
if TEST:
    print("test run: nothing saved;", keys[:3].tolist(), n_el[s_ok].sum(0)); raise SystemExit
np.savez_compressed(OUT / "rings.npz", keys=keys, n_el=n_el, f_sum=f_sum, f_one=f_one, g_sum=g_sum, g_n=g_n, g_one=g_one,
                    st_mean=st_mean, st_ghap=st_ghap, s_ok=s_ok, lat=lat, lon=lon, src=src, kmax=KMAX)
print("saved", OUT / "rings.npz", "forecasts", len(keys))
