#!/usr/bin/env python3
"""Data of the 'where the air comes from' figure, from FLEXGREEN's Green's functions of 2022 (HRES runs).
  1. over land (0.5 deg cells with land, area-weighted), every 4th start day of 2022, leads 1-7: the share of the
     arriving air (survival-weighted) that came from above the mixed layer of the start day, and the median distance of
     the particle origins from the receptor (great circle); per region too (boxes of tail_events.py)
  2. Beijing, target 25 January 2022: particle origins (256-particle store) of the 3 x 3 cells around the city for
     every lead, and the share from aloft
  3. Beijing series: mean of the 12 CNEMC stations (10 January - 5 February), and the forecasts of the 25th issued
     1-7 days ahead (members, station mean) by each model of case_beijing.sh
    python figdata_transport.py [A=<run> C=<run> ...]   -> data/transport_fig.npz"""
import glob
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
RUNS = HERE.parent / "global1km" / "runs"
NR, NC = 288, 720
RLAT = 84.0 - 0.125 - 0.5 * np.arange(NR)
RLON = -180.0 + 0.125 + 0.5 * np.arange(NC)
REG = {"Global": None, "East Asia": (54, 18, 97, 146), "South Asia": (37, 5, 60, 97), "Europe": (72, 35, -25, 45),
       "North America": (72, 15, -170, -50), "Africa": (37, -35, -20, 52)}
T = 24                                                                   # 25 January 2022 (0-based day of year)


def npy_read(path, index):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); sh, _, dt = np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    inner = int(np.prod(sh[len(index):])); k = int(np.ravel_multi_index(index, sh[:len(index)]))
    fd = os.open(path, os.O_RDONLY)
    try:
        parts, got, n = [], 0, inner * dt.itemsize
        while got < n:
            b = os.pread(fd, min(n - got, 1 << 30), hdr + k * n + got); parts.append(b); got += len(b)
    finally:
        os.close(fd)
    return np.frombuffer(b"".join(parts), dt).reshape(sh[len(index):])


def gc_km(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon2 - lon1)
    return 6371.0 * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1))


def main():
    land = np.load(G / "meta.npz")["land"].astype(bool).reshape(NR, 2, NC, 2).any((1, 3))
    LA2, LO2 = np.meshgrid(RLAT, RLON, indexing="ij")
    W = np.cos(np.deg2rad(LA2))
    RM = [land if b is None else land & (LA2 <= b[0]) & (LA2 > b[1]) & (LO2 >= b[2]) & (LO2 < b[3]) for b in REG.values()]
    days = list(range(0, 358, 4))
    aloft = np.full((len(days), 7, len(REG)), np.nan); dist = np.full((len(days), 7, len(REG)), np.nan)
    amap = np.zeros((7, NR, NC)); nmap = 0
    org_path = G / "green" / "hres" / "origins_2022.npy"
    for i, d in enumerate(days):
        for h in range(1, 8):
            o = npy_read(org_path, (d, h - 1)).astype(np.float32)        # (288, 720, 32, 4)
            lon, lat, zr, sv = np.moveaxis(o, -1, 0)
            sv = np.nan_to_num(sv).clip(0, 1)
            s = sv.sum(-1)
            a = np.where(s > 0, (sv * (zr > 1)).sum(-1) / np.maximum(s, 1e-6), np.nan)
            dk = np.nanmedian(gc_km(LA2[..., None], LO2[..., None], lat, lon), -1)
            ok = np.isfinite(a) & (s > 0)
            for g, m in enumerate(RM):
                mm = m & ok
                aloft[i, h - 1, g] = (W * np.nan_to_num(a) * mm).sum() / (W * mm).sum()
                dist[i, h - 1, g] = (W * np.nan_to_num(dk) * mm).sum() / (W * mm).sum()
            amap[h - 1] += np.where(ok & land, a, 0)
        nmap += 1
        if i % 10 == 0:
            print(f"day {d}: aloft by lead (land) {np.round(aloft[i, :, 0], 2)}, distance {np.round(dist[i, :, 0])} km", flush=True)
    # Beijing origins (256 particles), every lead, target 25 January
    bj = {}
    p256 = G / "green256" / "hres" / "origins_2022.npy"
    for h in range(1, 8):
        try:
            o = np.stack([npy_read(p256, (T - h, h - 1, r)) [591:594] for r in (87, 88, 89)]).astype(np.float32)   # (3, 3, P, 4)
        except Exception as e:
            print("Beijing origins", h, e); continue
        bj[h] = o.reshape(-1, 4)
    # Beijing series and forecasts of the 25th
    st = np.load(G / "stations_2022.npz")
    sel = np.where((st["source"] == 1) & (np.abs(st["lat"] - 39.9) < 0.3) & (np.abs(st["lon"] - 116.4) < 0.4))[0]
    obs = np.nanmean(st["daily"][sel].astype(np.float32), 0)                 # (365,)
    fc = {}
    for arg in sys.argv[1:]:
        lab, run = arg.split("=")
        E = np.full((7, 16), np.nan)
        for p in sorted(glob.glob(str(RUNS / run / "case_beijing" / "part_*.npz"))):
            z = np.load(p)
            for n, (d, h, t) in enumerate(z["keys"]):
                if int(t) == T:
                    m = np.nanmean(z["st_ens"][n][sel].astype(np.float32), 0)    # members: station mean
                    E[int(h) - 1, :len(m)] = m[:16]
        fc[lab] = E
        print(lab, "forecast of the 25th, mean of members by lead:", np.round(np.nanmean(E, 1)))
    out = dict(days=np.array(days), leads=np.arange(1, 8), regions=np.array(list(REG)), aloft=aloft, dist=dist,
               aloft_map=(amap / max(nmap, 1)).astype(np.float32), obs=obs, obs_first_doy=0,
               **{f"bj_org_{h}": v for h, v in bj.items()}, **{f"fc_{k}": v for k, v in fc.items()})
    np.savez(HERE / "data" / "transport_fig.npz", **out)
    m = np.nanmean(aloft, 0)
    for g, r in enumerate(REG):
        print(f"{r:14s} share from aloft by lead: " + " ".join(f"{v:.2f}" for v in m[:, g])
              + " | median origin distance (km): " + " ".join(f"{v:.0f}" for v in np.nanmean(dist, 0)[:, g]))
    print("Beijing 25 Jan, share from aloft by lead:",
          {h: round(float((o[:, 3] * (o[:, 2] > 1)).sum() / max(o[:, 3].sum(), 1e-6)), 2) for h, o in bj.items()})


if __name__ == "__main__":
    main()
