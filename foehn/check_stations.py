#!/usr/bin/env python3
"""Checks of every network of stations_2022.npz against GHAP 1 km: time alignment (median over stations of the TEMPORAL
correlation station(t) vs GHAP(t + L), highest at L = 0; a spatial test across stations is misleading in small or
homogeneous countries, e.g. Korea), location (the station pixel correlates better than pixels 30 km away), bias.
    source world/env.sh; python check_stations.py"""
import os
from pathlib import Path

import numpy as np

G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
NETS = {0: "EEA", 1: "CNEMC", 2: "OpenAQ", 3: "EPA", 4: "AirKorea", 5: "Japan", 6: "India"}
with open(G / "ghap1km_tiles.npy", "rb") as f:
    np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); HDR = f.tell()
FD = os.open(G / "ghap1km_tiles.npy", os.O_RDONLY)


def pix(k, rows, cols):
    off = (((k * 29 + rows // 512) * 71 + cols // 512) * 512 + rows % 512) * 512 + cols % 512
    v = np.array([np.frombuffer(os.pread(FD, 2, HDR + 2 * int(o)), np.float16)[0] for o in off], np.float32)
    v[v <= 0] = np.nan
    return v


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 20 else np.nan


st = np.load(G / "stations_2022.npz")
lat, lon, src, daily = st["lat"], st["lon"], st["source"], st["daily"]
r = np.floor((84 - lat) / 0.01).astype(int); c = np.floor((lon + 180) / 0.01).astype(int)
days = list(range(12, 360, 29))
rng = np.random.default_rng(0)
for s, name in NETS.items():
    idx = np.where(src == s)[0]
    if len(idx) == 0:
        continue
    sel = rng.choice(idx, min(80, len(idx)), replace=False)
    g = np.stack([pix(1826 + d, r[sel], c[sel]) for d in range(-1, 365)], 1)             # (stations, days -1 .. 364)
    lagc = {}
    for L in (-1, 0, 1):
        cs = [corr(daily[i, 1:364], g[n, 2 + L:365 + L]) for n, i in enumerate(sel)]
        lagc[L] = np.nanmedian(cs)
    sp = {0: [], 30: []}; bias = []
    for t in days:
        o = daily[sel, t]; g0 = g[:, t + 1]
        sp[0].append(corr(g0, o))
        sp[30].append(np.nanmean([corr(pix(1826 + t, r[sel] + dr, c[sel] + dc), o) for dr, dc in ((30, 0), (-30, 0), (0, 30), (0, -30))]))
        bias.append(np.nanmean(g0 - o))
    S = {k: np.nanmean(v) for k, v in sp.items()}
    ok = lagc[0] > max(lagc[-1], lagc[1]) and S[0] >= S[30]
    print(f"[{'PASS' if ok else 'FAIL'}] {name:8s} {int((src == s).sum()):5d} stations | temporal corr with GHAP: day-1 {lagc[-1]:.3f}, "
          f"same day {lagc[0]:.3f}, day+1 {lagc[1]:.3f} | spatial corr at pixel {S[0]:.3f} vs 30 km away {S[30]:.3f} | "
          f"GHAP - obs {np.nanmean(bias):+.1f} ug/m3", flush=True)
