#!/usr/bin/env python3
"""Case study kept for every model: the clean-air outbreak of 25 January 2022 over Beijing (case_beijing.sh). For every
lead (start day 25 - h January), the forecast at the 12 CNEMC stations of Beijing (ensemble mean, members, P(> 75)), the
observation, GHAP at the stations, and what FLEXGREEN (256 particles) said about the arriving air: share from aloft,
transported PM2.5 of day d and mean survival. Stations only evaluate.
    python case_beijing_summary.py <label>=<run>[:<eval dir, default case_beijing>] ...
    e.g. python case_beijing_summary.py A=green_flow_shift25_global_L7_noant_whres_ghres 2D=fp_flow_shift25_global_L7_noant:eval_global"""
import glob
import os
import sys

import numpy as np

G = "/scratch/project_462001140/ammar/eccv/data/world/global1km"
T = 24                                                   # target day: 25 January 2022 (day of year, 0-based)
st = np.load(f"{G}/stations_2022.npz")
SEL = np.where((st["source"] == 1) & (np.abs(st["lat"] - 39.9) < 0.3) & (np.abs(st["lon"] - 116.4) < 0.4))[0]
OBS = st["daily"][SEL].astype(np.float32)


def npy_read(path, index):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); sh, _, dt = np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    inner = int(np.prod(sh[len(index):])); k = int(np.ravel_multi_index(index, sh[:len(index)]))
    fd = os.open(path, os.O_RDONLY); b = os.pread(fd, inner * dt.itemsize, hdr + k * inner * dt.itemsize); os.close(fd)
    return np.frombuffer(b, dt).reshape(sh[len(index):])


def transport(d, h, r=88, c=592):
    """share of the arriving air from aloft, transported PM2.5 of day d (as the model reads it), mean survival"""
    o = npy_read(f"{G}/green256/hres/origins_2022.npy", (d, h - 1, r, c)).astype(np.float32); lon, lat, zr, sv = o.T
    pm = npy_read(f"{G}/pm.npy", (1826 + d,)).astype(np.float32)
    f = np.nan_to_num((pm - 15) / 20, nan=0.0).reshape(288, 2, 720, 2).mean((1, 3)); fw = np.concatenate([f, f[:, :1]], 1)
    x = ((lon + 180 - 0.125) % 360) / 0.5; y = np.clip((84 - 0.125 - lat) / 0.5, 0, 287)
    x0 = np.floor(x).astype(int); y0 = np.clip(np.floor(y).astype(int), 0, 286); wx, wy = x - x0, y - y0
    v = ((1 - wy) * ((1 - wx) * fw[y0, x0] + wx * fw[y0, x0 + 1]) + wy * ((1 - wx) * fw[y0 + 1, x0] + wx * fw[y0 + 1, x0 + 1])) * 20 + 15
    w = sv * (zr <= 1)
    return (sv * (zr > 1)).sum() / sv.sum(), np.sum(w * v) / max(np.sum(w), 1e-6), sv.mean()


def forecasts(run, ev):
    out = {}
    for p in sorted(glob.glob(f"runs/{run}/{ev}/part_*.npz")):
        z = np.load(p)
        for n, (d, h, t) in enumerate(z["keys"]):
            if int(t) == T:
                out[int(h)] = (z["st_ens"][n][SEL].astype(np.float32), z["st_ghap"][n][SEL].astype(np.float32))
    return out


runs = []
for arg in sys.argv[1:]:
    label, spec = arg.split("=")
    run, ev = (spec.split(":") + ["case_beijing"])[:2]
    runs.append((label, forecasts(run, ev)))
print(f"Beijing, 25 January 2022: {len(SEL)} CNEMC stations | observed 22-25 Jan: "
      + " -> ".join(f"{np.nanmean(OBS[:, t]):.0f}" for t in range(21, 25)) + " ug/m3")
print(f"{'lead':5s} {'start':7s} {'aloft':>6s} {'PM tr.':>7s} {'surv.':>6s} {'PM day d':>9s} | "
      + " | ".join(f"{lab}: mean [min-max] P>75" for lab, _ in runs))
for h in range(1, 8):
    d = T - h
    try:
        al, pmt, sv = transport(d, h)
        tr = f"{al:6.0%} {pmt:7.0f} {sv:6.2f}"
    except Exception:
        tr = f"{'-':>6s} {'-':>7s} {'-':>6s}"
    pd = np.nanmean(OBS[:, d])
    row = []
    for lab, fc in runs:
        if h in fc:
            E = fc[h][0]; mem = np.nanmean(E, 0)
            row.append(f"{np.nanmean(mem):5.0f} [{np.nanmin(mem):.0f}-{np.nanmax(mem):.0f}] {np.mean(mem > 75):.2f}")
        else:
            row.append("   -")
    print(f"J+{h:<3d} {d + 1:2d} Jan {tr} {pd:9.0f} | " + " | ".join(row))
g = [np.nanmean(fc[h][1]) for _, fc in runs for h in fc][:1]
print(f"observed on the 25th: {np.nanmean(OBS[:, T]):.0f} ug/m3" + (f" | GHAP at the stations {g[0]:.0f}" if g else ""))
