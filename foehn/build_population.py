#!/usr/bin/env python3
"""Population of every pixel of the global 1 km grid (0.01 deg, 84 N .. 60 S, 180 W .. 180 E), from GHS-POP R2023A
(epoch 2020, 30 arc-seconds, residents per cell): exact area-weighted re-binning (uniform density inside each source
cell), done separably with cumulative sums, so that population totals are conserved.
  pop1km_global.npy  (14400, 36000) float32   residents per 1 km pixel
  pop_pixels.npz     flat indices (row * 36000 + col) of the pixels scored for exposure: populated pixels (>= MIN_POP
                     residents) and the pixels of the monitoring stations, with the population they hold
    source world/env.sh; python build_population.py [build | pixels]"""
import sys
from pathlib import Path

import numpy as np

G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
SRC = Path("/scratch/project_462001140/ammar/eccv/data/world/population/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif")
GH, GW, RES = 14400, 36000, 0.01
MIN_POP = 100.0


def positions(edges, origin, step):
    """fractional source-cell coordinates of target edges"""
    p = (edges - origin) / step
    i = np.floor(p).astype(np.int64)
    return i, p - i


def rebin_axis(a, i, f, axis):
    """integral of the piecewise-constant source a (float64) between consecutive target edges (source coords i + f)."""
    a = np.moveaxis(a, axis, -1)
    c = np.concatenate([np.zeros(a.shape[:-1] + (1,), a.dtype), np.cumsum(a, -1)], -1)
    pad = np.concatenate([a, np.zeros(a.shape[:-1] + (1,), a.dtype)], -1)       # f = 0 at the last edge
    cum = c[..., i] + f * pad[..., i]
    return np.moveaxis(np.diff(cum, axis=-1), -1, axis)


def build():
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(SRC) as s:
        t = s.transform
        dx, x0, dy, y0 = t.a, t.c, -t.e, t.f
        W, H = s.width, s.height
        print("source", W, H, "dx", dx, "x0", x0, "dy", dy, "y0", y0, flush=True)
        ix, fx = positions(-180.0 + RES * np.arange(GW + 1), x0, dx)            # longitude edges
        iy, fy = positions(y0 - (84.0 - RES * np.arange(GH + 1)), 0.0, dy)      # latitude edges (distance from top)
        assert ix.min() >= 0 and ix.max() < W and iy.min() >= 0 and iy.max() < H, (ix.min(), ix.max(), iy.min(), iy.max())
        r0, r1 = int(iy.min()), int(iy.max()) + 1                               # source rows needed
        mid = np.zeros((r1 - r0, GW), np.float32)
        src_total = 0.0
        for a in range(r0, r1, 1024):
            n = min(1024, r1 - a)
            blk = np.maximum(s.read(1, window=Window(0, a, W, n)).astype(np.float64), 0.0)
            src_total += blk.sum()
            mid[a - r0:a - r0 + n] = rebin_axis(blk, ix, fx, 1)
        print(f"source population in rows {r0}-{r1}: {src_total / 1e9:.4f} billion", flush=True)
    out = np.zeros((GH, GW), np.float32)
    for c in range(0, GW, 2000):
        out[:, c:c + 2000] = rebin_axis(mid[:, c:c + 2000].astype(np.float64), iy - r0, fy, 0)
    np.save(G / "pop1km_global.npy", out)
    print(f"world population on the grid: {out.sum(dtype=np.float64) / 1e9:.4f} billion; negative pixels: {(out < -1e-3).sum()}")


def pixels():
    pop = np.load(G / "pop1km_global.npy", mmap_mode="r")
    st = np.load(G / "stations_2022.npz")
    sr = np.clip(np.floor((84.0 - st["lat"]) / RES).astype(np.int64), 0, GH - 1)
    sc = np.clip(np.floor((st["lon"] + 180.0) / RES).astype(np.int64), 0, GW - 1)
    idx, pv, tot = [], [], 0.0
    for r in range(0, GH, 1200):
        blk = np.asarray(pop[r:r + 1200], np.float32)
        tot += float(blk.sum(dtype=np.float64))
        k = np.flatnonzero(blk >= MIN_POP)
        idx.append(k + r * GW); pv.append(blk.ravel()[k])
    idx, pv = np.concatenate(idx), np.concatenate(pv)
    st_idx = np.unique(sr * GW + sc)
    allidx = np.union1d(idx, st_idx)
    popall = np.asarray(pop).ravel()[allidx]
    np.savez(G / "pop_pixels.npz", idx=allidx.astype(np.int64), pop=popall.astype(np.float32), min_pop=MIN_POP)
    print(f"{len(idx):,} pixels with >= {MIN_POP:g} residents hold {pv.sum() / tot:.1%} of {tot / 1e9:.2f} billion; "
          f"+ station pixels -> {len(allidx):,} pixels saved")


if __name__ == "__main__":
    {"build": build, "pixels": pixels}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
