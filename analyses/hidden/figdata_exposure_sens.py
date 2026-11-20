#!/usr/bin/env python3
"""Sensitivity of the count of people hidden by a coarse grid (data only, no forecast): a modified COPY of
figures/figdata_exposure.py (the original is not touched; its result data/exposure_2022.npz is reproduced as a check).

Every day of 2022: the daily PM2.5 of GHAP at 1 km (pixels with a retrieval) against the same field averaged over blocks,
weighted by the residents of every 1 km pixel (GHS-POP 2020). Variants:
  blocks    40 x 40 pixels (0.4 deg, the paper), 25 x 25 (0.25 deg) and 10 x 10 (0.1 deg);
  origin    the 40-pixel blocks shifted by 10, 20 and 30 pixels in both directions (periodic in longitude);
  margin    hidden with a margin: pixel above thr + delta AND block mean not above thr - delta (two-sided), and pixel above
            thr + delta with the block mean not above thr (pixel only); delta = 5, 10, 20 % of the threshold and 1, 2, 5,
            10 ug m-3 (40-pixel blocks);
  monitors  the part of the exposed and of the hidden residents who live in the 1 km pixel of a monitor of
            stations_2022.npz (8,110 monitors: six official networks and OpenAQ), or within 2 pixels of one (40-pixel blocks).
Thresholds: WHO 2021 24-h guideline (15) and interim targets (25, 37.5, 50, 75 ug m-3).
Regions: the boxes of the paper (membership of the 40 km cell centre, as figdata_exposure.py) and the country groups of
regions.py (country of the pixel).
Only the pixels with residents are gathered (a pixel without residents adds nothing to any sum); the block means use every
pixel with a retrieval, as in the original.
    python figdata_exposure_sens.py prep ; srun -n 24 python figdata_exposure_sens.py ; python figdata_exposure_sens.py merge"""
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from regions import BOXES, GROUP_NAMES, pixel_groups  # noqa: E402

G = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
REF = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/figures/data/exposure_2022.npz")
OUT = HERE / "data" / "exposure_sens"
GH, GW, T, NTR, NTC = 14400, 36000, 512, 29, 71
I2022 = 1826
THR = np.array([15.0, 25.0, 37.5, 50.0, 75.0], np.float32)
VARIANTS = ((40, 0), (25, 0), (10, 0), (40, 10), (40, 20), (40, 30))          # (pixels per block side, origin shift)
MARG = (("rel", 0.05), ("rel", 0.10), ("rel", 0.20), ("abs", 1.0), ("abs", 2.0), ("abs", 5.0), ("abs", 10.0))
REG = list(BOXES) + [f"G:{n}" for n in GROUP_NAMES]


def read_rows(path, n):
    """row n of a .npy (pread, in 1 GB pieces)  [copied from figdata_exposure.py]"""
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); sh, _, dt = np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    size = int(np.prod(sh[1:])) * dt.itemsize
    fd = os.open(path, os.O_RDONLY); parts, got = [], 0
    try:
        while got < size:
            b = os.pread(fd, min(size - got, 1 << 30), hdr + n * size + got); parts.append(b); got += len(b)
    finally:
        os.close(fd)
    return np.frombuffer(b"".join(parts), dt).reshape(sh[1:])


def ghap_day(i):
    a = read_rows(G / "ghap1km_tiles.npy", i)                                 # (29, 71, 512, 512) float16
    return a.transpose(0, 2, 1, 3).reshape(NTR * T, NTC * T)[:GH, :GW]


def prep():
    pop = np.load(G / "pop1km_global.npy", mmap_mode="r")
    idx = []
    tot = 0.0
    for r0 in range(0, GH, 1200):
        p = np.asarray(pop[r0:r0 + 1200], np.float32)
        assert np.isfinite(p).all() and p.min() >= 0, "population has negative or missing values"
        tot += float(p.sum(dtype=np.float64))
        idx.append(np.flatnonzero(p > 0) + r0 * GW)
    idx = np.concatenate(idx)
    w = np.asarray(pop, np.float32).ravel()[idx]
    r, c = idx // GW, idx % GW
    lat = 84.0 - 0.4 * (r // 40 + 0.5); lon = -180.0 + 0.4 * (c // 40 + 0.5)       # centre of the 40 km cell (region_masks)
    bits = np.zeros(len(idx), np.int64)
    for j, b in enumerate(list(BOXES.values())[1:]):
        bits |= ((lat <= b[0]) & (lat > b[1]) & (lon >= b[2]) & (lon < b[3])).astype(np.int64) << j
    grp = pixel_groups(r, c)
    combo, lab = np.unique(bits + 256 * grp, return_inverse=True)
    M = np.zeros((len(REG), len(combo)))
    M[0] = 1.0
    for j in range(len(BOXES) - 1):
        M[1 + j] = (combo % 256 >> j) & 1
    for g in range(len(GROUP_NAMES)):
        M[len(BOXES) + g] = combo // 256 == g
    st = np.load(G / "stations_2022.npz")
    sr = np.clip(np.floor((84.0 - st["lat"]) / 0.01).astype(int), 0, GH - 1)
    sc = np.clip(np.floor((st["lon"] + 180.0) / 0.01).astype(int), 0, GW - 1)
    near = np.zeros((GH, GW), bool); near[sr, sc] = True
    st1 = near.ravel()[idx]
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            near[np.clip(sr + dr, 0, GH - 1), (sc + dc) % GW] = True
    st5 = near.ravel()[idx]
    (HERE / "data").mkdir(exist_ok=True)
    np.savez(HERE / "data" / "pop_pos.npz", idx=idx.astype(np.int64), pop=w, lab=lab.astype(np.int32), M=M, st1=st1, st5=st5,
             regions=np.array(REG), n_stations=len(sr))
    print(f"prep: {len(idx)} pixels with residents, {w.sum(dtype=np.float64) / 1e6:.1f} M residents (all pixels {tot / 1e6:.1f} M), "
          f"{len(combo)} region combinations; residents in a monitor pixel {w[st1].sum(dtype=np.float64) / 1e6:.1f} M, "
          f"within 2 pixels {w[st5].sum(dtype=np.float64) / 1e6:.1f} M", flush=True)


def block_means(xf, ok, F, s):
    """means of xf over the valid pixels of F x F blocks whose origin is shifted by s pixels (periodic in longitude);
    -> (block rows, block columns); the base case (s = 0) is the expression of figdata_exposure.py"""
    nb = (GH - s) // F
    if s:
        xf = np.roll(xf, -s, axis=1)[s:s + F * nb]; ok = np.roll(ok, -s, axis=1)[s:s + F * nb]
    x4 = xf.reshape(nb, F, GW // F, F); o4 = ok.reshape(nb, F, GW // F, F)
    with np.errstate(all="ignore"):
        return np.where(o4, x4, 0).sum((1, 3)) / o4.sum((1, 3))


def run():
    rank, world = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(HERE / "data" / "pop_pos.npz")
    idx, pop, lab, M, st1, st5 = z["idx"], z["pop"].astype(np.float64), z["lab"].astype(np.int64), z["M"], z["st1"], z["st5"]
    NC = M.shape[1]; r, c = idx // GW, idx % GW
    ref = np.load(REF)["res"]                                                  # (365, 9 boxes, 4 terms, 5 thr)
    days = list(range(365))[rank::world]
    test = os.environ.get("DAYS")
    days = [int(v) for v in test.split(",")] if test else days
    tab = np.zeros((len(days), len(VARIANTS), len(REG), len(THR), 4))          # neither, pixel only (hidden), block only, both
    marg = np.zeros((len(days), len(REG), len(THR), len(MARG), 2))             # two-sided, pixel only
    stn = np.zeros((len(days), len(REG), len(THR), 2, 2))                      # (monitor pixel, within 2 px) x (above, hidden)
    lost = np.zeros((len(days), len(VARIANTS)))                                # residents with a retrieval outside the blocks
    worst = 0.0
    for n, d in enumerate(days):
        t0 = time.time()
        xf = ghap_day(I2022 + d).astype(np.float32)
        ok = xf > 0
        xv = xf.ravel()[idx]
        w = np.where(xv > 0, pop, 0.0)
        for v, (F, s) in enumerate(VARIANTS):
            cell = block_means(xf, ok, F, s)
            inside = (r >= s) & (r < s + F * cell.shape[0])
            cv = np.full(len(idx), np.nan, np.float32)
            cv[inside] = cell[(r[inside] - s) // F, ((c[inside] - s) % GW) // F]
            wv = np.where(np.isfinite(cv), w, 0.0)
            lost[n, v] = w.sum() - wv.sum()
            for k, thr in enumerate(THR):
                a1, a4 = xv > thr, cv > thr
                tab[n, v, :, k] = M @ np.bincount(lab * 4 + a1 + 2 * a4, weights=wv, minlength=NC * 4).reshape(NC, 4)
                if v == 0:
                    hid = a1 & ~a4
                    for j, (kind, val) in enumerate(MARG):
                        dl = np.float32(val * thr if kind == "rel" else val)
                        hi = xv > thr + dl
                        marg[n, :, k, j, 0] = M @ np.bincount(lab, weights=wv * (hi & ~(cv > thr - dl)), minlength=NC)
                        marg[n, :, k, j, 1] = M @ np.bincount(lab, weights=wv * (hi & ~a4), minlength=NC)
                    for q, sm in enumerate((st1, st5)):
                        stn[n, :, k, q, 0] = M @ np.bincount(lab[sm], weights=(wv * a1)[sm], minlength=NC)
                        stn[n, :, k, q, 1] = M @ np.bincount(lab[sm], weights=(wv * hid)[sm], minlength=NC)
            del cell
        # check against figdata_exposure.py (boxes, 40-pixel blocks): above at 1 km, above at 40 km, hidden, spurious
        mine = np.stack([tab[n, 0, :9, :, 1] + tab[n, 0, :9, :, 3], tab[n, 0, :9, :, 2] + tab[n, 0, :9, :, 3],
                         tab[n, 0, :9, :, 1], tab[n, 0, :9, :, 2]], 1)          # (9, 4, thr)
        dev = np.abs(mine - ref[d]) / np.maximum(ref[d], 1e5)
        worst = max(worst, float(dev.max()))
        k = list(THR).index(37.5)
        print(f"rank {rank}: day {d}: hidden > 37.5: {tab[n, 0, 0, k, 1] / 1e6:.2f} M (40 km; ref {ref[d, 0, 2, k] / 1e6:.2f}), "
              f"{tab[n, 1, 0, k, 1] / 1e6:.2f} M (25 km), {tab[n, 2, 0, k, 1] / 1e6:.2f} M (10 km), shifts "
              + "/".join(f"{tab[n, v, 0, k, 1] / 1e6:.1f}" for v in (3, 4, 5))
              + f" | max rel. dev. from the original {dev.max():.2e} | {time.time() - t0:.0f} s", flush=True)
    if not test:
        np.savez(OUT / f"part_{rank}.npz", days=np.array(days), tab=tab, marg=marg, stn=stn, lost=lost, worst=worst)
    print(f"rank {rank} done, worst relative deviation from exposure_2022.npz {worst:.2e}", flush=True)


def merge():
    P = [np.load(p) for p in sorted(OUT.glob("part_*.npz"))]
    days = np.concatenate([p["days"] for p in P]); o = np.argsort(days)
    assert len(days) == 365 and (days[o] == np.arange(365)).all(), len(days)
    cat = lambda k: np.concatenate([p[k] for p in P])[o]
    np.savez(HERE / "data" / "exposure_sens_2022.npz", days=days[o], tab=cat("tab"), marg=cat("marg"), stn=cat("stn"),
             lost=cat("lost"), regions=np.array(REG), thr=THR, variants=np.array(VARIANTS),
             margins=np.array([f"{k}:{v:g}" for k, v in MARG]),
             tab_cols=np.array(["neither", "pixel above only (hidden)", "block above only (reverse)", "both above"]),
             worst=max(float(p["worst"]) for p in P))
    print("saved", HERE / "data" / "exposure_sens_2022.npz", "| worst relative deviation from exposure_2022.npz:",
          max(float(p["worst"]) for p in P))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    prep() if arg == "prep" else merge() if arg == "merge" else run()
