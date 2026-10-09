#!/usr/bin/env python3
"""Analog retrieval for Deja-vu Flow. For every start day i and lead h, find the K past situations (j, h) of the
training library (2017-2020) closest to (PM today, weather of the forecast day):
    feature(i, h) = [log1p PM_i, weather_{i+h}] pooled 4x4, standardised per block, PCA-64.
Library entries never come from within 15 days of the query (no leakage for training queries).
Output data/analogs.npz: start[D, 4, K] (analog start day j; its successor j + h is the flow source), dist[D, 4, K]."""
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
K, POOL, NPC, EXCL = 16, 4, 64, 15
MET_CH = [0, 1, 2, 3, 12, 17, 29, 30, 33]  # u10, v10, t2m, msl, u850, v850, z500, BLH mean, rain

pm = np.load(DATA / "pm.npy", mmap_mode="r")
met = np.load(DATA / "met.npy", mmap_mode="r")
meta = np.load(DATA / "meta.npz")
year, land = meta["year"], meta["land"]
D, H, W = pm.shape


def pool(a):  # (..., 168, 280) -> (..., 42, 70) mean over 4x4 blocks
    return a.reshape(*a.shape[:-2], H // POOL, POOL, W // POOL, POOL).mean((-3, -1))


valid = np.array([np.isfinite(pm[i][land]).mean() > 0.8 for i in range(D)])
pm_feat = pool(np.where(land, np.log1p(np.nan_to_num(np.asarray(pm), nan=0.0)), 0.0)).reshape(D, -1)
met_feat = pool(np.asarray(met[:, MET_CH], np.float32)).reshape(D, -1)
train_days = np.where(valid & (year <= 2020))[0]

start = np.full((D, 4, K), -1, np.int32)
dist = np.full((D, 4, K), np.inf, np.float32)
for h in (1, 2, 3, 4):
    lib = train_days[(train_days + h < D)]
    lib = lib[valid[lib + h] & (year[lib + h] <= 2020)]
    q = np.where(valid[:D - h] & valid[h:])[0]
    def feats(days):
        return pm_feat[days], met_feat[days + h]
    lp, lm = feats(lib)
    mu_p, sd_p = lp.mean(0), lp.std(0) + 1e-6
    mu_m, sd_m = lm.mean(0), lm.std(0) + 1e-6
    def embed(days):
        p, m = feats(days)
        p, m = (p - mu_p) / sd_p / np.sqrt(p.shape[1]), (m - mu_m) / sd_m / np.sqrt(m.shape[1])  # equal block weight
        return np.concatenate([p, m], 1)
    L = embed(lib)
    mu = L.mean(0)
    _, _, vt = np.linalg.svd(L - mu, full_matrices=False)
    P = vt[:NPC].T
    Lz = (L - mu) @ P
    Qz = (embed(q) - mu) @ P
    d2 = (Qz ** 2).sum(1)[:, None] + (Lz ** 2).sum(1)[None] - 2 * Qz @ Lz.T
    d2[np.abs(q[:, None] - lib[None]) < EXCL] = np.inf
    order = np.argsort(d2, 1)[:, :K]
    start[q, h - 1] = lib[order]
    dist[q, h - 1] = np.sqrt(np.maximum(np.take_along_axis(d2, order, 1), 0))
    print(f"h={h}: library {len(lib)}, queries {len(q)}, median nearest distance {np.median(dist[q, h - 1, 0]):.3f}", flush=True)
np.savez(DATA / "analogs.npz", start=start, dist=dist, valid=valid)
print("saved", DATA / "analogs.npz")
