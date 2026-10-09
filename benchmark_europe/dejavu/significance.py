#!/usr/bin/env python3
"""Is the footprint gain significant? Paired moving-block bootstrap over start days (blocks of 7 days, the persistence
of pollution regimes), flow with vs without footprint, real HRES weather, 2022, per-day records of 3 seeds each
(sig_job.sh -> runs/*/records_2022_hres_rec8.npy, 8 members, 8 steps as the main table; seeds pooled by summing their scores for each start day).
Day sets: all days; documented transported episodes (Saharan dust 11-21 and 25-31 March, wildfires 7-21 July and
6-14 August 2022; start days); distribution tails: start days whose target day is in the top 10 % / 5 % of the
fraction of land above 25 ug/m3 (episode extent), per lead.
Reports the relative change of RMSE, CRPS and CSI>25 of the ensemble mean with a 95 % interval and a one-sided p-value."""
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent / "runs"
EPISODES = [(69, 79), (83, 89), (187, 201), (217, 225)]   # day of year (0 = 1 January), start days
B, BLOCK = 10000, 7
import os
TAG = os.environ.get("REC_TAG", "rec8")
COLS = dict(i=0, doy=1, h=2, n=3, sse=4, crps=5, phit=6, puni=7, mean=8, frac=9, hit=10, uni=11)   # CSI of the ensemble mean


def load(base):
    recs = [np.load(RUNS / f"{base}{s}" / f"records_2022_hres_{TAG}.npy") for s in ("", "_s43", "_s44")]
    key = lambda r: {(int(x[1]), int(x[2])): x for x in r}
    ks = [key(r) for r in recs]
    common = sorted(set.intersection(*[set(k) for k in ks]))
    return {c: np.sum([k[c] for k in ks], 0) for c in common}, ks[0]   # seed-summed scores, one seed for day metadata


def metrics(rows):
    n, sse, crps, hit, uni = (rows[:, COLS[c]].sum(0) for c in ("n", "sse", "crps", "hit", "uni"))
    return np.array([np.sqrt(sse / n), crps / n, hit / max(uni, 1)])


def main():
    no, meta = load("noise_t31")
    fp, _ = load("noise_t31_fpstab")
    rng = np.random.default_rng(0)
    print(f"{'set':22s} {'lead':5s} {'days':>5s} | {'RMSE %':>18s} | {'CRPS %':>18s} | {'CSI %':>18s}")
    for h in (1, 2, 3, 4):
        keys = [k for k in no if k[1] == h and k in fp]
        frac = np.array([meta[k][COLS["frac"]] for k in keys])
        sets = {"all days": np.ones(len(keys), bool),
                "transported episodes": np.array([any(a <= k[0] <= b for a, b in EPISODES) for k in keys]),
                "top 10 % extent": frac >= np.quantile(frac, 0.90),
                "top 5 % extent": frac >= np.quantile(frac, 0.95)}
        for name, sel in sets.items():
            idx = np.where(sel)[0]
            A = np.array([no[keys[i]] for i in idx]); F = np.array([fp[keys[i]] for i in idx])
            rel = lambda a, f: 100 * (metrics(f) - metrics(a)) / metrics(a)
            point = rel(A, F)
            nb, L = len(idx), min(BLOCK, len(idx))
            boot = np.empty((B, 3))
            for b in range(B):   # moving blocks of consecutive start days (in time order)
                starts = rng.integers(0, max(nb - L, 0) + 1, int(np.ceil(nb / L)))
                j = np.concatenate([np.arange(s, s + L) for s in starts])[:nb]
                boot[b] = rel(A[j], F[j])
            lo, hi = np.percentile(boot, [2.5, 97.5], 0)
            p_better = [(boot[:, 0] >= 0).mean(), (boot[:, 1] >= 0).mean(), (boot[:, 2] <= 0).mean()]
            cell = lambda c: f"{point[c]:+5.1f} [{lo[c]:+5.1f},{hi[c]:+5.1f}] p={p_better[c]:.3f}"
            print(f"{name:22s} T+{h:<3d} {nb:5d} | {cell(0)} | {cell(1)} | {cell(2)}")


if __name__ == "__main__":
    main()
