#!/usr/bin/env python3
"""Step 3: calibration against the 1 km estimates where people live. Reads (read-only) the stored members of the final
model at the populated 1 km pixels (eval_pop_crps_e01: pop_<rank>.npy (forecasts, 8,112,762 pixels, 4 members),
pop_ghap_<rank>.npy = GHAP of the target day there, part_<rank>.npz keys; 179 start days x leads 1 and 3) and writes
per-forecast sums by region (the nine boxes of eval_global_green.REGIONS, a pixel may lie in two boxes, + all pixels):
  n, var, sse, err, sae, crps (fair, four members), rank histogram of the estimate among the members (ties at random),
  counts by (members above the threshold, estimate above the threshold) for 15, 37.5 and 75 ug m-3,
  the same weighted by the residents of each pixel (GHS-POP 2020, pop_pixels.npz), and, over all pixels, n / var / sse in
  bins of ensemble spread. Pixels: >= 100 residents (the station pixels added to the list with fewer are left out),
  estimate of the target day valid, four members finite.
Per pixel and lead it also counts the valid days and the exceedances of the estimate, for the Brier score of the
frequency of 2022 at each pixel (the best constant forecast of a pixel).
    python s3_pop.py [workers]   -> data/sums_pop.npz"""
import glob
import sys
from multiprocessing import Pool

import numpy as np

from rel_common import DATA, FINAL, G, THR

EV = FINAL / "eval_pop_crps_e01"
REGIONS = {"Europe": (72, 35, -25, 45), "North America": (72, 15, -170, -50), "South America": (15, -56, -95, -30),
           "Africa": (37, -35, -20, 52), "Middle East": (42, 12, 35, 63), "South Asia": (37, 5, 60, 97),
           "East Asia": (54, 18, 97, 146), "Southeast Asia": (18, -11, 92, 142), "Oceania": (-10, -48, 110, 180)}
SD_EDGES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 1e9])
NB = len(SD_EDGES) - 1
M = 4
W4 = (2 * np.arange(M) - M + 1).astype(np.float32)
LEADS = (1, 3)


def setup():
    pp = np.load(G / "pop_pixels.npz")
    idx, pop = pp["idx"], pp["pop"].astype(np.float64)
    assert np.array_equal(idx, np.load(EV / "pop_idx.npy"))
    row, col = idx // 36000, idx % 36000
    lat = 84.0 - 0.01 * (row + 0.5); lon = -180.0 + 0.01 * (col + 0.5)
    combo = np.zeros(len(idx), np.int64)
    for r, (n, s, w, e) in enumerate(REGIONS.values()):
        combo |= (((lat <= n) & (lat > s) & (lon >= w) & (lon < e)).astype(np.int64) << r)
    u, lab = np.unique(combo, return_inverse=True)
    A = np.stack([[(c >> r) & 1 for r in range(len(REGIONS))] + [1] for c in u]).astype(np.float64)   # (labels, 10)
    return lab.astype(np.int64), A, pop, float(pp["min_pop"])


LAB, A, POP, MINPOP = setup()
NL = A.shape[0]
POPOK = POP >= MINPOP


def one_rank(rank):
    keys = np.load(EV / f"part_{rank}.npz")["keys"]
    if len(keys) == 0:
        return None
    ens = np.load(EV / f"pop_{rank}.npy", mmap_mode="r"); gh = np.load(EV / f"pop_ghap_{rank}.npy", mmap_mode="r")
    assert ens.shape[0] == len(keys) == gh.shape[0] and ens.shape[2] == M
    rng = np.random.default_rng(1000 + rank)
    out = []
    npx = len(LAB)
    pix_n = np.zeros((len(LEADS), npx), np.uint8); pix_o = np.zeros((len(LEADS), 3, npx), np.uint8)
    for k, (d, h, t) in enumerate(keys):
        E = np.asarray(ens[k], np.float32); g = np.asarray(gh[k], np.float32)
        ok = np.isfinite(g) & np.isfinite(E).all(1) & POPOK
        E = np.where(ok[:, None], E, 0.0); g = np.where(ok, g, 0.0)
        Es = np.sort(E, 1)
        mean = E.mean(1); var = E.var(1, ddof=1)
        err = mean - g
        crps = np.abs(E - g[:, None]).mean(1) - (Es * W4).sum(1) / (M * (M - 1))
        lt = (E < g[:, None]).sum(1); eq = (E == g[:, None]).sum(1)
        rk = np.clip(lt + (rng.integers(0, 1 << 30, npx) % (eq + 1)), 0, M)
        okf = ok.astype(np.float64); wp = POP * okf
        R = {}
        for nm, w in (("", okf), ("w", wp)):
            for q, x in (("n", None), ("var", var), ("sse", err * err), ("err", err), ("sae", np.abs(err)), ("crps", crps)):
                R[nm + q] = np.bincount(LAB, weights=w if x is None else w * x, minlength=NL)
            R[nm + "rank"] = np.bincount(LAB * (M + 1) + rk, weights=w, minlength=NL * (M + 1)).reshape(NL, M + 1)
            rel = []
            for T in THR:
                kT = (E > T).sum(1); o = (g > T).astype(np.int64)
                rel.append(np.bincount(LAB * (2 * (M + 1)) + kT * 2 + o, weights=w, minlength=NL * 2 * (M + 1)).reshape(NL, M + 1, 2))
            R[nm + "rel"] = np.stack(rel, 1)                                  # (labels, 3, k, o)
        sdb = np.clip(np.digitize(np.sqrt(var), SD_EDGES) - 1, 0, NB - 1)
        R["sd_n"] = np.bincount(sdb, weights=okf, minlength=NB); R["sd_var"] = np.bincount(sdb, weights=okf * var, minlength=NB)
        R["sd_sse"] = np.bincount(sdb, weights=okf * err * err, minlength=NB)
        li = LEADS.index(int(h))
        pix_n[li] += ok
        for j, T in enumerate(THR):
            pix_o[li, j] += (g > T) & ok
        out.append((tuple(int(v) for v in (d, h, t)), R))
    return out, pix_n, pix_o


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    ranks = sorted(int(p.split("_")[-1][:-4]) for p in glob.glob(str(EV / "part_*.npz")))
    rows, PN, PO = [], None, None
    with Pool(workers) as pool:
        for res in pool.imap_unordered(one_rank, ranks):
            if res is None:
                continue
            out, pix_n, pix_o = res
            rows += out
            PN = pix_n.astype(np.uint16) if PN is None else PN + pix_n
            PO = pix_o.astype(np.uint16) if PO is None else PO + pix_o
            print("ranks done:", len(rows), "forecasts", flush=True)
    rows.sort(key=lambda r: (r[0][1], r[0][0]))
    keys = np.array([r[0] for r in rows], np.int32)
    S = {k: np.stack([np.tensordot(A.T, r[1][k], axes=(1, 0)) if not k.startswith("sd_") else r[1][k] for r in rows])
         for k in rows[0][1]}                                                 # (forecasts, 10 regions, ...)
    # Brier score of the frequency of 2022 at each pixel (per lead and threshold; sum over pixels of n f (1 - f)), by region
    pixref = np.zeros((len(LEADS), 3, A.shape[1])); pixref_u = np.zeros_like(pixref)
    for li in range(len(LEADS)):
        n = PN[li].astype(np.float64)
        for j in range(3):
            f = PO[li, j] / np.maximum(n, 1)
            b = n * f * (1 - f)
            pixref[li, j] = A.T @ np.bincount(LAB, weights=b, minlength=NL)
            pixref_u[li, j] = A.T @ np.bincount(LAB, weights=b * n / np.maximum(n - 1, 1), minlength=NL)   # each day left out
    np.savez(DATA / "sums_pop.npz", keys=keys, regions=np.array(list(REGIONS) + ["All populated pixels"]), thr=np.array(THR),
             sd_edges=SD_EDGES, leads=np.array(LEADS), pixref=pixref, pixref_loo=pixref_u,
             n_pixels=np.array([int((PN[li] > 0).sum()) for li in range(len(LEADS))]),
             residents=np.array([float(POP[PN[li] > 0].sum()) for li in range(len(LEADS))]), **S)
    for h in LEADS:
        k = keys[:, 1] == h
        n, var, sse = (S[q][k, -1].sum() for q in ("n", "var", "sse"))
        rk = S["rank"][k, -1].sum(0)
        print(f"lead {h}: {int(k.sum())} forecasts, {int(n):,} pixel-days | spread/error {np.sqrt((M + 1) / M * var / sse):.3f} | "
              f"rank histogram {np.round(rk / rk.sum(), 3)} | CRPS {S['crps'][k, -1].sum() / n:.3f} RMSE {np.sqrt(sse / n):.3f}")


if __name__ == "__main__":
    main()
