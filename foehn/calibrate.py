#!/usr/bin/env python3
"""Post-processing calibration of the Foehn ensemble, fitted on GHAP only (validation year 2021; monitoring stations
never enter the fit): EMOS in log(1 + x) (Gneiting et al. 2005), per lead and region, Gaussian predictive distribution
with mean a + b Lbar and sd sqrt(c^2 + d^2 s^2) (Lbar, s: mean and sd of the log members; corrects the level-dependent
bias and the under-dispersion), fitted by the closed-form CRPS; members by ensemble copula coupling (ECC-Q, Schefzik et
al. 2013): each member receives the predictive quantile of its rank, keeping the spatial structure of the raw members.
(a, b, alpha) minimise the fair CRPS against GHAP at a random sample of the populated pixels of every 2021 forecast
(eval_global_fp_year.py --year 2021 --pop). The fitted transform is then applied to the 2022 forecasts (members at the
stations and at the populated pixels) and scored there: stations stay an independent evaluation.
    python calibrate.py fit runs/<run>/eval_2021 [--sample 400000]  -> calib.json in that directory
    python calibrate.py apply runs/<run>/eval_2021/calib.json runs/<run>/eval_global      -> station scores, raw vs calibrated"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri

G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
REGIONS = {"Europe": (72, 35, -25, 45), "North America": (72, 15, -170, -50), "South America": (15, -56, -95, -30),
           "Africa": (37, -35, -20, 52), "Middle East": (42, 12, 35, 63), "South Asia": (37, 5, 60, 97),
           "East Asia": (54, 18, 97, 146), "Southeast Asia": (18, -11, 92, 142), "Oceania": (-10, -48, 110, 180)}


def region_of(lat, lon):
    r = np.full(lat.shape, -1, int)
    for k, (n, s, w, e) in enumerate(REGIONS.values()):
        r[(r < 0) & (lat <= n) & (lat >= s) & (lon >= w) & (lon <= e)] = k
    return r


def fair_crps(E, y):
    """E (N, M), y (N) -> mean fair CRPS (sorted-member formula)"""
    M = E.shape[1]
    Es = np.sort(E, 1)
    w = 2 * np.arange(1, M + 1) - M - 1
    pair = (Es * w).sum(1) / (M * (M - 1))
    return float(np.mean(np.abs(E - y[:, None]).mean(1) - pair))


def emos_params(L, p):
    """log-space EMOS: mean a + b Lbar, sd sqrt(c^2 + d^2 s^2) (s: ensemble sd of the log members)"""
    a, b, c, d = p
    return a + b * L.mean(1), np.sqrt(c * c + d * d * L.var(1, ddof=1) + 1e-6)


def crps_gauss(mu, sd, y):
    z = (y - mu) / sd
    return float(np.mean(sd * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))))


def transform(E, p):
    """EMOS + ensemble copula coupling (ECC-Q): member m receives the predictive quantile of its rank at each pixel,
    which keeps the spatial structure (rank order) of the raw members."""
    L = np.log1p(np.maximum(E, 0))
    mu, sd = emos_params(L, p)
    M = E.shape[1]
    q = mu[:, None] + sd[:, None] * ndtri((np.arange(1, M + 1) - 0.5) / M)[None]    # sorted quantiles
    rank = np.argsort(np.argsort(L, 1), 1)
    return np.expm1(np.take_along_axis(q, rank, 1))


def fit_emos(E, y):
    L, Ly = np.log1p(np.maximum(E, 0)), np.log1p(np.maximum(y, 0))
    f = lambda p: crps_gauss(*emos_params(L, p), Ly)
    res = minimize(f, x0=[0.0, 1.0, 0.1, 1.0], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=800))
    return res.x, f


def pixel_latlon(idx):
    r, c = idx // 36000, idx % 36000
    return 84.0 - 0.01 * (r + 0.5), -180.0 + 0.01 * (c + 0.5)


def load_pop(d):
    parts = sorted(glob.glob(str(Path(d) / "part_*.npz")))
    keys, E, T = [], [], []
    for p in parts:
        r = p.split("_")[-1].split(".")[0]
        keys.append(np.load(p)["keys"]); E.append(np.load(Path(d) / f"pop_{r}.npy", mmap_mode="r"))
        T.append(np.load(Path(d) / f"popt_{r}.npy", mmap_mode="r") if (Path(d) / f"popt_{r}.npy").exists() else None)
    return keys, E, T, np.load(Path(d) / "pop_idx.npy")


def fit(a):
    keys, E, T, idx = load_pop(a.dir)
    lat, lon = pixel_latlon(idx)
    reg = region_of(lat, lon)
    rng = np.random.default_rng(0)
    out = {}
    for h in range(1, 8):
        xs, ys, rs = [], [], []
        for k, e, t in zip(keys, E, T):
            for n in np.where(k[:, 1] == h)[0]:
                sel = rng.choice(len(idx), max(1, a.sample // max(1, sum((kk[:, 1] == h).sum() for kk in keys))), replace=False)
                x, y = np.asarray(e[n][sel], np.float32), np.asarray(t[n][sel], np.float32)
                ok = np.isfinite(y) & np.isfinite(x).all(1) & (reg[sel] >= 0)
                xs.append(x[ok]); ys.append(y[ok]); rs.append(reg[sel][ok])
        X, Y, R = np.concatenate(xs), np.concatenate(ys), np.concatenate(rs)
        for k_, name in enumerate(REGIONS):
            m = R == k_
            if m.sum() < 2000:
                continue
            p, _ = fit_emos(X[m], Y[m])
            c_raw, c_cal = fair_crps(X[m], Y[m]), fair_crps(transform(X[m], p), Y[m])
            out[f"{h}|{name}"] = dict(p=[float(v) for v in p], n=int(m.sum()), crps_raw=c_raw, crps_cal=c_cal)
            print(f"lead {h} {name:15s} n {m.sum():7d} | a {p[0]:+.3f} b {p[1]:.3f} c {abs(p[2]):.3f} d {abs(p[3]):.2f} | "
                  f"CRPS vs GHAP {c_raw:.3f} -> {c_cal:.3f}", flush=True)
    Path(a.dir, "calib.json").write_text(json.dumps(out, indent=1))


def apply(a):
    cal = json.loads(Path(a.calib).read_text())
    parts = [np.load(p) for p in sorted(glob.glob(str(Path(a.dir) / "part_*.npz")))]
    keys = np.concatenate([p["keys"] for p in parts]); ens = np.concatenate([p["st_ens"] for p in parts]).astype(np.float32)
    st = np.load(G / "stations_2022.npz"); obs = st["daily"]
    reg = region_of(st["lat"], st["lon"])
    print("lead | stations: CRPS raw -> calibrated | spread/skill raw -> calibrated")
    for h in range(1, 8):
        rows = np.where(keys[:, 1] == h)[0]
        cr, cc, ss_r, ss_c = [], [], [], []
        for n in rows:
            t = int(keys[n, 2]); y = obs[:, t]; E = ens[n]
            ok = np.isfinite(y) & np.isfinite(E).all(1) & (reg >= 0)
            if ok.sum() == 0:
                continue
            Ec = E.copy()
            for k_, name in enumerate(REGIONS):
                m = ok & (reg == k_)
                p = cal.get(f"{h}|{name}")
                if p is not None and m.any():
                    Ec[m] = transform(E[m], p["p"])
            cr.append(fair_crps(E[ok], y[ok]) * ok.sum()); cc.append(fair_crps(Ec[ok], y[ok]) * ok.sum())
            ss_r.append(((E[ok].var(1, ddof=1)).sum(), ((E[ok].mean(1) - y[ok]) ** 2).sum(), ok.sum()))
            ss_c.append(((Ec[ok].var(1, ddof=1)).sum(), ((Ec[ok].mean(1) - y[ok]) ** 2).sum(), ok.sum()))
        N = sum(s[2] for s in ss_r)
        sr = lambda S: np.sqrt(sum(s[0] for s in S) / N) / np.sqrt(sum(s[1] for s in S) / N)
        print(f"{h} | {sum(cr) / N:.2f} -> {sum(cc) / N:.2f} | {sr(ss_r):.2f} -> {sr(ss_c):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "apply"])
    ap.add_argument("dir")
    ap.add_argument("calib", nargs="?")
    ap.add_argument("--sample", type=int, default=400000, help="pixels per lead")
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(a)
    else:
        a.calib, a.dir = a.dir, a.calib
        apply(a)
