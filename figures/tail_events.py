#!/usr/bin/env python3
"""Local pollution extremes (tail events), after the TailWeather labelling method (pm25_labeling_global.py: percentile
climatology per cell and day of year, +-15-day window; exceedance of the local P90 = event; severity tiers moderate /
severe / extreme at intensity scores 0.90 / 0.95 / 0.99). With that method's 99-point CDF, the tiers are exactly the
exceedances of the local P90, P95 and P99, so only these three thresholds are stored. Persistence filter: none (the
method's default, min-days = 1).
  thresholds: GHAP 0.25 deg block means (pm.npy) of the training years 2017-2020 only (the test year never enters),
              -> data/world/global1km/clim/tail_thresholds_2017_2020.npy (365, 3, 576, 1440) float16, on pm.npy points
  evaluate:   2022 forecasts against the GHAP block means of the target day, every system on the same land cells and
              days (Foehn runs: ensemble mean and one member, map25 / map25_m0 of the evaluations; CAMS, Aurora: leads
              1-4, bilinear; persistence: GHAP of day d). Per lead, tier and region: hits, misses, false alarms, correct
              negatives -> POD, FAR, CSI, frequency bias and SEDI (Ferro & Stephenson 2011, base-rate independent).
    python tail_events.py thresholds
    python tail_events.py evaluate <label>=<run>[:<eval dir>] ...   -> data/tail_events_<first label>.npz"""
import glob
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
RUNS = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/runs")
AUR = HERE.parent / "benchmark" / "aurora" / "out_global"
THR = G / "clim" / "tail_thresholds_2017_2020.npy"
WINDOW, PCTS, TIERS = 15, (90, 95, 99), ("moderate (>P90)", "severe (>P95)", "extreme (>P99)")
NR, NC = 576, 1440


def npy_read(path, i):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); sh, _, dt = np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    n = int(np.prod(sh[1:])); fd = os.open(path, os.O_RDONLY)
    b = os.pread(fd, n * dt.itemsize, hdr + i * n * dt.itemsize); os.close(fd)
    return np.frombuffer(b, dt).reshape(sh[1:])


def nanpct(x, qs):
    """np.nanpercentile(x, qs, axis=0) (linear method), vectorised: sort with NaN last, interpolate between ranks"""
    xs = np.sort(x, axis=0)
    n = np.isfinite(x).sum(0)
    out = []
    for q in qs:
        pos = q / 100.0 * np.maximum(n - 1, 0)
        lo = np.floor(pos).astype(np.int64); hi = np.minimum(lo + 1, np.maximum(n - 1, 0)); w = (pos - lo).astype(np.float32)
        vlo = np.take_along_axis(xs, lo[None], 0)[0]; vhi = np.take_along_axis(xs, hi[None], 0)[0]
        out.append(vlo + w * (vhi - vlo))
    res = np.stack(out); res[:, n == 0] = np.nan
    return res


def thresholds():
    doy = np.load(G / "meta.npz")["doy"][:1461]                      # 2017-2020
    H = np.stack([npy_read(G / "pm.npy", i).astype(np.float16) for i in range(1461)])
    out = np.lib.format.open_memmap(THR, mode="w+", dtype=np.float16, shape=(365, 3, NR, NC))
    for t in range(365):
        dd = np.abs(doy - t); dd = np.minimum(dd, 365 - dd)
        x = H[np.where(dd <= WINDOW)[0]].astype(np.float32)          # ~124 values per cell
        x[~(x > 0)] = np.nan
        q = nanpct(x, PCTS)                                          # (3, 576, 1440)
        q[:, np.isfinite(x).sum(0) < 60] = np.nan                    # too few valid values: no threshold
        out[t] = q.astype(np.float16)
        if t % 30 == 0:
            print("day", t, flush=True)
    out.flush(); print("saved", THR)


def to_centres(a):                                                    # pm.npy points (84 - 0.25 k) -> map25 block centres
    b = np.concatenate([a, a[..., -1:, :]], -2); b = 0.5 * (b[..., :-1, :] + b[..., 1:, :])
    c = np.concatenate([b, b[..., :1]], -1); return 0.5 * (c[..., :-1] + c[..., 1:])


def evaluate(args):
    import zarr
    LATC = 84.0 - 0.25 * np.arange(NR) - 0.125; LONC = -180.0 + 0.25 * np.arange(NC) + 0.125
    LA2, LO2 = np.meshgrid(LATC, LONC, indexing="ij")
    REG = {"Global": None, "East Asia": (54, 18, 97, 146), "South Asia": (37, 5, 60, 97), "Europe": (72, 35, -25, 45),
           "North America": (72, 15, -170, -50), "Africa": (37, -35, -20, 52)}
    RM = [np.ones((NR, NC), bool) if b is None else (LA2 <= b[0]) & (LA2 > b[1]) & (LO2 >= b[2]) & (LO2 < b[3]) for b in REG.values()]

    def bil(field, lat0, dlat, lon0, dlon):
        nlat, nlon = field.shape
        fi = (lat0 - LA2) / dlat; fj = ((LO2 - lon0) % 360.0) / dlon
        i0 = np.clip(np.floor(fi).astype(int), 0, nlat - 2); j0 = np.floor(fj).astype(int) % nlon; j1 = (j0 + 1) % nlon
        wi, wj = np.clip(fi - i0, 0, 1), fj - np.floor(fj)
        return ((1 - wi) * ((1 - wj) * field[i0, j0] + wj * field[i0, j1]) + wi * ((1 - wj) * field[i0 + 1, j0] + wj * field[i0 + 1, j1])).astype(np.float32)

    runs = []
    for arg in args:
        label, spec = arg.split("="); run, ev = (spec.split(":") + ["eval_global"])[:2]
        idx = {}
        for p in sorted(glob.glob(str(RUNS / run / ev / "part_*.npz"))):
            for n, (d, h, t) in enumerate(np.load(p)["keys"]):
                idx[(int(d), int(h))] = (p, n)
        runs.append((label, idx))
    common = sorted(set.intersection(*[set(r[1]) for r in runs]))
    print("common forecasts:", len(common), flush=True)
    thr = np.load(THR, mmap_mode="r")
    cams = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
    aur = {}
    for p in glob.glob(str(AUR / "part_*.npz")):
        z = np.load(p); alat, alon = z["lat"], z["lon"]
        for d, pr in zip(z["days"], z["pred"]):
            aur[int(d)] = pr
    cache = {}

    def arr(path):
        if path not in cache:
            if len(cache) > 12:
                cache.pop(next(iter(cache)))
            z = np.load(path); cache[path] = (z["map25"], z["map25_ghap"], z["map25_m0"] if "map25_m0" in z.files else None)
        return cache[path]
    pers = {}
    for (d, h), v in runs[0][1].items():
        pers.setdefault(d + h, v)
    SYS = [f"{l} (mean)" for l, _ in runs] + [f"{runs[0][0]} (one member)", "CAMS", "Aurora", "persistence"]
    C = np.zeros((len(REG), 7, 3, len(SYS), 4))                     # hits, misses, false alarms, correct negatives
    for c_, (d, h) in enumerate(common):
        t = d + h
        if t >= 365 or (h <= 4 and d not in aur):
            continue
        X = [arr(idx[(d, h)][0])[0][idx[(d, h)][1]].astype(np.float32) for _, idx in runs]
        p0, n0 = runs[0][1][(d, h)]; m25, y, mem = arr(p0); y = y[n0].astype(np.float32)
        X.append(mem[n0].astype(np.float32) if mem is not None else np.full_like(y, np.nan))
        X.append(bil(np.asarray(cams[d, 0, h], np.float32), 90.0, 0.4, 0.0, 0.4) if h <= 4 else np.full_like(y, np.nan))
        X.append(bil(aur[d][h - 1].astype(np.float32), float(alat[0]), float(alat[0] - alat[1]), float(alon[0]), float(alon[1] - alon[0]))
                 if h <= 4 else np.full_like(y, np.nan))
        X.append(arr(pers[d][0])[1][pers[d][1]].astype(np.float32) if d in pers else np.full_like(y, np.nan))
        T = to_centres(np.asarray(thr[min(t, 364)], np.float32))     # (3, 576, 1440)
        nr = len(runs)
        use = list(range(nr + 1)) + ([nr + 1, nr + 2] if h <= 4 else []) + [nr + 3]
        ok = np.isfinite(y) & (y > 0) & np.isfinite(T).all(0) & np.all([np.isfinite(X[k]) for k in use], 0)
        for k_t in range(3):
            obs = y > T[k_t]
            for k in use:
                fc = X[k] > T[k_t]
                for g, rm in enumerate(RM):
                    m = ok & rm
                    C[g, h - 1, k_t, k] += [(fc & obs & m).sum(), (~fc & obs & m).sum(), (fc & ~obs & m).sum(), (~fc & ~obs & m).sum()]
        if c_ % 200 == 0:
            print(c_, "/", len(common), flush=True)
    for g, rn in enumerate(REG):
        print(f"\n=== {rn}: local tail events of 2022 (thresholds GHAP 2017-2020) — POD / FAR / CSI / bias / SEDI")
        for k_t, tn in enumerate(TIERS):
            print(f"  {tn}")
            for h in range(1, 8):
                row = []
                for k, sn in enumerate(SYS):
                    a, b, c, dn = C[g, h - 1, k_t, k]
                    if a + b + c + dn == 0:
                        continue
                    H_ = a / max(a + b, 1); F_ = c / max(c + dn, 1)
                    H_, F_ = np.clip(H_, 1e-6, 1 - 1e-6), np.clip(F_, 1e-6, 1 - 1e-6)
                    sedi = (np.log(F_) - np.log(H_) - np.log(1 - F_) + np.log(1 - H_)) / (np.log(F_) + np.log(H_) + np.log(1 - F_) + np.log(1 - H_))
                    row.append(f"{sn}: {a/max(a+b,1):.2f}/{c/max(a+c,1):.2f}/{a/max(a+b+c,1):.2f}/{(a+c)/max(a+b,1):.2f}/{sedi:.2f}")
                print(f"    J+{h}  " + " | ".join(row))
    np.savez(HERE / "data" / f"tail_events_{runs[0][0]}.npz", C=C, systems=np.array(SYS), tiers=np.array(TIERS), regions=np.array(list(REG)))
    print("saved")


if __name__ == "__main__":
    if sys.argv[1] == "thresholds":
        thresholds()
    else:
        evaluate(sys.argv[2:])
