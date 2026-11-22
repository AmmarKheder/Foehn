#!/usr/bin/env python3
"""COPY of experiments/figures/tail_events.py (its 'evaluate' part), with the issuance of the baselines as a switch.
Nothing else is changed in the definitions: thresholds = GHAP 0.25 deg block means of 2017-2020 per cell and calendar day
(+-15 days), tiers P90 / P95 / P99, truth = GHAP block means of the target day, every system on the same land cells and
days, counts of hits / misses / false alarms / correct negatives per region, lead and tier.

  ISSUANCE=d   as submitted to Nature: CAMS run of 00 UTC of the START day d, lead day h (cams[d, 0, h]); Aurora
               initialised at 12 UTC of day d from the analyses of 00 and 12 UTC of day d (benchmark/aurora/out_global);
               baselines for leads 1-4.  With this switch the script must reproduce data/tail_events_<label>.npz exactly.
  ISSUANCE=d1  equal issuance, the indexing of compare_fair.py: CAMS run of 00 UTC of day d+1, lead day h-1
               (cams[d + 1, 0, h - 1]); Aurora initialised at 00 UTC of day d+1 from the analyses of 12 UTC d and 00 UTC
               d+1 (out_12h_all; daily mean of day d+h = 0.25 x 00 UTC + 0.5 x 12 UTC + 0.25 x 00 UTC of the next day,
               the first 00 UTC of lead 1 being the analysis); baselines for leads 1-5.

Added: counts per start day (CD) for the block bootstrap; DUMP=1 writes the fields on the scored cells
(fields_<ISSUANCE>_h<lead>.npy: truth, Foehn mean, one member, CAMS, Aurora, persistence; days x cells) for the
frequency-matched analysis (land_freqmatch.py).
    ISSUANCE=d1 DUMP=1 python tail_events_eqiss.py <label>=<run>[:<eval dir>] ...
    -> ../data/tail_events_<first label>_<ISSUANCE>.npz"""
import glob
import os
import sys
from pathlib import Path

import numpy as np

FIG = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures"))
G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
RUNS = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs"))
AURD = FIG.parent / "benchmark" / "aurora"
THR = G / "clim" / "tail_thresholds_2017_2020.npy"
SFX = f"_{os.environ['FOEHN_DESIGN']}" if os.environ.get("FOEHN_DESIGN") else ""   # A16 rebuild: design label (unset = unchanged)
OUT = Path(__file__).resolve().parent.parent / f"data{SFX}"
ISS = os.environ.get("ISSUANCE", "d1"); assert ISS in ("d", "d1")
HB = 4 if ISS == "d" else 5                                           # last lead with the baselines
DUMP = os.environ.get("DUMP", "0") == "1"
WINDOW, PCTS, TIERS = 15, (90, 95, 99), ("moderate (>P90)", "severe (>P95)", "extreme (>P99)")
NR, NC = 576, 1440


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
    print("issuance:", ISS, "| common forecasts:", len(common), flush=True)
    thr = np.load(THR, mmap_mode="r")
    cams = zarr.open((os.environ["FOEHN_WORK"] + "/eccv/data/world/cams_forecast_daily/2022.zarr"), mode="r")
    aur = {}
    if ISS == "d":                                                    # as submitted: analyses of day d (out_global)
        for p in glob.glob(str(AURD / "out_global" / "part_*.npz")):
            z = np.load(p); alat, alon = z["lat"], z["lon"]
            for d, pr in zip(z["days"], z["pred"]):
                aur[int(d)] = pr
    else:                                                             # equal issuance: initialised at 00 UTC of day d+1
        need = {d + 1 for d, h in common}
        for p in glob.glob(str(AURD / "out_12h_all" / "part_*.npz")):
            z = np.load(p); alat, alon = z["lat"], z["lon"]
            for d, pr in zip(z["days"], z["pred"]):
                if int(d) in need:
                    aur[int(d)] = pr.copy()                           # (10, 450, 900): 12, 24, ..., 120 h after 00 UTC
    print("Aurora initialisations loaded:", len(aur), flush=True)
    ab = lambda f: bil(np.asarray(f, np.float32), float(alat[0]), float(alat[0] - alat[1]), float(alon[0]), float(alon[1] - alon[0]))
    ccache = {}

    def cams_day(run_day):                                            # (5, 451, 900): the five lead days of one 00 UTC run
        if run_day not in ccache:
            if len(ccache) > 4:
                ccache.pop(next(iter(ccache)))
            ccache[run_day] = np.asarray(cams[run_day, 0], np.float32)
        return ccache[run_day]

    def analysis00(d):
        """CAMS analysis PM2.5 at 00 UTC of day d (Aurora's initial condition), ug m-3, on the rows of Aurora's grid"""
        z = np.load(AURD / "raw" / f"{d:03d}.npz")
        return (z["surf"][0, 5, :len(alat)] * 1e9).astype(np.float32)

    def have(d):
        return d in aur if ISS == "d" else ((d + 1) in aur and d + 1 <= 364)

    def baselines(d, h):
        if ISS == "d":
            return bil(cams_day(d)[h], 90.0, 0.4, 0.0, 0.4), ab(aur[d][h - 1])
        A = aur[d + 1]; k0 = 2 * (h - 1)                              # index of 24 (h - 1) + 12 h
        first = ab(analysis00(d + 1)) if h == 1 else ab(A[k0 - 1])
        return bil(cams_day(d + 1)[h - 1], 90.0, 0.4, 0.0, 0.4), 0.25 * first + 0.5 * ab(A[k0]) + 0.25 * ab(A[k0 + 1])

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
    days = sorted({d for d, h in common}); di = {d: i for i, d in enumerate(days)}
    C = np.zeros((len(REG), 7, 3, len(SYS), 4))                     # hits, misses, false alarms, correct negatives
    CD = np.zeros((len(days),) + C.shape)                             # the same per start day (block bootstrap)
    NOK = np.zeros((len(days), 7), np.int64)
    base = None; dumps = {}
    nr = len(runs)
    DSYS = ["truth (GHAP)", f"{runs[0][0]} (mean)", f"{runs[0][0]} (one member)", "CAMS", "Aurora", "persistence"]
    DK = [None, 0, nr, nr + 1, nr + 2, nr + 3]                        # index in X of each dumped system
    for c_, (d, h) in enumerate(common):
        t = d + h
        if t >= 365 or (h <= HB and not have(d)):
            continue
        X = [arr(idx[(d, h)][0])[0][idx[(d, h)][1]].astype(np.float32) for _, idx in runs]
        p0, n0 = runs[0][1][(d, h)]; m25, y, mem = arr(p0); y = y[n0].astype(np.float32)
        X.append(mem[n0].astype(np.float32) if mem is not None else np.full_like(y, np.nan))
        if h <= HB:
            cf, af = baselines(d, h); X += [cf, af]
        else:
            X += [np.full_like(y, np.nan), np.full_like(y, np.nan)]
        X.append(arr(pers[d][0])[1][pers[d][1]].astype(np.float32) if d in pers else np.full_like(y, np.nan))
        T = to_centres(np.asarray(thr[min(t, 364)], np.float32))     # (3, 576, 1440)
        use = list(range(nr + 1)) + ([nr + 1, nr + 2] if h <= HB else []) + [nr + 3]
        ok = np.isfinite(y) & (y > 0) & np.isfinite(T).all(0) & np.all([np.isfinite(X[k]) for k in use], 0)
        NOK[di[d], h - 1] = ok.sum()
        ii = np.flatnonzero(ok.ravel())                               # the same counts as the original, on the scored cells only
        yv = y.ravel()[ii]; Tv = T.reshape(3, -1)[:, ii]; rmv = [rm.ravel()[ii] for rm in RM]
        for k_t in range(3):
            obs = yv > Tv[k_t]
            for k in use:
                code = obs * 2 + (X[k].ravel()[ii] > Tv[k_t])        # 3 hit, 2 miss, 1 false alarm, 0 correct negative
                for g in range(len(RM)):
                    b = np.bincount(code[rmv[g]], minlength=4)
                    v = [b[3], b[2], b[1], b[0]]
                    C[g, h - 1, k_t, k] += v; CD[di[d], g, h - 1, k_t, k] += v
        if DUMP and ok.any():
            if base is None:
                base = ii.copy(); np.save(OUT / f"fields_{ISS}_cells.npy", base)
                for hh in range(1, 8):
                    mm = np.lib.format.open_memmap(OUT / f"fields_{ISS}_h{hh}.npy", mode="w+", dtype=np.float32,
                                                   shape=(len(DSYS), len(days), len(base)))
                    mm[:] = np.nan; dumps[hh] = mm
            assert np.array_equal(ii, base), f"the scored cells change at start day {d}, lead {h}"
            for s_, k in enumerate(DK):
                dumps[h][s_, di[d]] = yv if k is None else X[k].ravel()[ii]
        if c_ % 200 == 0:
            print(c_, "/", len(common), flush=True)
    for mm in dumps.values():
        mm.flush()
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
    f = OUT / f"tail_events_{runs[0][0]}_{ISS}.npz"
    np.savez(f, C=C, CD=CD, NOK=NOK, days=np.array(days), systems=np.array(SYS), tiers=np.array(TIERS), regions=np.array(list(REG)),
             issuance=np.array(ISS), dump_systems=np.array(DSYS))
    print("saved", f)


if __name__ == "__main__":
    evaluate(sys.argv[1:])
