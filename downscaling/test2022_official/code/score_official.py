#!/usr/bin/env python3
"""Amendment A7 (protocol_amendments/A7_official_stations_2022.md), steps (e)-(g): the SCORING process (world venv, numpy
+ zarr; no model code). The only process that reads the station observations and GHAP 2022 together with the
predictions; evaluation only, nothing is fitted or selected.

Station set: the 6,248 official stations of test2022_official/pos/st_meta_official.npz (six networks, >= 30 valid days,
pixel in the downscaler's mask), observations eccv/data/world/global1km/stations_2022.npz (the paper's file).
Tables (identical station-days inside a table and a 'where'):
  S  same-day mode, 365 days: ds_sameday, cams0_bil, bcsd, clim_ens, clim_med, clim_paper (pixel only), ghap
  C  forecast mode, 214 issue days x lead days 1-5: chain, chain_t6fc, chain_rtseed, coarse, camsfc, aurora, camsfc_ds,
     bcsd_fc, persist_ds, and at the valid day ds_sameday, cams0_bil, bcsd, clim_ens, clim_med, clim_paper, ghap
  F  table C on the 107 issue days that have a Foehn forecast, plus foehn (pixel only)
Scores, groups, contrasts and the 7-day moving-block bootstrap over start dates: A7(f). Paper systems read as
figures/compare_fair.py reads them (bilinear at the station coordinates, equal issuance).
Checks of the reading (A7(g)): --paper-check recomputes the paper's station comparison with these loaders.
    python -I score_official.py [--selftest] [--no-paper-check] [--procs 8]"""
import argparse
import datetime as dt
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
O = ROOT / "test2022_official"
E = Path("/scratch/project_462001804/ammar/eccv")
G = E / "data/world/global1km"
FIG = E / "cran-pm/experiments/figures"
G1 = E / "cran-pm/experiments/global1km"
AURD = E / "cran-pm/experiments/benchmark/aurora"
FOEHN_EVAL = G1 / "runs/green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4/eval_final_crps_e01"
PRED04 = G1 / "runs/cams04_flow_shift2.6_emis/cams12h"
CAMS_ZARR = str(E / "data/world/cams_forecast_daily/2022.zarr")
NETS = {0: "Europe (EEA)", 1: "China (CNEMC)", 3: "USA (EPA)", 4: "Korea (AirKorea)", 5: "Japan (NIES)", 6: "India (CPCB)"}
GROUPS = list(NETS.values()) + ["All six networks"]
Y0 = dt.date(2022, 1, 1)
I0, NI, NL = 120, 214, 5
WHERE = ("pixel", "ring1km", "ring3km", "ring10km")
NBOOT, BLOCK = 10000, 7
QN = ("n", "se", "e", "crps", "var", "below", "above")
T00 = time.time()


def log(*a):
    print(f"[{time.time() - T00:6.0f} s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ small tools
def bil(field, lat0, dlat, lon0, dlon, lat, lon):
    """compare_fair.bil: bilinear on a regular grid (row 0 at lat0, descending), periodic in longitude"""
    nlat, nlon = field.shape[-2:]
    fi = (lat0 - lat) / dlat
    fj = ((lon - lon0) % 360.0) / dlon
    i0 = np.clip(np.floor(fi).astype(int), 0, nlat - 2)
    j0 = np.floor(fj).astype(int) % nlon
    j1 = (j0 + 1) % nlon
    wi, wj = np.clip(fi - i0, 0, 1), fj - np.floor(fj)
    return ((1 - wi) * ((1 - wj) * field[..., i0, j0] + wj * field[..., i0, j1])
            + wi * ((1 - wj) * field[..., i0 + 1, j0] + wj * field[..., i0 + 1, j1])).astype(np.float32)


def reduce_ens(X, y):
    """X (..., M, S) members, y (..., S) -> dict(mean, crps (fair, Ferro 2014; |x - y| for M = 1), var (unbiased), M)"""
    X = np.asarray(X, np.float32)
    M = X.shape[-2]
    mean = X.mean(-2)
    if M == 1:
        z = np.zeros_like(mean)
        return dict(mean=mean, crps=np.abs(mean - y), var=z, below=z, above=z, M=1)
    e = np.abs(X - y[..., None, :]).mean(-2)
    xs = np.sort(X, axis=-2)
    w = (2.0 * np.arange(M, dtype=np.float32) - M + 1.0)[:, None]
    crps = e - 2.0 * (w * xs).sum(-2) / (2.0 * M * (M - 1))
    lo, hi = np.take(xs, 0, -2), np.take(xs, M - 1, -2)
    return dict(mean=mean, crps=crps.astype(np.float32), var=X.var(-2, ddof=1), below=(y < lo).astype(np.float32),
                above=(y > hi).astype(np.float32), M=M)


def det(x, y):
    x = np.asarray(x, np.float32)
    z = np.zeros_like(x)
    return dict(mean=x, crps=np.abs(x - y), var=z, below=z, above=z, M=1)


def take(arm, idx):
    """same-day arm (365, S) -> forecast layout (U, K, S) at the valid days idx (U, K)"""
    return {q: (arm[q][idx] if q != "M" else arm[q]) for q in arm}


def block_idx(nd, seed=0):
    """compare_fair.py l.166-169: 7-day moving blocks over the start dates"""
    rng = np.random.default_rng(seed)
    return (rng.integers(0, nd - BLOCK + 1, size=(NBOOT, -(-nd // BLOCK)))[:, :, None] + np.arange(BLOCK)).reshape(NBOOT, -1)[:, :nd]


# ------------------------------------------------------------------------------------------------ table scoring
class Table:
    """arms: name -> dict(mean, crps, var (U, K, S), M); y (U, K, S); gmask (G, S) bool; ref: arm of the 'share better' column"""

    def __init__(self, name, arms, y, gmask, ref, min_days, lead_names, note):
        self.name, self.arms, self.y, self.gmask, self.ref, self.min_days = name, arms, y, gmask, ref, min_days
        self.leads, self.note = lead_names, note
        ok = np.isfinite(y)
        for a in arms.values():
            ok &= np.isfinite(a["mean"]) & np.isfinite(a["crps"]) & np.isfinite(a["var"])
        self.ok = ok
        U, K, S = y.shape
        self.units = np.where(ok.any((1, 2)))[0]                       # start dates with data
        self.idx = block_idx(len(self.units)) if len(self.units) >= BLOCK else None
        self.sums = {n: self._sums(a, ok) for n, a in arms.items()}
        self.st = {n: self._station(a, ok) for n, a in arms.items()}

    def _sums(self, a, ok):
        """(G, U, K, 7): n, sum e2, sum e, sum crps, sum var, number of observations below the lowest / above the highest
        member, over the stations of each group"""
        e = np.where(ok, a["mean"] - self.y, 0.0).astype(np.float64)
        q = [ok.astype(np.float64), e * e, e, np.where(ok, a["crps"], 0.0).astype(np.float64),
             np.where(ok, a["var"], 0.0).astype(np.float64), np.where(ok, a["below"], 0.0).astype(np.float64),
             np.where(ok, a["above"], 0.0).astype(np.float64)]
        out = np.zeros((len(self.gmask),) + ok.shape[:2] + (len(q),))
        for j, x in enumerate(q):
            for g, m in enumerate(self.gmask):
                out[g, :, :, j] = x[:, :, m].sum(2)
        return out

    def _station(self, a, ok):
        """per lead and station over the start dates: n, rmse, r, amplitude ratio (K, S)"""
        x = np.where(ok, a["mean"], 0.0).astype(np.float64)
        y = np.where(ok, self.y, 0.0).astype(np.float64)
        n = ok.sum(0).astype(np.float64)
        with np.errstate(all="ignore"):
            sx, sy = x.sum(0), y.sum(0)
            vx = (x * x).sum(0) / n - (sx / n) ** 2
            vy = (y * y).sum(0) / n - (sy / n) ** 2
            cov = (x * y).sum(0) / n - sx * sy / n ** 2
            r = cov / np.sqrt(vx * vy)
            amp = np.sqrt(vx / vy)
            rmse = np.sqrt(((x - y) ** 2).sum(0) / n)
        return dict(n=n, rmse=rmse, r=r, amp=amp)

    def arm_scores(self, name):
        s, st, M = self.sums[name], self.st[name], self.arms[name]["M"]
        ref = self.st.get(self.ref)
        out = {}
        for g, gn in enumerate(GROUPS):
            out[gn] = {}
            for k, ln in enumerate(self.leads + (["all"] if len(self.leads) > 1 else [])):
                q = s[g, :, k].sum(0) if k < len(self.leads) else s[g].sum((0, 1))
                n = q[0]
                if n == 0:
                    out[gn][ln] = None
                    continue
                rmse = float(np.sqrt(q[1] / n))
                r = dict(n=int(n), rmse=rmse, bias=float(q[2] / n), crps=float(q[3] / n), M=M,
                         spread_skill=float(np.sqrt((M + 1) / M * q[4] / n) / rmse) if M > 1 else None)
                if M > 1:                                                      # observations inside the member range
                    r.update(share_obs_below_lowest_member=float(q[5] / n), share_obs_above_highest_member=float(q[6] / n),
                             coverage_member_range=float(1 - (q[5] + q[6]) / n), coverage_expected_if_exchangeable=(M - 1) / (M + 1))
                if k < len(self.leads):
                    good = self.gmask[g] & (st["n"][k] >= self.min_days)
                    r["n_stations_ge_min_days"] = int(good.sum())
                    if good.sum() >= 5:
                        rr, aa = st["r"][k][good], st["amp"][k][good]
                        r["median_station_r"] = float(np.nanmedian(rr))
                        r["median_station_amp"] = float(np.nanmedian(aa))
                        if ref is not None:
                            r[f"share_stations_rmse_below_{self.ref}"] = float((st["rmse"][k][good] < ref["rmse"][k][good]).mean())
                out[gn][ln] = r
        return out

    def contrast(self, A, B, sums_a=None, sums_b=None):
        """relative RMSE and CRPS change A / B - 1 with the moving-block bootstrap; identical station-days by construction"""
        sa = self.sums[A] if sums_a is None else sums_a
        sb = self.sums[B] if sums_b is None else sums_b
        out = {}
        for g, gn in enumerate(GROUPS):
            out[gn] = {}
            for k, ln in enumerate(self.leads + (["all"] if len(self.leads) > 1 else [])):
                a = sa[g, :, k] if k < len(self.leads) else sa[g].sum(1)
                b = sb[g, :, k] if k < len(self.leads) else sb[g].sum(1)
                a, b = a[self.units], b[self.units]
                n = a[:, 0].sum()
                if n == 0 or b[:, 1].sum() == 0:
                    out[gn][ln] = None
                    continue
                assert np.array_equal(a[:, 0], b[:, 0])
                r = dict(n=int(n), rmse_A=float(np.sqrt(a[:, 1].sum() / n)), rmse_B=float(np.sqrt(b[:, 1].sum() / n)),
                         crps_A=float(a[:, 3].sum() / n), crps_B=float(b[:, 3].sum() / n),
                         bias_A=float(a[:, 2].sum() / n), bias_B=float(b[:, 2].sum() / n))
                r["rel_rmse"] = float(np.sqrt(a[:, 1].sum() / b[:, 1].sum()) - 1)
                r["rel_crps"] = float(a[:, 3].sum() / b[:, 3].sum() - 1)
                if self.idx is not None:
                    ix = self.idx
                    with np.errstate(all="ignore"):
                        bs = np.sqrt(a[ix, 1].sum(1) / b[ix, 1].sum(1)) - 1
                        bc = a[ix, 3].sum(1) / b[ix, 3].sum(1) - 1
                    r["rel_rmse_ci"] = [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]
                    r["rel_crps_ci"] = [float(x) for x in np.nanpercentile(bc, [2.5, 97.5])]
                out[gn][ln] = r
        return out

    def contrast_submask(self, A, B_arm, bname):
        """A against an arm outside the table (own common days: the table's days where B is finite too)"""
        ok = self.ok & np.isfinite(B_arm["mean"])
        return dict(note=f"own common station-days: those of the table where {bname} is finite",
                    groups=self.contrast(A, bname, self._sums(self.arms[A], ok), self._sums(B_arm, ok)))

    def result(self, contrasts):
        res = {"table": self.name, "note": self.note, "arms_in_common_mask": list(self.arms), "n_start_dates": int(len(self.units)),
               "n_station_days": int(self.ok.sum()), "n_stations": int(self.ok.any((0, 1)).sum()),
               "min_common_days_for_station_statistics": self.min_days,
               "arms": {n: self.arm_scores(n) for n in self.arms}, "contrasts": {}}
        for A, B in contrasts:
            if A in self.arms and B in self.arms:
                res["contrasts"][f"{A} - {B}"] = self.contrast(A, B)
        return res


# ------------------------------------------------------------------------------------------------ data
class Data:
    def __init__(self, a):
        self.a = a
        st = np.load(G / "stations_2022.npz")
        self.LAT, self.LON, self.SRC = st["lat"], st["lon"], st["source"]
        self.OBS = st["daily"].astype(np.float32)                              # (8110, 365)
        meta = np.load(O / "pos/st_meta_official.npz")
        self.id, self.pix, self.src = meta["id"], meta["pix"], meta["src"]
        self.lat, self.lon = self.LAT[self.id], self.LON[self.id]
        assert np.array_equal(self.src, self.SRC[self.id])
        self.S = self.id.size
        self.obs = self.OBS[self.id].T.copy()                                  # (365, S)
        self.CLIM = np.load(G / "clim" / "st_clim_2022.npy").astype(np.float32)    # (365, 8110)
        self.gmask = np.stack([self.src == k for k in NETS] + [np.ones(self.S, bool)])
        self.vidx = I0 + np.arange(NI)[:, None] + np.arange(NL)[None]          # valid day of (issue, lead)
        import zarr
        self.cams = zarr.open(CAMS_ZARR, mode="r")
        self.an0 = {}
        self.fake = False

    def analysis00(self, d):
        """compare_fair.analysis00: CAMS analysis PM2.5 at 00 UTC of day d at all 8110 stations"""
        if d not in self.an0:
            z = np.load(AURD / "raw" / f"{d:03d}.npz")
            self.an0[d] = bil(z["surf"][0, 5] * 1e9, float(z["lat"][0]), float(z["lat"][0] - z["lat"][1]), float(z["lon"][0]),
                              float(z["lon"][1] - z["lon"][0]), self.LAT, self.LON)
        return self.an0[d]

    def load_aurora(self):
        self.aur_e = {}
        for p in sorted(glob.glob(str(AURD / "out_12h_all" / "part_*.npz"))):
            z = np.load(p)
            la_, lo_ = z["lat"], z["lon"]
            for d, pr in zip(z["days"], z["pred"]):
                self.aur_e[int(d)] = bil(pr.astype(np.float32), float(la_[0]), float(la_[0] - la_[1]), float(lo_[0]),
                                         float(lo_[1] - lo_[0]), self.LAT, self.LON)          # (10, 8110)
        log(f"Aurora (00 UTC runs, out_12h_all): {len(self.aur_e)} runs")

    def aurora_daily(self, I, L):
        """compare_fair.py l.134-137 with d + 1 = I, h = L"""
        A = self.aur_e[I]
        k0 = 2 * (L - 1)
        first = self.analysis00(I) if L == 1 else A[k0 - 1]
        return 0.25 * first + 0.5 * A[k0] + 0.25 * A[k0 + 1]

    def cams_daily(self, I):
        """CAMS forecast of the 00 UTC run of day I at all stations, lead days 0..4 -> (5, 8110)"""
        return bil(np.asarray(self.cams[I, 0], np.float32), 90.0, 0.4, 0.0, 0.4, self.LAT, self.LON)

    def load_foehn(self):
        """st_ens (8110, 4) of every forecast (start day d, lead h), rows aligned with stations_2022.npz"""
        self.foehn, self.foehn_parts, self.GH = {}, {}, {}
        for p in sorted(glob.glob(str(FOEHN_EVAL / "part_*.npz"))):
            z = np.load(p)
            e, gh = z["st_ens"], z["st_ghap"]
            assert e.shape[1] == self.LAT.size
            for n, (d, h, t) in enumerate(z["keys"]):
                self.foehn[(int(d), int(h))] = e[n].astype(np.float32)
                self.foehn_parts[(int(d), int(h))] = (p, n)
                if int(t) not in self.GH:
                    self.GH[int(t)] = gh[n].astype(np.float32)
        log(f"Foehn 1 km (eval_final_crps_e01): {len(self.foehn)} forecasts, start days {len({d for d, h in self.foehn})}")


# ------------------------------------------------------------------------------------------------ checks
def paper_check(D):
    """A7(g)-1: compare_fair.py recomputed with these loaders (Foehn 1 km, CAMS and Aurora at equal issuance, leads 1-4) on
    the paper's own start days and mask (its ten systems finite, with its coordinate-key station mapping)."""
    ref = np.load(FIG / "data" / "compare_fair_crps_e01.npz")
    key = lambda la, lo: np.round(la, 4) * 1e6 + np.round(lo, 4)                # noqa: E731
    refk = {v: i for i, v in enumerate(key(D.LAT, D.LON))}
    sets = {}
    for f in sorted(G.glob("stations_2022*.npz")):
        z = np.load(f)
        sets[len(z["lat"])] = np.array([refk.get(v, -1) for v in key(z["lat"], z["lon"])])
    idx = sets[D.LAT.size]
    NS = D.LAT.size
    official = np.isin(D.SRC, list(NETS))
    gid = np.array([list(NETS).index(s) if s in NETS else -1 for s in D.SRC])
    cache = {}

    def coarse(d, h, res):                                                    # compare_fair.coarse, verbatim logic
        p, n = D.foehn_parts[(d, h)]
        if p not in cache:
            if len(cache) > 6:
                cache.pop(next(iter(cache)))
            cache[p] = np.load(p)["map25"]
        m = cache[p][n].astype(np.float32)
        if res == 0.5:
            with np.errstate(all="ignore"):
                m = np.nanmean(m.reshape(288, 2, 720, 2), (1, 3))
        lat0, dl = (84.0 - 0.125, 0.25) if res == 0.25 else (84.0 - 0.25, 0.5)
        ok = np.isfinite(m).astype(np.float32)
        with np.errstate(all="ignore"):
            return bil(np.nan_to_num(m) * ok, lat0, dl, -180.0 + dl / 2, dl, D.LAT, D.LON) / bil(ok, lat0, dl, -180.0 + dl / 2, dl, D.LAT, D.LON)

    aur_d_days = set()
    for p in glob.glob(str(AURD / "out_global" / "part_*.npz")):
        aur_d_days.update(int(x) for x in np.load(p)["days"])
    SYS = (0, 4, 6)                                                           # Foehn 1 km, CAMS d+1, Aurora 00 UTC d+1
    acc = np.zeros((len(NETS) + 1, 4, 3, 3))                                  # n, sum e, sum e2
    common = sorted(sorted(D.foehn), key=lambda dh: D.foehn_parts[dh])         # as the paper: order irrelevant for sums
    GHk = {}                                                                  # the paper's GH: via the key mapping
    for t, g in D.GH.items():
        g_ = np.full(NS, np.nan, np.float32)
        g_[idx[idx >= 0]] = g[idx >= 0]
        GHk[t] = g_
    camsc = {}
    for (d, h) in common:
        if h > 4 or d + h >= 365 or d + 1 > 364 or d not in aur_d_days or (d + 1) not in D.aur_e or d not in GHk:
            continue
        t = d + h
        y = D.OBS[:, t]
        fo = np.full(NS, np.nan, np.float32)
        fo[idx[idx >= 0]] = D.foehn[(d, h)][idx >= 0].mean(-1)
        if d + 1 not in camsc:
            camsc[d + 1] = D.cams_daily(d + 1)
        X = np.stack([fo, coarse(d, h, 0.25), coarse(d, h, 0.5), camsc[d + 1][h - 1], D.aurora_daily(d + 1, h), D.OBS[:, d],
                      GHk[d], D.CLIM[t]])
        ok = official & np.isfinite(y) & np.all(np.isfinite(X), 0)
        for j, k in enumerate((0, 3, 4)):
            e = X[k] - y
            for g in range(len(NETS) + 1):
                m = ok & ((gid == g) if g < len(NETS) else True)
                acc[g, h - 1, j] += [m.sum(), e[m].sum(), (e[m] ** 2).sum()]
    out = {"file": str(FIG / "data" / "compare_fair_crps_e01.npz"), "systems": ["Foehn (1 km)", "CAMS (run d+1, equal issuance)",
           "Aurora (00 UTC d+1, equal issuance)"], "rows": [], "max_abs_rmse_diff": 0.0, "n_equal": True,
           "official_stations_without_foehn_value_in_the_paper_mapping": int((official & (np.isfinite(D.OBS).sum(1) >= 30)
                                                                               & ~np.isin(np.arange(NS), idx[idx >= 0])).sum())}
    for g in range(len(NETS) + 1):
        for h in range(4):
            for j, k in enumerate(SYS):
                mine = float(np.sqrt(acc[g, h, j, 2] / acc[g, h, j, 0]))
                theirs = float(np.sqrt(ref["S"][g, h, k, 2] / ref["S"][g, h, k, 0]))
                out["max_abs_rmse_diff"] = max(out["max_abs_rmse_diff"], abs(mine - theirs))
                out["n_equal"] &= bool(int(acc[g, h, j, 0]) == int(ref["S"][g, h, k, 0]))
                if g == len(NETS):
                    out["rows"].append(dict(lead=h + 1, system=out["systems"][j], rmse_recomputed=mine, rmse_file=theirs,
                                            n_recomputed=int(acc[g, h, j, 0]), n_file=int(ref["S"][g, h, k, 0])))
    out["pass"] = bool(out["max_abs_rmse_diff"] < 1e-3 and out["n_equal"])
    log(f"paper check: max |RMSE diff| {out['max_abs_rmse_diff']:.2e}, counts equal {out['n_equal']} -> pass {out['pass']}")
    for r in out["rows"][:3]:
        log("   ", r)
    return out


def coarse_fields(i):
    """the eight 0.4 deg members' daily means of lead days 1..5 of issue index i (A7(d); prep.build_pm) -> (5, 8, 361, 900)"""
    pred = np.load(PRED04 / "pred.npy", mmap_mode="r")
    an = np.load(O / "inputs/cams_an_2022.npy", mmap_mode="r")
    P = np.asarray(pred[i], np.float32)
    an0 = np.asarray(an[I0 + i - 119, 0], np.float32)
    inst = lambda h: np.repeat(an0[None], 8, 0) if h == 0 else P[:, h // 12 - 1]   # noqa: E731
    return np.stack([np.maximum(0.25 * inst(24 * (L - 1)) + 0.5 * inst(24 * L - 12) + 0.25 * inst(24 * L), 0.0) for L in range(1, NL + 1)])


def cams12h_check(D):
    """A7(g)-2: the station RMSE of the 0.4 deg model in figures/data/cams12h_stations.npz (eval_cams12h_stations.py: day
    d + n from the instants 24n, 24n + 12, 24n + 24 h = lead day n + 1 here), recomputed from pred.npy with this reader on
    that script's station-days (observation of the valid day and of d - 1 finite; its Aurora and CAMS fields are global)."""
    ref = np.load(FIG / "data" / "cams12h_stations.npz")
    keep = np.isin(D.SRC, list(NETS)) & (D.LAT < 84.0) & (D.LAT > -60.0)
    lat, lon, obs = D.LAT[keep], D.LON[keep], D.OBS[keep]
    acc = np.zeros((4, 2))
    for i in range(NI):
        d = I0 + i
        F5 = coarse_fields(i)                                                  # (5, 8, 361, 900)
        for n in range(1, 5):
            if d + n >= 365:
                continue
            em = bil(F5[n], 84.0, 0.4, -180.0, 0.4, lat, lon).mean(0)         # lead day n + 1
            y = obs[:, d + n]
            ok = np.isfinite(y) & np.isfinite(em) & np.isfinite(obs[:, d - 1])
            acc[n - 1] += [((em - y)[ok] ** 2).sum(), ok.sum()]
    rows, mx = [], 0.0
    for n in range(4):
        mine = float(np.sqrt(acc[n, 0] / acc[n, 1]))
        theirs = float(np.sqrt(ref["acc"][0, n, 6, 0] / ref["acc"][0, n, 6, 3]))
        mx = max(mx, abs(mine - theirs))
        rows.append(dict(paper_lead=f"J+{n + 1}", lead_day_here=n + 2, rmse_recomputed=mine, rmse_file=theirs,
                         n_recomputed=int(acc[n, 1]), n_file=int(ref["acc"][0, n, 6, 3])))
    out = {"file": str(FIG / "data" / "cams12h_stations.npz"), "rows": rows, "max_abs_rmse_diff": mx,
           "pass": bool(mx < 1e-2 and all(r["n_recomputed"] == r["n_file"] for r in rows))}
    log(f"cams12h check: max |RMSE diff| {mx:.2e} -> pass {out['pass']}; {rows[0]}")
    return out


def frozen_check(D, sd_pix):
    """A7(g)-4: same-day RMSE of the five shared networks in the frozen test's pixel-station convention (monitors of a
    network sharing a pixel averaged per day; pooled over pixel-days) against test2022_ghap/results/st_monitor."""
    ref = json.loads((ROOT / "test2022_ghap/results/st_monitor/scores_st_2022.json").read_text())
    names = {1: "CNEMC", 3: "EPA_AQS", 4: "AirKorea", 5: "Japan_NIES", 6: "India_CAAQMS"}
    arms = {"cdffm": "sameday_ds", "cams_bilinear": "sameday_cams0", "bcsd": "sameday_bcsd"}
    rows, mx = [], 0.0
    for k, nm in names.items():
        m = D.src == k
        px, inv = np.unique(D.pix[m], return_inverse=True)
        o = D.obs[:, m].astype(np.float64)                                    # (365, stations of the network)
        fin = np.isfinite(o)
        s = np.zeros((365, px.size))
        c = np.zeros((365, px.size))
        for j in range(o.shape[1]):
            s[:, inv[j]] += np.where(fin[:, j], o[:, j], 0.0)
            c[:, inv[j]] += fin[:, j]
        T = np.where(c > 0, s / np.maximum(c, 1), np.nan)                     # (365, pixel-stations)
        for theirs, mine in arms.items():
            x = np.maximum(sd_pix[mine][:, :, px].astype(np.float32), 0.0).mean(1)       # (365, px) ensemble mean at the pixel
            ok = np.isfinite(T) & np.isfinite(x)
            rm = float(np.sqrt(((x - T)[ok] ** 2).mean()))
            rf = ref["arms"][theirs]["leads"]["all"][f"{nm}/Global"]
            mx = max(mx, abs(rm - rf["rmse"]))
            rows.append(dict(network=nm, arm=theirs, rmse_recomputed=rm, rmse_frozen_test=rf["rmse"], n_recomputed=int(ok.sum()),
                             n_frozen_test=rf["n_pixel_days"], pixel_stations=int(px.size), pixel_stations_frozen=rf["n_pixels"]))
    out = {"file": str(ROOT / "test2022_ghap/results/st_monitor/scores_st_2022.json"), "rows": rows, "max_abs_rmse_diff": mx,
           "pass": bool(mx < 5e-3 and all(r["n_recomputed"] == r["n_frozen_test"] for r in rows))}
    log(f"frozen-test check: max |RMSE diff| {mx:.2e} -> pass {out['pass']}; {rows[0]}")
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(O / "preds"))
    ap.add_argument("--selftest", action="store_true", help="mechanics only: synthetic observations and GHAP, no real score")
    ap.add_argument("--no-paper-check", action="store_true")
    ap.add_argument("--min-days", type=int, default=30)
    ap.add_argument("--force", action="store_true", help="score although a check of A7(g) failed (after it is understood and logged)")
    ap.add_argument("--only-paper-checks", action="store_true", help="A7(g) checks 1-2 only: the paper's systems with the paper's "
                    "observations, no value of the downscaler or of the chain is read")
    a = ap.parse_args()
    P = Path(a.preds)
    D = Data(a)
    if a.only_paper_checks:
        with open(O / "UNSEALING_A7.md", "a") as f:
            f.write(f"- {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Helsinki), job {os.environ.get('SLURM_JOB_ID', '-')}: [A7] "
                    f"score_official.py --only-paper-checks reads the 2022 observations ({G / 'stations_2022.npz'}) with the paper's own "
                    f"systems only (Foehn 1 km, CAMS forecast, Aurora, the 0.4 deg model) to recompute the paper's stored station numbers "
                    f"(A7(g) checks 1-2). No downscaler or chain value is read.\n")
        D.load_aurora()
        D.load_foehn()
        out = {"paper_compare_fair": paper_check(D), "paper_cams12h_stations": cams12h_check(D)}
        (O / "results").mkdir(exist_ok=True)
        (O / "results/paper_checks.json").write_text(json.dumps(out, indent=1))
        log(f"wrote {O / 'results/paper_checks.json'}")
        return
    done_s, done_c = np.load(P / "done_sameday.npy").astype(bool), np.load(P / "done_chain.npy").astype(bool)
    if not a.selftest:
        assert done_s.all() and done_c.all(), "predictions incomplete: no scoring"
        with open(O / "UNSEALING_A7.md", "a") as f:
            f.write(f"- {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Helsinki), job {os.environ.get('SLURM_JOB_ID', '-')}: [A7] "
                    f"score_official.py reads the 2022 observations of the six official networks ({G / 'stations_2022.npz'}) and "
                    f"test2022_official/truth/ghap_official_2022.npy to score the frozen runs_ghap/pmG_final (same-day and "
                    f"forecast-mode predictions of test2022_official/preds, complete before this reading). Evaluation only.\n")
    else:
        rng = np.random.default_rng(1)
        fin = np.isfinite(D.OBS)
        D.OBS = np.where(fin, np.maximum(D.CLIM.T * np.exp(0.5 * rng.standard_normal(D.OBS.shape)), 0.1), np.nan).astype(np.float32)
        D.obs = D.OBS[D.id].T.copy()
        D.fake = True
        log("SELFTEST: synthetic observations (climatology x log-normal noise on the real missing-data pattern); no real score")
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "selftest_FAKE_observations": bool(a.selftest), "preds": str(P),
           "protocol": str(ROOT / "protocol_amendments/A7_official_stations_2022.md"),
           "system": json.loads((ROOT / "test2022_ghap/SYSTEM.json").read_text()),
           "stations": json.loads((O / "pos/st_qc_official.json").read_text()), "groups": GROUPS, "where": list(WHERE),
           "bootstrap": f"{BLOCK}-day moving blocks over start dates, {NBOOT} draws, seed 0, percentile 95 % (compare_fair.py)",
           "caution": "forecast-mode arms: the composition in channel 0 is a forecast; the weather, fires and transport channels "
                      "of the downscaler are analysis-type fields of the valid day (A7(d), A7(h))", "checks": {}, "tables": {}}
    D.load_aurora()
    D.load_foehn()
    if not a.no_paper_check and not a.selftest:
        res["checks"]["paper_compare_fair"] = paper_check(D)
        res["checks"]["paper_cams12h_stations"] = cams12h_check(D)
    # ---- arrays ---------------------------------------------------------------------------------------------
    names_s = {"ds_sameday": "sameday_ds", "cams0_bil": "sameday_cams0", "bcsd": "sameday_bcsd", "clim_ens": "sameday_climens",
               "clim_med": "sameday_climmed"}
    names_c = ("chain", "chain_t6fc", "chain_rtseed", "camsfc_ds", "bcsd_fc")
    mm = {k: np.load(P / f"{k}.npy", mmap_mode="r") for k in list(names_s.values()) + list(names_c)}
    if a.selftest or not (O / "truth/ghap_official_2022.npy").exists():
        assert a.selftest, "truth/ghap_official_2022.npy missing"
        ghap_all = (10.0 * np.exp(0.5 * np.random.default_rng(2).standard_normal((365, mm["sameday_ds"].shape[2], 4)))).astype(np.float32)
    else:
        ghap_all = np.load(O / "truth/ghap_official_2022.npy")
    if not a.selftest:
        res["checks"]["frozen_test_pixel_station_rmse"] = frozen_check(D, {k: np.asarray(mm[k][..., 0]) for k in
                                                                           ("sameday_ds", "sameday_cams0", "sameday_bcsd")})
    bad = [k for k, c in res["checks"].items() if not c["pass"]]
    if bad and not a.force:
        (O / "results").mkdir(exist_ok=True)
        (O / "results/failed_checks.json").write_text(json.dumps({k: res["checks"][k] for k in bad}, indent=1))
        raise SystemExit(f"A7(g): failed checks {bad} (results/failed_checks.json): no score is computed until they are understood")
    obs_v = D.obs[D.vidx]                                                      # (NI, NL, S)
    days_s = np.where(done_s)[0]
    iss = np.where(done_c)[0]
    # coarse systems at the station coordinates (used for every 'where')
    log("coarse systems at the stations ...")
    coarse = np.full((NI, NL, 8, D.S), np.nan, np.float32)
    camsfc = np.full((NI, NL, D.S), np.nan, np.float32)
    aurora = np.full((NI, NL, D.S), np.nan, np.float32)
    foehn = np.full((NI, NL, 4, D.S), np.nan, np.float32)
    for i in iss:
        I = I0 + int(i)
        coarse[i] = bil(coarse_fields(int(i)), 84.0, 0.4, -180.0, 0.4, D.lat, D.lon)
        camsfc[i] = D.cams_daily(I)[:, D.id]
        for L in range(1, NL + 1):
            aurora[i, L - 1] = D.aurora_daily(I, L)[D.id]
            if (I - 1, L) in D.foehn:
                foehn[i, L - 1] = D.foehn[(I - 1, L)][D.id].T
    clim_paper_s = D.CLIM[:, D.id]                                             # (365, S)
    obs_pers = D.obs[I0 - 1 + np.arange(NI)][:, None, :].repeat(NL, 1)         # observation of day I - 1
    A_coarse = reduce_ens(coarse, obs_v)
    A_camsfc, A_aurora, A_foehn = det(camsfc, obs_v), det(aurora, obs_v), reduce_ens(foehn, obs_v)
    A_obs_pers = det(obs_pers, obs_v)
    res["foehn_issue_days"] = int(np.isfinite(foehn).any((1, 2, 3)).sum())
    log(f"coarse systems done; issue days with a Foehn forecast: {res['foehn_issue_days']}")
    C_S = [("ds_sameday", "cams0_bil"), ("ds_sameday", "bcsd"), ("ds_sameday", "clim_paper"), ("ds_sameday", "clim_med"),
           ("ds_sameday", "clim_ens"), ("ds_sameday", "ghap"), ("bcsd", "cams0_bil"), ("cams0_bil", "clim_paper"), ("cams0_bil", "clim_med")]
    C_C = [("chain", b) for b in ("coarse", "camsfc", "aurora", "clim_paper", "clim_med", "clim_ens", "bcsd_fc", "ds_sameday",
                                  "camsfc_ds", "persist_ds", "ghap", "foehn")] + \
          [("chain_t6fc", b) for b in ("chain", "coarse", "camsfc", "aurora", "bcsd_fc", "clim_paper", "clim_med", "clim_ens", "foehn")] + \
          [("chain_rtseed", "chain"), ("coarse", "camsfc"), ("coarse", "aurora"), ("camsfc_ds", "camsfc"), ("camsfc_ds", "coarse"),
           ("bcsd_fc", "coarse"), ("bcsd_fc", "camsfc"), ("aurora", "camsfc"), ("persist_ds", "camsfc"),
           ("ds_sameday", "cams0_bil"), ("ds_sameday", "bcsd"), ("ds_sameday", "clim_paper"), ("ds_sameday", "clim_med"),
           ("ds_sameday", "clim_ens"), ("ds_sameday", "camsfc"), ("foehn", "camsfc"), ("foehn", "aurora"), ("foehn", "coarse"),
           ("foehn", "clim_paper"), ("foehn", "ds_sameday"), ("camsfc", "clim_paper"), ("coarse", "clim_paper")]
    for w, wn in enumerate(WHERE):
        log(f"=== where: {wn}")
        sv = lambda k: np.asarray(mm[k][..., w])[..., D.pix].astype(np.float32)      # noqa: E731  (..., M, S)
        # ---- same-day arms (365, S)
        sd_raw = {n: sv(f) for n, f in names_s.items()}                        # (365, M, S)
        if a.selftest:                                                         # mechanics only: fill the days not predicted
            near = days_s[np.abs(np.arange(365)[:, None] - days_s[None]).argmin(1)]
            sd_raw = {n: x[near] for n, x in sd_raw.items()}
        SD = {n: reduce_ens(x, D.obs) for n, x in sd_raw.items()}
        SD["ghap"] = det(ghap_all[:, D.pix, w], D.obs)
        if w == 0:
            SD["clim_paper"] = det(clim_paper_s, D.obs)
        armsS = {n: {q: (x[q][:, None, :] if q != "M" else x[q]) for q in x} for n, x in SD.items()}
        tS = Table("S", armsS, D.obs[:, None, :], D.gmask, "cams0_bil", a.min_days, ["sameday"],
                   "same-day mode, the 365 days of 2022: the downscaler with the frozen test's inputs (first-day mean of the CAMS "
                   "00 UTC run of the same day, ERA5, GFAS, transport); start dates = valid days")
        rS = tS.result(C_S)
        # ---- forecast arms (NI, NL, S)
        armsC = {n: reduce_ens(sv(n), obs_v) for n in names_c}
        armsC["coarse"], armsC["camsfc"], armsC["aurora"] = A_coarse, A_camsfc, A_aurora
        pers = [reduce_ens(sd_raw["ds_sameday"][I0 - 1 + np.arange(NI)], obs_v[:, L]) for L in range(NL)]
        armsC["persist_ds"] = {q: (np.stack([p_[q] for p_ in pers], 1) if q != "M" else pers[0]["M"]) for q in pers[0]}
        for n in SD:
            armsC[n] = take(SD[n], D.vidx)
        for v_ in armsC.values():
            for q in ("mean", "crps", "var"):
                v_[q] = np.where(done_c[:, None, None], v_[q], np.nan).astype(np.float32)
        noteC = ("forecast mode, issue days 2022-05-01 .. 2022-11-30 (00 UTC), lead day L = hours 24(L-1)..24L, valid day = issue "
                 "day + L - 1. chain*, camsfc_ds, bcsd_fc: the COMPOSITION in channel 0 is a forecast; the WEATHER, FIRES and "
                 "TRANSPORT channels of the downscaler are analysis-type fields of the valid day (chain_t6fc: transported PM2.5 "
                 "from the forecast itself). ds_sameday, cams0_bil, bcsd, ghap: fields of the valid day (analysis-type, not "
                 "forecasts beyond lead day 1). coarse, camsfc, aurora: values at the station coordinates on every 'where'.")
        tC = Table("C", armsC, obs_v, D.gmask, "camsfc", a.min_days, [str(L) for L in range(1, NL + 1)], noteC)
        rC = tC.result(C_C)
        rC["contrasts_own_days"] = {"chain - obs_persist": tC.contrast_submask("chain", A_obs_pers, "obs_persist"),
                                    "camsfc - obs_persist": tC.contrast_submask("camsfc", A_obs_pers, "obs_persist")}
        out = {"S": rS, "C": rC}
        PR_C = ("chain", "chain_t6fc", "camsfc_ds", "bcsd_fc", "ds_sameday", "clim_med", "ghap")
        PR_S = ("ds_sameday", "bcsd", "clim_med", "ghap")
        if w == 0:
            keepC, keepS = {n: armsC[n] for n in PR_C}, {n: armsS[n] for n in PR_S}
        else:                                                                  # A7(f) addition: each 1 km arm at the pixel against its ring mean
            ac = {f"{n}@{k}": v for n in PR_C for k, v in (("pixel", keepC[n]), ("ring", armsC[n]))}
            out["C_pixel_vs_ring"] = Table("C pixel vs ring", ac, obs_v, D.gmask, None, a.min_days, [str(L) for L in range(1, NL + 1)],
                                           "table C, the same arm at the station pixel (A) and as its ring mean (B), stations that have the ring"
                                           ).result([(f"{n}@pixel", f"{n}@ring") for n in PR_C])
            as_ = {f"{n}@{k}": v for n in PR_S for k, v in (("pixel", keepS[n]), ("ring", armsS[n]))}
            out["S_pixel_vs_ring"] = Table("S pixel vs ring", as_, D.obs[:, None, :], D.gmask, None, a.min_days, ["sameday"],
                                           "table S, the same arm at the station pixel (A) and as its ring mean (B), stations that have the ring"
                                           ).result([(f"{n}@pixel", f"{n}@ring") for n in PR_S])
        if w == 0:
            armsF = dict(armsC)
            armsF["foehn"] = A_foehn
            tF = Table("F", armsF, obs_v, D.gmask, "camsfc", a.min_days, [str(L) for L in range(1, NL + 1)],
                       noteC + " Table F: the issue days that have a Foehn forecast (start day = issue day - 1, every second "
                       "day), plus foehn (4 members, forecast weather, needs the 1 km estimates of the two days before the issue).")
            out["F"] = tF.result(C_C)
            # GHAP's own agreement with the stations (description of the target), per network
            pin = {}
            for g, gn in enumerate(GROUPS):
                pin[gn] = {}
                for w2, wn2 in enumerate(WHERE):
                    gh = ghap_all[:, D.pix, w2][:, D.gmask[g]]
                    o = D.obs[:, D.gmask[g]]
                    ok = np.isfinite(gh) & np.isfinite(o)
                    n = ok.sum(0)
                    good = n >= 30
                    with np.errstate(all="ignore"):
                        x, y = np.where(ok, gh, 0.0).astype(np.float64), np.where(ok, o, 0.0).astype(np.float64)
                        r = ((x * y).sum(0) / n - x.sum(0) * y.sum(0) / n ** 2) / np.sqrt(((x * x).sum(0) / n - (x.sum(0) / n) ** 2)
                                                                                        * ((y * y).sum(0) / n - (y.sum(0) / n) ** 2))
                    pin[gn][wn2] = {"stations": int(good.sum()),
                                    "share_station_days_within_0p5": float((np.abs(gh - o)[ok] <= 0.5).mean()) if ok.any() else None,
                                    "share_stations_R_gt_0p999": float((r[good] > 0.999).mean()) if good.any() else None,
                                    "median_station_R": float(np.nanmedian(r[good])) if good.any() else None,
                                    "rmse": float(np.sqrt(((gh - o)[ok] ** 2).mean())) if ok.any() else None}
            res["ghap_vs_stations_2022"] = pin
        res["tables"][wn] = out
        log(f"    S: {rS['n_station_days']:,} station-days, {rS['n_stations']} stations | C: {rC['n_station_days']:,}, "
            f"{rC['n_stations']} stations, {rC['n_start_dates']} start dates" + (f" | F: {out['F']['n_station_days']:,}, "
            f"{out['F']['n_start_dates']} start dates" if w == 0 else ""))
    tag = "selftest_scores" if a.selftest else "scores_official_2022"
    (O / "results").mkdir(exist_ok=True)
    (O / "results" / f"{tag}.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    log(f"wrote {O / 'results' / (tag + '.json')}")


if __name__ == "__main__":
    main()
