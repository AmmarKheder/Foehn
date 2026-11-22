#!/usr/bin/env python3
"""A16 law check (CPU only, reads only): is the 'ar' design (fresh Philox noise per start day, AR(1) across leads) the same
per-forecast law as the stored fixed-field design, with the stored results being the luck of one draw of fixed fields?

Modes (each writes only under P/a16/lawcheck, through a temporary file, fsync, read-back, rename):
  noise  (a)+(e) the input fields: stored noise_tiles16.npy (PCG64 seeds 1000+m, float32 -> float16, prep_test.py) against the
         Philox fields of a16/arnoise22.py (re-implemented here, checked against a16/golden22.json) at the same tiles, leads
         1, 3, 5, 7, as they enter the sampler (float16).                                          -> part_noise.json
  pop    (c)+(d) sixteen stored members (a13/warn16/data/run16 + a14/data/run16_13 + run16all task 0) against the sixteen
         'ar' members (a16/data/run16) on the same 358 forecasts at the 8.1 M populated pixels: per forecast, per run,
         member sums, Gram matrices (raw and log1p), counts above 37.5/75/150, log histograms, per region, at the station
         pixels (coordinates only), and member-departure fingerprints at 60,000 fixed pixels.      -> pop_stats.npz
  ncp    (d)+(e) North China Plain box of the East Asia case study (N 41.5 S 35.5 W 113.5 E 121.5), 28 starts x 7 leads,
         16 members, stored and _ar: spatial band variances per member, levels, Gram, counts.      -> ncp_stats.npz
  final  every statistic, jackknife over member indices (the field draw), quartet distributions, lead drifts; plus the
         4-member runs F1/G1 per lead (part files).                                                -> lawcheck.json
"""
import glob
import hashlib
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

P = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
LC = P / "a16" / "lawcheck"
G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
RUN = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/"
           "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"))
GW = 36000
THR = (37.5, 75.0, 150.0)
RHO = 0.59
NB, LO, HI = 1000, np.log(0.1), np.log(3000.0)
STEP = (HI - LO) / NB
STORED16 = [P / "a13/warn16/data/run16", P / "a14/data/run16_13"]
TASK0_DIR, TASK0 = P / "a13/warn16/data/run16all", [(2, 1), (2, 3), (4, 1), (4, 3)]
AR16 = P / "a16/data/run16"
REGIONS = {  # name: (north, south, west, east)
    "EastAsia": (50, 20, 100, 145), "SouthAsia": (35, 5, 60, 100), "Europe": (72, 35, -12, 45),
    "NorthAmerica": (72, 15, -170, -50), "SouthAmerica": (15, -60, -90, -30), "Africa": (35, -35, -20, 55),
    "MiddleEast": (42, 12, 35, 60), "SoutheastAsia": (20, -11, 95, 145), "Korea": (39, 33, 124.5, 130),
    "NCP": (41.5, 35.5, 113.5, 121.5), "KanoBox": (28, 6, 0, 26), "GirondeBox": (47, 43, -3, 1)}
RNAMES = list(REGIONS)
NCP_ROWS, NCP_COLS = (50, 650), (150, 950)       # inside the East Asia crop (rows 4200:5100, cols 29200:31100)
TILES = [(8, 57), (8, 58), (9, 57), (9, 58), (11, 52), (14, 35)]


# ------------------------------------------------------------------ atomic writes
def _commit(tmp, path, check):
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    check(tmp)
    os.replace(tmp, path)
    fd = os.open(str(Path(path).parent), os.O_RDONLY)
    os.fsync(fd); os.close(fd)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    return x


def save_json(path, obj):
    path = Path(path); assert path.parent == LC, path
    txt = json.dumps(jsonable(obj), indent=1) + "\n"
    tmp = str(path) + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(txt); f.flush()
    def chk(t):
        assert open(t).read() == txt, t
    _commit(tmp, path, chk)


def save_npz(path, **arr):
    path = Path(path); assert path.parent == LC, path
    tmp = str(path) + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **arr)

    def chk(t):
        z = np.load(t)
        for k, v in arr.items():
            assert np.array_equal(z[k], v, equal_nan=v.dtype.kind == "f"), (t, k)
    _commit(tmp, path, chk)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ (a) noise
def innovation(year, doy, m, k, tr, tc):            # a16/arnoise22.py innovation(), verbatim arithmetic
    ss = np.random.SeedSequence([1521, int(year), int(doy), int(m), int(k), int(tr), int(tc)])
    return np.random.Generator(np.random.Philox(ss)).standard_normal((512, 512), dtype=np.float32)


def coeffs(h, rho):                                   # a16/arnoise22.py coeffs(), verbatim
    s = np.sqrt(1.0 - rho * rho)
    return np.array([rho ** (h - 1)] + [s * rho ** (h - k) for k in range(2, h + 1)], np.float64).astype(np.float32)


def z_from(eps, h, rho=RHO):                          # ARNoise._arith: float32 accumulation, float16 result
    c = coeffs(h, rho); acc = np.zeros((512, 512), np.float32)
    for k in range(1, h + 1):
        acc += c[k - 1] * eps[k - 1]
    return acc.astype(np.float16)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


KX = np.fft.fftfreq(512)
KR = np.sqrt(KX[:, None] ** 2 + KX[None, :] ** 2)
with np.errstate(divide="ignore"):
    LAM = np.where(KR > 0, 1.0 / KR, np.inf)
BANDS_SPEC = {">=100km": LAM >= 100, "20-100km": (LAM >= 20) & (LAM < 100), "5-20km": (LAM >= 5) & (LAM < 20),
              "2-5km": (LAM >= 2) & (LAM < 5), "<2km": LAM < 2}


def field_stats(arrs):
    """arrs: list of 2-D float16 fields as they enter the sampler (cast to float32 there)"""
    a = np.stack([np.asarray(x, np.float32) for x in arrs])
    v = a.reshape(-1).astype(np.float64)
    mu, var = v.mean(), v.var()
    zc = (v - mu) / np.sqrt(var)
    out = dict(fields=len(arrs), values=v.size, dtype=str(np.asarray(arrs[0]).dtype), mean=mu, var=var,
               skew=float(np.mean(zc ** 3)), excess_kurtosis=float(np.mean(zc ** 4) - 3))
    for t in (2, 3, 4):
        out[f"P_abs_gt{t}"] = float(np.mean(np.abs(v) > t))
    out["P_abs_gt2_normal"], out["P_abs_gt3_normal"], out["P_abs_gt4_normal"] = 0.0455003, 0.0026998, 6.3342e-05
    out["max_abs"] = float(np.abs(v).max())
    qs = [1e-4, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999]
    out["quantiles"] = dict(zip([str(q) for q in qs], np.quantile(v, qs).tolist()))
    def lagc(x, y):
        x = x.astype(np.float64).ravel(); y = y.astype(np.float64).ravel()
        return float(np.corrcoef(x, y)[0, 1])
    out["lag1_corr_cols"] = float(np.mean([lagc(x[:, :-1], x[:, 1:]) for x in a]))
    out["lag1_corr_rows"] = float(np.mean([lagc(x[:-1], x[1:]) for x in a]))
    out["lag2_corr_cols"] = float(np.mean([lagc(x[:, :-2], x[:, 2:]) for x in a]))
    ps = np.mean([np.abs(np.fft.fft2(x - x.mean())) ** 2 / x.size for x in a], 0)
    out["spectrum_band_power_over_white"] = {k: float(ps[m].mean()) for k, m in BANDS_SPEC.items()}
    for b in (8, 32, 128):                             # variance of b x b block means times b^2 (white: 1)
        bm = a.reshape(len(a), 512 // b, b, 512 // b, b).mean((2, 4), dtype=np.float64)
        out[f"blockmean_var_x_area_{b}px"] = float(bm.var() * b * b)
        out[f"blockmean_n_{b}px"] = int(bm.size)
    return out


def ncp_noise_power(get_tile):
    """large-scale power of a field in the NCP box (global rows 4250:4850, cols 29350:30150) from tiles 8-9 x 57-58"""
    blk = np.block([[get_tile(8, 57), get_tile(8, 58)], [get_tile(9, 57), get_tile(9, 58)]]).astype(np.float64)
    x = blk[4250 - 4096:4850 - 4096, 29350 - 29184:30150 - 29184]
    out = {"var_1km": x.var()}
    for b in (5, 20, 100, 200):
        bm = x.reshape(600 // b, b, 800 // b, b).mean((1, 3))
        out[f"ms_blockmean_x_area_{b}km"] = float((bm ** 2).mean() * b * b)   # mean square (white: 1), no centring
    out["mean_x_sqrtN"] = float(x.mean() * np.sqrt(x.size))
    return out


def mode_noise():
    t0 = time.time()
    res = {"what": "(a) and (e): input noise fields as they enter the sampler (float16 tiles cut by tile_window, cast "
                   "to float32 on the GPU). Tiles (tile row, tile column) of the (29, 71, 512, 512) layout: " + str(TILES)}
    gold = json.loads((P / "a16/golden22.json").read_text())
    gk = ((2022, 0, 0, 1, 0, 0), (2022, 356, 3, 7, 28, 70), (2022, 2, 15, 3, 14, 35), (2022, 19, 1, 2, 7, 60),
          (2022, 189, 12, 5, 10, 40))
    ok = {}
    for key in gk:
        e = innovation(*key); nm = "eps_" + "_".join(map(str, key))
        ok[nm] = sha(e) == gold[nm + "/f32"] and sha(e.astype(np.float16)) == gold[nm + "/f16"]
    ok["z_2022_2_lead3_m15_tile14_35"] = sha(z_from([innovation(2022, 2, 15, k, 14, 35) for k in (1, 2, 3)], 3)) == \
        gold["z_2022_2_lead3_rho0.59_philox_m15_tile14_35/f16"]
    ok["z_2022_0_lead7_m0_tile7_30"] = sha(z_from([innovation(2022, 0, 0, k, 7, 30) for k in range(1, 8)], 7)) == \
        gold["z_2022_0_lead7_rho0.59_philox_m0_tile7_30/f16"]
    res["reimplementation_equals_golden22"] = dict(all=all(ok.values()), items=ok, numpy=np.__version__,
                                                   source=str(P / "a16/golden22.json"))
    assert all(ok.values()), ok
    s16 = np.load(G / "noise_tiles16.npy", mmap_mode="r")
    s4 = np.load(G / "noise_tiles.npy", mmap_mode="r")
    res["stored_files"] = {"noise_tiles16.npy": dict(shape=s16.shape, dtype=str(s16.dtype)),
                           "noise_tiles.npy": dict(shape=s4.shape, dtype=str(s4.dtype)),
                           "recipe": "prep_test.py noise: member m: rng = default_rng(1000 + m) (PCG64); for tile row r: "
                                     "rng.standard_normal((71, 512, 512), float32).astype(float16)"}
    res["stored4_equals_stored16_members0_3_at_tiles"] = bool(all(np.array_equal(s4[m, tr, tc], s16[m, tr, tc])
                                                                  for m in range(4) for tr, tc in TILES))
    # the recipe reproduces the stored bytes (members 0, 7, 15, tile row 9, columns 57 and 58)
    rec = {}
    for m in (0, 7, 15):
        rng = np.random.default_rng(1000 + m)
        for r in range(10):
            row = rng.standard_normal((71, 512, 512), dtype=np.float32).astype(np.float16)
        rec[f"m{m}"] = bool(np.array_equal(row[57], s16[m, 9, 57]) and np.array_equal(row[58], s16[m, 9, 58]))
    res["stored_recipe_reproduces_bytes_tile_row9"] = rec
    stored = [np.asarray(s16[m, tr, tc]) for m in range(16) for tr, tc in TILES]
    res["stored16"] = field_stats(stored)
    res["stored16"]["corr_between_members_max_abs"] = float(max(
        abs(np.corrcoef(np.asarray(s16[a, tr, tc], np.float64).ravel(), np.asarray(s16[b, tr, tc], np.float64).ravel())[0, 1])
        for a, b in [(0, 1), (2, 9), (5, 15), (3, 4)] for tr, tc in TILES[:2]))
    # philox: starts 0, 2, 10, 20, members 0-15, leads 1, 3, 5, 7
    starts = (0, 2, 10, 20)
    eps = {}
    for d in starts:
        for m in range(16):
            for tr, tc in TILES:
                eps[(d, m, tr, tc)] = [innovation(2022, d, m, k, tr, tc) for k in range(1, 8)]
    zz = {h: {kk: z_from(v, h) for kk, v in eps.items()} for h in (1, 3, 5, 7)}
    res["philox_sample"] = dict(year=2022, starts=starts, members="0-15", tiles=TILES, rho=RHO)
    for h in (1, 3, 5, 7):
        r = field_stats(list(zz[h].values()))
        c = lambda A, B: float(np.corrcoef(A.astype(np.float64).ravel(), B.astype(np.float64).ravel())[0, 1])
        r["corr_with_lead1_same_start_member"] = float(np.mean([c(zz[h][k], zz[1][k]) for k in list(zz[h])[:96]]))
        r["corr_with_lead1_expected_rho^(h-1)"] = RHO ** (h - 1)
        r["corr_between_starts_0_2_same_member_max_abs"] = float(max(abs(c(zz[h][(0, m, tr, tc)], zz[h][(2, m, tr, tc)]))
                                                                    for m in range(16) for tr, tc in TILES[:2]))
        r["corr_between_members_0_1_max_abs"] = float(max(abs(c(zz[h][(d, 0, tr, tc)], zz[h][(d, 1, tr, tc)]))
                                                          for d in starts for tr, tc in TILES[:2]))
        r["sum_coeff_squared_float32"] = float(np.sum(coeffs(h, RHO).astype(np.float64) ** 2))
        res[f"philox_lead{h}"] = r
    # two-sample comparison of the value distributions (sup |F1 - F2| on the float16 values, exact over all values)
    def ks(a, b):
        a = np.sort(np.concatenate([x.astype(np.float32).ravel() for x in a]))
        b = np.sort(np.concatenate([x.astype(np.float32).ravel() for x in b]))
        u = np.unique(np.concatenate([a[:: 997], b[:: 997], [-4, -3, -2, -1, 0, 1, 2, 3, 4]]))
        Fa = np.searchsorted(a, u, side="right") / a.size; Fb = np.searchsorted(b, u, side="right") / b.size
        n = a.size * b.size / (a.size + b.size)
        return dict(D=float(np.abs(Fa - Fb).max()), D_crit_0p05=float(1.358 / np.sqrt(n)), n_a=a.size, n_b=b.size,
                    note="pixels within a field are independent under both designs; D evaluated on a grid of values")
    res["KS_stored16_vs_philox_lead1"] = ks(stored, list(zz[1].values()))
    res["KS_stored16_vs_philox_lead7"] = ks(stored, list(zz[7].values()))
    # large-scale power of the noise inside the NCP box: stored members (fixed for all forecasts) vs ar (fresh per start)
    ncp_s = {m: ncp_noise_power(lambda tr, tc: np.asarray(s16[m, tr, tc])) for m in range(16)}
    ncp_a = {}
    for d in range(28):
        for m in range(16):
            ncp_a[(d, m)] = ncp_noise_power(lambda tr, tc: innovation(2022, d, m, 1, tr, tc).astype(np.float16))
    keys = list(ncp_s[0])
    res["ncp_box_noise_power"] = {
        "what": "per field, NCP box (600 x 800 px): mean square of b-km block means x b^2 (white: 1, chi-square/dof), "
                "mean x sqrt(N) (white: N(0,1)); stored: the 16 fixed fields; ar: lead-1 innovation of 28 starts x 16 members",
        "stored_per_member": {k: [ncp_s[m][k] for m in range(16)] for k in keys},
        "ar_lead1_summary": {k: dict(mean=float(np.mean([v[k] for v in ncp_a.values()])),
                                     sd=float(np.std([v[k] for v in ncp_a.values()], ddof=1)),
                                     q025=float(np.quantile([v[k] for v in ncp_a.values()], 0.025)),
                                     q975=float(np.quantile([v[k] for v in ncp_a.values()], 0.975)), n=len(ncp_a))
                             for k in keys},
        "ar_lead1_per_start_member": {k: [[ncp_a[(d, m)][k] for m in range(16)] for d in range(28)] for k in keys},
        "stored_summary": {k: dict(mean=float(np.mean([ncp_s[m][k] for m in range(16)])),
                                   sd=float(np.std([ncp_s[m][k] for m in range(16)], ddof=1)), n=16) for k in keys}}
    res["seconds"] = time.time() - t0
    save_json(LC / "part_noise.json", res)
    print("noise done", time.time() - t0, flush=True)


# ------------------------------------------------------------------ (c) populated pixels
def index16(dirs, only=None):
    out = {}
    for D in dirs:
        for fk in sorted(Path(D).glob("keys16_*.npy")):
            tag = fk.stem.split("_")[1]
            done = set(json.loads((Path(D) / f"done16_{tag}.json").read_text()))
            for n, (d, h, t) in enumerate(np.load(fk)):
                if n in done and (only is None or (int(d), int(h)) in only):
                    key = (int(d), int(h)); assert key not in out, key
                    out[key] = (str(Path(D) / f"pop16_{tag}.npy"), n, int(t))
    return out


def gram(X):
    Gm = np.zeros((X.shape[1], X.shape[1]), np.float64)
    for c in range(0, X.shape[0], 1 << 20):
        x = X[c:c + (1 << 20)].astype(np.float64); Gm += x.T @ x
    return Gm


def basic(X):
    return dict(N=np.int64(X.shape[0]), S1=X.sum(0, dtype=np.float64), G=gram(X),
                C=np.stack([(X > t).sum(0) for t in THR]).astype(np.int64))


_POP = {}


def _pop_init(sel_regions, sel_st, fp_idx):
    os.environ["OMP_NUM_THREADS"] = "1"
    _POP.update(sel_regions=sel_regions, sel_st=sel_st, fp_idx=fp_idx)


def _row(f, n):
    a = np.load(f, mmap_mode="r")
    return np.array(a[n], np.float32)


def _pop_one(job):
    (d, h), (fs, ns), (fa, na) = job
    Xs, Xa = _row(fs, ns), _row(fa, na)
    ok = np.isfinite(Xs).all(1) & np.isfinite(Xa).all(1)
    out = {"ok": int(ok.sum()), "n_all": int(ok.size), "nonfinite_s": int((~np.isfinite(Xs).all(1)).sum()),
           "nonfinite_a": int((~np.isfinite(Xa).all(1)).sum())}
    off = (np.arange(16) * (NB + 2))[None, :]
    for tag, X in (("s", Xs), ("a", Xa)):
        Xo = X[ok]
        out[tag] = basic(Xo)
        L = np.log1p(np.maximum(Xo, 0))
        out[tag]["LS1"], out[tag]["LG"] = L.sum(0, dtype=np.float64), gram(L)
        with np.errstate(divide="ignore"):
            ix = np.clip(np.floor((np.log(np.maximum(Xo, 1e-30)) - LO) / STEP).astype(np.int32) + 1, 0, NB + 1)
        out[tag]["H"] = np.bincount((ix + off).ravel(), minlength=16 * (NB + 2)).reshape(16, NB + 2).astype(np.int64)
        del L, ix
        reg = []
        for sel in _POP["sel_regions"]:
            Xr = X[sel[ok[sel]]]
            reg.append(basic(Xr))
        out[tag]["R"] = reg
        st = _POP["sel_st"]
        out[tag]["ST"] = basic(X[st[ok[st]]])
        fp = _POP["fp_idx"]
        Lf = np.log1p(np.maximum(X[fp], 0))
        dep = Lf - (Lf.sum(1, keepdims=True) - Lf) / 15.0                 # departure from the mean of the 15 others (log1p)
        out[tag]["FP"] = np.where(ok[fp][:, None], dep, np.nan).T.astype(np.float16)
    return (d, h), out


def mode_pop(workers):
    from multiprocessing import get_context
    t0 = time.time()
    S = index16(STORED16)
    S.update(index16([TASK0_DIR], only=set(TASK0)))
    A = index16([AR16])
    keys = sorted(set(S) & set(A))
    assert len(keys) == 358 and set(S) == set(A), (len(S), len(A), len(keys))
    assert all(S[k][2] == A[k][2] for k in keys)
    pop = np.load(G / "pop_pixels.npz")["idx"]
    lat = 84.0 - 0.01 * (pop // GW + 0.5); lon = -180.0 + 0.01 * (pop % GW + 0.5)
    sel_regions = [np.flatnonzero((lat <= n) & (lat > s) & (lon >= w) & (lon < e)) for n, s, w, e in REGIONS.values()]
    with np.load(G / "stations_2022.npz") as st:                       # coordinates only (pixel extraction)
        slat, slon = st["lat"], st["lon"]
    sr = np.clip(np.floor((84.0 - slat) / 0.01).astype(int), 0, 14399); sc = np.clip(np.floor((slon + 180.0) / 0.01).astype(int), 0, GW - 1)
    pix = sr.astype(np.int64) * GW + sc
    pos = np.clip(np.searchsorted(pop, pix), 0, len(pop) - 1)
    sel_st = np.unique(pos[pop[pos] == pix])
    fp_idx = np.sort(np.random.default_rng(20261121).choice(len(pop), 60000, replace=False))
    jobs = [(k, S[k][:2], A[k][:2]) for k in keys]
    F = len(keys)
    R = len(RNAMES)
    arr = {}
    for t in ("s", "a"):
        arr.update({f"{t}_N": np.zeros(F, np.int64), f"{t}_S1": np.zeros((F, 16)), f"{t}_G": np.zeros((F, 16, 16)),
                    f"{t}_C": np.zeros((F, 3, 16), np.int64), f"{t}_LS1": np.zeros((F, 16)), f"{t}_LG": np.zeros((F, 16, 16)),
                    f"{t}_H": np.zeros((F, 16, NB + 2), np.int64),
                    f"{t}_RN": np.zeros((F, R), np.int64), f"{t}_RS1": np.zeros((F, R, 16)), f"{t}_RG": np.zeros((F, R, 16, 16)),
                    f"{t}_RC": np.zeros((F, R, 3, 16), np.int64),
                    f"{t}_STN": np.zeros(F, np.int64), f"{t}_STS1": np.zeros((F, 16)), f"{t}_STG": np.zeros((F, 16, 16)),
                    f"{t}_STC": np.zeros((F, 3, 16), np.int64), f"{t}_FP": np.zeros((F, 16, len(fp_idx)), np.float16)})
    ok_n, nonfin = np.zeros(F, np.int64), np.zeros((F, 2), np.int64)
    pos_of = {k: i for i, k in enumerate(keys)}
    ctx = get_context("fork")
    with ctx.Pool(workers, initializer=_pop_init, initargs=(sel_regions, sel_st, fp_idx)) as pool:
        for c, (k, o) in enumerate(pool.imap_unordered(_pop_one, jobs, chunksize=1)):
            i = pos_of[k]
            ok_n[i] = o["ok"]; nonfin[i] = (o["nonfinite_s"], o["nonfinite_a"])
            for t in ("s", "a"):
                b = o[t]
                arr[f"{t}_N"][i], arr[f"{t}_S1"][i], arr[f"{t}_G"][i], arr[f"{t}_C"][i] = b["N"], b["S1"], b["G"], b["C"]
                arr[f"{t}_LS1"][i], arr[f"{t}_LG"][i], arr[f"{t}_H"][i], arr[f"{t}_FP"][i] = b["LS1"], b["LG"], b["H"], b["FP"]
                for r, rb in enumerate(b["R"]):
                    arr[f"{t}_RN"][i, r], arr[f"{t}_RS1"][i, r], arr[f"{t}_RG"][i, r], arr[f"{t}_RC"][i, r] = \
                        rb["N"], rb["S1"], rb["G"], rb["C"]
                arr[f"{t}_STN"][i], arr[f"{t}_STS1"][i], arr[f"{t}_STG"][i], arr[f"{t}_STC"][i] = \
                    b["ST"]["N"], b["ST"]["S1"], b["ST"]["G"], b["ST"]["C"]
            if c % 20 == 0:
                print(f"pop {c + 1}/{F} {time.time() - t0:.0f}s", flush=True)
    save_npz(LC / "pop_stats.npz", keys=np.array(keys, np.int32), ok_n=ok_n, nonfinite=nonfin, fp_idx=fp_idx,
             n_station_pixels=np.int64(len(sel_st)), region_names=np.array(RNAMES), region_pixels=np.array([len(s) for s in sel_regions]),
             stored_files=np.array([f"{S[k][0]}:{S[k][1]}" for k in keys]), ar_files=np.array([f"{A[k][0]}:{A[k][1]}" for k in keys]),
             **arr)
    print("pop done", time.time() - t0, flush=True)


# ------------------------------------------------------------------ NCP box
def _ncp_one(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    from scipy.ndimage import uniform_filter
    d, h = job
    out = {}
    Xs = np.load(RUN / "eval_case_eastasia_crps_e01" / f"crop0_d{d}_h{h}.npz")["members"]
    Xa = np.load(RUN / "eval_case_eastasia_crps_e01_ar" / f"crop0_d{d}_h{h}.npz")["members"]
    r0, r1 = NCP_ROWS; c0, c1 = NCP_COLS
    Xs = Xs[:, r0:r1, c0:c1].astype(np.float64); Xa = Xa[:, r0:r1, c0:c1].astype(np.float64)
    ok = np.isfinite(Xs).all(0) & np.isfinite(Xa).all(0)
    w = ok.astype(np.float64)
    W = {s: uniform_filter(w, s, mode="constant") for s in (5, 20, 80, 300)}
    out["ok"] = int(ok.sum())
    for tag, X in (("s", Xs), ("a", Xa)):
        X = np.where(ok[None], X, 0.0)
        bands = np.zeros((16, 6))
        for m in range(16):
            U = {s: uniform_filter(X[m], s, mode="constant") / np.maximum(W[s], 1e-12) for s in (5, 20, 80, 300)}
            v = X[m][ok]
            comp = [v - U[5][ok], U[5][ok] - U[20][ok], U[20][ok] - U[80][ok], U[80][ok] - U[300][ok], U[300][ok], v]
            bands[m] = [c.var() for c in comp]
        V = X[:, ok].T
        out[tag] = dict(bands=bands, S1=V.sum(0), G=V.T @ V, C=np.stack([(V > t).sum(0) for t in THR]))
    return (d, h), out


def mode_ncp(workers):
    from multiprocessing import get_context
    t0 = time.time()
    jobs = [(d, h) for d in range(28) for h in range(1, 8)]
    F = len(jobs)
    arr = {f"{t}_{k}": np.zeros(sh) for t in ("s", "a") for k, sh in
           (("bands", (F, 16, 6)), ("S1", (F, 16)), ("G", (F, 16, 16)), ("C", (F, 3, 16)))}
    okn = np.zeros(F, np.int64)
    pos = {k: i for i, k in enumerate(jobs)}
    with get_context("fork").Pool(workers) as pool:
        for k, o in pool.imap_unordered(_ncp_one, jobs, chunksize=1):
            i = pos[k]; okn[i] = o["ok"]
            for t in ("s", "a"):
                for kk in ("bands", "S1", "G", "C"):
                    arr[f"{t}_{kk}"][i] = o[t][kk]
    save_npz(LC / "ncp_stats.npz", keys=np.array(jobs, np.int32), ok=okn,
             band_names=np.array(["<5km", "5-20km", "20-80km", "80-300km", ">300km", "total"]), **arr)
    print("ncp done", time.time() - t0, flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    if mode == "noise":
        mode_noise()
    elif mode == "pop":
        mode_pop(nw)
    elif mode == "ncp":
        mode_ncp(nw)
    elif mode == "final":
        import lawfinal
        lawfinal.main()
