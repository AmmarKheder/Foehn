#!/usr/bin/env python3
"""A16 law check, independent re-check (CPU only, reads only). Writes only P/a16/lawcheck/v_law.json and
P/a16/lawcheck/v_cells.npz, each through a temporary file, fsync, read-back and rename. Question: did the 'ar' design change
the single-lead law of a member (lower, smoother, less dispersed), or are the moves the luck of the stored fixed fields?

Modes (python -B v_law.py <mode> [workers] [--smoke]):
  noise  (job container, numpy 1.26.3) the input noise through the REAL code path: a16/arnoise22.ARNoise (philox, rho 0.59)
         and the stored noise_tiles16.npy, both cut by eval_global_green.tile_window into two overlapping production windows
         (10 and 11: k0 136, j0 928 and 1160), starts 101 and 263, leads 1, 3, 7 (not used by lawcheck.py); moments,
         tails, lag-1, block-mean power up to 1408 px, cross-member and cross-lead correlation, overlap consistency; global
         512-px tile means of the 16 stored fields and of 6 philox fields (leads 1 and 3) at all 2,059 tiles.
  pop    16 stored vs 16 'ar' members, the 358 forecasts, 8.1 M populated pixels -> sums per 1-degree cell and lead (and per
         10-degree block and forecast): levels, between-member variances (16, 8, quartets 0-3 and 4-7), log variance,
         exceedance counts split by member halves, structure functions at 1, 4, 16 km (total and member departure);
         station pixels (coordinates only, no station value is read).
  crop   East Asia case study (16 vs 16, 28 starts x 7 leads): 100-km patches, Gaussian band variances, spread, level.
  map25  F1 vs G1 (4 members, 1,253 forecasts, leads 1-7): 0.25-degree spread and member-0 maps -> 1-degree cells per lead.
  final  draw-robust (equal weight over cells, normaliser free of the stored draw) and hotspot-weighted ratios ar/stored,
         spatial block bootstraps, correlograms of the stored luck, lead drifts.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

P = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
LC = P / "a16" / "lawcheck"
G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
RUN = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/"
           "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"))
OJ, OZ = LC / "v_law.json", LC / "v_cells.npz"
GW = 36000
RHO = 0.59
NC = 144 * 360                      # 1-degree cells from 84N, 180W
SMOKE = "--smoke" in sys.argv


# ------------------------------------------------------------------ atomic writes (own files only)
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


def _fsync_dir(d):
    fd = os.open(str(d), os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


def write_json_section(name, obj):
    if SMOKE:
        print("SMOKE (not written)", name, json.dumps(jsonable(obj))[:3000])
        return
    cur = json.loads(OJ.read_text()) if OJ.exists() else {}
    cur[name] = jsonable(obj)
    txt = json.dumps(cur, indent=1) + "\n"
    tmp = LC / f"v_law.json.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(txt)
        f.flush()
        os.fsync(f.fileno())
    assert tmp.read_text() == txt, tmp
    os.replace(tmp, OJ)
    _fsync_dir(LC)


def write_arrays(prefix, arrs):
    if SMOKE:
        print("SMOKE (not written)", prefix, {k: np.asarray(v).shape for k, v in arrs.items()})
        return
    new = {}
    if OZ.exists():
        with np.load(OZ) as z:
            new = {k: z[k] for k in z.files if not k.startswith(prefix + "_")}
    new.update({f"{prefix}_{k}": np.asarray(v) for k, v in arrs.items()})
    tmp = LC / f"v_cells.tmp{os.getpid()}.npz"
    np.savez(tmp, **new)
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    with np.load(tmp) as z:
        assert sorted(z.files) == sorted(new), tmp
        for k, v in new.items():
            assert np.array_equal(z[k], v, equal_nan=v.dtype.kind == "f"), (tmp, k)
    os.replace(tmp, OZ)
    _fsync_dir(LC)


# ------------------------------------------------------------------ noise (real code path)
W10, W11 = (136, 928), (136, 1160)
OV = (29000, 30240)                 # global columns shared by windows 10 (23200..30240) and 11 (29000..36040)
BLOCKS = (16, 128, 352, 704, 1408)


def _wstats(x):
    a = x.astype(np.float32)        # as on the GPU: float16 tile window, .float()
    v = a.astype(np.float64)
    H, W = a.shape
    o = dict(n=v.size, s1=v.sum(), s2=np.dot(v.ravel(), v.ravel()))
    v2 = v * v
    o["s3"] = float(np.dot(v2.ravel(), v.ravel()))
    o["s4"] = float(np.dot(v2.ravel(), v2.ravel()))
    del v2
    o["c3"], o["c4"] = int((np.abs(a) > 3).sum()), int((np.abs(a) > 4).sum())
    o["lagc"], o["nlagc"] = float(np.einsum("ij,ij->", v[:, :-1], v[:, 1:])), v[:, :-1].size
    o["lagr"], o["nlagr"] = float(np.einsum("ij,ij->", v[:-1], v[1:])), v[:-1].size
    for b in BLOCKS:
        bm = v.reshape(H // b, b, W // b, b).mean((1, 3))
        o[f"bp{b}"], o[f"nb{b}"] = float((bm ** 2).sum() * b * b), bm.size
    o["bm128"] = v.reshape(H // 128, 128, W // 128, 128).mean((1, 3)).astype(np.float32)
    return o


def _noise_job(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    import hashlib
    import arnoise22 as an
    egg = an.egg
    kind = job[0]
    if kind in ("st", "ar"):
        _, d, h, m, (k0, j0) = job
        nz = egg.Npy(egg.G / "noise_tiles16.npy") if kind == "st" else an.ARNoise(2022, d, h, RHO, "philox", 16)
        x = egg.tile_window(nz, m, 25 * k0, 25 * j0, egg.PH, egg.PW)
        assert x.dtype == np.float16 and x.shape == (egg.PH, egg.PW), (x.dtype, x.shape)
        o = _wstats(x)
        ov = np.ascontiguousarray(x[:, OV[0] - 25 * j0:OV[1] - 25 * j0])
        assert ov.shape[1] == OV[1] - OV[0]
        o["ov_sha"] = hashlib.sha256(ov.tobytes()).hexdigest()
        return job, o
    if kind == "gst":
        _, m = job
        a = np.load(G / "noise_tiles16.npy", mmap_mode="r")[m]
        mu, va = np.empty((29, 71)), np.empty((29, 71))
        for tr in range(29):
            r = np.asarray(a[tr], np.float32)
            mu[tr], va[tr] = r.mean((1, 2), dtype=np.float64), r.var((1, 2), dtype=np.float64)
        return job, dict(mu=mu, va=va)
    if kind == "gar":
        _, d, m, h = job
        nz = an.ARNoise(2022, d, h, RHO, "philox", 16)
        mu, va = np.empty((29, 71)), np.empty((29, 71))
        for tr in range(29):
            for tc in range(71):
                t = nz.tile(m, tr, tc)
                assert t.dtype == np.float16
                t = t.astype(np.float32)
                mu[tr, tc], va[tr, tc] = t.mean(dtype=np.float64), t.var(dtype=np.float64)
        return job, dict(mu=mu, va=va)
    raise ValueError(kind)


def _pool_moments(rows):
    n = sum(r["n"] for r in rows)
    s1, s2, s3, s4 = (sum(r[k] for r in rows) / n for k in ("s1", "s2", "s3", "s4"))
    mu = s1
    m2 = s2 - mu * mu
    m3 = s3 - 3 * mu * s2 + 2 * mu ** 3
    m4 = s4 - 4 * mu * s3 + 6 * mu * mu * s2 - 3 * mu ** 4
    c3, c4 = sum(r["c3"] for r in rows) / n, sum(r["c4"] for r in rows) / n
    out = dict(windows=len(rows), values=n, mean=mu, mean_z=mu * np.sqrt(n), var=m2, var_se_white=np.sqrt(2.0 / n),
               skew=m3 / m2 ** 1.5, excess_kurtosis=m4 / m2 ** 2 - 3, P_abs_gt3=c3, P_abs_gt3_normal=0.0026998,
               P_abs_gt3_z=(c3 - 0.0026998) / np.sqrt(0.0026998 / n), P_abs_gt4=c4, P_abs_gt4_normal=6.3342e-05,
               P_abs_gt4_z=(c4 - 6.3342e-05) / np.sqrt(6.3342e-05 / n),
               lag1_cols=(sum(r["lagc"] for r in rows) / sum(r["nlagc"] for r in rows) - mu * mu) / m2,
               lag1_rows=(sum(r["lagr"] for r in rows) / sum(r["nlagr"] for r in rows) - mu * mu) / m2)
    for b in BLOCKS:
        nb = sum(r[f"nb{b}"] for r in rows)
        out[f"blockpower_{b}px"] = dict(value=sum(r[f"bp{b}"] for r in rows) / nb, se_white=np.sqrt(2.0 / nb), blocks=nb)
    return out


def _xcorr(bms):
    """bms: list of (33, 55) block-mean maps of different members: mean r^2 x blocks over pairs (independent: 1)"""
    X = np.stack([b.ravel().astype(np.float64) for b in bms])
    X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    C = X @ X.T / X.shape[1]
    iu = np.triu_indices(len(bms), 1)
    return dict(pairs=len(iu[0]), mean_r2_x_n=float((C[iu] ** 2).mean() * X.shape[1]),
                se_white=float(np.sqrt(2.0 / len(iu[0]))), max_abs_r=float(np.abs(C[iu]).max()), n=X.shape[1])


def _tile_summary(fields):
    """fields: list of dict(mu, va) on the (29, 71) tile grid; z = tile mean x 512 (white: N(0, 1))"""
    Z = np.stack([f["mu"] * 512.0 for f in fields])
    V = np.stack([f["va"] for f in fields])
    out = dict(fields=len(fields), tiles=Z[0].size, var_z=float(Z.var()), mean_z=float(Z.mean()),
               var_z_se_white=float(np.sqrt(2.0 / Z.size)))
    for k in (2, 4):
        R, C = 29 // k * k, 71 // k * k
        zk = Z[:, :R, :C].reshape(len(fields), R // k, k, C // k, k).sum((2, 4)) / k
        out[f"var_z_{k}x{k}tiles"] = float((zk ** 2).mean())
        out[f"var_z_{k}x{k}tiles_se_white"] = float(np.sqrt(2.0 / zk.size))
    c = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    out["corr_EW_adjacent_tiles"] = float(np.mean([c(z[:, :-1], z[:, 1:]) for z in Z]))
    out["corr_NS_adjacent_tiles"] = float(np.mean([c(z[:-1], z[1:]) for z in Z]))
    out["corr_adjacent_se_white"] = float(1.0 / np.sqrt(Z[0].size * len(fields)))
    out["tile_var_mean"] = float(V.mean())
    out["tile_var_dispersion_over_white"] = float(V.var() / (2.0 / (512 * 512 - 1)))
    if len(fields) > 2:
        X = Z.reshape(len(fields), -1)
        X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
        Cm = X @ X.T / X.shape[1]
        iu = np.triu_indices(len(fields), 1)
        out["cross_field_mean_r2_x_n"] = float((Cm[iu] ** 2).mean() * X.shape[1])
        out["cross_field_mean_r2_x_n_se_white"] = float(np.sqrt(2.0 / len(iu[0])))
        out["cross_field_max_abs_r"] = float(np.abs(Cm[iu]).max())
    return out


def mode_noise(nw):
    from multiprocessing import get_context
    t0 = time.time()
    sys.path.insert(0, str(P / "a16"))
    import arnoise22 as an           # noqa: F401  (asserts numpy 1.26.3; imported before the fork)
    win = [tuple(int(v) for v in w) for w in np.load(RUN / "eval_final_crps_e01" / "part_0.npz")["windows"]]
    assert win[10] == W10 and win[11] == W11, (win[10], win[11])
    starts, leads = (101, 263), (1, 3, 7)
    jobs = [("st", None, None, m, w) for m in range(16) for w in (W10, W11)]
    jobs += [("ar", d, h, m, w) for d in starts for h in leads for m in range(16) for w in (W10, W11)]
    gar = [(37, 0), (37, 9), (149, 3), (149, 14), (301, 6), (301, 11)]
    jobs += [("gst", m) for m in range(16)] + [("gar", d, m, h) for d, m in gar for h in (1, 3)]
    if SMOKE:
        jobs = [j for j in jobs if (j[0] == "st" and j[3] < 2) or (j[0] == "ar" and j[1] == 101 and j[2] in (1, 3) and j[3] < 2)
                or j == ("gst", 0) or j == ("gar", 37, 0, 1)]
    res = {}
    with get_context("fork").Pool(nw) as pool:
        for c, (j, o) in enumerate(pool.imap_unordered(_noise_job, jobs, chunksize=1)):
            res[j] = o
            if c % 20 == 0:
                print(f"noise {c + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    out = {"what": "input noise as it enters the sampler (float16 window of tile_window, cast to float32), through the real "
                   "arnoise22.ARNoise / noise_tiles16.npy; windows 10 and 11 of the production layout (k0, j0) = "
                   f"{W10}, {W11}; ar starts {starts}, leads {leads}, members 0-15; stored members 0-15 (fixed)",
           "numpy": np.__version__}
    st = [o for j, o in res.items() if j[0] == "st"]
    out["stored16_windows"] = _pool_moments(st)
    for h in leads:
        rows = [o for j, o in res.items() if j[0] == "ar" and j[2] == h]
        if rows:
            out[f"ar_lead{h}_windows"] = _pool_moments(rows)
    # per-field large-scale power (the 'luck' of one field): stored members vs fresh ar fields
    for b in (704, 1408):
        sv = np.array([o[f"bp{b}"] / o[f"nb{b}"] for j, o in res.items() if j[0] == "st"])
        av = np.array([o[f"bp{b}"] / o[f"nb{b}"] for j, o in res.items() if j[0] == "ar"])
        out[f"per_field_blockpower_{b}px"] = dict(stored_mean=sv.mean(), stored_sd=sv.std(ddof=1), stored_n=len(sv),
                                                  ar_mean=av.mean() if len(av) else None,
                                                  ar_sd=av.std(ddof=1) if len(av) > 1 else None, ar_n=len(av),
                                                  white_sd=np.sqrt(2.0 / res[jobs[0]][f"nb{b}"]))
    # cross-member independence (128-px block means), per window
    xc = {}
    for w in (W10, W11):
        bms = [res[j]["bm128"] for j in jobs if j[0] == "st" and j[4] == w]
        if len(bms) > 1:
            xc[f"stored_w{w[1]}"] = _xcorr(bms)
        for d in starts:
            for h in leads:
                bms = [res[j]["bm128"] for j in jobs if j[0] == "ar" and j[1] == d and j[2] == h and j[4] == w]
                if len(bms) > 1:
                    xc[f"ar_d{d}_h{h}_w{w[1]}"] = _xcorr(bms)
    out["cross_member_128px"] = xc
    # cross-lead correlation of the same member (ar): rho^(h-1) expected; and between starts (0 expected)
    cl = {}
    for h in leads[1:]:
        r = [float(np.corrcoef(res[("ar", d, 1, m, w)]["bm128"].ravel(), res[("ar", d, h, m, w)]["bm128"].ravel())[0, 1])
             for d in starts for m in range(16) for w in (W10, W11) if ("ar", d, h, m, w) in res and ("ar", d, 1, m, w) in res]
        if r:
            cl[f"lead1_vs_lead{h}"] = dict(mean_r=float(np.mean(r)), expected=RHO ** (h - 1), n=len(r))
    r = [float(np.corrcoef(res[("ar", starts[0], 1, m, w)]["bm128"].ravel(), res[("ar", starts[1], 1, m, w)]["bm128"].ravel())[0, 1])
         for m in range(16) for w in (W10, W11) if ("ar", starts[1], 1, m, w) in res and ("ar", starts[0], 1, m, w) in res]
    if r:
        cl["start101_vs_start263_lead1"] = dict(mean_r=float(np.mean(r)), max_abs=float(np.max(np.abs(r))), n=len(r))
    out["cross_lead_128px"] = cl
    # the same pixel gets the same value in both overlapping windows
    ov = []
    for j in jobs:
        if j[0] in ("st", "ar") and j[4] == W10:
            j2 = j[:4] + (W11,)
            if j2 in res:
                ov.append(res[j]["ov_sha"] == res[j2]["ov_sha"])
    out["overlap_windows_10_11_identical"] = dict(all=bool(all(ov)), n=len(ov))
    out["global_tiles_stored16"] = _tile_summary([res[("gst", m)] for m in range(16) if ("gst", m) in res])
    for h in (1, 3):
        f = [res[("gar", d, m, h)] for d, m in gar if ("gar", d, m, h) in res]
        if f:
            out[f"global_tiles_ar_lead{h}"] = _tile_summary(f)
    out["seconds"] = time.time() - t0
    write_json_section("noise", out)
    print("noise done", time.time() - t0, flush=True)


# ------------------------------------------------------------------ populated pixels, 16 vs 16
STATS = ["n", "m16", "m4", "v16", "v8", "v4", "v4b", "lv16", "c37lo", "c37hi", "c75lo", "c75hi", "c150lo", "c150hi",
         "np1", "St1", "Sd1", "np4", "St4", "Sd4", "np16", "St16", "Sd16"]
SI = {s: i for i, s in enumerate(STATS)}
LAGS = (1, 4, 16)
STORED16 = [P / "a13/warn16/data/run16", P / "a14/data/run16_13"]
TASK0_DIR, TASK0 = P / "a13/warn16/data/run16all", {(2, 1), (2, 3), (4, 1), (4, 3)}
AR16 = P / "a16/data/run16"
_PP = {}


def index16(dirs, only=None):
    out = {}
    for D in dirs:
        for fk in sorted(Path(D).glob("keys16_*.npy")):
            tag = fk.stem[len("keys16_"):]
            done = set(int(v) for v in json.loads((Path(D) / f"done16_{tag}.json").read_text()))
            for n, (d, h, t) in enumerate(np.load(fk).tolist()):
                if n in done and (only is None or (d, h) in only):
                    assert (d, h) not in out, (D, d, h)
                    assert t == d + h, (D, d, h, t)
                    out[(d, h)] = (str(Path(D) / f"pop16_{tag}.npy"), n)
    return out


def _pp_init(cell, cblk, pairs, st):
    os.environ["OMP_NUM_THREADS"] = "1"
    _PP.update(cell=cell, cblk=cblk, pairs=pairs, st=st)


def _pp_stats(X, ok):
    cell = _PP["cell"]
    out = np.zeros((len(STATS), NC))
    i = np.flatnonzero(ok)
    c = cell[i]
    Y = X[i]

    def acc(name, w):
        out[SI[name]] = np.bincount(c, weights=np.asarray(w, np.float64), minlength=NC)
    acc("n", np.ones(len(i)))
    mean16 = Y.mean(1, dtype=np.float64)
    acc("m16", mean16)
    acc("m4", Y[:, :4].mean(1, dtype=np.float64))
    acc("v16", Y.var(1, ddof=1))
    acc("v8", Y[:, :8].var(1, ddof=1))
    acc("v4", Y[:, :4].var(1, ddof=1))
    acc("v4b", Y[:, 4:8].var(1, ddof=1))
    L = np.log1p(np.maximum(Y, 0))
    acc("lv16", L.var(1, ddof=1))
    del L
    for t, nm in ((37.5, "c37"), (75.0, "c75"), (150.0, "c150")):
        acc(nm + "lo", (Y[:, :8] > t).sum(1))
        acc(nm + "hi", (Y[:, 8:] > t).sum(1))
    del Y
    fm = np.zeros(X.shape[0])
    fm[i] = mean16
    for lag, (pi, pj) in zip(LAGS, _PP["pairs"]):
        okp = ok[pi] & ok[pj]
        a, b = pi[okp], pj[okp]
        D = X[a] - X[b]
        St = (D * D).sum(1, dtype=np.float64)
        del D
        dm = fm[a] - fm[b]
        Sd = St - 16.0 * dm * dm
        cc = cell[a]
        out[SI[f"np{lag}"]] = np.bincount(cc, minlength=NC)
        out[SI[f"St{lag}"]] = np.bincount(cc, weights=St, minlength=NC)
        out[SI[f"Sd{lag}"]] = np.bincount(cc, weights=Sd, minlength=NC)
    blk = np.stack([np.bincount(_PP["cblk"], weights=o, minlength=540) for o in out])
    st = _PP["st"]
    sv = np.stack([mean16_at(X, st), X[st].var(1, ddof=1), X[st, :8].var(1, ddof=1), X[st, :4].var(1, ddof=1)])
    return out, blk, sv


def mean16_at(X, st):
    return X[st].mean(1, dtype=np.float64)


def _pop_one(job):
    (d, h), (fs, ns), (fa, na) = job
    Xs = np.array(np.load(fs, mmap_mode="r")[ns], np.float32)
    Xa = np.array(np.load(fa, mmap_mode="r")[na], np.float32)
    assert Xs.shape == Xa.shape and Xs.shape[1] == 16, (Xs.shape, Xa.shape)
    ok = np.isfinite(Xs).all(1) & np.isfinite(Xa).all(1)
    rs, bs, ss = _pp_stats(Xs, ok)
    ra, ba, sa = _pp_stats(Xa, ok)
    return (d, h), np.stack([rs, ra]), np.stack([bs, ba]), np.stack([ss, sa]), ok[_PP["st"]], int(ok.sum())


def mode_pop(nw):
    from multiprocessing import get_context
    t0 = time.time()
    S = index16(STORED16)
    S.update(index16([TASK0_DIR], only=TASK0))
    A = index16([AR16])
    want = {(d, h) for d in range(0, 357, 2) for h in (1, 3)}
    assert set(S) == want and set(A) == want, (len(S), len(A), len(set(S) ^ want), len(set(A) ^ want))
    keys = sorted(want)
    if SMOKE:
        keys = [(2, 1), (100, 3)]
    idx = np.load(G / "pop_pixels.npz")["idx"]
    assert np.all(np.diff(idx) > 0)
    row, col = idx // GW, idx % GW
    cell = ((row // 100) * 360 + col // 100).astype(np.int64)
    cr, cc = np.arange(NC) // 360, np.arange(NC) % 360
    cblk = ((cr // 10) * 36 + cc // 10).astype(np.int64)
    pairs = []
    for lag in LAGS:
        j = np.searchsorted(idx, idx + lag)
        jj = np.minimum(j, len(idx) - 1)
        good = (j < len(idx)) & (idx[jj] == idx + lag) & (col + lag < GW)
        pairs.append((np.flatnonzero(good).astype(np.int64), jj[good].astype(np.int64)))
    with np.load(G / "stations_2022.npz") as z:      # coordinates only
        slat, slon = z["lat"], z["lon"]
    sr = np.clip(np.floor((84.0 - slat) / 0.01).astype(np.int64), 0, 14399)
    sc = np.clip(np.floor((slon + 180.0) / 0.01).astype(np.int64), 0, GW - 1)
    pix = sr * GW + sc
    pos = np.clip(np.searchsorted(idx, pix), 0, len(idx) - 1)
    st = np.unique(pos[idx[pos] == pix])
    F = len(keys)
    CELLS = np.zeros((2, 2, len(STATS), NC))          # lead index (1, 3), run (stored, ar), stat, cell
    BLK = np.zeros((F, 2, len(STATS), 540))
    STV = np.zeros((2, 2, 4, len(st)))
    STN = np.zeros((2, len(st)))
    okn = np.zeros(F, np.int64)
    pos_of = {k: n for n, k in enumerate(keys)}
    jobs = [(k, S[k], A[k]) for k in keys]
    with get_context("fork").Pool(nw, initializer=_pp_init, initargs=(cell, cblk, pairs, st)) as pool:
        for c, (k, r, b, s, okst, n) in enumerate(pool.imap_unordered(_pop_one, jobs, chunksize=1)):
            li = (1, 3).index(k[1])
            CELLS[li] += r
            BLK[pos_of[k]] = b
            STV[li] += np.where(okst[None, None], np.nan_to_num(s), 0.0)
            STN[li] += okst
            okn[pos_of[k]] = n
            if c % 20 == 0:
                print(f"pop {c + 1}/{F} {time.time() - t0:.0f}s", flush=True)
    write_arrays("pop", dict(keys=np.array(keys, np.int32), cells=CELLS, blk=BLK, stv=STV, stn=STN, okn=okn,
                             st_pos=st, st_idx=idx[st], npairs=np.array([len(p[0]) for p in pairs])))
    print("pop done", time.time() - t0, flush=True)


# ------------------------------------------------------------------ East Asia crop, 16 vs 16, leads 1-7
PR0, PC0, PS, NPR, NPC = 50, 150, 100, 8, 17       # patch grid inside the crop; NCP box = patch rows 0-5, cols 0-7
CSTATS = ["n", "m16", "v16", "v4", "b0", "b1", "b2", "c75"]


def _crop_one(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    from scipy.ndimage import gaussian_filter
    d, h = job
    Xs = np.load(RUN / "eval_case_eastasia_crps_e01" / f"crop0_d{d}_h{h}.npz")["members"].astype(np.float64)
    Xa = np.load(RUN / "eval_case_eastasia_crps_e01_ar" / f"crop0_d{d}_h{h}.npz")["members"].astype(np.float64)
    assert Xs.shape == Xa.shape == (16, 900, 1900), (Xs.shape, Xa.shape)
    ok = np.isfinite(Xs).all(0) & np.isfinite(Xa).all(0)
    pr, pc = np.arange(900)[:, None] - PR0, np.arange(1900)[None, :] - PC0
    inside = (pr >= 0) & (pr < NPR * PS) & (pc >= 0) & (pc < NPC * PS)
    pid = np.where(inside, (pr // PS) * NPC + pc // PS, -1)
    sel = ok & inside
    p = pid[sel]
    w = ok.astype(np.float64)
    Gw = {s: np.maximum(gaussian_filter(w, s, mode="constant", truncate=3.0), 1e-6) for s in (2, 8, 32)}
    out = np.zeros((2, len(CSTATS), NPR * NPC))
    for r, X in enumerate((Xs, Xa)):
        X = np.where(ok[None], X, 0.0)
        b = np.zeros((3, 900, 1900))
        for m in range(16):
            g = {s: gaussian_filter(X[m], s, mode="constant", truncate=3.0) / Gw[s] for s in (2, 8, 32)}
            b[0] += (X[m] - g[2]) ** 2
            b[1] += (g[2] - g[8]) ** 2
            b[2] += (g[8] - g[32]) ** 2
        vals = dict(n=np.ones(sel.sum()), m16=X.mean(0)[sel], v16=X.var(0, ddof=1)[sel], v4=X[:4].var(0, ddof=1)[sel],
                    b0=b[0][sel] / 16, b1=b[1][sel] / 16, b2=b[2][sel] / 16, c75=(X > 75).sum(0)[sel])
        for k, nm in enumerate(CSTATS):
            out[r, k] = np.bincount(p, weights=vals[nm], minlength=NPR * NPC)
    return (d, h), out


def mode_crop(nw):
    from multiprocessing import get_context
    t0 = time.time()
    jobs = [(d, h) for d in range(28) for h in range(1, 8)]
    if SMOKE:
        jobs = [(0, 1), (5, 7)]
    arr = np.zeros((28, 7, 2, len(CSTATS), NPR * NPC))
    with get_context("fork").Pool(nw) as pool:
        for (d, h), o in pool.imap_unordered(_crop_one, jobs, chunksize=1):
            arr[d, h - 1] = o
    write_arrays("crop", dict(arr=arr))
    print("crop done", time.time() - t0, flush=True)


# ------------------------------------------------------------------ F1 vs G1, 4 members, 0.25 deg maps, leads 1-7
def mode_map25():
    t0 = time.time()
    CEL = np.zeros((7, 2, 3, NC))                      # lead, run, (n, sd, m0), 1-degree cell
    BL = {}
    seen = set()
    cr, cc = np.arange(NC) // 360, np.arange(NC) % 360
    cblk = ((cr // 10) * 36 + cc // 10).astype(np.int64)
    ranks = range(2) if SMOKE else range(64)
    for r in ranks:
        zs = np.load(RUN / "eval_final_crps_e01" / f"part_{r}.npz")
        za = np.load(RUN / "eval_final_crps_e01_ar" / f"part_{r}.npz")
        ks, ka = zs["keys"], za["keys"]
        assert np.array_equal(ks, ka), r
        sds, sda, m0s, m0a = zs["map25_sd"], za["map25_sd"], zs["map25_m0"], za["map25_m0"]
        for i, (d, h, t) in enumerate(ks.tolist()):
            assert (d, h) not in seen and 1 <= h <= 7 and t == d + h
            seen.add((d, h))
            a = [np.asarray(x[i], np.float64) for x in (sds, sda, m0s, m0a)]
            ok = np.isfinite(a[0]) & np.isfinite(a[1]) & np.isfinite(a[2]) & np.isfinite(a[3])
            agg = lambda v: np.where(ok, v, 0.0).reshape(144, 4, 360, 4).sum((1, 3)).ravel()
            n = agg(np.ones_like(a[0]))
            for run in (0, 1):
                CEL[h - 1, run, 0] += n
                CEL[h - 1, run, 1] += agg(a[run])
                CEL[h - 1, run, 2] += agg(a[2 + run])
            BL[(d, h)] = np.stack([np.stack([np.bincount(cblk, weights=v, minlength=540) for v in (n, agg(a[run]), agg(a[2 + run]))])
                                   for run in (0, 1)])
    if not SMOKE:
        assert len(seen) == 1253, len(seen)
    keys = sorted(BL)
    write_arrays("m25", dict(cells=CEL, keys=np.array(keys, np.int32), blk=np.stack([BL[k] for k in keys])))
    print("map25 done", time.time() - t0, flush=True)


# ------------------------------------------------------------------ final analysis
def boot(bid, a, b, B=2000, seed=1):
    """R = sum a / sum b over cells; percentile interval from a bootstrap over the blocks bid"""
    ub, inv = np.unique(bid, return_inverse=True)
    A = np.bincount(inv, weights=a, minlength=len(ub))
    Bs = np.bincount(inv, weights=b, minlength=len(ub))
    R = A.sum() / Bs.sum()
    rng = np.random.default_rng(seed)
    dr = rng.integers(0, len(ub), (B, len(ub)))
    rb = A[dr].sum(1) / Bs[dr].sum(1)
    lo, hi = np.quantile(rb, [0.025, 0.975])
    return dict(R=R, ci=[lo, hi], se=float(rb.std()), blocks=len(ub), cells=len(a),
                detectable_80pct_power=float(2.8 * rb.std() / R))


def bid_of(cells, deg):
    r, c = cells // 360, cells % 360
    return (r // deg) * 1000 + c // deg


def correlogram(e, valid):
    """correlation of e between 1-degree cells at E-W and N-S lags (cells on the 144 x 360 grid)"""
    E = np.full(NC, np.nan)
    E[valid] = e
    E = E.reshape(144, 360)
    out = {}
    for lag in (1, 2, 4, 8, 16):
        for nm, (x, y) in {"EW": (E[:, :-lag], E[:, lag:]), "NS": (E[:-lag], E[lag:])}.items():
            m = np.isfinite(x) & np.isfinite(y)
            out[f"{nm}{lag}"] = [float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())] if m.sum() > 10 else None
    return out


def final():
    z = np.load(OZ)
    res = {"what": "ratios are ar/stored. 'eq' = draw-robust: every cell weighted equally through a normaliser free of the "
                   "stored draw (ar 16-member value of the same cell), sum(T_a/N)/sum(T_s/N); under an unchanged law the stored "
                   "luck of distant cells is independent, so it averages out over many cells. 'hot' = sum T_a / sum T_s "
                   "(dominated by the polluted cells, as the global ratios of lawcheck.py). Intervals: bootstrap over spatial "
                   "blocks of 10, 20, 30 degrees (cells). detectable_80pct_power = 2.8 se / R."}
    # ---------------- pop
    if "pop_cells" in z.files:
        C = z["pop_cells"]
        pop = {}
        for li, h in enumerate((1, 3)):
            s, a = C[li, 0], C[li, 1]
            n = s[SI["n"]]
            assert np.array_equal(n, a[SI["n"]])
            nf = 179
            valid = (n / nf >= 200) & (a[SI["v16"]] > 0) & (s[SI["v16"]] > 0)      # 2 cells with v16 = 0 dropped
            cells = np.flatnonzero(valid)
            Na = a[:, valid]
            Ns = s[:, valid]
            out = {"cells": int(valid.sum()), "pixel_share_in_cells": float(n[valid].sum() / n.sum())}
            for deg in (10, 20, 30):
                bid = bid_of(cells, deg)
                o = {}
                # levels and variances, normaliser = ar 16-member value of the cell
                for nm, norm in (("m16", "m16"), ("m4", "m16"), ("v16", "v16"), ("v8", "v16"), ("v4", "v16"), ("v4b", "v16"),
                                 ("lv16", "lv16")):
                    Nn = Na[SI[norm]]
                    ok = Nn > 0
                    o[f"eq_{nm}"] = boot(bid[ok], Na[SI[norm], ok] / Nn[ok], Ns[SI[nm], ok] / Nn[ok])
                    o[f"hot_{nm}"] = boot(bid, Na[SI[nm]], Ns[SI[nm]])
                # controls inside ar: quartet 0-3 and 8 members against the ar 16 (expected 1: fresh draws)
                o["ctrl_ar_v4_over_ar_v16_eq"] = boot(bid, Na[SI["v4"]] / Na[SI["v16"]], np.ones(len(cells)))
                # polluted cells only (ar level >= 35 ug m-3)
                pol = Na[SI["m16"]] / Na[SI["n"]] >= 35
                for nm in ("v16", "v4", "m16"):
                    norm = "m16" if nm == "m16" else "v16"
                    o[f"eq_polluted_{nm}"] = boot(bid[pol], Na[SI[norm], pol] / Na[SI[norm], pol], Ns[SI[nm], pol] / Na[SI[norm], pol])
                o["polluted_cells"] = int(pol.sum())
                # exceedances: ratio on one half of the members, normaliser and selection from the other half (both runs)
                for t in ("c37", "c75", "c150"):
                    for half, other in (("lo", "hi"), ("hi", "lo")):
                        Nn = 0.5 * (Na[SI[t + other]] + Ns[SI[t + other]])
                        sel = Nn >= 50
                        o[f"eq_{t}{half}"] = boot(bid[sel], Na[SI[t + half], sel] / Nn[sel], Ns[SI[t + half], sel] / Nn[sel])
                        o[f"eq_{t}{half}"]["min_other_half_count"] = 50
                    o[f"hot_{t}"] = boot(bid, Na[SI[t + "lo"]] + Na[SI[t + "hi"]], Ns[SI[t + "lo"]] + Ns[SI[t + "hi"]])
                # structure functions (smoothness), total and member departure
                for lag in LAGS:
                    for k in ("St", "Sd"):
                        nm = f"{k}{lag}"
                        Nn = Na[SI[nm]]
                        ok = (Na[SI[f"np{lag}"]] >= 179 * 50) & (Nn > 0)
                        o[f"eq_{nm}"] = boot(bid[ok], Nn[ok] / Nn[ok], Ns[SI[nm], ok] / Nn[ok])
                        o[f"hot_{nm}"] = boot(bid, Na[SI[nm]], Ns[SI[nm]])
                out[f"blocks{deg}"] = o
            # magnitude and range of the stored luck per cell: e = log(T_s / T_a)
            for nm in ("v16", "v4", "m16"):
                norm = "m16" if nm == "m16" else "v16"
                e = np.log(Ns[SI[nm]] / Na[SI[norm]])
                out[f"luck_{nm}"] = dict(sd_cells=float(e.std()), mean=float(e.mean()), correlogram=correlogram(e, valid))
            ec = np.log(Na[SI["v4"]] / Na[SI["v16"]])
            out["luck_ctrl_ar_v4"] = dict(sd_cells=float(ec.std()), mean=float(ec.mean()), correlogram=correlogram(ec, valid))
            pop[f"lead{h}"] = out
        # station pixels (coordinates only; model values only): equal weight per station pixel
        STV, STN, sidx = z["pop_stv"], z["pop_stn"], z["pop_st_idx"]
        srow, scol = sidx // GW, sidx % GW
        stp = {}
        for li, h in enumerate((1, 3)):
            o = {"station_pixels": int((STN[li] > 0).sum())}
            ok = (STN[li] >= 100) & (STV[li, 1, 1] > 0) & (STV[li, 1, 0] > 0)
            for deg in (5, 10):
                bid = (srow[ok] // (100 * deg)) * 1000 + scol[ok] // (100 * deg)
                v16a = STV[li, 1, 1, ok]
                for k, nm in ((1, "v16"), (2, "v8"), (3, "v4")):
                    o[f"eq_{nm}_blocks{deg}"] = boot(bid, v16a / v16a, STV[li, 0, k, ok] / v16a)
                    o[f"hot_{nm}_blocks{deg}"] = boot(bid, STV[li, 1, k, ok], STV[li, 0, k, ok])
                o[f"eq_m16_blocks{deg}"] = boot(bid, STV[li, 1, 0, ok] / STV[li, 1, 0, ok], STV[li, 0, 0, ok] / STV[li, 1, 0, ok])
            stp[f"lead{h}"] = o
        pop["station_pixels"] = stp
        res["pop"] = pop
    # ---------------- map25 (4 members, leads 1-7)
    if "m25_cells" in z.files:
        C = z["m25_cells"]
        m25 = {}
        nmax = 179 * 16
        valid = (C[:, 0, 0] >= nmax * 0.5).all(0)
        cells = np.flatnonzero(valid)
        for deg in (10, 30):
            bid = bid_of(cells, deg)
            ub, inv = np.unique(bid, return_inverse=True)
            o = {}
            for k, nm in ((1, "sd"), (2, "m0")):
                Rh, Hh, AB = [], [], []
                for h in range(7):
                    a, s = C[h, 1, k, valid], C[h, 0, k, valid]
                    ok = a > 0
                    o[f"eq_{nm}_lead{h + 1}"] = boot(bid[ok], np.ones(ok.sum()), s[ok] / a[ok])
                    o[f"hot_{nm}_lead{h + 1}"] = boot(bid, a, s)
                    AB.append((np.bincount(inv, weights=np.where(ok, 1.0, 0.0), minlength=len(ub)),
                               np.bincount(inv, weights=np.where(ok, s / np.where(ok, a, 1), 0.0), minlength=len(ub))))
                # lead drift of the eq ratio, bootstrap over blocks paired across leads
                rng = np.random.default_rng(7)
                dr = rng.integers(0, len(ub), (2000, len(ub)))
                Rb = np.stack([A[dr].sum(1) / Bb[dr].sum(1) for A, Bb in AB])         # (7, B)
                R0 = np.array([A.sum() / Bb.sum() for A, Bb in AB])
                hh = np.arange(1, 8) - 4.0
                slope = lambda R: np.polyfit(hh, np.log(R), 1)[0]
                sb = np.array([slope(Rb[:, j]) for j in range(Rb.shape[1])])
                o[f"eq_{nm}_lead_drift"] = dict(R_by_lead=R0, R7_over_R1=R0[6] / R0[0],
                                                R7_over_R1_ci=np.quantile(Rb[6] / Rb[0], [0.025, 0.975]),
                                                slope_log_per_lead=slope(R0), slope_ci=np.quantile(sb, [0.025, 0.975]))
            m25[f"blocks{deg}"] = o
        m25["cells"] = int(valid.sum())
        e = np.log(C[0, 0, 1, valid] / C[0, 1, 1, valid])
        m25["luck_sd_lead1"] = dict(sd_cells=float(e.std()), correlogram=correlogram(e, valid))
        e7 = np.log(C[6, 0, 1, valid] / C[6, 1, 1, valid])
        m25["luck_sd_corr_lead1_lead7"] = float(np.corrcoef(e, e7)[0, 1])
        res["map25_F1_G1"] = m25
    # ---------------- crop (16 members, leads 1-7)
    if "crop_arr" in z.files:
        A = z["crop_arr"]                                   # (28, 7, 2, stats, patches)
        T = A.sum(0)                                        # (7, 2, stats, patches)
        n = T[:, 0, 0]
        valid = (n >= 28 * 3000).all(0)
        pr, pc = np.arange(NPR * NPC) // NPC, np.arange(NPR * NPC) % NPC
        ncp = (pr <= 5) & (pc <= 7)
        cr = {"patches_valid": int(valid.sum()), "ncp_patches_valid": int((valid & ncp).sum())}
        bid = (pr // 3) * 100 + pc // 3
        for nm in ("b0", "b1", "b2", "v16", "v4", "m16"):
            k = CSTATS.index(nm)
            norm = CSTATS.index("v16") if nm == "v4" else k
            per = {}
            for h in range(7):
                a, s = T[h, 1, norm], T[h, 0, k]
                per[f"lead{h + 1}"] = dict(all=boot(bid[valid], np.ones(valid.sum()), s[valid] / a[valid]),
                                           ncp=boot(bid[valid & ncp], np.ones((valid & ncp).sum()), s[valid & ncp] / a[valid & ncp]),
                                           ncp_hot=float(a[valid & ncp].sum() / s[valid & ncp].sum()))
            Rl = np.array([per[f"lead{h + 1}"]["all"]["R"] for h in range(7)])
            per["slope_log_per_lead_all"] = float(np.polyfit(np.arange(1, 8), np.log(Rl), 1)[0])
            # bootstrap over start days (field draw NOT included: stored fields are fixed) for the slope
            rng = np.random.default_rng(3)
            sb = []
            for _ in range(1000):
                ii = rng.integers(0, 28, 28)
                Tb = A[ii].sum(0)
                Rb = [1.0 / np.mean(Tb[h, 0, k, valid] / Tb[h, 1, norm, valid]) for h in range(7)]
                sb.append(np.polyfit(np.arange(1, 8), np.log(Rb), 1)[0])
            per["slope_ci_starts_only"] = np.quantile(sb, [0.025, 0.975])
            cr[nm] = per
        res["crop_eastasia"] = cr
    write_json_section("analysis", res)
    print("final done", flush=True)


def mode_index():
    """Field-draw uncertainty with the member index as the unit (the 16 stored fields are exchangeable draws): read-only
    use of the per-forecast aggregates of lawcheck.py (pop_stats.npz: S1, Gram matrices, counts per member index), own
    code. Per-index level ratios and their lead 1 -> 3 drift; per-index exceedance ratios; pair groups (within indices
    0-7, within 8-15, across); rank of the paper's octet 0-7 among the 12,870 stored octets; and, as a check of the
    spatial-block bootstrap of 'analysis', the across-index spread of a single field's global level and drift."""
    from itertools import combinations
    z = np.load(LC / "pop_stats.npz")
    keys = z["keys"]
    out = {"source": str(LC / "pop_stats.npz") + " (s_* stored run16 sets, a_* a16/data/run16; 358 forecasts, 8.1 M pixels)"}
    lv = {}
    for lead in (1, 3):
        sel = keys[:, 1] == lead
        N = float(z["s_N"][sel].sum())
        assert N == float(z["a_N"][sel].sum())
        r = z["a_S1"][sel].sum(0) / z["s_S1"][sel].sum(0)
        lv[lead] = r
        o = {"level_ratio_by_index": r, "level_ratio_mean_over_indices": r.mean(), "se_over_indices": r.std(ddof=1) / 4,
             "per_field_sd": r.std(ddof=1)}
        for k, t in enumerate((37.5, 75.0, 150.0)):
            cs, ca = z["s_C"][sel, k].sum(0).astype(float), z["a_C"][sel, k].sum(0).astype(float)
            q = ca / cs
            o[f"exceed{t}"] = dict(ratio_all=ca.sum() / cs.sum(), by_index=q, mean_over_indices=q.mean(),
                                   se_over_indices=q.std(ddof=1) / 4, detectable_80pct_power=2.8 * q.std(ddof=1) / 4,
                                   ratio_indices_0_7=ca[:8].sum() / cs[:8].sum(), ratio_indices_8_15=ca[8:].sum() / cs[8:].sum(),
                                   indices_below_1=int((q < 1).sum()))
        Gs, Ga = z["s_G"][sel].sum(0) / N, z["a_G"][sel].sum(0) / N
        D = lambda Gm: 0.5 * (np.diag(Gm)[:, None] + np.diag(Gm)[None, :] - 2 * Gm)     # pixel mean of (x_a - x_b)^2 / 2
        Ds, Da = D(Gs), D(Ga)
        lo, hi, al = list(range(8)), list(range(8, 16)), list(range(16))
        grp = lambda Dm, A, B: np.mean([Dm[a, b] for a in A for b in B if (A is not B) or a < b])
        o["var_between_pair_groups_ar_over_stored"] = {nm: grp(Da, A, B) / grp(Ds, A, B) for nm, (A, B) in
                                                       {"within_0_7": (lo, lo), "within_8_15": (hi, hi), "across": (lo, hi),
                                                        "all_16": (al, al)}.items()}
        o["stored_within_8_15_over_within_0_7"] = grp(Ds, hi, hi) / grp(Ds, lo, lo)
        o["ar_within_8_15_over_within_0_7"] = grp(Da, hi, hi) / grp(Da, lo, lo)
        pv = lambda Dm, I: np.mean([Dm[a, b] for a, b in combinations(I, 2)])
        for size in (4, 8):
            sets = list(combinations(range(16), size))
            vs = np.array([pv(Ds, I) for I in sets])
            va = np.array([pv(Da, I) for I in sets])
            i0 = sets.index(tuple(range(size)))
            obs = va[i0] / vs[i0]
            null = va[i0] / vs
            o[f"sets_of_{size}"] = dict(n_sets=len(sets), stored_rank_of_first=float((vs < vs[i0]).mean()),
                                        ratio_first_ar_over_stored=obs, null_ratio_q025_q975=np.quantile(null, [0.025, 0.975]),
                                        share_as_far_from_1=float((np.abs(np.log(null)) >= abs(np.log(obs))).mean()),
                                        ar_sets_cv=va.std() / va.mean(), stored_sets_cv=vs.std() / vs.mean())
        o["per_index_mean_pair_var_stored_over_ar_mean"] = np.array([np.mean([Ds[m, k] for k in range(16) if k != m])
                                                                     for m in range(16)]) / Da[np.triu_indices(16, 1)].mean()
        out[f"lead{lead}"] = o
    d = np.log(lv[3] / lv[1])
    out["level_drift_lead1_to_3_by_index"] = dict(log_drift=d, mean=d.mean(), se_over_indices=d.std(ddof=1) / 4,
                                                  per_field_sd=d.std(ddof=1), index0=d[0])
    write_json_section("index_analysis", out)
    print("index done", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 16
    {"noise": lambda: mode_noise(nw), "pop": lambda: mode_pop(nw), "crop": lambda: mode_crop(nw),
     "map25": mode_map25, "final": final, "index": mode_index}[mode]()
