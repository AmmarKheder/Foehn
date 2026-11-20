#!/usr/bin/env python3
"""GHAP V1 store of the CAMS -> GHAP 1 km downscaler (PROTOCOL.md Amendment A2) in data/processed/ghap/.

Source (read only): eccv/data/world/global1km/ghap1km_global.npy (2,191 days 2017-01-01 .. 2022-12-31, 14,400 x 36,000
float16; row r at 83.995 - 0.01 r, column c at -179.995 + 0.01 c; 0 = sea / no retrieval). 2022 rows are NEVER read
here (sealed until the freeze, A2(b)); the 2022 test truth is extracted by test2022_ghap/extract_truth.py after it.

The lattice is cut into 24 latitude bands (edges r = 25 k + 13: one row off the 0.25 deg cell-row boundary r = 25 k + 12, so
the 23 edge cells are split between two bands and their cell means are merged by count in 'finish'); each band job
reads one contiguous byte range per day.

  scan  --band b   pass 1: per lattice pixel of the band, the number of valid days (GHAP > 0 and finite) over
                   2017-01-01 .. 2021-12-31 (cnt_all) and in 2021 (cnt21); over the training days 2017-2020 the one-pass
                   sums of the exact per-pixel harmonic fit of y = log1p(max(round(GHAP), 1)) on the pixel's valid days:
                   S (9) = sum y g_k, Q (5) = sum y^2 b_k, H (13) = sum_valid h_m (indicator harmonics up to 6), and the
                   per-pixel training maximum of k; per-day counts. -> build/scan_bNN.npz/.json
  merge            mask (>= 95 % valid days 2017-2021, nearest 0.25 deg point inside the grid), exact per-pixel K = 2
                   fit (mean, residual variance), sigma floor, statics s1-s6, cell ids, band quotas, CAMS/T6
                   normalisation on 2017-2020, empty point / hold-out files, tile-padding check -> grid/, ghap_clim/,
                   manifest_parts/ghap_merge.json
  points --band b  pass 2: 4 M cos(lat)-weighted points per training day (band quota), the 2021 hold-out full fields
                   (int16 k, -1 = hole), training-day 0.25 deg cell means of y (BCSD input), hole counts.
  finish           band outputs of pass 2 -> manifest.json (T_G, P_max, hole statistics), cellmean_train_G.npz
World venv (numpy, zarr), partition small: runs_ghap/build_*.sh."""
import argparse
import datetime as dt
import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
OUT = ROOT / "data/processed/ghap"
G1K = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
SRC = G1K / "ghap1km_global.npy"
TILES = G1K / "ghap1km_tiles.npy"
STATIC1KM = "/scratch/project_462001804/ammar/eccv/data/world/static/static_1km.zarr"
GLOBAL = ROOT / "data/processed/global"
NR, NC = 14400, 36000
LAT0, LON0, RES = 83.995, -179.995, 0.01          # row r: LAT0 - RES r (descending), column c: LON0 + RES c
D0, ND = dt.date(2017, 1, 1), 2191
TRAIN0, TRAIN1 = dt.date(2017, 1, 1), dt.date(2020, 12, 31)
SCAN1 = dt.date(2021, 12, 31)                     # validity scan 2017-01-01 .. 2021-12-31 (2022 never read)
HOLD_YEAR = 2021
NBANDS = 24
MASK_FRAC = 0.95
NPTS, SEED = 4_000_000, 20261028
H25, W25 = 576, 1440
R_MAX = 14386                                     # (2 r + 26) // 50 <= 575: nearest 0.25 deg point inside the grid
THR = (100, 200, 300, 400, 440, 500, 600, 800, 1000, 1500, 2000, 3000)
CAND_FRAC = 0.5                                   # pass-1 sums saved for pixels valid on >= 50 % of the scan days


def band_edges(J=NBANDS):
    return [0] + [25 * round(m * H25 / J) + 13 for m in range(1, J)] + [NR]


def doy(d):
    return (d - dt.date(d.year, 1, 1)).days


def bvec(dd):
    th = 2.0 * np.pi * np.asarray(dd, np.float64) / 365.0
    return np.stack([np.ones_like(th), np.cos(th), np.sin(th), np.cos(2 * th), np.sin(2 * th)], -1)


def hvec(dd, K):
    th = 2.0 * np.pi * np.asarray(dd, np.float64) / 365.0
    return np.stack([np.ones_like(th)] + [f(k * th) for k in range(1, K + 1) for f in (np.cos, np.sin)], -1)


def gvec(dd):
    return hvec(dd, 4)


def npy_header(path):
    with open(path, "rb") as f:
        ver = np.lib.format.read_magic(f)
        shape, fo, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        return tuple(shape), np.dtype(dtype), f.tell()


def create(path, shape, dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": np.dtype(dtype).str, "fortran_order": False, "shape": tuple(shape)})
        hdr = f.tell()
        f.truncate(hdr + int(np.prod(shape)) * np.dtype(dtype).itemsize)


def pwrite_slice(path, row, col0, block, _cache={}):
    """write block (1-D) into row `row`, columns col0.. of the 2-D .npy file at path"""
    if path not in _cache:
        _cache[path] = npy_header(path)
    shape, dtype, hdr = _cache[path]
    block = np.ascontiguousarray(block, dtype=dtype)
    assert block.ndim == 1 and col0 + block.size <= shape[1], (path, block.shape, col0, shape)
    mv = memoryview(block.view(np.uint8))
    off = hdr + (row * shape[1] + col0) * dtype.itemsize
    fd = os.open(path, os.O_WRONLY)
    try:
        done = 0
        while done < len(mv):
            done += os.pwrite(fd, mv[done:], off + done)
    finally:
        os.close(fd)


def save(path, **arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arr)
    os.replace(tmp, path)


class BandReader:
    """rows r0:r1 of day d of ghap1km_global.npy (one contiguous pread), prefetched in a thread"""

    def __init__(self, r0, r1, days, n_slots=3):
        shape, dtype, self.hdr = npy_header(SRC)
        assert shape == (ND, NR, NC) and dtype == np.float16, (shape, dtype)
        assert all(D0 <= d <= SCAN1 for d in days), "2022 is sealed until the freeze (PROTOCOL.md A2(b))"
        self.r0, self.r1, self.days = r0, r1, list(days)
        self.n = (r1 - r0) * NC
        self.fd = os.open(SRC, os.O_RDONLY)
        self.q = queue.Queue(n_slots)
        self.read_s = 0.0
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _read(self, d):
        a = np.empty(self.n, np.float16)
        mv = memoryview(a.view(np.uint8))
        off = self.hdr + (((d - D0).days * NR + self.r0) * NC) * 2
        pos = 0
        while pos < len(mv):
            k = os.preadv(self.fd, [mv[pos:]], off + pos)
            if k <= 0:
                raise IOError(f"short read {d}")
            pos += k
        return a

    def _run(self):
        try:
            for d in self.days:
                t0 = time.time()
                a = self._read(d)
                self.read_s += time.time() - t0
                self.q.put((d, a))
            self.q.put(None)
        except BaseException as e:  # noqa: BLE001
            self.q.put(e)

    def __iter__(self):
        while True:
            x = self.q.get()
            if x is None:
                return
            if isinstance(x, BaseException):
                raise RuntimeError("reader failed") from x
            yield x


# ------------------------------------------------------------------------------------------------------------- scan
def cmd_scan(a):
    t0 = time.time()
    edges = band_edges()
    r0, r1 = edges[a.band], edges[a.band + 1]
    npx = (r1 - r0) * NC
    days = [D0 + dt.timedelta(k) for k in range((SCAN1 - D0).days + 1)]
    if a.test:                                    # code test: 20 training days + 4 days of 2021, output build_test/
        days = days[::73][:20] + [dt.date(2021, 3, 1), dt.date(2021, 6, 1), dt.date(2021, 9, 1), dt.date(2021, 12, 1)]
    bdir = OUT / ("build_test" if a.test else "build")
    cnt_all = np.zeros(npx, np.uint16)
    cnt21 = np.zeros(npx, np.uint16)
    maxk = np.zeros(npx, np.float32)
    S = np.zeros((9, npx), np.float64)
    Q = np.zeros((5, npx), np.float64)
    H = np.zeros((13, npx), np.float64)
    BLK = 16
    Y = np.zeros((BLK, npx), np.float32)
    V = np.zeros((BLK, npx), np.float32)
    bdoys = []
    per_day = {}

    def flush():
        nb = len(bdoys)
        if nb == 0:
            return
        g = gvec(bdoys).astype(np.float32)          # (nb, 9)
        b = bvec(bdoys).astype(np.float32)          # (nb, 5)
        h = hvec(bdoys, 6).astype(np.float32)       # (nb, 13)
        S[...] += g.T @ Y[:nb]
        Q[...] += b.T @ (Y[:nb] * Y[:nb])
        H[...] += h.T @ V[:nb]
        bdoys.clear()

    rd = BandReader(r0, r1, days)
    ntr = 0
    for i, (d, a16) in enumerate(rd):
        af = a16.astype(np.float32)
        fin = np.isfinite(af)
        valid = fin & (af > 0)
        rec = {"nvalid": int(np.count_nonzero(valid)), "n_nonfinite": int(npx - np.count_nonzero(fin)),
               "n_negative": int(np.count_nonzero(af < 0))}
        cnt_all += valid
        if d.year == HOLD_YEAR:
            cnt21 += valid
        if TRAIN0 <= d <= TRAIN1:
            k = np.rint(af)
            np.maximum(k, 1.0, out=k)
            k[~valid] = 0.0
            np.maximum(maxk, k, out=maxk)
            mx = float(k.max())
            rec["max_k"] = mx
            if mx > THR[0]:
                rec["n_gt"] = {str(t): int(np.count_nonzero(k > t)) for t in THR if t < mx}
            j = len(bdoys)
            y = Y[j]
            np.log1p(k, out=y)
            y[~valid] = 0.0
            V[j] = valid
            bdoys.append(doy(d))
            ntr += 1
            if len(bdoys) == BLK:
                flush()
        per_day[str(d)] = rec
        if i % 100 == 0:
            print(f"  band {a.band} {d}: valid {rec['nvalid']:,} | {time.time() - t0:.0f} s (read {rd.read_s:.0f} s)",
                  flush=True)
    flush()
    assert (a.test or ntr == (TRAIN1 - TRAIN0).days + 1) and len(per_day) == len(days)
    assert np.array_equal(H[0], np.rint(H[0])) and int(H[0].max()) <= ntr
    cand = np.flatnonzero(cnt_all >= int(np.ceil(CAND_FRAC * len(days))))
    rows = r0 + cand // NC
    gid = (rows * NC + cand % NC).astype(np.uint32)
    save(bdir / f"scan_b{a.band:02d}.npz", r0=np.int64(r0), r1=np.int64(r1), cnt_all=cnt_all, cnt21=cnt21,
         maxk=maxk.astype(np.float16), cand_gid=gid, S=S[:, cand], Q=Q[:, cand], H=H[:, cand])
    info = {"band": a.band, "rows": [r0, r1], "n_scan_days": len(days), "n_train_days": ntr,
            "n_candidates": int(cand.size), "seconds": round(time.time() - t0), "read_seconds": round(rd.read_s),
            "per_day": per_day}
    (bdir / f"scan_b{a.band:02d}.json").write_text(json.dumps(info))
    print(f"[scan] band {a.band} rows {r0}:{r1}: {cand.size:,} candidates, {time.time() - t0:.0f} s", flush=True)


# -------------------------------------------------------------------------------------------------------------- ice
LC_ZIP = Path("/scratch/project_462001804/ammar/eccv/data/world/static/esa_cci_landcover_2020.zip")
ICE_CLASS, ICE_PCT = 220, 50


def cmd_ice(a):
    """share (%) of ESA CCI / C3S land cover 2020 (300 m) cells of class 220 'permanent snow and ice' whose centres fall
    in each GHAP 1 km lattice pixel -> grid/ice_pct_lattice.npy (14400, 36000) uint8 (255 = no land-cover cell).
    PROTOCOL.md A2(a) (coordinator's finding: GHAP ~10 ug m-3 on the Greenland ice sheet where CAMS gives 0.4)."""
    import zipfile
    import netCDF4
    t0 = time.time()
    with zipfile.ZipFile(LC_ZIP) as z:
        name = z.namelist()[0]
        buf = z.read(name)
    ds = netCDF4.Dataset(name, mode="r", memory=buf)
    v = ds["lccs_class"]
    v.set_auto_maskandscale(False)
    lat = np.asarray(ds["lat"][:], np.float64)
    lon = np.asarray(ds["lon"][:], np.float64)
    assert v.dimensions[-2:] == ("lat", "lon") and v.shape[-2:] == (lat.size, lon.size), (v.dimensions, v.shape)
    rr = np.floor((LAT0 + RES / 2 - lat) / RES + 1e-9).astype(np.int64)        # GHAP row of each land-cover row
    cc = np.floor((lon - (LON0 - RES / 2)) / RES + 1e-9).astype(np.int64) % NC
    rows = np.flatnonzero((rr >= 0) & (rr < NR))
    tot = np.zeros(NR * NC, np.uint16)
    ice = np.zeros(NR * NC, np.uint16)
    BLK = 720
    for i0 in range(rows[0], rows[-1] + 1, BLK):
        i1 = min(rows[-1] + 1, i0 + BLK)
        x = np.asarray(v[0, i0:i1, :] if v.ndim == 3 else v[i0:i1, :])
        base = rr[i0] * NC
        flat = (rr[i0:i1] * NC - base)[:, None] + cc[None, :]
        nb = (rr[i1 - 1] - rr[i0] + 1) * NC
        tot[base:base + nb] += np.bincount(flat.ravel(), minlength=nb).astype(np.uint16)
        ice[base:base + nb] += np.bincount(flat.ravel(), weights=(x == ICE_CLASS).ravel(), minlength=nb).astype(np.uint16)
        if (i0 - rows[0]) % (BLK * 10) == 0:
            print(f"  land-cover rows {i0}:{i1} -> GHAP rows {rr[i0]}:{rr[i1 - 1]} ({time.time() - t0:.0f} s)", flush=True)
    ds.close()
    pct = np.where(tot > 0, np.rint(100.0 * ice / np.maximum(tot, 1)), 255).astype(np.uint8)
    (OUT / "grid").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "grid/ice_pct_lattice.npy", pct.reshape(NR, NC))
    info = {"source": str(LC_ZIP), "member": name, "class": ICE_CLASS, "rule_pct_ge": ICE_PCT,
            "lc_cells_per_pixel_min_max": [int(tot[tot > 0].min()), int(tot.max())],
            "n_lattice_pixels_ice_ge50": int(((pct >= ICE_PCT) & (pct <= 100)).sum()), "seconds": round(time.time() - t0)}
    (OUT / "manifest_parts").mkdir(parents=True, exist_ok=True)
    (OUT / "manifest_parts/ghap_ice.json").write_text(json.dumps(info, indent=1))
    print("[ice]", info, flush=True)


# ------------------------------------------------------------------------------------------------------------ merge
def product_coeffs():
    """C (5, 5, 9): b_a b_c = sum_k C[a, c, k] g_k (exact, as build_global_lghap.product_coeffs)"""
    th = np.arange(64) * 365.0 / 64
    B, Gm = bvec(th), gvec(th)
    C = np.zeros((5, 5, 9))
    for a_ in range(5):
        for c in range(5):
            sol, *_ = np.linalg.lstsq(Gm, B[:, a_] * B[:, c], rcond=None)
            C[a_, c] = sol
            assert np.abs(Gm @ sol - B[:, a_] * B[:, c]).max() < 1e-10
    return C


def triple_coeffs():
    """D (5, 5, 5, 13): b_a b_c b_e = sum_m D[a, c, e, m] h_m (harmonics up to 6, exact)"""
    th = np.arange(128) * 365.0 / 128
    B, Hm = bvec(th), hvec(th, 6)
    D = np.zeros((5, 5, 5, 13))
    for a_ in range(5):
        for c in range(5):
            for e in range(5):
                f = B[:, a_] * B[:, c] * B[:, e]
                sol, *_ = np.linalg.lstsq(Hm, f, rcond=None)
                assert np.abs(Hm @ sol - f).max() < 1e-10
                D[a_, c, e] = sol
    return D


def fit_exact(S, Q, H, C, D):
    """per-pixel OLS of y on b over the pixel's valid days (design from the indicator harmonics H), then OLS of the
    squared residual on b (= climstats.fit_harmonic on each pixel's own valid days). S (9, n), Q (5, n), H (13, n)"""
    XtX = np.einsum("ack,kn->nac", C, H[:9])
    beta = np.linalg.solve(XtX, S[:5].T[..., None])[..., 0]                     # (n, 5)
    M = np.einsum("ack,kn->nac", C, S)                                          # sum_t b_a b_c y
    T = np.einsum("acem,mn->nace", D, H)                                        # sum_t b_a b_c b_e
    rhs = Q.T - 2.0 * np.einsum("nac,nc->na", M, beta) + np.einsum("nace,nc,ne->na", T, beta, beta, optimize=True)
    gamma = np.linalg.solve(XtX, rhs[..., None])[..., 0]
    return beta, gamma


def selftest_fit(C, D):
    """synthetic check of fit_exact against a direct per-pixel least squares with random holes"""
    rng = np.random.default_rng(5)
    doys = np.array([doy(TRAIN0 + dt.timedelta(k)) for k in range((TRAIN1 - TRAIN0).days + 1)])
    n = 300
    y = 2.5 + 0.4 * np.cos(2 * np.pi * doys / 365)[None] + rng.gamma(2.0, 0.3, (n, doys.size))
    valid = rng.random((n, doys.size)) > np.linspace(0, 0.3, n)[:, None]
    valid[: n // 3] = True
    g, b, h = gvec(doys), bvec(doys), hvec(doys, 6)
    yv = np.where(valid, y, 0.0)
    S, Q, H = g.T @ yv.T, b.T @ (yv * yv).T, h.T @ valid.T.astype(float)
    beta, gamma = fit_exact(S, Q, H, C, D)
    err_b = err_g = 0.0
    for i in range(n):
        v = valid[i]
        bi, *_ = np.linalg.lstsq(b[v], y[i, v], rcond=None)
        r2 = (y[i, v] - b[v] @ bi) ** 2
        gi, *_ = np.linalg.lstsq(b[v], r2, rcond=None)
        err_b = max(err_b, np.abs(bi - beta[i]).max())
        err_g = max(err_g, np.abs(gi - gamma[i]).max())
    assert err_b < 1e-8 and err_g < 1e-8, (err_b, err_g)
    return {"n_pixels": n, "max_abs_diff_mean": err_b, "max_abs_diff_var": err_g}


def read_rows(path, rows):
    shape, dtype, off = npy_header(path)
    rs = tuple(shape[1:])
    nb = int(np.prod(rs)) * dtype.itemsize
    out = np.empty((len(rows),) + rs, dtype)
    with open(path, "rb") as f:
        for i, r in enumerate(rows):
            f.seek(off + int(r) * nb)
            out[i] = np.frombuffer(f.read(nb), dtype).reshape(rs)
    return out


def input_check(days):
    """finite CAMS, ERA5 (Foehn met.npy), fires (injh store) and T6 on each day -> {date: [bad inputs]}"""
    from concurrent.futures import ThreadPoolExecutor
    met = G1K / "met.npy"
    fires = GLOBAL / "coarse/fires_injh_2017_2022.npy"

    def one(d):
        r = (d - D0).days
        return d, [n for n, x in (("cams", read_rows(GLOBAL / "coarse/cams_pm25_025.npy", [r])[0]),
                                  ("met", read_rows(met, [r])[0]), ("fires", read_rows(fires, [r])[0]),
                                  ("t6", read_rows(GLOBAL / f"transport/t6_{d.year}.npy", [doy(d)])[0]))
                   if not np.isfinite(x).all()]
    with ThreadPoolExecutor(16) as ex:
        return {str(d): b for d, b in ex.map(one, days) if b}


def norm_stats(days):
    """CAMS log1p mean/std (every 2nd training day) and T6 channel mean/std (every training day), global, unweighted:
    build_global_manifest.norm_stats on the GHAP training days (complete GHAP field, finite inputs)"""
    s1 = s2 = n = 0.0
    for d in days[::2]:
        x = np.log1p(read_rows(GLOBAL / "coarse/cams_pm25_025.npy", [(d - D0).days])[0].astype(np.float64))
        s1 += x.sum(); s2 += (x * x).sum(); n += x.size
    cm = s1 / n
    t1 = np.zeros(6); t2 = np.zeros(6); tn = 0
    for d in days:
        x = read_rows(GLOBAL / f"transport/t6_{d.year}.npy", [doy(d)])[0].astype(np.float64)
        assert np.isfinite(x).all(), d
        t1 += x.sum((1, 2)); t2 += (x * x).sum((1, 2)); tn += x[0].size
    tm = t1 / tn
    return {"note": f"GHAP training days ({len(days)} days of {TRAIN0} .. {TRAIN1}; CAMS every 2nd day, T6 every day), global, unweighted",
            "cams_log1p_mean_std": [float(cm), float(np.sqrt(s2 / n - cm * cm))],
            "t6_channel_mean": tm.round(6).tolist(), "t6_channel_std": np.sqrt(t2 / tn - tm * tm).round(6).tolist()}


def tile_check():
    """ghap1km_tiles.npy blocks == ghap1km_global.npy rows (padding zeros) on sample blocks of 2 days <= 2021"""
    tshape, _, thdr = npy_header(TILES)
    gshape, _, ghdr = npy_header(SRC)
    assert tshape == (ND, 29, 71, 512, 512), tshape
    res = []
    fdt, fdg = os.open(TILES, os.O_RDONLY), os.open(SRC, os.O_RDONLY)
    for di in (100, 1000):
        for tr, tc in ((5, 10), (8, 40), (13, 60), (28, 70), (20, 3)):
            blk = np.frombuffer(os.pread(fdt, 512 * 512 * 2, thdr + (((di * 29 + tr) * 71 + tc) * 512 * 512) * 2),
                                np.float16).reshape(512, 512)
            ref = np.zeros((512, 512), np.float16)
            nr_, nc_ = min(512, NR - tr * 512), min(512, NC - tc * 512)
            for i in range(nr_):
                off = ghdr + ((di * NR + tr * 512 + i) * NC + tc * 512) * 2
                ref[i, :nc_] = np.frombuffer(os.pread(fdg, nc_ * 2, off), np.float16)
            res.append({"day": str(D0 + dt.timedelta(di)), "tile": [tr, tc], "valid_rows_cols": [nr_, nc_],
                        "equal": bool(np.array_equal(blk.view(np.uint16), ref.view(np.uint16))),
                        "n_positive": int((ref > 0).sum())})
    os.close(fdt); os.close(fdg)
    assert all(r["equal"] for r in res), res
    return res


def cmd_merge(a):
    t0 = time.time()
    edges = band_edges()
    nscan = (SCAN1 - D0).days + 1
    thr = int(np.ceil(MASK_FRAC * nscan))
    C, D = product_coeffs(), triple_coeffs()
    st = selftest_fit(C, D)
    print("fit self-test", st, flush=True)
    gids, Ss, Qs, Hs, mk, c21, call = [], [], [], [], [], [], []
    cnt_hist = np.zeros(nscan + 1, np.int64)
    band_info = []
    day_valid = {}
    ice_lat = np.load(OUT / "grid/ice_pct_lattice.npy", mmap_mode="r")
    ice_stats = []
    frac025 = np.zeros((H25 + 1) * W25)
    n025 = np.zeros((H25 + 1) * W25)
    for b in range(NBANDS):
        z = np.load(OUT / f"build/scan_b{b:02d}.npz")
        r0, r1 = int(z["r0"]), int(z["r1"])
        assert (r0, r1) == (edges[b], edges[b + 1])
        cnt = z["cnt_all"]
        cnt_hist += np.bincount(cnt, minlength=nscan + 1)
        rr = (r0 + np.arange(r1 - r0))
        r25 = np.repeat((2 * rr + 26) // 50, NC)
        c25 = np.tile(((2 * np.arange(NC) + 26) // 50) % W25, r1 - r0)
        pos = cnt > 0
        fl = (r25 * W25 + c25)[pos]
        frac025 += np.bincount(fl, weights=cnt[pos] / nscan, minlength=frac025.size)
        n025 += np.bincount(fl, minlength=n025.size)
        gid = z["cand_gid"].astype(np.int64)
        loc = gid - r0 * NC
        rows = gid // NC
        icep = ice_lat[rows, gid % NC]
        is_ice = (icep >= ICE_PCT) & (icep <= 100)
        keep = (cnt[loc] >= thr) & (rows <= R_MAX) & ~is_ice
        n_south = int(((cnt[loc] >= thr) & (rows > R_MAX)).sum())
        ice_rm = (cnt[loc] >= thr) & (rows <= R_MAX) & is_ice
        if ice_rm.any():
            yb = z["S"][0, ice_rm] / np.maximum(z["H"][0, ice_rm], 1)
            lat_i = LAT0 - RES * rows[ice_rm]
            lon_i = LON0 + RES * (gid[ice_rm] % NC)
            grl = (lat_i >= 59) & (lon_i >= -75) & (lon_i <= -10)
            ice_stats.append((int(ice_rm.sum()), int(grl.sum()), float(np.expm1(yb).sum()), float(np.expm1(yb[grl]).sum())))
        gids.append(gid[keep].astype(np.uint32))
        Ss.append(z["S"][:, keep]); Qs.append(z["Q"][:, keep]); Hs.append(z["H"][:, keep])
        mk.append(z["maxk"][loc[keep]]); c21.append(z["cnt21"][loc[keep]]); call.append(cnt[loc[keep]])
        info = json.loads((OUT / f"build/scan_b{b:02d}.json").read_text())
        for dd, rec in info["per_day"].items():
            day_valid[dd] = day_valid.get(dd, 0) + rec["nvalid"]
        band_info.append({"band": b, "rows": [r0, r1], "lat": [LAT0 - RES * r0, LAT0 - RES * (r1 - 1)],
                          "n_mask": int(keep.sum()), "n_ever_valid": int(pos.sum()), "n_ice_removed": int(ice_rm.sum()),
                          "n_valid_all_scan_days": int((cnt == nscan).sum()), "n_excluded_south_of_grid": n_south,
                          "max_k_train_field": float(max(v.get("max_k", 0) for v in info["per_day"].values())),
                          "n_nonfinite_total": int(sum(v["n_nonfinite"] for v in info["per_day"].values())),
                          "n_negative_total": int(sum(v["n_negative"] for v in info["per_day"].values()))})
        print(f"  band {b}: mask {keep.sum():,} ({time.time() - t0:.0f} s)", flush=True)
    n_ice = sum(x[0] for x in ice_stats)
    n_grl = sum(x[1] for x in ice_stats)
    ice_info = {"rule": f"ESA CCI/C3S land cover 2020 300 m: >= {ICE_PCT} % of the cells in the pixel are class {ICE_CLASS} "
                         "(permanent snow and ice) -> removed from the mask (PROTOCOL.md A2(a))",
                "n_removed": n_ice, "n_removed_greenland_box": n_grl,
                "mean_train_ghap_expm1_ybar_removed": sum(x[2] for x in ice_stats) / max(n_ice, 1),
                "mean_train_ghap_expm1_ybar_greenland": sum(x[3] for x in ice_stats) / max(n_grl, 1)}
    print("ice exclusion", ice_info, flush=True)
    mask = np.concatenate(gids)
    assert np.all(np.diff(mask.astype(np.int64)) > 0)
    # training days: complete GHAP field (>= 99 % of the median valid count) and finite inputs (A2(a), 15:20)
    medv = float(np.median(list(day_valid.values())))
    incomplete = {dd: n for dd, n in day_valid.items() if n < 0.99 * medv}
    alltrain = [TRAIN0 + dt.timedelta(k) for k in range((TRAIN1 - TRAIN0).days + 1)]
    cand_days = [dd for dd in alltrain if str(dd) not in incomplete]
    bad_inputs = input_check(cand_days)
    train_days = [dd for dd in cand_days if str(dd) not in bad_inputs]
    assert not [dd for dd in incomplete if dd.startswith(str(HOLD_YEAR))], "incomplete hold-out day"
    print(f"training days {len(train_days)} of {len(alltrain)}: {len(incomplete)} incomplete GHAP days, inputs not finite "
          f"{bad_inputs}", flush=True)
    S, Q, H = np.concatenate(Ss, 1), np.concatenate(Qs, 1), np.concatenate(Hs, 1)
    del Ss, Qs, Hs
    maxk = np.concatenate(mk).astype(np.float32)
    cnt21 = np.concatenate(c21)
    cntall = np.concatenate(call)
    Nm = int(mask.size)
    ntrain = (TRAIN1 - TRAIN0).days + 1
    ntv = H[0]
    assert np.array_equal(ntv, np.rint(ntv))
    print(f"mask {Nm:,} pixels; train valid days min {ntv.min():.0f}; {time.time() - t0:.0f} s", flush=True)
    rr = (mask // NC).astype(np.int64)
    cc = (mask % NC).astype(np.int64)
    lat = LAT0 - RES * rr
    r25 = (2 * rr + 26) // 50
    c25 = ((2 * cc + 26) // 50) % W25
    assert r25.max() <= H25 - 1
    assert np.abs(84.0 - 0.25 * r25 - lat).max() <= 0.125 + 1e-9
    cell = (r25 * W25 + c25).astype(np.int32)
    (OUT / "grid").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "grid/ghap_mask.npy", mask)
    np.save(OUT / "grid/cell_id.npy", cell)
    np.save(OUT / "grid/valid_frac_025.npy", np.where(n025 > 0, frac025 / np.maximum(n025, 1), np.nan)
            .reshape(H25 + 1, W25).astype(np.float32))
    np.savez(OUT / "grid/mask_counts.npz", cnt_all=cntall, cnt21=cnt21, n_train_valid=ntv.astype(np.uint16),
             max_k_train=maxk.astype(np.float16))
    # exact per-pixel fit in chunks
    cm = np.zeros((Nm, 5), np.float32)
    cv = np.zeros((Nm, 5), np.float32)
    full_days = [dd for dd in alltrain if str(dd) not in incomplete]
    Hc = hvec([doy(dd) for dd in full_days], 6).sum(0)
    nohole = np.flatnonzero(np.abs(H - Hc[:, None]).max(0) < 1e-6)
    for lo in range(0, Nm, 1_000_000):
        hi = min(Nm, lo + 1_000_000)
        b_, g_ = fit_exact(S[:, lo:hi], Q[:, lo:hi], H[:, lo:hi], C, D)
        cm[lo:hi], cv[lo:hi] = b_, g_
    # common-design check on no-hole pixels (the LGHAP builder's formula)
    Bt = bvec([doy(dd) for dd in full_days])
    A = np.linalg.inv(Bt.T @ Bt)
    sub = nohole[np.random.default_rng(7).choice(nohole.size, min(20000, nohole.size), replace=False)] if nohole.size else nohole
    beta_c = (A @ S[:5, sub]).T
    Tt = np.einsum("ta,tc,te->ace", Bt, Bt, Bt)
    Mm = np.einsum("ack,kn->acn", C, S[:, sub])
    rhs = Q[:, sub] - 2.0 * np.einsum("acn,nc->an", Mm, beta_c) + np.einsum("ace,nc,ne->an", Tt, beta_c, beta_c)
    gam_c = (A @ rhs).T
    ver = {"selftest_synthetic_holes": st, "n_nohole_checked": int(sub.size),
           "max_abs_diff_mean_vs_common_design": float(np.abs(beta_c - cm[sub]).max()) if sub.size else None,
           "max_abs_diff_var_vs_common_design": float(np.abs(gam_c - cv[sub]).max()) if sub.size else None}
    print("fit check", ver, flush=True)
    del S, Q, H
    (OUT / "ghap_clim").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "ghap_clim/clim_mean_G.npy", cm)
    np.save(OUT / "ghap_clim/clim_var_G.npy", cv)
    # sigma floor (PROTOCOL 1.2 rule)
    mid = np.array([14, 45, 73, 104, 134, 165, 195, 226, 257, 287, 318, 348])
    rs = np.random.default_rng(3).choice(Nm, min(Nm, 5_000_000), replace=False)
    var_mid = cv[rs].astype(np.float64) @ bvec(mid).T
    sig_mid = np.sqrt(np.maximum(var_mid, 0.0))
    floor = float(np.percentile(sig_mid, 1))
    mu_mid = cm[rs].astype(np.float64) @ bvec(mid).T
    # statics
    import zarr
    el = zarr.open_group(STATIC1KM, mode="r")["elevation"]
    assert el.shape == (18000, NC), el.shape
    E = np.asarray(el[600 + int(rr.min()):600 + int(rr.max()) + 1, :], np.float32)
    elev = E[rr - int(rr.min()), cc].astype(np.float64)
    del E
    n_elev_nan = int((~np.isfinite(elev)).sum())
    elev[~np.isfinite(elev)] = 0.0
    cmd_ = cm.astype(np.float64)
    cvd = cv.astype(np.float64)
    mu0 = cmd_[:, 0]
    cells, cellc = np.unique(cell, return_inverse=True)
    cnt = np.bincount(cellc, minlength=cells.size)
    cmean = np.bincount(cellc, weights=mu0, minlength=cells.size) / np.maximum(cnt, 1)
    sig_at = lambda dd: np.sqrt(np.maximum(cvd @ bvec(dd), floor ** 2))
    raw = np.stack([mu0, mu0 - cmean[cellc], np.hypot(cmd_[:, 1], cmd_[:, 2]), sig_at(15), sig_at(196), elev], 1)
    smean, sstd = raw.mean(0), np.maximum(raw.std(0, ddof=1), 1e-6)
    np.save(OUT / "ghap_clim/static_G.npy", ((raw - smean) / sstd).astype(np.float16))
    sig_ann = np.sqrt(np.maximum(cvd[:, 0], floor ** 2))
    print(f"clim + statics written, {time.time() - t0:.0f} s", flush=True)
    # band quotas (cos-lat weight share, largest remainder)
    edges_mask = np.searchsorted(mask, np.array(edges, np.int64) * NC)
    wsum = np.array([np.cos(np.deg2rad(lat[edges_mask[b]:edges_mask[b + 1]])).sum() for b in range(NBANDS)])
    q = NPTS * wsum / wsum.sum()
    quota = np.floor(q).astype(np.int64)
    quota[np.argsort(-(q - quota))[:NPTS - quota.sum()]] += 1
    assert quota.sum() == NPTS and all(quota[b] <= edges_mask[b + 1] - edges_mask[b] for b in range(NBANDS))
    offset = np.concatenate([[0], np.cumsum(quota)[:-1]])
    # output files of pass 2
    files = {}
    for y in range(TRAIN0.year, TRAIN1.year + 1):
        nd = 366 if y % 4 == 0 else 365
        for kind, dtype in (("idx", np.uint32), ("val", np.int16)):
            pth = OUT / f"ghap_points/{kind}_{y}.npy"
            create(pth, (nd, NPTS), dtype)
        files[str(y)] = {"first": f"{y}-01-01", "last": f"{y}-12-31", "days": nd, "idx": f"ghap_points/idx_{y}.npy",
                         "val": f"ghap_points/val_{y}.npy"}
    create(OUT / f"ghap_eval/full_{HOLD_YEAR}.npy", (365, Nm), np.int16)
    tc = tile_check()
    norm = norm_stats(train_days)
    info = {
        "built": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0),
        "source": str(SRC), "lattice": {"rows": NR, "cols": NC, "lat_row0": LAT0, "lon_col0": LON0, "res": RES,
                                        "row_lat": "83.995 - 0.01 r (descending)", "flat_id": "r * 36000 + c"},
        "mask": {"Nm": Nm, "rule": f"GHAP > 0 and finite on >= {MASK_FRAC:.0%} of the {nscan} days {D0} .. {SCAN1} "
                 f"(>= {thr} days) and row <= {R_MAX} (nearest 0.25 deg point inside the 576-row grid) and not permanent "
                 f"ice (ESA CCI 2020 class {ICE_CLASS} >= {ICE_PCT} %)",
                 "threshold_days": thr, "n_pixels_valid_every_scan_day": int((cntall == nscan).sum()),
                 "n_mask_pixels_with_train_holes": int((ntv < ntrain).sum()),
                 "n_mask_pixels_with_2021_holes": int((cnt21 < 365).sum()),
                 "lat_range": [float(lat.min()), float(lat.max())], "n_cells_025": int(cells.size),
                 "pixels_per_cell_median": float(np.median(cnt)),
                 "n_mask_north_of_60N": int((lat > 60).sum()), "ice_exclusion": ice_info,
                 "lattice_valid_count_hist": {"ever": int(cnt_hist[1:].sum()), "ge50pct": int(cnt_hist[int(np.ceil(0.5 * nscan)):].sum()),
                                              "ge95pct": int(cnt_hist[thr:].sum()), "all": int(cnt_hist[nscan])},
                 "bands": band_info},
        "clim": {"y": "log1p(max(round(GHAP), 1))", "K": 2, "fit": "exact per-pixel OLS over the pixel's valid training days "
                 "(indicator harmonics), then OLS of the squared residual (climstats.fit_harmonic per pixel)",
                 "training_days": ntrain, "verification": ver,
                 "mean_y_p1_p50_p99_midmonth": np.percentile(mu_mid, [1, 50, 99]).round(4).tolist(),
                 "sigma_p1_p50_p99_midmonth": np.percentile(sig_mid, [1, 50, 99]).round(4).tolist(),
                 "annual_sigma_p1_p50_p99": np.percentile(sig_ann, [1, 50, 99]).round(4).tolist(),
                 "frac_var_fit_nonpositive_midmonth": float((var_mid <= 0).mean()),
                 "frac_annual_var_at_floor": float((cvd[:, 0] <= floor ** 2).mean())},
        "static": {"columns": ["mu0", "mu0_minus_cell_mean", "seasonal_amplitude", "sigma_doy15", "sigma_doy196", "elevation_m"],
                   "raw_mean": smean.round(5).tolist(), "raw_std": sstd.round(5).tolist(), "sigma_floor": floor,
                   "elevation_source": f"{STATIC1KM} elevation row 600 + r", "n_elevation_nan_set_0": n_elev_nan,
                   "elevation_min_max_m": [float(elev.min()), float(elev.max())],
                   "years_used": "2017-2020 climatology only"},
        "max_k_train_field_mask": float(maxk.max()),
        "points": {"n_per_day": NPTS, "seed": f"np.random.default_rng([{SEED}, year, doy, band])", "weight": "cos(lat)",
                   "method": "Efraimidis-Spirakis keys log(u)/w among the VALID mask pixels of the day, per latitude band "
                             "(quota = band share of the mask cos(lat) weight, largest remainder), sorted ascending",
                   "band_edges_rows": edges, "band_mask_index": edges_mask.tolist(), "band_quota": quota.tolist(),
                   "band_offset": offset.tolist(), "files": files},
        "eval": {"file": f"ghap_eval/full_{HOLD_YEAR}.npy", "shape": [365, Nm], "row": f"doy of {HOLD_YEAR}",
                 "value": "int16 max(round(GHAP), 1); -1 = hole"},
        "split_G": {"train": [str(TRAIN0), str(TRAIN1), ntrain], "train_days": [str(dd) for dd in train_days],
                    "n_train_days": len(train_days), "excluded_days": {"incomplete_ghap_valid_count": incomplete,
                                                                       "inputs_not_finite": bad_inputs,
                                                                       "median_valid_count": medv},
                    "buffer": ["2021-01-01", "2021-01-07"],
                    "val": ["2021-01-08", "2021-12-31"], "test_sealed": ["2022-01-01", "2022-12-31"]},
        "normalisation": norm, "tile_check": tc,
    }
    (OUT / "manifest_parts").mkdir(parents=True, exist_ok=True)
    (OUT / "manifest_parts/ghap_merge.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps({k: info[k] for k in ("mask", "clim", "static")}, default=str)[:4000], flush=True)
    print(f"merge done in {time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------------------------------------- points
def cmd_points(a):
    t0 = time.time()
    man = json.loads((OUT / "manifest_parts/ghap_merge.json").read_text())
    pts = man["points"]
    edges = pts["band_edges_rows"]
    b = a.band
    r0, r1 = edges[b], edges[b + 1]
    mask = np.load(OUT / "grid/ghap_mask.npy", mmap_mode="r")
    lo, hi = pts["band_mask_index"][b], pts["band_mask_index"][b + 1]
    gid = np.asarray(mask[lo:hi]).astype(np.int64)
    assert gid.size == 0 or (gid[0] >= r0 * NC and gid[-1] < r1 * NC)
    loc = gid - r0 * NC
    n = gid.size
    w = np.cos(np.deg2rad(LAT0 - RES * (gid // NC)))
    quota, off = pts["band_quota"][b], pts["band_offset"][b]
    cid = np.asarray(np.load(OUT / "grid/cell_id.npy", mmap_mode="r")[lo:hi])
    ucell, cinv = np.unique(cid, return_inverse=True)
    train = [dt.date.fromisoformat(x) for x in man["split_G"]["train_days"]]
    hold = [dt.date(HOLD_YEAR, 1, 1) + dt.timedelta(k) for k in range(365)]
    if a.test:
        train, hold = train[::200], hold[::120]
    cmean = np.full((len(train), ucell.size), np.nan, np.float32)
    ccnt = np.zeros((len(train), ucell.size), np.uint16)
    per_day = {}
    tpos = {d: i for i, d in enumerate(train)}
    evalf = OUT / f"ghap_eval/full_{HOLD_YEAR}.npy"
    for d, a16 in BandReader(r0, r1, train + hold):
        v = a16[loc].astype(np.float32)
        valid = np.isfinite(v) & (v > 0)
        k = np.maximum(np.rint(v), 1.0)
        k[~valid] = 0.0
        assert k.max() < 32767
        ki = k.astype(np.int16)
        nv = int(valid.sum())
        rec = {"n_holes": int(n - nv), "max_k": float(k.max()) if n else 0.0}
        if d in tpos:
            t = tpos[d]
            rng = np.random.default_rng([SEED, d.year, doy(d), b])
            key = rng.random(n)
            np.log(key, out=key)
            key /= w
            key[~valid] = -np.inf
            assert nv >= quota, (d, nv, quota)
            sel = np.argpartition(key, n - quota)[n - quota:]
            sel.sort()
            assert valid[sel].all()
            if not a.test:
                pwrite_slice(OUT / f"ghap_points/idx_{d.year}.npy", doy(d), off, (lo + sel).astype(np.uint32))
                pwrite_slice(OUT / f"ghap_points/val_{d.year}.npy", doy(d), off, ki[sel])
            rec["max_k_points"] = float(k[sel].max()) if quota else 0.0
            y = np.log1p(k) * valid
            cs = np.bincount(cinv, weights=y, minlength=ucell.size)
            cn = np.bincount(cinv, weights=valid.astype(np.float64), minlength=ucell.size)
            cmean[t] = np.where(cn > 0, cs / np.maximum(cn, 1), np.nan)
            ccnt[t] = cn
        else:
            ki[~valid] = -1
            if not a.test:
                pwrite_slice(evalf, doy(d), lo, ki)
        per_day[str(d)] = rec
        if len(per_day) % 200 == 1:
            print(f"  band {b} {d}: holes {rec['n_holes']:,} ({time.time() - t0:.0f} s)", flush=True)
    tag = "build_test" if a.test else "build"
    save(OUT / f"{tag}/points_b{b:02d}.npz", cells=ucell.astype(np.int32), cellmean=cmean, cellcount=ccnt,
         cell_npix=np.bincount(cinv, minlength=ucell.size).astype(np.int32))
    (OUT / f"{tag}/points_b{b:02d}.json").write_text(json.dumps({"band": b, "n_mask": n, "quota": quota,
                                                                  "seconds": round(time.time() - t0), "per_day": per_day}))
    print(f"[points] band {b}: {n:,} mask pixels, quota {quota:,}, {time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------------------------------------- finish
def cmd_finish(a):
    man = json.loads((OUT / "manifest_parts/ghap_merge.json").read_text())
    Nm = man["mask"]["Nm"]
    cells, means, counts, npix = [], [], [], []
    holes = {}
    tmax_pts = 0.0
    band_holes = []
    for b in range(NBANDS):
        z = np.load(OUT / f"build/points_b{b:02d}.npz")
        cells.append(z["cells"]); means.append(z["cellmean"]); counts.append(z["cellcount"]); npix.append(z["cell_npix"])
        info = json.loads((OUT / f"build/points_b{b:02d}.json").read_text())
        nh_tr = nh_21 = 0
        for d, r in info["per_day"].items():
            holes[d] = holes.get(d, 0) + r["n_holes"]
            tmax_pts = max(tmax_pts, r.get("max_k_points", 0.0))
            if d.startswith(str(HOLD_YEAR)):
                nh_21 += r["n_holes"]
            else:
                nh_tr += r["n_holes"]
        ntr = len(man["split_G"]["train_days"])
        band_holes.append({"band": b, "lat": man["mask"]["bands"][b]["lat"], "n_mask": info["n_mask"],
                           "hole_frac_train": nh_tr / max(1, info["n_mask"] * ntr),
                           "hole_frac_2021": nh_21 / max(1, info["n_mask"] * 365)})
    # band edges r = 25 k + 13 put one row of the boundary cell (r = 25 k + 12) in the band above: the cells at the 23
    # edges appear in two bands -> count-weighted merge of their daily means
    cells = np.concatenate(cells).astype(np.int64)
    mean = np.concatenate(means, 1).astype(np.float64)
    cnt = np.concatenate(counts, 1).astype(np.float64)
    npx = np.concatenate(npix).astype(np.int64)
    ucells, inv = np.unique(cells, return_inverse=True)
    S_ = np.zeros((mean.shape[0], ucells.size))
    C_ = np.zeros((mean.shape[0], ucells.size))
    for j in range(cells.size):
        k = inv[j]
        S_[:, k] += np.nan_to_num(mean[:, j]) * cnt[:, j]
        C_[:, k] += cnt[:, j]
    print(f"cells: {cells.size} band entries -> {ucells.size} unique ({cells.size - ucells.size} split at band edges)", flush=True)
    cells = ucells
    mean = np.where(C_ > 0, S_ / np.maximum(C_, 1), np.nan).astype(np.float32)
    counts = [C_.astype(np.uint16)]
    npix = [np.bincount(inv, weights=npx, minlength=ucells.size).astype(np.int32)]
    assert np.all(np.diff(cells) > 0)
    train = [dt.date.fromisoformat(x) for x in man["split_G"]["train_days"]]
    np.savez(OUT / "ghap_clim/cellmean_train_G.npz", cells=cells, dates=np.array([np.datetime64(d) for d in train]),
             mean=mean, count=counts[0], npix=npix[0])
    T_G = float(tmax_pts)
    P_max = 1000.0 if T_G < 1000 else float(np.ceil(1.5 * T_G / 100.0) * 100.0)
    hf = np.array(list(holes.values()), np.float64) / Nm
    man["holes"] = {"per_day_mask_holes": holes, "frac_pixel_days_train": float(sum(v for d, v in holes.items() if not d.startswith(str(HOLD_YEAR))) / (Nm * len(train))),
                    "frac_pixel_days_2021": float(sum(v for d, v in holes.items() if d.startswith(str(HOLD_YEAR))) / (Nm * 365)),
                    "max_day_frac": float(hf.max()), "n_days_with_holes": int((hf > 0).sum()), "by_band": band_holes}
    man["T_G_train_max_points"] = T_G
    man["P_max_ug"] = P_max
    man["G0a_rule"] = "PROTOCOL.md A2(c): T_G = largest k in the training point files 2017-2020; P_max = 1000 if T_G < 1000 else ceil(1.5 T_G / 100) * 100"
    man["cellmean"] = {"file": "ghap_clim/cellmean_train_G.npz", "n_cells": int(cells.size), "days": len(train),
                       "value": "daily mean of y over the VALID mask pixels of each 0.25 deg cell (NaN if none)"}
    man["finished"] = time.strftime("%Y-%m-%d %H:%M")
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({k: man[k] for k in ("T_G_train_max_points", "P_max_ug")}), json.dumps(
        {k: v for k, v in man["holes"].items() if k != "per_day_mask_holes"})[:3000], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "ice", "merge", "points", "finish"])
    ap.add_argument("--band", type=int, default=0)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    {"scan": cmd_scan, "ice": cmd_ice, "merge": cmd_merge, "points": cmd_points, "finish": cmd_finish}[a.cmd](a)
