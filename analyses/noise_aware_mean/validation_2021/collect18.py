#!/usr/bin/env python3
"""A18 step 1 (CPU only, stored data only, no model run). Gathers, at a fixed random subsample of the 8,112,762 populated
pixels (G/pop_pixels.npz; numpy default_rng(20261121).choice(8112762, NPIX, replace=False), sorted; NPIX = 1,000,000 for
the A18 run), everything score18.py reads, into a18/work (removed after the score):
  fc16.npy  (2 leads [1, 3], 179 forecasts, N, 16) float16 ug m-3: the FIXED-design 16-member forecasts of 2022
            (a13/warn16/data/run16, a14/data/run16_13, a13/warn16/data/run16all). Used ONLY to fit the noise law.
  fc64.npy  (2, 15, N, 64) float16: the 64-field run of 2021 (a14/r2/data/run64), 15 strong-episode start days.
  feat.npy  (64 fields, N, 12) float32: the LINEAR features of a17/diag/collect.py (columns 0-11 there, same code):
            0 z at the pixel, 1 mean of the aligned 2 x 2 block, 2-11 centred box means of width 3, 5, 9, 17, 33, 65, 129,
            257, 513, 1025 px; each multiplied by sqrt(pixels in the box) (unit variance under N(0, 1) white noise).
            Fields 0-15 from G/noise_tiles16.npy (the fields of the 2022 runs and fields 0-15 of run64), 16-63 from
            G/noise_tiles64.npy (fields 0-15 of that file are asserted equal to noise_tiles16.npy, row 14, as a17/diag).
  truth.npy (2, 15, N) float32 (--part truth or all ONLY): the 1 km GHAP estimates of each target day of run64
            (G/ghap1km_tiles.npy, day index 1461 + target day of year, as a14/r2/smoke_check.py), NaN where not > 0.
            Only days of 2021 can be read (asserted). Never read in --part members.
  meta.npz  pixel positions, flat indices, residents, land share in a 51 x 51 px box (G/land1km_global.npy; coast < 0.9,
            inland >= 0.999 in score18.py, the definition of a17/review_diag), forecast keys, read-back checks.
An audit hook refuses any path with 'station', 'openaq' or 'monitor', any '_ar' folder, a16/data, a16/results, the GHAP
file unless --part truth/all, and any write outside a18 (or /tmp). No GPU.
    python -B -u collect18.py --part members|truth|all [--npix 1000000] [--workers 16] [--work DIR]"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

A18_TAG = os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution/a18/"
ALLOW = {"truth": False}


def _audit(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = os.fsdecode(args[0])
    pl = p.lower()
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    wr = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or \
         (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND)))
    if not wr and (pl.endswith((".py", ".pyc", ".so")) or "/site-packages/" in pl):      # code, not data
        return
    if "station" in pl or "openaq" in pl or "monitor" in pl or pl.rstrip("/").endswith("_ar") or "_ar/" in pl \
            or "/a16/data" in pl or "/a16/results" in pl:
        raise PermissionError(f"A18: refused path {p}")
    if "ghap" in pl and not ALLOW["truth"]:
        raise PermissionError(f"A18: truth refused in this mode: {p}")
    if wr:
        rp = os.path.realpath(p)
        if not (A18_TAG in rp or A18_TAG in os.path.abspath(p) or rp.startswith("/tmp/") or rp.startswith("/dev/")):
            raise PermissionError(f"A18: write outside a18 refused: {p}")


sys.addaudithook(_audit)
P = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
DIRS16 = [P / "a13/warn16/data/run16", P / "a14/data/run16_13", P / "a13/warn16/data/run16all"]
DIR64 = P / "a14/r2/data/run64"
NPIX, GH, GW, TILE, NTR, NTC = 8112762, 14400, 36000, 512, 29, 71
SEED = 20261121
WID = (3, 5, 9, 17, 33, 65, 129, 257, 513, 1025)
BASE, NDAY21 = 1461, 365                     # index of 2021-01-01 in ghap1km_tiles.npy (days of 2017-2022)
S = {}


def header(path):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f)
        shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        return shape, dtype, f.tell()


def pread_all(path, off, nbytes):
    fd = os.open(path, os.O_RDONLY)
    try:
        buf, got = bytearray(nbytes), 0
        mv = memoryview(buf)
        while got < nbytes:
            n = os.preadv(fd, [mv[got:got + (1 << 30)]], off + got)
            assert n > 0, (path, off, got)
            got += n
    finally:
        os.close(fd)
    return buf


def index(dirs, M):
    out = {}
    for D in dirs:
        for fk in sorted(glob.glob(str(D / f"keys{M}_*.npy"))):
            tag = Path(fk).stem.split("_", 1)[1]
            done = set(json.loads((D / f"done{M}_{tag}.json").read_text()))
            for n, (d, h, t) in enumerate(np.load(fk)):
                if n in done:
                    key = (int(d), int(h))
                    assert key not in out, key
                    out[key] = (str(D / f"pop{M}_{tag}.npy"), n, int(t))
    return out


def read_fc(job):
    which, li, ti, path, row = job
    shape, dtype, hdr = header(path)
    assert shape[1] == NPIX and dtype == np.float16, (path, shape, dtype)
    M = shape[2]
    rowbytes = NPIX * M * 2
    x = np.frombuffer(pread_all(path, hdr + row * rowbytes, rowbytes), np.float16).reshape(NPIX, M)
    arr = np.load(S["work"] / f"{which}.npy", mmap_mode="r+")
    arr[li, ti] = x[S["sub"]]
    arr.flush()
    del arr
    return which, li, ti


def full_map(path, k):
    shape, dtype, hdr = header(path)
    n = NTR * NTC * TILE * TILE
    a = np.frombuffer(pread_all(path, hdr + k * n * 2, n * 2), np.float16)
    return a.reshape(NTR, NTC, TILE, TILE).transpose(0, 2, 1, 3).reshape(NTR * TILE, NTC * TILE)[:GH, :GW].astype(np.float32)


def box_sums(cs, r, c, w):
    h = w // 2
    r0, r1 = np.clip(r - h, 0, GH), np.clip(r + h + 1, 0, GH)
    c0, c1 = np.clip(c - h, 0, GW), np.clip(c + h + 1, 0, GW)
    s = cs[r1, c1] - cs[r0, c1] - cs[r1, c0] + cs[r0, c0]
    return s, ((r1 - r0) * (c1 - c0)).astype(np.float64)


def integral(z):
    cs = np.zeros((GH + 1, GW + 1), np.float64)
    np.cumsum(z, 0, dtype=np.float64, out=cs[1:, 1:])
    np.cumsum(cs[1:, 1:], 1, out=cs[1:, 1:])
    return cs


def feats(m):
    """the 12 linear features of a17/diag/collect.py feats() (its columns 0-11), unchanged arithmetic"""
    src = G / ("noise_tiles16.npy" if m < 16 else "noise_tiles64.npy")
    z = full_map(src, m)
    r, c = S["r"], S["c"]
    f = np.zeros((len(r), 12), np.float32)
    f[:, 0] = z[r, c]
    rb, cb = r // 2 * 2, c // 2 * 2
    f[:, 1] = (z[rb, cb] + z[rb + 1, cb] + z[rb, cb + 1] + z[rb + 1, cb + 1]) / 2.0     # mean * sqrt(4)
    cs = integral(z)
    del z
    for j, w in enumerate(WID):
        s, n = box_sums(cs, r, c, w)
        f[:, 2 + j] = s / np.sqrt(n)
    del cs
    arr = np.load(S["work"] / "feat.npy", mmap_mode="r+")
    arr[m] = f
    arr.flush()
    del arr
    print(f"field {m} done {time.strftime('%H:%M:%S')} sd {f[:, [0, 2, 8]].std(0).round(3).tolist()}", flush=True)
    return m


def land51():
    land = np.load(G / "land1km_global.npy", mmap_mode="r")
    assert land.shape == (GH, GW) and land.dtype == np.bool_, (land.shape, land.dtype)
    cs = integral(np.asarray(land, np.float32))
    s, n = box_sums(cs, S["r"], S["c"], 51)
    return (s / n).astype(np.float32)


def truth_day(job):
    li, ti, target = job
    day = BASE + int(target)
    assert BASE <= day < BASE + NDAY21, ("not a day of 2021", target)
    path = G / "ghap1km_tiles.npy"
    shape, dtype, hdr = header(path)
    assert shape == (2191, NTR, NTC, TILE, TILE) and dtype == np.float16, (shape, dtype)
    n = NTR * NTC * TILE * TILE
    a = np.frombuffer(pread_all(path, hdr + day * n * 2, n * 2), np.float16).reshape(NTR, NTC, TILE, TILE)
    r, c = S["r"], S["c"]
    t = np.asarray(a[r // TILE, c // TILE, r % TILE, c % TILE], np.float32)
    t = np.where(np.isfinite(t) & (t > 0), t, np.float32(np.nan))
    arr = np.load(S["work"] / "truth.npy", mmap_mode="r+")
    arr[li, ti] = t
    arr.flush()
    del arr
    return li, ti


def save_npz(path, **kw):
    tmp = path.parent / f"{path.stem}.tmp{os.getpid()}.npz"
    np.savez(tmp, **kw)
    fd = os.open(tmp, os.O_RDONLY); os.fsync(fd); os.close(fd)
    os.replace(tmp, path)
    z = np.load(path)
    for k, v in kw.items():
        assert np.array_equal(z[k], np.asarray(v)), k


def members(workers, t0):
    W = S["work"]
    K16, K64 = index(DIRS16, 16), index([DIR64], 64)
    keys16 = {h: sorted(k for k in K16 if k[1] == h) for h in (1, 3)}
    keys64 = {h: sorted(k for k in K64 if k[1] == h) for h in (1, 3)}
    T16, T64 = len(keys16[1]), len(keys64[1])
    assert T16 == len(keys16[3]) == 179 and T64 == len(keys64[3]) == 15, (T16, len(keys16[3]), T64)
    assert [d for d, _ in keys64[1]] == [d for d, _ in keys64[3]]
    N = len(S["sub"])
    for nm, shp, dt in (("fc16", (2, T16, N, 16), np.float16), ("fc64", (2, T64, N, 64), np.float16),
                        ("feat", (64, N, 12), np.float32)):
        np.lib.format.open_memmap(W / f"{nm}.npy", mode="w+", dtype=dt, shape=shp).flush()
    jobs = [("fc16", li, ti, *K16[k][:2]) for li, h in enumerate((1, 3)) for ti, k in enumerate(keys16[h])]
    jobs += [("fc64", li, ti, *K64[k][:2]) for li, h in enumerate((1, 3)) for ti, k in enumerate(keys64[h])]
    with get_context("fork").Pool(workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(read_fc, jobs, chunksize=1)):
            if n % 50 == 0:
                print(f"forecast rows {n + 1}/{len(jobs)} {time.time() - t0:.0f} s", flush=True)
    same = []
    for m in range(16):
        a = np.load(G / "noise_tiles16.npy", mmap_mode="r")[m, 14]
        b = np.load(G / "noise_tiles64.npy", mmap_mode="r")[m, 14]
        same.append(bool(np.array_equal(np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16))))
    assert all(same), same
    with get_context("fork").Pool(max(1, min(8, workers // 2))) as pool:
        list(pool.imap_unordered(feats, range(64), chunksize=1))
    print(f"features done {time.time() - t0:.0f} s", flush=True)
    lf = land51()
    chk = {}
    for nm, K, keys in (("fc16", K16, keys16), ("fc64", K64, keys64)):
        arr = np.load(W / f"{nm}.npy", mmap_mode="r")
        for li, h in enumerate((1, 3)):
            for ti in (0, len(keys[h]) // 2, len(keys[h]) - 1):
                path, row, _ = K[keys[h][ti]]
                src = np.load(path, mmap_mode="r")[row]
                chk[f"{nm}_l{h}_t{ti}"] = bool(np.array_equal(np.asarray(src[S["sub"]]).view(np.uint16),
                                                              np.asarray(arr[li, ti]).view(np.uint16)))
    ft = np.load(W / "feat.npy", mmap_mode="r")
    chk["feat_finite"] = bool(all(np.isfinite(np.asarray(ft[m])).all() for m in range(64)))
    chk["feat_sd_box129_in_0.9_1.1"] = bool(all(0.9 < float(np.asarray(ft[m, :, 8]).std()) < 1.1 for m in range(64)))
    chk["feat_fields_distinct"] = bool(all(not np.array_equal(np.asarray(ft[m]), np.asarray(ft[m - 1])) for m in range(1, 64)))
    chk["noise64_fields0_15_equal_16"] = all(same)
    chk["land51_in_0_1"] = bool(np.isfinite(lf).all() and lf.min() >= 0 and lf.max() <= 1)
    print(json.dumps(chk), flush=True)
    assert all(chk.values()), chk
    keys16a = np.array([[d, h, K16[(d, h)][2]] for h in (1, 3) for d, _ in keys16[h]], np.int32).reshape(2, T16, 3)
    keys64a = np.array([[d, h, K64[(d, h)][2]] for h in (1, 3) for d, _ in keys64[h]], np.int32).reshape(2, T64, 3)
    save_npz(W / "meta.npz", sub=S["sub"], idx=S["idx"], r=S["r"], c=S["c"], pop=S["pop"], land51=lf, seed=SEED,
             npix=len(S["sub"]), keys16=keys16a, keys64=keys64a, checks=json.dumps(chk))


def truth(workers, t0):
    W = S["work"]
    meta = np.load(W / "meta.npz")
    assert np.array_equal(meta["sub"], S["sub"]), "meta.npz is of another pixel sample"
    k64 = meta["keys64"]
    N = len(S["sub"])
    np.lib.format.open_memmap(W / "truth.npy", mode="w+", dtype=np.float32, shape=(2, k64.shape[1], N)).flush()
    jobs = [(li, ti, int(k64[li, ti, 2])) for li in range(2) for ti in range(k64.shape[1])]
    with get_context("fork").Pool(max(1, min(8, workers // 2))) as pool:
        list(pool.imap_unordered(truth_day, jobs, chunksize=1))
    arr = np.load(W / "truth.npy", mmap_mode="r")
    li, ti, tg = jobs[len(jobs) // 2]
    S2 = dict(S)
    re = np.asarray(np.load(G / "ghap1km_tiles.npy", mmap_mode="r")[BASE + tg][S2["r"] // TILE, S2["c"] // TILE,
                                                                             S2["r"] % TILE, S2["c"] % TILE], np.float32)
    re = np.where(np.isfinite(re) & (re > 0), re, np.float32(np.nan))
    ok = bool(np.array_equal(np.isnan(re), np.isnan(np.asarray(arr[li, ti]))) and
              np.array_equal(np.nan_to_num(re), np.nan_to_num(np.asarray(arr[li, ti]))))
    print(json.dumps({"truth_readback": ok, "truth_days": sorted({j[2] for j in jobs})}), flush=True)
    assert ok
    print(f"truth done {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("members", "truth", "all"), required=True)
    ap.add_argument("--npix", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--work", default=str(P / "a18" / "work"))
    a = ap.parse_args()
    ALLOW["truth"] = a.part in ("truth", "all")
    t0 = time.time()
    W = Path(a.work)
    assert A18_TAG in os.path.realpath(W) + "/" or A18_TAG in str(W) + "/", W
    W.mkdir(parents=True, exist_ok=True)
    S["work"] = W
    pp = np.load(G / "pop_pixels.npz")
    idx, pop = pp["idx"], pp["pop"]
    assert len(idx) == NPIX
    sub = np.sort(np.random.default_rng(SEED).choice(NPIX, a.npix, replace=False))
    S["sub"], S["idx"], S["pop"] = sub, idx[sub], pop[sub]
    S["r"], S["c"] = (idx[sub] // GW).astype(np.int64), (idx[sub] % GW).astype(np.int64)
    if a.part in ("members", "all"):
        members(a.workers, t0)
    if a.part in ("truth", "all"):
        truth(a.workers, t0)
    print(f"all done {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
