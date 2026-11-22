#!/usr/bin/env python3
"""A17 diag, step 1 (CPU only, stored forecasts only, no model run). Gathers, at a fixed random subsample of the 8,112,762
populated pixels (pop_pixels.npz), everything the analysis needs:
  work/fc16.npy  (2 leads [1, 3], 179 forecasts, N pixels, 16 members) float16, ug m-3: the FIXED-design 16-member
                 forecasts of 2022 (a13/warn16/data/run16, a14/data/run16_13, a13/warn16/data/run16all; leads 1 and 3)
  work/fc64.npy  (2 leads, 15 forecasts, N, 64) float16: the 64-field run of 2021 (a14/r2/data/run64)
  work/feat.npy  (64 fields, N, 17) float32: features of each stored noise field at the pixels (fields 0-15 from
                 noise_tiles16.npy, the fields of the 2022 runs; 16-63 from noise_tiles64.npy, used by run64)
                 0  z at the pixel (1 km)            1  mean of the aligned 2 x 2 block (the network's space-to-depth unit)
                 2-11 centred box means of width 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025 px (1 px = 0.01 deg ~ 1 km)
                 12-16 centred box means of z^2 - 1, widths 3, 9, 33, 129, 513 (local energy: EVEN in z)
                 every feature multiplied by sqrt(pixels in the box) (z features) or sqrt(pixels / 2) (energy): unit
                 variance under N(0, 1) white noise
  work/meta.npz  pixel positions, flat indices, residents, keys of every forecast row
No truth, no station, no monitor file is read (audit hook). Nothing outside P/a17/diag is written.
    python collect.py [workers]"""
import glob
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0]).lower()
        if p.endswith((".py", ".pyc", ".so")) or "/site-packages/" in p:
            return
        if "station" in p or "openaq" in p or "monitor" in p or p.rstrip("/").endswith("_ar") or "_ar/" in p \
                or "/a16/data" in p or "/a16/results" in p:
            raise PermissionError(f"A17 diag: refused path {p}")


sys.addaudithook(_audit)
P = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
OUT = P / "a17" / "diag" / "work"
G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
DIRS16 = [P / "a13/warn16/data/run16", P / "a14/data/run16_13", P / "a13/warn16/data/run16all"]
DIR64 = P / "a14/r2/data/run64"
NPIX, GH, GW, TILE, NTR, NTC = 8112762, 14400, 36000, 512, 29, 71
NSUB, SEED = 300_000, 20261121
WID = (3, 5, 9, 17, 33, 65, 129, 257, 513, 1025)
EWID = (3, 9, 33, 129, 513)
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
    arr = np.load(OUT / f"{which}.npy", mmap_mode="r+")
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
    src = G / ("noise_tiles16.npy" if m < 16 else "noise_tiles64.npy")
    z = full_map(src, m)
    r, c = S["r"], S["c"]
    f = np.zeros((len(r), 17), np.float32)
    f[:, 0] = z[r, c]
    rb, cb = r // 2 * 2, c // 2 * 2
    f[:, 1] = (z[rb, cb] + z[rb + 1, cb] + z[rb, cb + 1] + z[rb + 1, cb + 1]) / 2.0     # mean * sqrt(4)
    cs = integral(z)
    for j, w in enumerate(WID):
        s, n = box_sums(cs, r, c, w)
        f[:, 2 + j] = s / np.sqrt(n)
    del cs
    np.square(z, out=z)
    cs = integral(z)
    del z
    for j, w in enumerate(EWID):
        s, n = box_sums(cs, r, c, w)
        f[:, 12 + j] = (s - n) / np.sqrt(2.0 * n)
    del cs
    arr = np.load(OUT / "feat.npy", mmap_mode="r+")
    arr[m] = f
    arr.flush()
    del arr
    print(f"field {m} done {time.strftime('%H:%M:%S')} sd {f[:, [0, 2, 7, 12]].std(0).round(3).tolist()}", flush=True)
    return m


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    t0 = time.time()
    pp = np.load(G / "pop_pixels.npz")
    idx, pop = pp["idx"], pp["pop"]
    assert len(idx) == NPIX
    sub = np.sort(np.random.default_rng(SEED).choice(NPIX, NSUB, replace=False))
    S["sub"] = sub
    S["r"], S["c"] = (idx[sub] // GW).astype(np.int64), (idx[sub] % GW).astype(np.int64)
    K16, K64 = index(DIRS16, 16), index([DIR64], 64)
    keys16 = {h: sorted(k for k in K16 if k[1] == h) for h in (1, 3)}
    keys64 = {h: sorted(k for k in K64 if k[1] == h) for h in (1, 3)}
    T16, T64 = len(keys16[1]), len(keys64[1])
    assert T16 == len(keys16[3]) == 179 and T64 == len(keys64[3]) == 15, (T16, len(keys16[3]), T64)
    for nm, shp, dt in (("fc16", (2, T16, NSUB, 16), np.float16), ("fc64", (2, T64, NSUB, 64), np.float16),
                        ("feat", (64, NSUB, 17), np.float32)):
        if not (OUT / f"{nm}.npy").exists():
            np.lib.format.open_memmap(OUT / f"{nm}.npy", mode="w+", dtype=dt, shape=shp).flush()
    jobs = [("fc16", li, ti, *K16[k][:2]) for li, h in enumerate((1, 3)) for ti, k in enumerate(keys16[h])]
    jobs += [("fc64", li, ti, *K64[k][:2]) for li, h in enumerate((1, 3)) for ti, k in enumerate(keys64[h])]
    with get_context("fork").Pool(workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(read_fc, jobs, chunksize=1)):
            if n % 50 == 0:
                print(f"forecast rows {n + 1}/{len(jobs)} {time.time() - t0:.0f} s", flush=True)
    print(f"forecasts done {time.time() - t0:.0f} s", flush=True)
    # fields 0-15 of noise_tiles64.npy must be the fields of noise_tiles16.npy (run64 used noise_tiles64 for all 64)
    same = []
    for m in range(16):
        a = np.load(G / "noise_tiles16.npy", mmap_mode="r")[m, 14]
        b = np.load(G / "noise_tiles64.npy", mmap_mode="r")[m, 14]
        same.append(bool(np.array_equal(np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16))))
    assert all(same), same
    with get_context("fork").Pool(max(1, min(8, workers // 2))) as pool:
        list(pool.imap_unordered(feats, range(64), chunksize=1))
    print(f"features done {time.time() - t0:.0f} s", flush=True)
    # read-back checks: 3 forecast rows of each run against a fresh read of the source, feature file finite
    chk = {}
    for nm, K, keys, M in (("fc16", K16, keys16, 16), ("fc64", K64, keys64, 64)):
        arr = np.load(OUT / f"{nm}.npy", mmap_mode="r")
        for li, h in enumerate((1, 3)):
            for ti in (0, len(keys[h]) // 2, len(keys[h]) - 1):
                path, row, _ = K[keys[h][ti]]
                src = np.load(path, mmap_mode="r")[row]
                ok = np.array_equal(np.asarray(src[sub]).view(np.uint16), np.asarray(arr[li, ti]).view(np.uint16))
                chk[f"{nm}_l{h}_t{ti}"] = bool(ok)
    ft = np.load(OUT / "feat.npy", mmap_mode="r")
    chk["feat_finite"] = bool(all(np.isfinite(np.asarray(ft[m])).all() for m in range(64)))
    chk["feat_nonzero"] = bool(all(np.asarray(ft[m, :, 7]).std() > 0.5 for m in range(64)))
    chk["noise64_fields0_15_equal_16"] = all(same)
    print(json.dumps(chk), flush=True)
    assert all(chk.values()), chk
    meta = dict(sub=sub, idx=idx[sub], r=S["r"], c=S["c"], pop=pop[sub], seed=SEED, wid=np.array(WID), ewid=np.array(EWID),
                keys16=np.array([[d, h, K16[(d, h)][2]] for h in (1, 3) for d, _ in keys16[h]], np.int32).reshape(2, T16, 3),
                keys64=np.array([[d, h, K64[(d, h)][2]] for h in (1, 3) for d, _ in keys64[h]], np.int32).reshape(2, T64, 3),
                src16=np.array([K16[k][0] + f":{K16[k][1]}" for h in (1, 3) for k in keys16[h]]),
                checks=json.dumps(chk))
    tmp = OUT / f"meta.tmp{os.getpid()}.npz"
    np.savez(tmp, **meta)
    fd = os.open(tmp, os.O_RDONLY); os.fsync(fd); os.close(fd)
    os.replace(tmp, OUT / "meta.npz")
    z = np.load(OUT / "meta.npz")
    assert np.array_equal(z["sub"], sub) and np.array_equal(z["keys16"], meta["keys16"])
    print(f"all done {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
