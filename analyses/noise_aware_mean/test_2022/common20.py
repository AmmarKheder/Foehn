#!/usr/bin/env python3
"""A20 common code (a20/A20_draft.md). Imported FIRST by every A20 script; CPU only, no GPU, no model run.

Hooks (install(mode), before any data file is opened):
  every mode   writes only under a20/ (and /tmp, /dev, /proc, ~/.cache): open-for-write, rename, remove, mkdir, link,
               truncate, chmod, ... anywhere else raise PermissionError (nothing outside a20 can be modified)
  'positions'  (stpos20.py) no OpenAQ, monitor or 'ghap' path; of the station files only stations_2022.npz, and the script
               reads only its 'lat' and 'lon' arrays (pixel positions, no measurement)
  'members'    (check20, collect20, fit20) no station, monitor or OpenAQ path, no path containing 'ghap' (no truth), no
               a20 score output: the collecting and fitting code never sees a station or a 1 km estimate
  'tests'      (score20.py --tests) station files only through score19.station_prep (the paper's scorer, compare_fair.py);
               no G2 'pop_ghap' file (the 1 km estimates at the populated pixels); every corrected quantity is compared with a
               synthetic or member-drawn truth only (score20 docstring)
  'decision'   (score20.py after the filing; refused unless a20/FILED holds score20's sha256)
Noise. innovation() and coeffs() are verbatim copies of a16/arnoise22.py (check20 SOURCE: same source text; NOISE: same
bytes as arnoise22.ARNoise(2022, d, h, 0.59, 'philox', 4 or 16).tile(m, tr, tc), the generator of G2 and G11); z_tiles()
is the arithmetic of ARNoise._arith (base 'philox') for several leads from one draw of the innovations; z_maps() lays the
tiles out as fullflow_global.full_map / tile_window do (check20 LAYOUT), so noise pixel (r, c) is 1 km pixel (r, c).
Features. feats_at() = collect18.feats() (a17/diag/collect.py columns 0-11): z at the pixel, the aligned 2 x 2 block,
centred boxes of 3 ... 1025 px, each times sqrt(pixels in the box) (unit variance under white N(0, 1) noise).
Law. Stats accumulates the sufficient statistics of the level law D_k = xbar (F_k - Fbar) . gamma (a17/diag m3_fit,
score18.fit_law) per (block of start days, subset, level bin), so that a cross-fitted law and its out-of-sample R^2, and
kappa^2 / tau^2 of A18 4.1, follow exactly from sums (no second pass). Application (A18 4.2, primary = iterated once):
g_k = F_k . gamma (uncentred, E[g] = 0), m1 = xbar (1 - gbar), cbar = xbar - m1 gbar."""
import sys

sys.dont_write_bytecode = True                                  # no __pycache__ (file quota)
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution"))
A20 = ROOT / "a20"
CODE, DATA, OUT = A20 / "code", A20 / "data", A20 / "code" / "out"
G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
RUN = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/runs/"
           "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12_crps4"))
G2D = RUN / "eval_pop_crps_e01_ar"                              # A16 run G2: members 0-3 at the populated pixels
G11D = ROOT / "a16" / "data" / "run16"                          # A16 run G11: sixteen members at the populated pixels
A19C = ROOT / "a19" / "code"
YEAR, RHO, ROOT_WORD = 2022, 0.59, 1521
NTR, NTC, TILE, GH, GW = 29, 71, 512, 14400, 36000
NPOP = 8112762
WID = (3, 5, 9, 17, 33, 65, 129, 257, 513, 1025)
NF = 12
DAYS = tuple(range(0, 357, 2))                                  # the 179 start days of G2 (and G11)
ND = len(DAYS)
LEADS = (1, 3)                                                  # the leads of G2
LEADS3 = (1, 2, 3)                                              # noise fields collected for members 0-3
NBLK = 6                                                        # cross-fitting: 6 contiguous blocks of start days
NP_SAMPLE = 1_000_000                                           # populated-pixel sample (pop >= min_pop)
PIX_SEED = 2026112020
NULL_STRIDE = 10                                                # null features: every 10th pixel of the sample + stations
LAM = 1e-3                                                      # relative ridge (a17/diag, score18)
LEV = np.array([0.0, 2.0, 5.0, 10.0, 15.0, 25.0, 37.5, 50.0, 75.0, 100.0, 150.0, 250.0])   # score18 LEV
NBIN = len(LEV) + 1
MINBIN = 1000
SIZES = np.array([1, 4, 9, 25, 81, 289, 1089, 4225, 16641, 66049, 263169, 1050625], np.float64)
CFEAT = np.sqrt(np.minimum.outer(SIZES, SIZES) / np.maximum.outer(SIZES, SIZES))   # nested boxes, white noise (score18)
NB_, BLK = 10000, 7                                             # compare_fair's moving-block bootstrap (score19)


def block_of(j):
    """cross-fitting block of start-day index j (0..178): contiguous, about two months each"""
    return int(j) * NBLK // ND


# ======================================================================== hooks
MODE = {"mode": None}
_WFLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_WEVENTS = {"os.rename": (0, 1), "os.remove": (0,), "os.rmdir": (0,), "os.mkdir": (0,), "os.link": (1,), "os.symlink": (1,),
            "os.truncate": (0,), "os.chmod": (0,), "os.chown": (0,), "os.utime": (0,), "shutil.copyfile": (1,),
            "shutil.copytree": (1,), "shutil.move": (0, 1), "shutil.rmtree": (0,)}
_A20_REAL = (os.path.realpath(str(A20)) + "/", (os.path.realpath(os.environ["FOEHN_WORK"]) + "/global_model_forecasting_air_pollution/a20/"),
             (os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution/a20/"))


def _write_ok(p):
    q = os.fsdecode(p)
    if not q.startswith("/"):
        try:
            q = os.path.join(os.getcwd(), q)
        except OSError:
            return False
    rp = os.path.realpath(q)
    ok = rp.startswith(_A20_REAL) or os.path.normpath(q).startswith(_A20_REAL) or rp.startswith(("/tmp/", "/dev/", "/proc/"))
    return ok or rp.startswith(os.path.realpath(os.path.expanduser("~/.cache")) + "/")


def _read_refused(pl, mode):
    if mode == "positions":
        if "openaq" in pl or "monitor" in pl or "ghap" in pl:
            return True
        return "station" in pl and not pl.endswith("/stations_2022.npz")
    if mode == "members":
        return ("station" in pl or "openaq" in pl or "monitor" in pl or "ghap" in pl
                or ("/a20/code/out/" in pl and os.path.basename(pl).startswith("score20")))
    if mode == "tests":
        return "pop_ghap" in pl
    return False


def _audit20(event, args):
    m = MODE["mode"]
    if m is None:
        return
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        wr = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or (isinstance(flags, int) and bool(flags & _WFLAGS))
        if wr and not _write_ok(p):
            raise PermissionError(f"A20: write outside a20 refused: {p}")
        pl = p.lower()
        if not wr and (pl.endswith((".py", ".pyc", ".so")) or "/site-packages/" in pl):      # code, not data
            return
        if _read_refused(pl, m):
            raise PermissionError(f"A20 ({m}): refused path {p}")
    elif event in _WEVENTS:
        for j in _WEVENTS[event]:
            a = args[j] if len(args) > j else None
            if isinstance(a, (str, bytes, os.PathLike)) and not _write_ok(a):
                raise PermissionError(f"A20: {event} outside a20 refused: {os.fsdecode(a)}")


_INSTALLED = []


def install(mode):
    assert mode in ("positions", "members", "tests", "decision"), mode
    assert MODE["mode"] in (None, mode), (MODE["mode"], mode)
    MODE["mode"] = mode
    if not _INSTALLED:
        sys.addaudithook(_audit20)
        _INSTALLED.append(True)


# ======================================================================== atomic writes, hashes
def fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, Path):
        return str(x)
    return x


def write_bytes(path, data):
    p = Path(path)
    assert _write_ok(str(p)) and str(p.resolve()).startswith(os.path.realpath(str(A20)) + "/"), p
    tmp = p.with_name(f"{p.name}.tmp_a20_{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    with open(tmp, "rb") as f:
        assert f.read() == data, tmp
    os.replace(tmp, p)
    fsync_dir(p.parent)
    with open(p, "rb") as f:
        assert f.read() == data, p


def save_json(path, obj):
    write_bytes(path, (json.dumps(jsonable(obj), indent=1) + "\n").encode())


def save_npz(path, **kw):
    p = Path(path)
    assert str(p.resolve()).startswith(os.path.realpath(str(A20)) + "/"), p
    tmp = p.with_name(f"{p.stem}.tmp_a20_{os.getpid()}.npz")
    with open(tmp, "wb") as f:
        np.savez(f, **kw); f.flush(); os.fsync(f.fileno())
    with zipfile.ZipFile(tmp) as zf:
        assert sorted(zf.namelist()) == sorted(k + ".npy" for k in kw), tmp
        assert zf.testzip() is None, f"A20: CRC read-back of {tmp} failed"
    os.replace(tmp, p)
    fsync_dir(p.parent)
    z = np.load(p, allow_pickle=False)
    for k, v in kw.items():
        a, b = z[k], np.asarray(v)
        assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), k


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def finalize_memmap(tmp, final):
    """a memmap written in place by the workers: fsync, rename to its final name, fsync of the folder"""
    fd = os.open(str(tmp), os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
    os.replace(tmp, final)
    fsync_dir(Path(final).parent)


def log(*a):
    print(f"[a20 {time.strftime('%H:%M:%S')}]", *a, flush=True)


# ======================================================================== noise: verbatim copies of a16/arnoise22.py
def innovation(year, doy, m, k, tr, tc):
    """eps_k of member m, start day doy, storage tile (tr, tc): a fresh Philox stream per 7-word key (fixed length)"""
    key = (ROOT_WORD, year, doy, m, k, tr, tc)
    for v in key:
        assert isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) and 0 <= int(v) < 2 ** 32, key
    ss = np.random.SeedSequence([int(v) for v in key])
    return np.random.Generator(np.random.Philox(ss)).standard_normal((TILE, TILE), dtype=np.float32)


def coeffs(h, rho):
    """a_h1 .. a_hh of z_h = sum_k a_hk eps_k: float64, returned as float32"""
    h, rho = int(h), float(rho)
    assert h >= 1 and 0.0 <= rho <= 1.0, (h, rho)
    s = np.sqrt(1.0 - rho * rho)
    return np.array([rho ** (h - 1)] + [s * rho ** (h - k) for k in range(2, h + 1)], np.float64).astype(np.float32)


def z_tiles(doy, m, tr, tc, leads, rho=RHO, year=YEAR):
    """{h: z(m, doy, h) tile (float16)} for h in leads, from ONE draw of eps_1 .. eps_max(leads); per lead the arithmetic of
    arnoise22.ARNoise._arith with base 'philox' (acc float32 = 0; acc += c_k * eps_k in k order, skipping c_k == 0; float16)"""
    eps = {k: innovation(year, doy, m, k, tr, tc) for k in range(1, max(leads) + 1)}
    out = {}
    for h in leads:
        c = coeffs(h, rho)
        acc = np.zeros((TILE, TILE), np.float32)
        for k in range(1, h + 1):
            ck = c[k - 1]
            if ck == 0:
                continue
            acc += ck * eps[k]
        out[h] = acc.astype(np.float16)
    return out


def z_maps(doy, m, leads, padded=False):
    """{h: global map of z(m, doy, h)} (float16), tiles laid out as fullflow_global.full_map: [:GH, :GW] unless padded"""
    maps = {h: np.empty((NTR * TILE, NTC * TILE), np.float16) for h in leads}
    for tr in range(NTR):
        for tc in range(NTC):
            t = z_tiles(doy, m, tr, tc, leads)
            for h in leads:
                maps[h][tr * TILE:(tr + 1) * TILE, tc * TILE:(tc + 1) * TILE] = t[h]
    return maps if padded else {h: v[:GH, :GW] for h, v in maps.items()}


# ======================================================================== features: the arithmetic of a18/code/collect18.py
def integral(z):
    """collect18.integral: cumulative sums in float64 over rows then columns. The row pass is written row by row (the same
    sequential float64 additions as np.cumsum along axis 0, without its full-size float64 temporary of a float16 map)"""
    cs = np.zeros((GH + 1, GW + 1), np.float64)
    for i in range(GH):
        np.add(cs[i, 1:], z[i], out=cs[i + 1, 1:])
    for i in range(1, GH + 1):
        np.cumsum(cs[i, 1:], out=cs[i, 1:])
    return cs


def box_sums(cs, r, c, w):
    h = w // 2
    r0, r1 = np.clip(r - h, 0, GH), np.clip(r + h + 1, 0, GH)
    c0, c1 = np.clip(c - h, 0, GW), np.clip(c + h + 1, 0, GW)
    s = cs[r1, c1] - cs[r0, c1] - cs[r1, c0] + cs[r0, c0]
    return s, ((r1 - r0) * (c1 - c0)).astype(np.float64)


def feats_at(z, r, c):
    """the 12 linear features of collect18.feats() at pixels (r, c) of the map z (float16 values read as float32, as the
    float32 map of collect18; integral in float64)"""
    f = np.zeros((len(r), NF), np.float32)
    zf = lambda rr, cc: np.asarray(z[rr, cc], np.float32)     # noqa: E731
    f[:, 0] = zf(r, c)
    rb, cb = r // 2 * 2, c // 2 * 2
    f[:, 1] = (zf(rb, cb) + zf(rb + 1, cb) + zf(rb, cb + 1) + zf(rb + 1, cb + 1)) / 2.0     # mean * sqrt(4)
    cs = integral(z)
    for j, w in enumerate(WID):
        s, n = box_sums(cs, r, c, w)
        f[:, 2 + j] = s / np.sqrt(n)
    del cs
    return f


# ======================================================================== stored runs
def g2_index():
    """(start, lead) -> (rank, row, target) of G2's part files (score19.part_index arithmetic; only 'keys' is read)"""
    out = {}
    for p in sorted(G2D.glob("part_*.npz")):
        assert ".tmp" not in p.name, p
        r = int(p.stem.split("_")[1])
        with np.load(p) as z:
            for n, (d, h, t) in enumerate(z["keys"].tolist()):
                assert (int(d), int(h)) not in out, (p, d, h)
                out[(int(d), int(h))] = (r, n, int(t))
    return out


def g11_index():
    """(start, lead) -> (pop16 file, row, target) of the finished rows of G11 (score19.index16)"""
    out = {}
    for fk in sorted(G11D.glob("keys16_*.npy")):
        tag = fk.stem.split("_", 1)[1]
        done = set(json.loads((G11D / f"done16_{tag}.json").read_text()))
        for n, (d, h, t) in enumerate(np.load(fk)):
            if n in done:
                key = (int(d), int(h))
                assert key not in out, key
                out[key] = (str(G11D / f"pop16_{tag}.npy"), n, int(t))
    return out


def positions():
    """the A20 sample (stpos20.py): rows 0..NP-1 the populated-pixel sample, rows NP.. the station pixels"""
    z = np.load(DATA / "pix20.npz")
    return {k: z[k] for k in z.files}


# ======================================================================== the law: sufficient statistics
class Stats:
    """per (block, subset, level bin): A = sum w xbar^2 Ft'Ft, c = sum w xbar Ft'D, sD = sum w D^2, sX = sum w xbar^2,
    n = sum w (pixel-forecasts); per (block, subset): U = sum w F'F (uncentred), nU = sum w (member-pixels).
    Sums run over the M members of each valid pixel-forecast; D = x - xbar, Ft = F - Fbar over those members."""

    def __init__(self, K, nsub, binned=True):
        self.K, self.nsub, self.binned = K, nsub, binned
        nb = NBIN if binned else 1
        self.A = np.zeros((NBLK, nsub, nb, K, K))
        self.c = np.zeros((NBLK, nsub, nb, K))
        self.sD = np.zeros((NBLK, nsub, nb))
        self.sX = np.zeros((NBLK, nsub, nb))
        self.n = np.zeros((NBLK, nsub, nb))
        self.U = np.zeros((NBLK, nsub, K, K))
        self.nU = np.zeros((NBLK, nsub))
        self.M = None

    def add(self, blk, sub, X, F, w):
        """X (n, M) members (finite), F (n, M, K) their features, w (n,) weights"""
        n, M = X.shape
        if self.M is None:
            self.M = M
        assert self.M == M and F.shape == (n, M, self.K), (F.shape, X.shape)
        if n == 0:
            return
        xb = X.mean(1)
        D = X - xb[:, None]
        Fm = F.mean(1, keepdims=True)
        Ft = F - Fm
        b = np.digitize(xb, LEV) if self.binned else np.zeros(n, np.int64)
        for bb in np.unique(b):
            s = np.flatnonzero(b == bb)
            ws, xs = w[s], xb[s]
            Fs = Ft[s].reshape(-1, self.K)
            wx2 = np.repeat(ws * xs * xs, M)
            self.A[blk, sub, bb] += (Fs * wx2[:, None]).T @ Fs
            self.c[blk, sub, bb] += Fs.T @ (np.repeat(ws * xs, M) * D[s].reshape(-1))
            self.sD[blk, sub, bb] += float(ws @ (D[s] ** 2).sum(1))
            self.sX[blk, sub, bb] += float(ws @ (xs * xs))
            self.n[blk, sub, bb] += float(ws.sum())
        Fu = F.reshape(-1, self.K)
        self.U[blk, sub] += (Fu * np.repeat(w, M)[:, None]).T @ Fu
        self.nU[blk, sub] += float(w.sum()) * M

    def merge(self, o):
        for k in ("A", "c", "sD", "sX", "n", "U", "nU"):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        self.M = self.M or o.M

    def arrays(self, pre):
        return {pre + k: getattr(self, k) for k in ("A", "c", "sD", "sX", "n", "U", "nU")}

    @classmethod
    def from_arrays(cls, z, pre, M):
        A = np.asarray(z[pre + "A"])
        o = cls(A.shape[-1], A.shape[1], binned=A.shape[2] > 1)
        for k in ("A", "c", "sD", "sX", "n", "U", "nU"):
            setattr(o, k, np.asarray(z[pre + k], np.float64))
        o.M = M
        return o


def solve(A, c):
    K = len(c)
    return np.linalg.solve(A + LAM * np.trace(A) / K * np.eye(K), c)


def cross_fit(S, sub):
    """gamma (NBLK, K): the law of block b fitted on every OTHER block (subset `sub`, all level bins); and the full fit"""
    A = S.A[:, sub].sum(1)                                       # (NBLK, K, K)
    c = S.c[:, sub].sum(1)
    gam = np.stack([solve(A.sum(0) - A[b], c.sum(0) - c[b]) for b in range(NBLK)])
    return gam, solve(A.sum(0), c.sum(0))


def sse(S, gam_by_block, sub):
    """(NBLK, nbin) residual sum of squares of each block under its own law (gam_by_block (NBLK, K))"""
    g = gam_by_block
    return (S.sD[:, sub] - 2 * np.einsum("bnk,bk->bn", S.c[:, sub], g)
            + np.einsum("bk,bnkl,bl->bn", g, S.A[:, sub], g))


def r2(S, gam_by_block, sub, bins=None):
    e = sse(S, gam_by_block, sub)
    t = S.sD[:, sub]
    if bins is not None:
        e, t = e[:, bins], t[:, bins]
    return float(1.0 - e.sum() / t.sum())


def fill_bins(v, n):
    """score18.fill_bins: a bin with fewer than MINBIN pixel-forecasts takes the value of the nearest bin that has them"""
    v = v.copy()
    ok = n >= MINBIN
    assert ok.any(), "no level bin has enough pixel-forecasts"
    good = np.flatnonzero(ok)
    for b in np.flatnonzero(~ok):
        v[b] = v[good[np.argmin(np.abs(good - b))]]
    return v


def q2_params(S, gam_by_block, sub):
    """per block b, from the OTHER blocks under gamma_b: sigmaL2 = gamma' U gamma / nU, kappa2(bin) = SSE / (M - 1) / sX,
    tau2(bin) = sD / (M - 1) / sX (score18.fit_law definitions, s2 over the M members with ddof 1)"""
    M = S.M
    out = dict(sL2=np.zeros(NBLK), sL2_analytic=np.zeros(NBLK), kappa2=np.zeros((NBLK, NBIN)), tau2=np.zeros((NBLK, NBIN)),
               kappa2_raw=np.zeros((NBLK, NBIN)), tau2_raw=np.zeros((NBLK, NBIN)), n_bin=np.zeros((NBLK, NBIN)))
    for b in range(NBLK):
        g = gam_by_block[b]
        oth = [x for x in range(NBLK) if x != b]
        U, nU = S.U[oth, sub].sum(0), S.nU[oth, sub].sum()
        out["sL2"][b] = g @ U @ g / nU
        out["sL2_analytic"][b] = g @ CFEAT @ g
        e = np.array([S.sD[x, sub] - 2 * S.c[x, sub] @ g + np.einsum("k,nkl,l->n", g, S.A[x, sub], g) for x in oth]).sum(0)
        sX, sD, n = S.sX[oth, sub].sum(0), S.sD[oth, sub].sum(0), S.n[oth, sub].sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            k2, t2 = e / (M - 1) / sX, sD / (M - 1) / sX
        out["kappa2_raw"][b], out["tau2_raw"][b], out["n_bin"][b] = k2, t2, n
        out["kappa2"][b], out["tau2"][b] = fill_bins(k2, n), fill_bins(t2, n)
    return out


def correct(X, g):
    """the noise-aware means of the members X (n, M) with their law values g (n, M) (A18 4.2):
    primary 'corr' = xbar - m1 gbar with m1 = xbar (1 - gbar) (iterated once); 'noiter' = xbar - xbar gbar;
    'fixpt' = xbar / max(1 + gbar, 0.5)"""
    xb = X.mean(1)
    gb = g.mean(1)
    m1 = xb - xb * gb
    return dict(plain=xb, corr=xb - m1 * gb, noiter=m1, fixpt=xb / np.maximum(1.0 + gb, 0.5), m1=m1, gbar=gb)


# ======================================================================== Q2 helpers (score18, verbatim arithmetic)
def warn(p):
    """tie-split warning (A18 C2): 1 if p > 1/2, 1/2 if p == 1/2, 0 otherwise"""
    return (p > 0.5).astype(np.float64) + 0.5 * (p == 0.5)


def lookup(tab, m):
    return tab[np.digitize(m, LEV)]


def p_logn(m, v, T):
    from scipy.special import ndtr
    s2 = np.log1p(v)
    s = np.sqrt(s2)
    return np.where(m > 0, ndtr((np.log(np.maximum(m, 1e-12)) - 0.5 * s2 - np.log(T)) / s), 0.0)


def p_gauss(m, v, T):
    from scipy.special import ndtr
    return np.where(m > 0, ndtr((m - T) / (np.sqrt(v) * np.maximum(m, 1e-12))), 0.0)


# ======================================================================== bootstrap (copy of score19.boot_counts)
def boot_counts(nd):
    """counts (10,000, nd) of compare_fair's moving-block bootstrap (= rel_common.block_weights): default_rng(0), blocks of 7"""
    rng = np.random.default_rng(0)
    idx = (rng.integers(0, nd - BLK + 1, size=(NB_, -(-nd // BLK)))[:, :, None] + np.arange(BLK)).reshape(NB_, -1)[:, :nd]
    W = np.zeros((NB_, nd))
    np.add.at(W, (np.arange(NB_)[:, None], idx), 1.0)
    return W


def ci(v):
    """point (index 0) and 95 % percentile interval of the resamples (1..)"""
    v = np.asarray(v, np.float64)
    b = v[1:]
    lo, hi = (np.nanpercentile(b, [2.5, 97.5]) if np.isfinite(b).any() else (np.nan, np.nan))
    return {"est": float(v[0]), "ci": [float(lo), float(hi)]}


def inv_ci(d):
    """an equivalent number from the summary of its inverse (interval inverted, inf where the inverse reaches 0)"""
    e, (lo, hi) = d["est"], d["ci"]
    f = lambda x: (1.0 / x if x > 0 else float("inf"))     # noqa: E731
    return {"est": f(e), "ci": [f(hi), f(lo)]}
