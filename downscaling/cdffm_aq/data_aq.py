"""cdffm_aq.data_aq -- data side of the CD-FFM adaptation (PLAN.md 2.1, 2.5, 2.6, 4.2-1, 5 "Data path").

Stores (same interface, so training, evaluation and the tests share one code path):
  SmokeStore  data/processed/smoke/ (Europe 35-60N 10W-35E; 151 training days 2020-10-01 .. 2021-02-28 as 2 M
              cos(lat)-weighted points per day; 24 hold-out days 2021-03-08 .. 03-31 as full fields) + the Foehn
              met.npy / fires.npy (read-only). Big .npy files are read row by row with os.pread(v) (no memory map:
              64 GB vmem limit on login nodes; the node page cache is shared by the ranks).
  ToyStore    synthetic arrays on the 72 x 144 toy grid, for the CPU shape tests.
Encoder input of day D (47 channels on the 0.25 deg grid, PLAN.md 2.1): 0 = (log1p CAMS PM2.5 day-0 - mu) / sd;
1-37 = ERA5 (met.npy, already standardised); 38-40 = GFAS fires (fires.npy, fixed encoding); 41-46 = T6 transport
((x - mean) / std, then bilinear x 2 with periodic longitude from its 0.5 deg cell centres). Normalisation numbers
come from training days only (manifest 'normalisation_convenience'). No emissions, GHAP, HRES, population, stations.
Target (PLAN.md 4.2-1): k = max(LGHAP, 1); training draws k + U(-0.5, 0.5); y = log1p(.); z = (y - mu_p(doy)) /
sigma_p(doy) from the per-pixel harmonic climatology (K = 0 in the smoke), sigma floor 0.1; PM = expm1(mu + sigma z).
sigma_k (store attribute, run argument --sigma-k, PROTOCOL.md A1(e)): 2 = sigma_p(doy) from the K = 2 harmonic fit of
the residual variance (runs before 2026-10-27); 0 = the annual sigma_p = sqrt(constant term) of that fit, the smoke
configuration (the seasonal fit collapses to the floor for 9 % of the pixels at some doy, which made the z targets
heavy-tailed and the members absurd after the exponential back-transform: logs/GLOBAL_PREP_STATUS.md [fix]).
z_to_ug(pmax=...) is the physical cap of the members (A1(e) safety net); the flow state bound is in model_aq.integrate.
"""
import datetime as dt
import json
import math
import os
import queue
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import SMOKE_ROOT

FOEHN_GLOBAL1KM = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
MET_FIRST = dt.date(2017, 1, 1)          # row 0 of met.npy / fires.npy
SIGMA_FLOOR = 0.1
LGHAP_LAT0, LGHAP_LON0, LGHAP_RES, LGHAP_NLON = -57.995, -179.995, 0.01, 36000
WHO_THRESHOLDS = (15.0, 25.0, 37.5, 50.0, 75.0)
RAW_NAMES = ("cams", "met", "fires", "t6")


def doy_of(d):
    """0-based day of year, as CD-FFM preprocess_fast.doy_of"""
    return (d - dt.date(d.year, 1, 1)).days


def parse_date(s):
    return dt.date.fromisoformat(str(s))


# ---------------------------------------------------------------------------------------------------------------
# .npy row reader
# ---------------------------------------------------------------------------------------------------------------

class NpyRows:
    """Rows of a C-order .npy file through os.pread / os.preadv (one system call per row, thread-safe)."""

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            version = np.lib.format.read_magic(f)
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(f)
            self.offset = f.tell()
        assert not fortran, f"{path}: Fortran order"
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        self.row_shape = self.shape[1:]
        self.row_bytes = int(np.prod(self.row_shape, dtype=np.int64)) * self.dtype.itemsize
        self.fd = os.open(self.path, os.O_RDONLY)

    def __len__(self):
        return self.shape[0]

    def read(self, i, out=None):
        """row i -> array of shape shape[1:]; with out (writable C-contiguous array of that shape and dtype) the
        bytes go straight into it (e.g. a pinned buffer)"""
        if not 0 <= i < self.shape[0]:
            raise IndexError(f"{self.path}: row {i} outside 0..{self.shape[0] - 1}")
        if out is None:
            out = np.empty(self.row_shape, self.dtype)
        assert out.dtype == self.dtype and out.shape == self.row_shape and out.flags.c_contiguous
        mv = memoryview(out).cast("B")
        pos, off = 0, self.offset + i * self.row_bytes
        while pos < self.row_bytes:
            n = os.preadv(self.fd, [mv[pos:]], off + pos)
            if n <= 0:
                raise IOError(f"{self.path}: short read at row {i}")
            pos += n
        return out


# ---------------------------------------------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------------------------------------------

class Norm:
    def __init__(self, cams_mu, cams_sd, t6_mean, t6_std):
        self.cams_mu, self.cams_sd = float(cams_mu), float(cams_sd)
        self.t6_mean = np.asarray(t6_mean, np.float32)
        self.t6_std = np.asarray(t6_std, np.float32)


class SmokeStore:
    """data/processed/smoke (see its manifest.json) + Foehn met.npy / fires.npy. Pixel index k = position in
    grid/lghap_mask.npy; days: training t = (date - 2020-10-01).days, hold-out t = (date - 2021-03-08).days."""
    name = "smoke"

    def __init__(self, root=SMOKE_ROOT, met_path=FOEHN_GLOBAL1KM / "met.npy", fires_path=FOEHN_GLOBAL1KM / "fires.npy"):
        from .model_aq import GRIDS
        root = Path(root)
        self.root = root
        man = json.loads((root / "manifest.json").read_text())
        self.manifest = man
        lg = man["lghap"]
        f0, _, n = lg["train_days"]
        self.train_dates = [parse_date(f0) + dt.timedelta(t) for t in range(int(n))]
        h0, _, nh = lg["holdout_days"]
        self.hold_dates = [parse_date(h0) + dt.timedelta(t) for t in range(int(nh))]
        self.cams_first = parse_date(man["cams"]["days"][0])
        self.t6_first = parse_date(man["t6"]["days"][0])
        self.grid_native = GRIDS["025"]
        mask = np.load(root / "grid/lghap_mask.npy").astype(np.int64)
        i, j = mask // LGHAP_NLON, mask % LGHAP_NLON
        self.lat = LGHAP_LAT0 + LGHAP_RES * i
        self.lon = LGHAP_LON0 + LGHAP_RES * j
        self.n_pix = mask.size
        self.clim_mean = np.load(root / "lghap_clim/clim_mean_smoke.npy")
        self.clim_var = np.load(root / "lghap_clim/clim_var_smoke.npy")
        self.static = np.load(root / "lghap_clim/static_smoke.npy")
        self.static_dim = self.static.shape[1]
        self.sigma_floor = float(lg["static"]["sigma_floor"])
        self.idx = NpyRows(root / "lghap_points/idx_smoke.npy")
        self.val = NpyRows(root / "lghap_points/val_smoke.npy")
        self.n_points = self.idx.shape[1]
        self.hold = NpyRows(root / "lghap_eval/holdout_smoke.npy")
        self.cams = NpyRows(root / "coarse/cams_pm25_025_smoke.npy")
        self.t6 = NpyRows(root / "transport/t6_smoke.npy")
        self.met = NpyRows(met_path)
        self.fires = NpyRows(fires_path)
        nc = man["normalisation_convenience"]
        self.norm = Norm(nc["cams_log1p_mean_std"][0], nc["cams_log1p_mean_std"][1], nc["t6_channel_mean"],
                         nc["t6_channel_std"])
        box = lg["box"]
        i0, j0 = box["lghap_rows_i"][0], box["lghap_cols_j"][0]
        self.tile_shape = tuple(box["tile_shape"])
        self.tile_row = (i - i0).astype(np.int64)
        self.tile_col = (j - j0).astype(np.int64)
        self.tile_row_lat = LGHAP_LAT0 + LGHAP_RES * (i0 + np.arange(self.tile_shape[0]))
        self.tile_col_lon0 = LGHAP_LON0 + LGHAP_RES * j0
        self.res_deg = LGHAP_RES
        assert len(self.idx) == len(self.train_dates) and len(self.hold) == len(self.hold_dates)
        assert self.clim_mean.shape == (self.n_pix, 5) and self.static.shape[0] == self.n_pix
        assert self.cams.row_shape == (self.grid_native.H, self.grid_native.W)
        assert self.met.row_shape == (37, self.grid_native.H, self.grid_native.W)
        assert self.t6.row_shape == (6, self.grid_native.H // 2, self.grid_native.W // 2)

    def raw_shapes(self):
        return {"cams": self.cams.row_shape, "met": self.met.row_shape, "fires": self.fires.row_shape,
                "t6": self.t6.row_shape}

    def raw_inputs(self, d, out=None):
        """the four raw float16 input arrays of day d (CAMS ug m-3, standardised ERA5, encoded fires, raw T6)"""
        out = out or {}
        rows = {"cams": (self.cams, (d - self.cams_first).days), "met": (self.met, (d - MET_FIRST).days),
                "fires": (self.fires, (d - MET_FIRST).days), "t6": (self.t6, (d - self.t6_first).days)}
        return {k: r.read(i, out.get(k)) for k, (r, i) in rows.items()}

    def points(self, t):
        """training day t: (positions k (n_points,) uint32, raw LGHAP integers (n_points,) int16)"""
        return self.idx.read(t), self.val.read(t)

    def holdout(self, t):
        """hold-out day t: raw LGHAP integers at every pixel (n_pix,) int16"""
        return self.hold.read(t)


class ToyStore:
    """Synthetic store with the SmokeStore interface on the 72 x 144 toy grid (CPU shape tests only)."""
    name = "toy"

    def __init__(self, n_train=6, n_hold=3, tile=(100, 200), n_points=4000, seed=0):
        from .model_aq import GRIDS
        g = np.random.default_rng(seed)
        self.seed = seed
        self.grid_native = GRIDS["toy"]
        self.train_dates = [dt.date(2020, 10, 1) + dt.timedelta(t) for t in range(n_train)]
        self.hold_dates = [dt.date(2021, 3, 8) + dt.timedelta(t) for t in range(n_hold)]
        self.tile_shape = tuple(tile)
        valid = g.random(tile) < 0.9
        rr, cc = np.nonzero(valid)
        self.tile_row, self.tile_col = rr.astype(np.int64), cc.astype(np.int64)
        self.res_deg = LGHAP_RES
        self.tile_row_lat = 45.005 + LGHAP_RES * np.arange(tile[0])
        self.tile_col_lon0 = 10.005
        self.lat = self.tile_row_lat[rr]
        self.lon = self.tile_col_lon0 + LGHAP_RES * cc
        self.n_pix = rr.size
        self.clim_mean = np.zeros((self.n_pix, 5), np.float32)
        self.clim_mean[:, 0] = 2.4 + 0.3 * g.standard_normal(self.n_pix)
        self.clim_var = np.zeros((self.n_pix, 5), np.float32)
        self.clim_var[:, 0] = 0.2 * (0.5 + g.random(self.n_pix))
        self.static = g.standard_normal((self.n_pix, 6)).astype(np.float16)
        self.static_dim = 6
        self.sigma_floor = SIGMA_FLOOR
        self.n_points = n_points
        self._idx = np.sort(np.stack([g.choice(self.n_pix, n_points, replace=False) for _ in range(n_train)]), 1)
        self._val = np.maximum(1, np.round(np.exp(2.4 + 0.5 * g.standard_normal((n_train, n_points))))).astype(np.int16)
        self._hold = np.maximum(1, np.round(np.exp(2.4 + 0.5 * g.standard_normal((n_hold, self.n_pix))))).astype(np.int16)
        self.norm = Norm(2.2, 0.9, np.zeros(6), np.ones(6))
        H, W = self.grid_native.H, self.grid_native.W
        self._shapes = {"cams": (H, W), "met": (37, H, W), "fires": (3, H, W), "t6": (6, H // 2, W // 2)}

    def raw_shapes(self):
        return dict(self._shapes)

    def raw_inputs(self, d, out=None):
        g = np.random.default_rng([self.seed, d.toordinal()])
        arr = {"cams": np.exp(2.2 + 0.9 * g.standard_normal(self._shapes["cams"])),
               "met": g.standard_normal(self._shapes["met"]), "fires": 0.1 * g.standard_normal(self._shapes["fires"]),
               "t6": np.abs(g.standard_normal(self._shapes["t6"]))}
        res = {}
        for k, a in arr.items():
            a = a.astype(np.float16)
            if out and k in out:
                out[k][...] = a
                a = out[k]
            res[k] = a
        return res

    def points(self, t):
        return self._idx[t].astype(np.uint32), self._val[t]

    def holdout(self, t):
        return self._hold[t]


def open_store(name, **kw):
    if name == "global":                       # cdffm_aq.store_global (kw: fires = injh | mami, see store_kwargs)
        from .store_global import GlobalStore
        return GlobalStore(**kw)
    if name == "ghap":                         # cdffm_aq.store_ghap (PROTOCOL.md A2: GHAP V1 target, split G)
        from .store_ghap import GhapStore
        return GhapStore(**kw)
    return SmokeStore(**kw) if name == "smoke" else ToyStore(**kw)


def store_kwargs(args):
    """open_store keyword arguments recorded in a run's args: the fires file of the global store (default injh, the
    real-time-compatible channel 40, PROTOCOL.md amendment 2026-10-26) and the sigma harmonics of the target (sigma_k,
    default 2 = the runs before A1(e)); nothing for smoke / toy (unchanged)"""
    if getattr(args, "store", None) == "global":
        sk = getattr(args, "sigma_k", None)
        return {"fires": getattr(args, "fires", None) or "injh", "split": getattr(args, "split", None) or "A",
                "sigma_k": 2 if sk is None else int(sk)}
    if getattr(args, "store", None) == "ghap":       # PROTOCOL.md A2 (split G; annual sigma unless stated)
        sk = getattr(args, "sigma_k", None)
        return {"fires": getattr(args, "fires", None) or "injh", "split": getattr(args, "split", None) or "G",
                "sigma_k": 0 if sk is None else int(sk)}
    return {}


def in_channels(transport):
    return 47 if transport == "t6" else 41


# ---------------------------------------------------------------------------------------------------------------
# Device side: per-pixel arrays, target transform, encoder input
# ---------------------------------------------------------------------------------------------------------------

class PixelArrays:
    """per-pixel arrays of a store on one device (PLAN.md 5: they sit on every GCD)"""

    def __init__(self, store, device):
        t = lambda a, dtype=None: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype)
        self.device = device
        self.lat = t(store.lat, torch.float32)
        self.lon = t(store.lon, torch.float32)
        self.w_area = torch.cos(torch.deg2rad(t(store.lat, torch.float64))).float()
        self.clim_mean = t(store.clim_mean, torch.float32)
        self.clim_var = t(store.clim_var, torch.float32)
        self.static = t(store.static)
        self.sigma_floor = store.sigma_floor
        self.sigma_k = int(getattr(store, "sigma_k", 2))          # 2: seasonal sigma (K = 2); 0: annual sigma
        self.norm = store.norm
        self.t6_mean = t(store.norm.t6_mean, torch.float32)
        self.t6_std = t(store.norm.t6_std, torch.float32)

    def mu_sigma(self, k, doy):
        """k (B, N) pixel indices, doy (B,) -> mu, sigma (B, N): climstats.eval_harmonic at the points"""
        ang = 2.0 * math.pi * doy.double() / 365.0
        b = torch.stack([torch.ones_like(ang), torch.cos(ang), torch.sin(ang), torch.cos(2 * ang),
                         torch.sin(2 * ang)], -1).float()[:, None, :]                   # (B, 1, 5)
        mu = (self.clim_mean[k] * b).sum(-1)
        cv = self.clim_var[k]
        var = (cv * b).sum(-1) if self.sigma_k == 2 else cv[..., 0]          # sigma_k 0: annual residual variance
        return mu, torch.sqrt(torch.clamp(var, min=self.sigma_floor ** 2))


def target_z(v, mu, sigma, u=None):
    """raw LGHAP integers v -> z; u: dequantisation noise U(-0.5, 0.5) (training) or None (evaluation truth)"""
    k = v.float().clamp(min=1.0)
    if u is not None:
        k = k + u
    return (torch.log1p(k) - mu) / sigma


def z_to_ug(z, mu, sigma, pmax=None):
    """z -> PM2.5 ug m-3 (expm1(mu + sigma z), clamped at 0; pmax: physical cap of the members, PROTOCOL.md A1(e))"""
    return torch.expm1(mu + sigma * z).clamp(min=0.0, max=pmax)


def upsample2_periodic(x):
    """(B, C, h, w) values at 0.5 deg cell centres -> (B, C, 2h, 2w) at the 0.25 deg points: bilinear (the points
    sit at -1/4 and +1/4 of a 0.5 deg cell = F.interpolate x 2, align_corners False), periodic in longitude"""
    xp = torch.cat([x[..., -1:], x, x[..., :1]], dim=-1)
    up = F.interpolate(xp, scale_factor=2, mode="bilinear", align_corners=False)
    return up[..., 2:2 + 2 * x.shape[-1]]


def assemble_inputs(raw, pix, grid_mode="025", transport="t6"):
    """raw: dict of (B, ...) float16 tensors on the device -> encoder input (B, 47 | 41, H, W) float32"""
    n = pix.norm
    cams = (torch.log1p(raw["cams"].float()) - n.cams_mu) / n.cams_sd
    x = torch.cat([cams[:, None], raw["met"].float(), raw["fires"].float()], dim=1)
    if grid_mode == "05":
        x = F.avg_pool2d(x, 2)
    if transport == "t6":
        t6 = (raw["t6"].float() - pix.t6_mean[:, None, None]) / pix.t6_std[:, None, None]
        if grid_mode == "025":
            t6 = upsample2_periodic(t6)
        x = torch.cat([x, t6], dim=1)
    return x


def raw_to_device(raw_np, device):
    """dict of numpy arrays of ONE day -> dict of (1, ...) tensors on the device"""
    return {k: torch.from_numpy(np.ascontiguousarray(v))[None].to(device) for k, v in raw_np.items()}


def bilinear_log1p_at(field, lat, lon, grid):
    """field (H, W) ug m-3 on the grid's points -> values at lat, lon (...): bilinear in log1p, periodic longitude
    (the B1 baseline, as the 0.4 -> 0.25 deg CAMS regrid)"""
    H, W = field.shape
    r = ((grid.lat0 - lat.double()) / grid.dlat).clamp(0.0, H - 1.0)
    c = torch.remainder((lon.double() - grid.lon0) / grid.dlon, W)
    r0 = torch.floor(r).clamp(max=H - 2)
    c0 = torch.floor(c)
    fr, fc = (r - r0).float(), (c - c0).float()
    r0, c0 = r0.long(), c0.long()
    c1 = torch.remainder(c0 + 1, W)
    L = torch.log1p(field.float().clamp(min=0.0))
    v = ((1 - fr) * (1 - fc) * L[r0, c0] + (1 - fr) * fc * L[r0, c1]
         + fr * (1 - fc) * L[r0 + 1, c0] + fr * fc * L[r0 + 1, c1])
    return torch.expm1(v)


# ---------------------------------------------------------------------------------------------------------------
# Training batches: one prefetch thread per rank
# ---------------------------------------------------------------------------------------------------------------

class BatchStream:
    """Draws B training days (with replacement) and N of their stored points (uniformly, i.e. area-weighted since the
    store holds cos(lat)-weighted draws) with this rank's own numpy generator ([seed, rank, start_step]), reads the
    raw inputs and the targets into a ring of (pinned) host buffers in a background thread; next() moves one batch
    to the device. Rank-distinct draws are what PLAN.md 5 asks (seed + rank)."""

    def __init__(self, store, B, N, seed, rank, device, start_step=0, n_slots=3):
        self.store, self.B, self.N, self.device = store, B, N, device
        self.rng = np.random.default_rng([seed, rank, start_step])
        self.pin = device.type == "cuda"
        shapes = store.raw_shapes()

        def buf(shape, dtype):
            x = torch.empty(shape, dtype=dtype)
            return x.pin_memory() if self.pin else x

        self.slots = [dict(raw={k: buf((B,) + tuple(s), torch.float16) for k, s in shapes.items()},
                           k=buf((B, N), torch.int64), v=buf((B, N), torch.int16), day=buf((B,), torch.int64),
                           doy=buf((B,), torch.int64)) for _ in range(n_slots)]
        self.free, self.ready = queue.Queue(), queue.Queue()
        for i in range(n_slots):
            self.free.put((i, None))
        self.wait_s = 0.0
        self.fill_s, self.fill_bytes = 0.0, 0         # loader throughput (prefetch thread): seconds and bytes read
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"prefetch-rank{rank}")
        self.thread.start()

    def _run(self):
        try:
            while True:
                i, ev = self.free.get()
                if i is None:
                    return
                if ev is not None:
                    ev.synchronize()
                import time
                t0 = time.time()
                self._fill(self.slots[i])
                self.fill_s += time.time() - t0
                self.ready.put(i)
        except BaseException as e:  # noqa: BLE001  (handed to the main thread)
            self.ready.put(e)

    def _fill(self, s):
        st = self.store
        days = self.rng.integers(0, len(st.train_dates), self.B)
        for b, t in enumerate(days):
            d = st.train_dates[int(t)]
            st.raw_inputs(d, out={k: s["raw"][k][b].numpy() for k in s["raw"]})
            idx, val = st.points(int(t))
            self.fill_bytes += sum(v[b].numel() * 2 for v in s["raw"].values()) + idx.nbytes + val.nbytes
            m = self.rng.integers(0, idx.shape[0], self.N)
            s["k"][b] = torch.from_numpy(idx[m].astype(np.int64))
            s["v"][b] = torch.from_numpy(val[m])
            s["day"][b] = int(t)
            s["doy"][b] = doy_of(d)

    def next(self):
        import time
        t0 = time.time()
        i = self.ready.get()
        self.wait_s += time.time() - t0
        if isinstance(i, BaseException):
            raise RuntimeError("prefetch thread failed") from i
        s = self.slots[i]
        cpu = self.device.type == "cpu"
        mv = lambda x: x.to(self.device, non_blocking=not cpu, copy=cpu)
        out = dict(raw={k: mv(v) for k, v in s["raw"].items()}, k=mv(s["k"]), v=mv(s["v"]), day=s["day"].clone(),
                   doy=mv(s["doy"]))
        ev = None
        if self.device.type == "cuda":
            ev = torch.cuda.Event()
            ev.record()
        self.free.put((i, ev))
        return out

    def close(self):
        self.free.put((None, None))


# ---------------------------------------------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------------------------------------------

def fair_crps(ens, y):
    """ens (M, P), y (P,) -> (P,): mean_m |X_m - y| - sum_{m,m'} |X_m - X_m'| / (2 M (M - 1)) (Ferro 2014, unbiased
    for the CRPS of the distribution the members are drawn from); M = 1: |X - y| (= MAE)"""
    M = ens.shape[0]
    e = (ens - y[None]).abs().mean(0)
    if M == 1:
        return e
    xs = torch.sort(ens, dim=0).values
    i = torch.arange(M, device=ens.device, dtype=ens.dtype)[:, None]
    pair = 2.0 * ((2.0 * i - M + 1.0) * xs).sum(0)            # sum over all ordered pairs of |x_i - x_j|
    return e - pair / (2.0 * M * (M - 1))


class Scores:
    """Weighted sums for one forecast (M members; M = 1 for a deterministic baseline) against the truth, in ug m-3 and
    in log1p; WHO 24-h threshold contingency (member 0 and ensemble mean) and Brier score of the member fraction.
    All entries are plain sums, so ranks and days add up (all_reduce)."""

    SPACES = ("ug", "log")

    def __init__(self, M, device, thresholds=WHO_THRESHOLDS):
        self.M, self.thr = M, tuple(thresholds)
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        self.s = {}
        for sp in self.SPACES:
            for k in ("n", "se1", "seM", "ae1", "aeM", "bias", "crps", "spread", "var"):
                self.s[f"{sp}/{k}"] = z(())
            self.s[f"{sp}/rank"] = z(M + 1)
        T = len(self.thr)
        for k in ("obs", "f1", "hit1", "fM", "hitM", "brier"):
            self.s[f"thr/{k}"] = z(T)

    def add(self, ens, y, w):
        """ens (M, P) ug m-3, y (P,) ug m-3 truth, w (P,) weights"""
        w = w.double()
        for sp in self.SPACES:
            X, Y = (ens, y) if sp == "ug" else (torch.log1p(ens), torch.log1p(y))
            X, Y = X.float(), Y.float()
            m = X.mean(0)
            a = self.s
            a[f"{sp}/n"] += w.sum()
            a[f"{sp}/se1"] += (w * (X[0] - Y).double() ** 2).sum()
            a[f"{sp}/seM"] += (w * (m - Y).double() ** 2).sum()
            a[f"{sp}/ae1"] += (w * (X[0] - Y).double().abs()).sum()
            a[f"{sp}/aeM"] += (w * (m - Y).double().abs()).sum()
            a[f"{sp}/bias"] += (w * (m - Y).double()).sum()
            a[f"{sp}/crps"] += (w * fair_crps(X, Y).double()).sum()
            if self.M > 1:
                a[f"{sp}/spread"] += (w * X.std(0, unbiased=False).double()).sum()
                a[f"{sp}/var"] += (w * X.var(0, unbiased=True).double()).sum()
            rank = (X < Y[None]).sum(0)
            # one-hot sum, not index_add_: float64 atomics into M + 1 bins serialise on MI250X (~3 s per call)
            bins = torch.arange(self.M + 1, device=w.device)[:, None]
            a[f"{sp}/rank"] += ((rank[None] == bins).double() * w[None]).sum(1)
        thr = torch.tensor(self.thr, device=y.device, dtype=y.dtype)[:, None]          # (T, 1)
        o = y[None] >= thr
        f1 = ens[0][None] >= thr
        fM = ens.mean(0)[None] >= thr
        p = (ens[None] >= thr[..., None]).float().mean(1)                                # (T, P)
        wt = w[None]
        a = self.s
        a["thr/obs"] += (wt * o).sum(1)
        a["thr/f1"] += (wt * f1).sum(1)
        a["thr/hit1"] += (wt * (o & f1)).sum(1)
        a["thr/fM"] += (wt * fM).sum(1)
        a["thr/hitM"] += (wt * (o & fM)).sum(1)
        a["thr/brier"] += (wt * (p.double() - o.double()) ** 2).sum(1)

    def all_reduce(self):
        import torch.distributed as dist
        for v in self.s.values():
            dist.all_reduce(v)

    def summary(self):
        out = {}
        M = self.M
        for sp in self.SPACES:
            g = lambda k: float(self.s[f"{sp}/{k}"])
            n = max(g("n"), 1e-300)
            rmseM = math.sqrt(g("seM") / n)
            r = {"n": g("n"), "crps_fair": g("crps") / n, "rmse_1member": math.sqrt(g("se1") / n), "rmse_ensmean": rmseM,
                 "mae_1member": g("ae1") / n, "mae_ensmean": g("aeM") / n, "bias": g("bias") / n}
            if M > 1:
                r["spread"] = g("spread") / n
                r["spread_skill"] = r["spread"] / max(rmseM, 1e-12)
                r["spread_skill_fortin"] = math.sqrt((M + 1) / M * g("var") / n) / max(rmseM, 1e-12)
                rk = self.s[f"{sp}/rank"]
                r["rank_hist"] = (rk / rk.sum().clamp(min=1e-300)).tolist()
            out[sp] = r
        n = max(float(self.s["ug/n"]), 1e-300)
        thr = {}
        for i, t in enumerate(self.thr):
            g = lambda k: float(self.s[f"thr/{k}"][i])
            obs = g("obs")
            row = {"base_rate": obs / n, "brier": g("brier") / n}
            for tag in ("1", "M"):
                hit, fc = g(f"hit{tag}"), g(f"f{tag}")
                miss, fa = obs - hit, fc - hit
                cn = n - hit - miss - fa
                H = hit / obs if obs > 0 else float("nan")
                Fr = fa / (fa + cn) if fa + cn > 0 else float("nan")
                row[f"pod_{tag}"] = H
                row[f"far_{tag}"] = fa / fc if fc > 0 else float("nan")
                row[f"csi_{tag}"] = hit / (hit + miss + fa) if hit + miss + fa > 0 else float("nan")
                row[f"sedi_{tag}"] = sedi(H, Fr)
                row[f"freq_bias_{tag}"] = fc / obs if obs > 0 else float("nan")
            thr[f"{t:g}"] = row
        out["thresholds_ug"] = thr
        return out


def sedi(H, F_):
    """symmetric extremal dependence index (Ferro and Stephenson 2011); NaN when H or F is 0 or 1"""
    try:
        if not (0 < H < 1 and 0 < F_ < 1):
            return float("nan")
        lf, lh, l1f, l1h = math.log(F_), math.log(H), math.log(1 - F_), math.log(1 - H)
        return (lf - lh - l1f + l1h) / (lf + lh + l1f + l1h)
    except (ValueError, TypeError):
        return float("nan")
