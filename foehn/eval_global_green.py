#!/usr/bin/env python3
"""Global test of Foehn-Green (fullflow_global_green.py: FLEXGREEN transport, 2022, leads 1-7 days), a copy of
eval_global_fp.py with the Green's-function inputs: HRES weather of the run of day d and HRES-driven Green's functions of
2022; ensemble member m gathers its latent tracers with its own half of the particles (the same for every window of a
forecast, so that the blended global member is one transport realisation).

Forecast issued at the end of day d. Inputs: PM2.5 of days d and d-1 (1 km), analysed weather of day d, emissions of the
month and fires of day d, and the ECMWF HRES run of 00 UTC day d for everything after day d (daily weather of the
target day, 3-hourly winds and precipitation of days d .. d+7 for the footprint; day-d precipitation from the first
24 h of the run). No information after the issue time enters the forecast.

Global maps without seams: every ensemble member uses ONE global N(0, 1) field (noise_tiles.npy) cut into each window,
and the 21 Europe-sized windows, which overlap by 800-1200 km, are blended with smooth weights (cosine ramps over 600 km
from each window edge). The blended global map is the product that is scored (runs/<run>/<out>/part_<rank>.npz):
  * recs: per (start day, lead, region), sums over land pixels with a valid target (GHAP 1 km), columns in rec_cols:
    errors of the ensemble mean, fair CRPS, ensemble variance, moments, Brier scores and CSI counts (25/35/75 ug/m3) and
    the errors of persistence (GHAP of day d) on the same pixels;
  * the stations of stations_2022.npz: every ensemble member and the GHAP value at the station pixel;
  * seams: mean jump between neighbouring pixels across every window edge divided by the mean jump elsewhere;
  * map25_sd / map25_m0: the 0.25 deg cell means of the 1 km ensemble spread and of member 0, for every forecast;
  * the ensemble-mean map at 0.25 deg (mean over the valid 1 km pixels of each cell, with the same mean of GHAP) for
    every forecast, at 5 km for every 7th start day (files), at 1 km for the start days of --save-1km (files);
  * --pop: every member at the 8.1 M pixels where >= 100 people live (94 % of the world population, GHS-POP 2020) and
    at the station pixels (pop_<rank>.npy, pixel list pop_idx.npy), for exposure and for stations added later, with GHAP of
    the target day there (pop_ghap_<rank>.npy) and the 40 km cell means (0.4 deg from 84N 180W, valid pixels) of the
    ensemble mean and of GHAP (cell40_<rank>.npy: forecasts, 2, 360, 900);
  * --annual: per lead, sums over the land pixels (list annual_idx.npy) of the ensemble mean and of GHAP on the pixel-days
    with a valid target, their count and the days above 37.5 ug/m3 of each (annual_<rank>.npz): annual means at 1 km.
Window inputs are read by DataLoader workers while the GPU samples.
    srun ... python eval_global_fp.py runs/<run> [--ckpt C] [--members 4 --steps 8 --every 1] [--doys 19 --leads 1 7]"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from fullflow_global import CH_, CW_, FH, FW, G, GH, GW, NTC, NTR, PH, PW, TILE, GlobalData, Npy, tile_window  # noqa: E402
from fullflow_global_fp import CTX_H, CTX_W, FullModuleFP, GlobalDataFP, ctx_read  # noqa: E402
from fullflow_global_green import FS_SCALE, NE, FullModuleGreen, GlobalDataGreen, green_block, green_rows  # noqa: E402
from fullflow import MU, SD, FullModule  # noqa: E402
from dejavu import theta_gaps  # noqa: E402

REGIONS = {"Europe": (72, 35, -25, 45), "North America": (72, 15, -170, -50), "South America": (15, -56, -95, -30),
           "Africa": (37, -35, -20, 52), "Middle East": (42, 12, 35, 63), "South Asia": (37, 5, 60, 97),
           "East Asia": (54, 18, 97, 146), "Southeast Asia": (18, -11, 92, 142), "Oceania": (-10, -48, 110, 180)}
THR = (25.0, 35.0, 75.0)
THR_ANN = 37.5                                    # --annual: WHO 2021 24-h interim target 3
RAMP = 600                                                     # px (km) of cosine ramp from each window edge
REC_COLS = (["start_doy", "lead", "region", "n", "sse", "sae", "sum_err", "crps", "var", "sum_t", "sum_f", "sum_tt", "sum_ff",
             "sum_tf", "sse_persist", "sae_persist"] + [f"{k}_{int(t)}" for t in THR for k in ("brier", "hits", "union")])


class TestFP(GlobalDataFP):
    """2022 test samples: the training inputs up to day d, the ECMWF HRES run of 00 UTC day d for days d+1 .. d+L.
    fp=False: the model without transport branch (inputs of fullflow_global.py). Ablations as in training: no_emis zeroes
    the emission and fire channels; wind_shift takes all path weather (winds, rain, stability) from the run issued
    wind_shift days later (earlier at the end of the record), as it took the ERA5 path weather in training."""

    def __init__(self, max_lead, fp=True, no_emis=False, wind_shift=0, era5=False, no_ant=False):
        super().__init__([2022], train=False, max_lead=max_lead, no_emis=no_emis, wind_shift=wind_shift, no_ant=no_ant)
        self.fp, self.era5 = fp, era5                                 # era5: diagnostic, analysed weather (perfect prognosis)
        st = np.load(G / "stats.npz")
        self.em, self.es = st["era5_mean"][:, None, None], st["era5_std"][:, None, None]
        self.rm, self.rs = float(st["extra_mean"][3]), float(st["extra_std"][3])
        self.doy = np.load(G / "meta.npz")["doy"]
        ok = np.load(G / "hres_days.npy")
        self.samples = [(i, h) for i, h in self.samples if ok[int(self.doy[i])]]
        self.hb, self.hw, self.hr0 = Npy(G / "hres_block.npy"), Npy(G / "hres_wind.npy"), Npy(G / "hres_rain0.npy")

    def item(self, i, h, k0, j0):
        it = (GlobalDataFP.item if self.fp else GlobalData.item)(self, i, h, k0, j0)   # forecast-time parts replaced below
        if self.era5:
            return it
        d, L = int(self.doy[i]), self.max_lead
        raw = self.hb.rows([(d, h - 1, c) for c in range(31)], k0, k0 + CH_)[..., j0:j0 + CW_].astype(np.float32)
        blk = np.concatenate([(raw[:30] - self.em) / self.es,
                              ((np.log1p(np.maximum(raw[30], 0) * 1000.0) - self.rm) / self.rs)[None]])
        it["coarse"][-31:] = torch.from_numpy(np.nan_to_num(blk).astype(np.float32))
        if not self.fp:
            return it
        ws, D = self.wind_shift, self.year.shape[0]
        dw = d + ((ws if i + ws + L < D else -ws) if ws else 0)       # run of the path weather (same shift as training)
        wind = ctx_read(self.hw, [(dw, s, ch) for s in range(8 * (L + 1)) for ch in range(4)], k0, j0, 2)
        it["wind"] = torch.from_numpy(np.nan_to_num(wind.reshape(8 * (L + 1), 4, CTX_H // 2, CTX_W // 2).astype(np.float32)).astype(np.float16))
        pool = lambda a: a.reshape(*a.shape[:-2], CTX_H // 2, 2, CTX_W // 2, 2).mean((-3, -1))
        rain = [np.log1p(np.maximum(ctx_read(self.hr0, [(dw,)], k0, j0)[0].astype(np.float32), 0) * 1000.0)]
        rain += [np.log1p(np.maximum(ctx_read(self.hb, [(dw, k - 1, 30)], k0, j0)[0].astype(np.float32), 0) * 1000.0)
                 for k in range(1, L + 1)]
        it["rain"] = torch.from_numpy(pool(np.stack(rain)).astype(np.float16))
        stab = it["stab"].numpy().astype(np.float32)                  # first day: analysed temperatures (as in training)
        for k in range(1, L + 1):                                     # following days: HRES temperatures (store units)
            stab[k] = theta_gaps(pool(ctx_read(self.hb, [(dw, k - 1, c) for c in (5, 6, 7)], k0, j0).astype(np.float32)))
        it["stab"] = torch.from_numpy(stab.astype(np.float16))
        return it


class TestGreen(GlobalDataGreen):
    """test samples of Foehn-Green (2022; or 2021, the validation year, scored against GHAP only): every particle of the
    Green's functions (members choose their share)"""

    def __init__(self, max_lead, particles=32, store="green", emis=False, year=2022, emis_store=None):
        super().__init__([year], train=False, max_lead=max_lead, weather="hres", green="hres", psub=particles, store=store,
                         emis=emis, emis_store=emis_store)
        if year == 2022:
            ok = np.load(G / "hres_days.npy")
            self.samples = [(i, h) for i, h in self.samples if ok[int(self.doy[i])]]
        else:                                       # 2017-2021: HRES runs of hres_block_train.npy (global day index)
            ok = np.load(G / "hres_train_days.npy")
            self.samples = [(i, h) for i, h in self.samples if ok[int(i)]]
        self.lag_stores = {}

    def add_lag(self, lag, year):
        """Green's functions of the start days driven by the HRES run of day d - lag (flexgreen.py --lag, store green_lag<lag>)"""
        root = G / f"green_lag{lag}" / "hres"
        self.lag_stores[lag] = dict(org=Npy(root / f"origins_{year}.npy"), fs=Npy(root / f"fire_{year}.npy"),
                                    es=Npy(root / f"emis_{year}.npy") if self.emis else None,
                                    done={int(q.name) for q in (root / f"done_{year}").glob("*")})

    def lagged(self, i, h, k0, j0, lag):
        """the inputs that change with the weather run: forecast-day weather of the run of day d - lag (its lead h + lag),
        particle origins, fire and emission terms of the transport driven by that run"""
        st = self.lag_stores[lag]
        d = int(self.doy[i])
        assert d in st["done"] and h + lag <= 7, (d, h, lag)
        r0, c0 = k0 // 2, j0 // 2
        cols = (c0 + np.arange(CW_ // 2)) % 720
        out = {"coarse_fc": torch.from_numpy(np.nan_to_num(self.hres_block(i - lag, h + lag, k0, j0)).astype(np.float32)),
               "org": torch.from_numpy(np.ascontiguousarray(green_block(st["org"], d, h, r0, r0 + CH_ // 2, c0, CW_ // 2)).astype(np.float32))}
        fs = green_rows(st["fs"], d, h, 0, 3)[:, r0:r0 + CH_ // 2][:, :, cols]
        out["fsens"] = torch.from_numpy(np.log1p(np.maximum(np.nan_to_num(fs), 0) * FS_SCALE[:, None, None]).astype(np.float32))
        if self.emis:
            es = green_rows(st["es"], d, h, 0, NE)[:, r0:r0 + CH_ // 2][:, :, cols]
            out["esens"] = torch.from_numpy(np.log1p(np.maximum(np.nan_to_num(es), 0)).astype(np.float32))
        return out


def member_particles(i, h, m, total=32, psub=16):
    """indices of the particles of member m for the forecast (i, h): fixed across windows"""
    return np.sort(np.random.default_rng((int(i) * 8 + int(h)) * 1009 + int(m)).choice(total, psub, replace=False))


class Windows(Dataset):
    """Inputs of the forecasts `jobs` = [(start index, lead, window)], with the noise of every member cut from the global
    fields, read in DataLoader workers (the target never enters the network)."""

    def __init__(self, ds, jobs, win, members, lags=()):
        self.ds, self.jobs, self.win, self.members, self.lags = ds, jobs, win, members, sorted({l for l in lags if l > 0})
        self.noise = Npy(G / ("noise_tiles.npy" if members <= 4 else "noise_tiles16.npy"))   # same seeds: members 0-3 identical
        assert members <= self.noise.shape[0], f"noise_tiles.npy has {self.noise.shape[0]} members"

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, n):
        i, h, w = self.jobs[n]
        k0, j0 = (int(v) for v in self.win[w])
        b = self.ds.item(i, h, k0, j0)
        del b["x1"], b["mask"]
        b["x0"] = torch.from_numpy(np.stack([tile_window(self.noise, m, 25 * k0, 25 * j0, PH, PW) for m in range(self.members)]))
        for l in self.lags:                                              # lagged weather runs (members with --lags l)
            for k, v in self.ds.lagged(i, h, k0, j0, l).items():
                b[f"lag{l}_{k}"] = v
        return b


def windows(land, min_land=0.05, stride=(136, 232)):
    H, W = land.shape
    ks = sorted(set(list(range(0, H - CH_, stride[0])) + [H - CH_]))
    js = sorted(set(list(range(0, W - CW_, stride[1])) + [W - CW_]))
    return [(k, j) for k in ks for j in js if land[k:k + CH_, j:j + CW_].mean() >= min_land]


def ramp(n, r):
    x = np.minimum(np.arange(n) + 0.5, n - np.arange(n) - 0.5) / r
    return np.clip(np.sin(0.5 * np.pi * np.clip(x, 0, 1)) ** 2, 1e-3, 1.0).astype(np.float32)


def crop_px(n, s, w, e):
    """rows r0:r1 and columns c0:c1 of the global 1 km grid inside a box in degrees."""
    return int(round((84 - n) * 100)), int(round((84 - s) * 100)), int(round((w + 180) * 100)), int(round((e + 180) * 100))


def full_map(npy, k):
    n = NTR * NTC * TILE * TILE
    return npy._read(k * n, n).reshape(NTR, NTC, TILE, TILE).transpose(0, 2, 1, 3).reshape(NTR * TILE, NTC * TILE)[:GH, :GW]


def region_slices():
    lat1 = 84.0 - 0.01 * (np.arange(GH) + 0.5); lon1 = -180.0 + 0.01 * (np.arange(GW) + 0.5)
    out = {}
    for name, (n, s, w, e) in REGIONS.items():
        r = np.where((lat1 <= n) & (lat1 > s))[0]; c = np.where((lon1 >= w) & (lon1 < e))[0]
        out[name] = (slice(int(r[0]), int(r[-1]) + 1), slice(int(c[0]), int(c[-1]) + 1))
    return out


def block_mean(x, ok, f):
    """mean of x over the pixels ok of every f x f block (NaN where none)."""
    H, W = x.shape[0] // f * f, x.shape[1] // f * f
    s = np.where(ok, x, 0.0)[:H, :W].reshape(H // f, f, W // f, f).sum((1, 3), dtype=np.float64)
    n = ok[:H, :W].reshape(H // f, f, W // f, f).sum((1, 3))
    return np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float16)


def cell_mean(x, ok):
    """mean over the valid pixels around each 0.25 deg point (fine rows 25 k - 12 .. 25 k + 12, as pm.npy)."""
    pad = lambda a, v: np.pad(a, ((12, 13), (12, 13)), constant_values=v)
    return block_mean(pad(x, np.nan), pad(ok, False), 25)[:GH // 25, :GW // 25]


def flow_grid(model, steps, refine_tail=1):
    """flow times of the sampler: the quantiles of the training distribution of tau (model.tau_grid), with the last two
    intervals (where the 1 km detail forms) cut into refine_tail sub-steps each (1 = the grid of training)."""
    g = np.asarray(model.tau_grid(steps), np.float64)
    if refine_tail > 1:
        head, tail = g[:-2], []
        for a_, b_ in ((g[-3], g[-2]), (g[-2], g[-1])):
            tail += list(np.linspace(a_, b_, refine_tail + 1)[1:])
        g = np.concatenate([head, tail])
    return g


@torch.no_grad()
def sample(model, b, x0, steps, refine_tail=1, solver="euler"):
    """probability-flow ODE on the quantile grid of the training tau (flow_grid). solver 'euler': the sampler used so far;
    'heun': second order on the same grid (two network evaluations per interval, one on the last), less discretization
    error. With the exact velocity of a Gaussian target, 12 Euler steps shrink the standard deviation by 13 %
    (tests/test_sampler_eta.py)."""
    grid = flow_grid(model, steps, refine_tail)
    x = x0
    for k in range(len(grid) - 1):
        t, t2 = float(grid[k]), float(grid[k + 1])
        v = model.velocity(x, b, torch.full((x.shape[0],), t, device=x.device))
        if solver == "heun" and t2 < 1.0:
            v = v.float()
            v2 = model.velocity(x + v * (t2 - t), b, torch.full((x.shape[0],), t2, device=x.device)).float()
            x = x + (v + v2) * (0.5 * (t2 - t))
        else:
            x = x + v * (t2 - t)                                     # exactly the previous arithmetic (Euler)
    return x


def region_row(E, t, p, M):
    """sums over the pixels of one region: members E (M, N), target t (N), persistence p (N)."""
    E = E.astype(np.float32)
    f = E.mean(0)
    err = f - t
    if M > 1:
        pair = sum(np.abs(E[a] - E[b]) for a in range(M) for b in range(a + 1, M)) / (M * (M - 1))   # fair CRPS
        crps, var = np.abs(E - t).mean(0) - pair, E.var(0, ddof=1)
    else:
        crps, var = np.abs(err), np.zeros_like(t)
    s = lambda v: float(np.sum(v, dtype=np.float64))
    row = [len(t), s(err ** 2), s(np.abs(err)), s(err), s(crps), s(var), s(t), s(f), s(t * t), s(f * f), s(t * f),
           s((p - t) ** 2), s(np.abs(p - t))]
    for th in THR:
        o, g = t > th, f > th
        row += [s(((E > th).mean(0) - o) ** 2), int((o & g).sum()), int((o | g).sum())]
    return row


def run(a):
    rank, world, local = (int(os.environ.get(k, 0)) for k in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID"))
    world = max(world, 1)
    name = Path(a.run).name
    no_emis = "_noemis" in name
    no_ant = "_noant" in name
    m_ = re.search(r"_windshift(\d+)", name); wind_shift = int(m_.group(1)) if m_ else 0
    ck = Path(a.ckpt) if a.ckpt else Path(a.run) / "final.ckpt"
    state = torch.load(ck, map_location="cpu", weights_only=False)
    fp = any(k.startswith("net.fp.") for k in state["state_dict"])
    green = any(k.startswith("net.enc.") for k in state["state_dict"]) and not fp
    assert green, "eval_global_green.py evaluates Foehn-Green checkpoints (use eval_global_fp.py otherwise)"
    assert state["hyper_parameters"]["max_lead"] == a.max_lead, state["hyper_parameters"]
    hp = state["hyper_parameters"]
    store, emis = a.store or hp.get("store", "green"), bool(hp.get("emis", False))   # --store: e.g. more particles
    P = int(Npy(G / store / "hres" / f"origins_{a.year}.npy").shape[-2])
    assert a.psub <= P, (a.psub, P)
    ds = TestGreen(a.max_lead, P, store, emis, a.year, hp.get("emis_store"))
    win = win_all = windows(np.load(G / "meta.npz")["land"])
    if a.crop_only:                              # case studies: only the windows that overlap the crop boxes (same noise)
        boxes = [crop_px(*c) for c in a.crop]
        win = [(k0, j0) for k0, j0 in win_all if any(25 * k0 < r1 and 25 * k0 + FH > r0 and 25 * j0 < c1 and 25 * j0 + FW > c0
                                                     for r0, r1, c0, c1 in boxes)]
    starts = sorted({i for i, h in ds.samples})
    starts = [i for i in starts if int(ds.doy[i]) in set(a.doys)] if a.doys else starts[::a.every]
    have, leads = set(ds.samples), a.leads or list(range(1, a.max_lead + 1))
    todo = [(i, h) for i in starts for h in leads if (i, h) in have]
    n_all, todo = len(todo), todo[rank::world]
    lags = a.lags or [0] * a.members
    assert len(lags) == a.members, (lags, a.members)
    for l in sorted(set(lags) - {0}):
        ds.add_lag(l, a.year)
    loader = DataLoader(Windows(ds, [(i, h, w) for i, h in todo for w in range(len(win))], win, a.members, lags), batch_size=None,
                        num_workers=a.workers, prefetch_factor=2 if a.workers else None)
    batches = iter(loader)                                           # workers start reading before CUDA is initialised
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    cls = FullModuleGreen
    model = cls.load_from_checkpoint(ck, map_location="cpu").to(dev).eval()
    if rank == 0:
        print(f"{name} | {ck.name} (epoch {state.get('epoch')}) | FLEXGREEN store {store}: {P} particles, {a.psub} per member"
              f"{' + emission term' if emis else ''}{' | NO TRANSPORT (ablation)' if hp.get('no_transport') else ''}"
              f"{' | no emissions' if no_emis else ''}{' | fires only (no anthropogenic inventory)' if no_ant else ''}"
              f"{f' | path weather shifted {wind_shift} d' if wind_shift else ''}"
              f"{' | DIAGNOSTIC: ERA5 weather after day d' if a.era5 else ''} | "
              f"{len(win)} windows, {a.members} members, {a.steps} steps | {n_all} forecasts on {world} GCDs",
              flush=True)
    del state

    FT, STT = Npy(G / "ghap1km_tiles.npy"), Npy(G / "static1km_tiles.npy")
    land = full_map(STT, 1) > 0
    wgt = np.outer(ramp(FH, RAMP), ramp(FW, RAMP))
    wgt_d = torch.from_numpy(wgt).to(dev)
    wsum = np.zeros((GH, GW), np.float32)
    for k0, j0 in win_all:
        wsum[25 * k0:25 * k0 + FH, 25 * j0:25 * j0 + FW] += wgt
    covered = wsum > 0
    inv_w = np.where(covered, 1.0 / np.maximum(wsum, 1e-12), 0.0).astype(np.float32)
    if a.crop_only:                              # keep the pixels whose every window was run: there the blend is the global one
        for k0, j0 in win:
            wsum[25 * k0:25 * k0 + FH, 25 * j0:25 * j0 + FW] -= wgt
        covered &= wsum <= 1e-4
    del wsum
    lc = land & covered
    regs = {} if a.crop_only else region_slices()
    stz = np.load(G / "stations_2022.npz")
    inside = (stz["lat"] < 84.0) & (stz["lat"] > -60.0)
    sr = np.clip(np.floor((84.0 - stz["lat"]) / 0.01).astype(int), 0, GH - 1)
    sc = np.clip(np.floor((stz["lon"] + 180.0) / 0.01).astype(int), 0, GW - 1)
    s_ok = inside & covered[sr, sc]
    if a.year != 2022:                              # stations only evaluate 2022: nothing stored for another year
        s_ok[:] = False
    edges_r = sorted({k * 25 for k, _ in win} | {k * 25 + FH for k, _ in win})
    edges_c = sorted({j * 25 for _, j in win} | {j * 25 + FW for _, j in win})
    edges_r = [e for e in edges_r if 0 < e < GH - 1]; edges_c = [e for e in edges_c if 0 < e < GW - 1]
    out = Path(a.run) / a.out; out.mkdir(exist_ok=True)
    pop_idx = np.load(G / "pop_pixels.npz")["idx"] if a.pop else None           # populated pixels + station pixels
    if a.pop and rank == 0:
        np.save(out / "pop_idx.npy", pop_idx)
    pop_ens, pop_tgt, pop_40 = [], [], []
    ann_idx = np.flatnonzero(lc) if a.annual else None                 # --annual: sums per lead over the land pixels
    if a.annual and rank == 0:
        np.save(out / "annual_idx.npy", ann_idx)
    ann = {h: {k: np.zeros(len(ann_idx), np.float32 if k in ("f", "g") else np.uint16) for k in ("f", "g", "n", "xf", "xg")}
           for h in sorted({h for _, h in todo})} if a.annual else {}
    acc = np.zeros((a.members, GH, GW), np.float16 if a.members > 8 else np.float32)   # 16 members: 17 GB instead of 33 GB per GCD
    recs, seams, keys, st_ens, st_ghap, m25, g25, sd25, m025 = [], [], [], [], [], [], [], [], []
    t_start = time.time()

    def save():
        np.savez(out / f"part_{rank}.npz", recs=np.array(recs, np.float64).reshape(-1, len(REC_COLS)), rec_cols=np.array(REC_COLS),
                 seams=np.array(seams, np.float64).reshape(-1, 3), keys=np.array(keys, np.int32).reshape(-1, 3),
                 st_ens=np.array(st_ens, np.float16).reshape(-1, len(sr), a.members),
                 st_ghap=np.array(st_ghap, np.float16).reshape(-1, len(sr)),
                 map25=np.array(m25, np.float16).reshape(-1, GH // 25, GW // 25),
                 map25_ghap=np.array(g25, np.float16).reshape(-1, GH // 25, GW // 25),
                 map25_sd=np.array(sd25, np.float16).reshape(-1, GH // 25, GW // 25),
                 map25_m0=np.array(m025, np.float16).reshape(-1, GH // 25, GW // 25),
                 windows=np.array(win), regions=np.array(list(REGIONS)), thresholds=np.array(THR),
                 info=np.array([name, ck.name, str(fp), str(no_emis), str(wind_shift), str(a.members), str(a.steps)]))
        if a.pop:                                                     # (forecasts in the order of keys, pixels, members)
            np.save(out / f"pop_{rank}.npy", np.stack(pop_ens))
            np.save(out / f"pop_ghap_{rank}.npy", np.stack(pop_tgt)); np.save(out / f"cell40_{rank}.npy", np.stack(pop_40))
        if a.annual:                                                  # the forecasts summed are those of keys
            np.savez(out / f"annual_{rank}.npz", leads=np.array(sorted(ann)), thr=np.array(THR_ANN),
                     **{f"{k}_{h}": v for h, A in ann.items() for k, v in A.items()})

    for n_done, (i, h) in enumerate(todo):
        d0, dt = int(ds.doy[i]), int(ds.doy[i + h])
        acc.fill(0.0)
        t_io = t_gpu = 0.0
        for (k0, j0) in win:
            t = time.time(); b = next(batches); t_io += time.time() - t
            t = time.time()
            r0, c0 = 25 * k0, 25 * j0
            bt = {k: (v[None] if v.dim() > 0 else v.reshape(1)).to(dev, non_blocking=True) for k, v in b.items() if k != "x0"}
            base = {k: v for k, v in bt.items() if not k.startswith("lag")}
            for m in range(a.members):
                bm = dict(base)
                if lags[m] > 0:                                          # this member's weather run and its transport
                    l = lags[m]
                    bm["coarse"] = base["coarse"].clone(); bm["coarse"][:, -31:] = bt[f"lag{l}_coarse_fc"]
                    for k in ("org", "fsens", "esens"):
                        if f"lag{l}_{k}" in bt:
                            bm[k] = bt[f"lag{l}_{k}"]
                org_m = bm["org"]
                bm["org"] = org_m[..., torch.from_numpy(member_particles(i, h, m, org_m.shape[-2], a.psub)).to(dev), :]
                x0 = b["x0"][m].to(dev).float()[None, None]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x = sample(model, bm, x0, a.steps, a.refine_tail, a.solver)
                acc[m, r0:r0 + FH, c0:c0 + FW] += (wgt_d * (x[0, 0, :FH, :FW].float() * SD + MU)).cpu().numpy()
            t_gpu += time.time() - t
        t = time.time()
        acc *= inv_w                                                  # blended members (0 outside the windows)
        mean = acc.mean(0, dtype=np.float32)                          # float32 even when acc is float16 (16 members)
        tgt = full_map(FT, i + h).astype(np.float32)
        per = full_map(FT, i).astype(np.float32)
        ok = lc & (tgt > 0)
        if a.annual:
            A = ann[h]; v = mean.ravel()[ann_idx]; g = tgt.ravel()[ann_idx]; k = g > 0
            A["f"] += np.where(k, v, 0); A["g"] += np.where(k, g, 0); A["n"] += k
            A["xf"] += k & (v > THR_ANN); A["xg"] += k & (g > THR_ANN)
            del v, g, k
        rows = {}
        for rid, (rname, (rs, cs)) in enumerate(regs.items()):
            mk = ok[rs, cs]
            if not mk.any():
                continue
            rows[rname] = [d0, h, rid] + region_row(acc[:, rs, cs][:, mk], tgt[rs, cs][mk], per[rs, cs][mk], a.members)
            recs.append(rows[rname])
        mean[~covered] = np.nan
        jr = [np.nanmean(np.abs(mean[e] - mean[e - 1])[lc[e] & lc[e - 1]]) for e in edges_r]
        jc = [np.nanmean(np.abs(mean[:, e] - mean[:, e - 1])[lc[:, e] & lc[:, e - 1]]) for e in edges_c]
        dr = np.abs(np.diff(mean, axis=0))[lc[1:] & lc[:-1]]; dc = np.abs(np.diff(mean, axis=1))[lc[:, 1:] & lc[:, :-1]]
        seam = 0.5 * (np.nanmean(jr) / np.nanmean(dr) + np.nanmean(jc) / np.nanmean(dc))
        seams.append((d0, h, seam))
        keys.append((d0, h, dt))
        st_ens.append(np.where(s_ok[:, None], acc[:, sr, sc].T, np.nan))
        st_ghap.append(np.where(s_ok & (tgt[sr, sc] > 0), tgt[sr, sc], np.nan))
        m25.append(cell_mean(mean, ok)); g25.append(cell_mean(tgt, ok))
        sd = np.zeros((GH, GW), np.float32)                          # ensemble spread (1 km), member by member: acc.std
        if a.members > 1:                                             # would make a full float32 copy (33 GB at 16 members)
            for m_ in range(a.members):
                dm = acc[m_].astype(np.float32); dm -= mean; dm *= dm; sd += dm
            del dm
            np.sqrt(sd / a.members, out=sd)
        sd25.append(cell_mean(sd, ok)); del sd                                                     # cell means of the spread
        m025.append(cell_mean(acc[0].astype(np.float32), ok))                                       # one member (variance)
        if a.pop:
            pop_ens.append(acc.reshape(a.members, -1)[:, pop_idx].T.astype(np.float16))
            pop_tgt.append(np.where(tgt.ravel()[pop_idx] > 0, tgt.ravel()[pop_idx], np.nan).astype(np.float16))
            pop_40.append(np.stack([block_mean(mean, ok, 40), block_mean(tgt, ok, 40)]))
        if d0 % 7 == 0:
            np.save(out / f"map5km_d{d0}_h{h}.npy", block_mean(mean, lc, 5))
        for b_, (cn, cs_, cw, ce) in enumerate(a.crop or []):           # every member at 1 km over the crop boxes
            if d0 in a.crop_days:
                cr0, cr1, cc0, cc1 = crop_px(cn, cs_, cw, ce)
                np.savez(out / f"crop{b_}_d{d0}_h{h}.npz", members=acc[:, cr0:cr1, cc0:cc1].astype(np.float16),
                         box=np.array([cn, cs_, cw, ce]), rows=np.array([cr0, cr1]), cols=np.array([cc0, cc1]))
        if d0 in a.save_1km:
            np.save(out / f"map1km_d{d0}_h{h}.npy", mean.astype(np.float16))
            np.save(out / f"map1km_d{d0}_h{h}_member0.npy", np.where(covered, acc[0], np.nan).astype(np.float16))
        t_post = time.time() - t
        rm = lambda r, c: np.sqrt(rows[r][REC_COLS.index(c)] / rows[r][3]) if r in rows else np.nan
        print(f"rank {rank}: start {d0} lead {h}: io wait {t_io:.0f}s gpu {t_gpu:.0f}s post {t_post:.0f}s | seam {seam:.3f} | RMSE "
              + " ".join(f"{r.split()[0]} {rm(r, 'sse'):.1f}/{rm(r, 'sse_persist'):.1f}" for r in ("Europe", "East Asia", "South Asia", "North America", "Africa"))
              + f" (model/persistence) | {n_done + 1}/{len(todo)} {time.time() - t_start:.0f}s", flush=True)
        if (n_done + 1) % 10 == 0 or n_done + 1 == len(todo):
            save()
    print("rank", rank, "done", len(todo), "forecasts", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default=None, help="checkpoint (default runs/<run>/final.ckpt)")
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--refine-tail", type=int, default=1, help="cut the last two flow intervals into N sub-steps each")
    ap.add_argument("--every", type=int, default=1, help="every n-th start day")
    ap.add_argument("--doys", type=int, nargs="*", default=None, help="only these start days (day of year, 0-based)")
    ap.add_argument("--leads", type=int, nargs="*", default=None)
    ap.add_argument("--max-lead", type=int, default=7)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-1km", type=int, nargs="*", default=[], help="start days whose 1 km maps are saved (1 GB each)")
    ap.add_argument("--out", default="eval_global")
    ap.add_argument("--era5", action="store_true", help="diagnostic: analysed (ERA5) weather instead of HRES after day d")
    ap.add_argument("--crop", type=float, nargs=4, action="append", metavar=("N", "S", "W", "E"),
                    help="box (degrees) whose members are all saved at 1 km for the start days of --crop-days")
    ap.add_argument("--crop-days", type=int, nargs="*", default=[])
    ap.add_argument("--crop-only", action="store_true",
                    help="case studies: run only the windows that overlap the crop boxes (identical values inside them; "
                         "stations and maps outside are NaN, no regional scores)")
    ap.add_argument("--annual", action="store_true", help="per lead, sums of the ensemble mean and of GHAP over the land pixels")
    ap.add_argument("--pop", action="store_true", help="store every member at the pixels of pop_pixels.npz (exposure, stations)")
    ap.add_argument("--psub", type=int, default=16, help="particles per member (as in training)")
    ap.add_argument("--solver", choices=["euler", "heun"], default="euler", help="ODE solver of the sampler (euler: as before)")
    ap.add_argument("--lags", type=int, nargs="*", default=None,
                    help="weather run of each member: 0 = run of day d, l = run of day d - l (lagged ensemble; leads <= 7 - l)")
    ap.add_argument("--year", type=int, default=2022, help="2021: validation year, scored against GHAP only (no stations)")
    ap.add_argument("--store", default=None, help="Green's function store (default: the one of the checkpoint), e.g. green256")
    run(ap.parse_args())
