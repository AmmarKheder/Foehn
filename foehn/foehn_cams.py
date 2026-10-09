#!/usr/bin/env python3
"""Foehn trained as Aurora: the CAMS experiment of Aurora (Bodnar et al. 2025, Nature 641:1180) with Foehn's network,
transport and flow matching. Everything that defines the task is Aurora's; everything that defines the model is Foehn's.
Task (Aurora, 0.4 deg CAMS):
  * grid: the native CAMS / Aurora 0.4 deg grid, no regridding (rows 84 N .. 60 S: the domain of the transport);
  * initial conditions: the CAMS analysis at t0 (00 UTC) and t0 - 12 h, the 13 pollutant fields of cams_series.npy
    (PM1, PM2.5, PM10, total columns of CO, NO, NO2, O3, SO2, and CO, NO, NO2, O3, SO2 at 1000 hPa), plus Aurora's static
    fields (land-sea mask, orography, the 4 emission maps of its static file);
  * target: PM2.5 of the CAMS analysis at t0 + 12 h, 24 h, ..., 120 h, instantaneous (no daily mean, no weighting);
  * training: initial times Oct 2017 - Apr 2022 whose targets precede 1 May 2022 (CAMS analysis, as Aurora's CAMS
    fine-tuning; Aurora also pretrains on EAC4 2003-2021, Foehn does not); the final checkpoint is used (no selection);
  * test: initialisations 00 UTC 1 May - 30 Nov 2022, against the CAMS analysis (eval_cams12h.py).
Model (Foehn):
  * the weather of the forecast is the IFS HRES run of day d (1 deg, 6-hourly, wb2/hres_B), as Foehn and as the CAMS
    forecasts themselves (which run on IFS); the fields at t0 and at the valid time, and the precipitation of the 12 h
    before it;
  * transport: FLEXGREEN in Aurora's protocol (flexgreen.py --instant --all-cells: 32 particles released at the valid time
    in every 0.5 deg cell, integrated back to t0; fires: GFAS of day d - 1, the last complete day at t0); 32 latent
    tracers made from the CAMS analysis at t0 and those fires are gathered at the particle origins (as Foehn), with the
    aloft fraction, the survival, the transported PM2.5 and the 3 fire terms (and, with --emis, the 6 emission terms);
  * network: Foehn's U-Net (residual blocks with GroupNorm, SiLU, FiLM on flow time and lead) and its global self-
    attention, here on the whole Earth at once: 0.4 -> 0.8 -> 1.6 -> 3.2 deg, 4 transformer blocks over the 46 x 114
    tokens of the globe; conditioning (weather, transport, fires) injected at 0.4 deg by bilinear sampling + 1x1 conv;
  * conditional flow matching, x1 = (PM2.5 - 15) / 20, Foehn's resolution-shifted schedule with Foehn's rule for the shift
    (linear size ratio to a 168 x 280 image: s = 2.6 here, 25 for the 1 km windows), AdamW + OneCycle, bf16;
  * forecasts: M members (independent noise, random half of the particles), Euler on the quantile grid with refined tail.
    python foehn_cams.py prep                                  (static fields + HRES statistics, CPU, once)
    python foehn_cams.py check [--split train]                 (one sample)
    python foehn_cams.py train [--epochs 40] [--emis] ...      (srun, 8 GCDs per node)
    python foehn_cams.py predict --run <run> [--members 8]     (srun; runs/<run>/cams12h/pred.npy)"""
import argparse
import glob
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "fullimage"))
sys.path.insert(0, str(ROOT.parent / "dejavu"))
from fullflow_global import G, Npy  # noqa: E402
from fullflow import AttnBlock, FullModule, Timer  # noqa: E402
from dejavu import MU, SD, Block, emb  # noqa: E402

CAMS, CMETA = G / "cams_series.npy", G / "cams_series_meta.npz"
GREEN12 = G / "green12h" / "hres"
WB = G / "wb2"
STATIC, HSTATS = G / "cams_static.npz", G / "cams_hres_stats.npz"
NY, NX = 361, 900                    # 0.4 deg points: lat 84 - 0.4 i, lon -180 + 0.4 j
PY, PX = 368, 912                    # padded to a multiple of 8: 7 rows south (masked), 12 columns wrapped (masked)
NR, NC = 288, 720                    # receptors: 0.5 deg cells, centres 84 - 0.125 - 0.5 r, -180 + 0.125 + 0.5 c
NL = 10                              # leads 12, 24, ..., 120 h
NCH = 13                             # CAMS pollutant channels (cams_series.npy), PM2.5 = channel 1
NST = 6                              # static: land-sea mask, orography, 4 emission maps (Aurora's static file)
NW = 53                              # HRES channels (static_1deg.npz): u, v, w, t, q, z at 8 levels, u10, v10, t2m, sp, tp6h
NWX = 2 * NW + 1                     # weather at t0 and at the valid time, precipitation of the 12 h before it
NG, NE = 32, 6                       # latent tracers, emission terms (as Foehn)
NGF = NG + 2 + 1 + 3                 # gathered tracers, aloft fraction, survival, transported PM2.5, fire terms
FS_SCALE = np.array([1e9, 1.0, 1e9], np.float32)
FIRST = np.datetime64("2017-10-01")
GI_FIRST = 273                       # global day index (meta.npz, days since 1 Jan 2017) of 1 Oct 2017
I2022 = 1826
T_TEST = 2 * int((np.datetime64("2022-05-01") - FIRST).astype(int))     # series index of 00 UTC 1 May 2022
TEST_DOYS = list(range(120, 334))   # initialisations 1 May - 30 Nov 2022


def read(npy, offset, count, shape):
    return npy._read(int(offset), int(count)).reshape(shape)


def wrap_pad(a):
    """(..., NY, NX) -> (..., PY, PX): 12 columns wrapped east of 180 E, 7 zero rows south of 60 S"""
    a = np.concatenate([a, a[..., :PX - NX]], -1)
    return np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, PY - NY), (0, 0)])


def clean_origins(o):
    """particles with a non-finite coordinate (FLEXGREEN: a few reach a pole, where the longitude is undefined) get zero
    survival, hence no weight in the gather, instead of propagating NaN (NaN x 0 = NaN)"""
    o = np.ascontiguousarray(o)
    bad = ~np.isfinite(o).all(-1)
    if bad.any():
        o[bad] = 0
    return o


class CamsData(Dataset):
    """split 'train': initial days from 2 Oct 2017 whose 10 targets all precede the test period, with their transport;
    'test': initialisations 1 May - 30 Nov 2022; 'any': every day with a transport (smoke tests). One sample = (day, lead)."""

    def __init__(self, split, psub=16, emis=False, train=True, lag=0):
        meta = np.load(G / "meta.npz")
        self.year, self.doy = meta["year"], meta["doy"]
        cm = np.load(CMETA)
        self.eps, self.cmu, self.csd = cm["eps"][:, None, None], cm["mean"][:, None, None], cm["std"][:, None, None]
        cdone = cm["days_done"]
        hs = np.load(HSTATS)
        self.hmu, self.hsd = hs["mean"][:, None, None], hs["std"][:, None, None]
        self.static = np.load(STATIC)["static"].astype(np.float32)
        self.psub, self.emis, self.train = psub, emis, train
        self.cams, self.fire = Npy(CAMS), Npy(G / "fires.npy")
        self.lag = lag                                   # weather test: HRES run of day d - lag (and its transport)
        gr = GREEN12 if lag == 0 else G / f"green12h_lag{lag}" / "hres"
        ys = sorted({int(p.name[8:12]) for p in gr.glob("origins_*.npy")})
        self.org = {y: Npy(gr / f"origins_{y}.npy") for y in ys}
        self.fs = {y: Npy(gr / f"fire_{y}.npy") for y in ys}
        self.es = {y: Npy(gr / f"emis_{y}.npy") for y in ys} if emis else {}
        gdone = {y: {int(p.name) for p in (gr / f"done_{y}").glob("*")} for y in ys}
        self.hb = {}
        if split == "test":
            days = [I2022 + d for d in TEST_DOYS]
        else:
            days = range(GI_FIRST + 1, len(self.year))
        self.samples = []
        for gi in days:
            y, d, sd = int(self.year[gi]), int(self.doy[gi]), gi - GI_FIRST
            if d not in gdone.get(y, ()) or not (cdone[sd - 1] and cdone[sd] and cdone[min(sd + 5, len(cdone) - 1)]):
                continue
            if split == "train" and 2 * sd + NL >= T_TEST:
                continue
            self.samples += [(gi, k) for k in range(1, NL + 1)]

    def __len__(self):
        return len(self.samples)

    def cams_norm(self, t):
        x = read(self.cams, t * NCH * NY * NX, NCH * NY * NX, (NCH, NY, NX)).astype(np.float32)
        return (np.log(np.maximum(x, 0) + self.eps) - self.cmu) / self.csd

    def pm25(self, t):
        return read(self.cams, (t * NCH + 1) * NY * NX, NY * NX, (NY, NX)).astype(np.float32)

    def hres(self, gi, k):
        L = getattr(self, "lag", 0)                      # run of day d - L: t0 is its step 4 L, the valid time 2 k + 4 L
        gi = gi - L
        y, d = int(self.year[gi]), int(self.doy[gi])
        m = int(str(np.datetime64(f"{y}-01-01") + np.timedelta64(d, "D"))[5:7])
        if (y, m) not in self.hb:
            self.hb[(y, m)] = (Npy(WB / f"hres_B_{y}_{m:02d}.npy"), list(np.load(WB / f"hres_B_{y}_{m:02d}_doy.npy")))
        npy, doys = self.hb[(y, m)]
        di, per = doys.index(d), NW * 181 * 360
        st = [read(npy, (di * 33 + s) * per, per, (NW, 181, 360)).astype(np.float32) for s in (4 * L, 2 * k + 4 * L)]
        st[0][NW - 1] = np.nan                           # initial state: no accumulated precipitation, as at a run's step 0
        tp = sum(read(npy, ((di * 33 + s) * NW + NW - 1) * 181 * 360, 181 * 360, (181, 360)).astype(np.float32)
                 for s in (2 * k - 2 + 4 * L, 2 * k - 1 + 4 * L))                      # mm in the 12 h before the valid time
        for a in st:
            a[NW - 1] = np.log1p(np.maximum(a[NW - 1], 0))
        x = np.concatenate([st[0], st[1], np.log1p(np.maximum(tp, 0))[None]])
        return np.nan_to_num((x - self.hmu) / self.hsd)

    def item(self, gi, k, member=0):
        y, d, sd = int(self.year[gi]), int(self.doy[gi]), gi - GI_FIRST
        t0 = 2 * sd
        c0, cm = self.cams_norm(t0), self.cams_norm(t0 - 1)
        pm0 = (self.pm25(t0) - MU) / SD
        tgt = (self.pm25(t0 + k) - MU) / SD
        fine = wrap_pad(np.concatenate([c0, cm, self.static]))
        mask = wrap_pad(np.ones((1, NY, NX), np.float32)); mask[..., NX:] = 0          # wrapped columns: not scored twice
        o = self.org[y]
        P = int(o.shape[4]); per = NR * NC * P * 4
        org = read(o, (d * NL + k - 1) * per, per, (NR, NC, P, 4))
        rng = np.random.default_rng(None if self.train else (int(gi) * 31 + int(k)) * 101 + int(member))
        sub = np.sort(rng.choice(P, self.psub, replace=False))
        fs = read(self.fs[y], (d * NL + k - 1) * 3 * NR * NC, 3 * NR * NC, (3, NR, NC))
        fire = np.nan_to_num(read(self.fire, (gi - 1) * 3 * 576 * 1440, 3 * 576 * 1440, (3, 576, 1440)).astype(np.float32))
        it = {"fine": torch.from_numpy(fine.astype(np.float16)), "x1": torch.from_numpy(wrap_pad(tgt[None])),
              "mask": torch.from_numpy(mask), "lead": torch.tensor(0.5 * k, dtype=torch.float32),
              "pm0": torch.from_numpy(pm0[None]), "wx": torch.from_numpy(self.hres(gi, k).astype(np.float32)),
              "fire": torch.from_numpy(fire.astype(np.float16)),
              "org": torch.from_numpy(clean_origins(org[:, :, sub])),                       # float16
              "fsens": torch.from_numpy(np.log1p(np.maximum(np.nan_to_num(fs), 0) * FS_SCALE[:, None, None]).astype(np.float32))}
        if self.emis:                                                                     # ug m-3 added since t0
            es = read(self.es[y], (d * NL + k - 1) * NE * NR * NC, NE * NR * NC, (NE, NR, NC))
            it["esens"] = torch.from_numpy(np.log1p(np.maximum(np.nan_to_num(es), 0)).astype(np.float32))
        return it

    def __getitem__(self, idx):
        gi, k = self.samples[idx]
        return self.item(gi, k)


def gather(f, org, lat0, dlat):
    """Foehn's parameter-free transport on a global lat-lon grid: f (B, C, n, m) at lat0 - dlat i, -180 + (360 / m) j;
    org (B, R, Q, P, 4) particle origins (lon, lat, z/h, survival) -> (B, C + 2, R, Q): survival-weighted mean of f over the
    origins in the mixed layer, fraction of the surviving air that came from above it, mean survival."""
    B, C, n, m = f.shape
    _, R, Q, P, _ = org.shape
    fw = torch.cat([f, f[..., :1]], -1)
    lon, lat, zr, sv = org.float().unbind(-1)
    gx = ((lon + 180.0) % 360.0) / 360.0 * 2.0 - 1.0
    gy = (lat0 - lat) / (dlat * (n - 1)) * 2.0 - 1.0
    smp = F.grid_sample(fw, torch.stack([gx, gy], -1).view(B, R * Q, P, 2).to(fw.dtype), mode="bilinear",
                        padding_mode="border", align_corners=True)
    sv = torch.nan_to_num(sv).clamp(0, 1)
    inm = (zr <= 1.0).to(sv.dtype) * sv
    out = (smp * inm.view(B, 1, R * Q, P).to(smp.dtype)).sum(-1) / inm.view(B, 1, R * Q, P).sum(-1).clamp(min=1e-3).to(smp.dtype)
    aloft = ((zr > 1.0).to(sv.dtype) * sv).sum(-1) / sv.sum(-1).clamp(min=1e-3)
    return torch.cat([out, aloft.view(B, 1, R * Q).to(out.dtype), sv.mean(-1).view(B, 1, R * Q).to(out.dtype)], 1).view(B, C + 2, R, Q)


def sampling_grid(lat0, dlat, lon0, n):
    """grid_sample coordinates of the padded 0.4 deg points (PY, PX) in a global grid of n rows at lat0 - dlat i and
    columns at lon0 + (360 / m) j with one wrapped column appended (align_corners=True)"""
    lat = torch.tensor(84.0 - 0.4 * np.arange(PY), dtype=torch.float32)
    lon = torch.tensor(-180.0 + 0.4 * np.arange(PX), dtype=torch.float32)
    gx = ((lon - lon0) % 360.0) / 360.0 * 2.0 - 1.0
    gy = (lat0 - lat) / (dlat * (n - 1)) * 2.0 - 1.0
    return torch.stack([gx[None].expand(PY, -1), gy[:, None].expand(-1, PX)], -1)[None]


def wrap_cols(a):
    return torch.cat([a, a[..., :1]], -1)


def rewrap(x):
    """the 12 padded columns east of 180 E repeat the first 12 (noise and flow state stay periodic)"""
    return torch.cat([x[..., :NX], x[..., :PX - NX]], -1)


class CamsNet(nn.Module):
    """Foehn's network on the 0.4 deg globe: stem + injection at 0.4 deg, residual FiLM blocks 0.4 -> 3.2 deg, global
    self-attention over 46 x 114 tokens, symmetric decoder with skips."""

    def __init__(self, emis=False, ch=(96, 128, 192, 256), edim=256, attn_depth=4, heads=8):
        super().__init__()
        self.emis = emis
        cin = 2 * NCH + NST + 1
        ccond = NWX + 3 + NGF + (NE if emis else 0)
        self.e = nn.Sequential(nn.Linear(128, edim), nn.SiLU(), nn.Linear(edim, edim))
        self.enc = nn.Sequential(nn.Conv2d(NCH + 3, NG, 3, padding=1), nn.SiLU(), nn.Conv2d(NG, NG, 3, padding=1))
        self.stem = nn.Conv2d(cin, ch[0], 3, padding=1)
        self.cproj = nn.Conv2d(ccond, ch[0], 1)
        self.down = nn.ModuleList([Block(ch[k], ch[k + 1], edim) for k in range(len(ch) - 1)])
        f = 2 ** (len(ch) - 1)
        self.pos = nn.Parameter(0.02 * torch.randn(1, ch[-1], PY // f, PX // f))
        self.attn = nn.ModuleList([AttnBlock(ch[-1], heads, edim) for _ in range(attn_depth)])
        self.up = nn.ModuleList([Block(2 * ch[k + 1], ch[k], edim) for k in reversed(range(len(ch) - 1))])
        self.out = nn.Conv2d(ch[0], 1, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.register_buffer("g_wx", sampling_grid(90.0, 1.0, -180.0, 181), persistent=False)
        self.register_buffer("g_fire", sampling_grid(84.0, 0.25, -180.0, 576), persistent=False)
        self.register_buffer("g_rec", sampling_grid(84.0 - 0.125, 0.5, -180.0 + 0.125, NR), persistent=False)

    def smp(self, a, g):
        return F.grid_sample(wrap_cols(a), g.expand(a.shape[0], -1, -1, -1).to(a.dtype), mode="bilinear",
                             padding_mode="border", align_corners=True)

    def forward(self, x, b, tau, lead):
        e = self.e(torch.cat([emb(tau), emb(lead / 4.0)], 1))
        fire = self.smp(b["fire"].float(), self.g_fire)                                 # (B, 3, PY, PX)
        f = self.enc(torch.cat([x[:, :NCH, :NY, :NX], fire[..., :NY, :NX]], 1))          # latent tracers at t0 (0.4 deg)
        g = gather(f, b["org"], 84.0, 0.4)                                               # (B, 34, NR, NC)
        pm_t = gather(b["pm0"].float(), b["org"], 84.0, 0.4)[:, :1]                      # transported PM2.5 (physics only)
        g = torch.cat([g, pm_t, b["fsens"].to(g.dtype)] + ([b["esens"].to(g.dtype)] if self.emis else []), 1)
        cond = torch.cat([self.smp(b["wx"].float(), self.g_wx), fire, self.smp(g, self.g_rec)], 1)
        h = self.stem(x) + self.cproj(cond.to(x.dtype))
        skips = []
        for blk in self.down:
            h = checkpoint(blk, h, e, use_reentrant=False)
            skips.append(h)
            h = F.avg_pool2d(h, 2)
        B, C, Hh, Ww = h.shape
        t = (h + self.pos).flatten(2).transpose(1, 2)
        for blk in self.attn:
            t = blk(t, e)
        h = t.transpose(1, 2).reshape(B, C, Hh, Ww)
        for blk, s in zip(self.up, reversed(skips)):
            h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = checkpoint(blk, torch.cat([h, s], 1), e, use_reentrant=False)
        return self.out(h)


class CamsModule(FullModule):
    """FullModule's flow matching (loss, shifted schedule, optimiser) around CamsNet."""

    def __init__(self, lr=2e-4, epochs=40, shift=2.6, shift_mix=0.5, psub=16, emis=False, reg=False):
        super(FullModule, self).__init__()                 # LightningModule.__init__ (no FullNet); super() lets Lightning find the args
        self.save_hyperparameters()
        self.net = CamsNet(emis=emis)

    def velocity(self, x, b, tau):
        return self.net(torch.cat([b["fine"].float(), x], 1), b, tau, b["lead"])

    def step(self, b):                                       # FullModule.step with periodic noise
        x1, m = b["x1"].float(), b["mask"].float()
        x0 = rewrap(torch.randn_like(x1))
        tau = self.draw_tau(x1.shape[0], x1.device)
        xt = (1 - tau[:, None, None, None]) * x0 + tau[:, None, None, None] * x1
        return (((self.velocity(xt, b, tau) - (x1 - x0)) ** 2) * m).sum() / m.sum().clamp(min=1)


def flow_grid(model, steps, refine_tail=1):
    """Foehn's sampler grid: quantiles of the training distribution of tau, last two intervals cut in refine_tail parts"""
    g = np.asarray(model.tau_grid(steps), np.float64)
    if refine_tail > 1:
        head, tail = g[:-2], []
        for a_, b_ in ((g[-3], g[-2]), (g[-2], g[-1])):
            tail += list(np.linspace(a_, b_, refine_tail + 1)[1:])
        g = np.concatenate([head, tail])
    return g


def train(a):
    pl.seed_everything(a.seed, workers=True)
    tr = CamsData("any" if a.smoke else "train", psub=a.psub, emis=a.emis, train=True)
    if a.limit < 1:
        tr.samples = tr.samples[:max(8, int(len(tr.samples) * a.limit))]
    dl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, persistent_workers=True, pin_memory=True,
                    drop_last=True)
    out = ROOT / "runs" / f"cams04_flow_shift{a.shift:g}{'_emis' if a.emis else ''}{'' if a.seed == 42 else f'_s{a.seed}'}{a.tag}"
    model = CamsModule(lr=a.lr, epochs=a.epochs, shift=a.shift, shift_mix=a.shift_mix, psub=a.psub, emis=a.emis)
    trainer = pl.Trainer(accelerator="gpu", devices=int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)),
                         num_nodes=int(os.environ.get("SLURM_NNODES", 1)), strategy="ddp", precision="bf16-mixed",
                         max_epochs=a.epochs, gradient_clip_val=1.0, log_every_n_steps=10, default_root_dir=str(out),
                         logger=CSVLogger(str(out), name="logs"), limit_val_batches=0,
                         callbacks=[Timer(), ModelCheckpoint(dirpath=str(out / "ckpts"), filename="{epoch:02d}", save_top_k=-1,
                                                             every_n_epochs=1)])
    if trainer.global_rank == 0:
        print(f"{out.name}: train {len(tr)} samples ({len(tr) // NL} initial days) params "
              f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    trainer.fit(model, dl, ckpt_path=a.resume)
    trainer.save_checkpoint(out / "final.ckpt")


def collate(items):
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


@torch.no_grad()
def predict(a):
    dev = torch.device("cuda", int(os.environ.get("SLURM_LOCALID", 0)))
    torch.cuda.set_device(dev)
    rank, world = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    run = ROOT / "runs" / a.run
    ckpt = Path(a.ckpt) if a.ckpt else run / "final.ckpt"
    model = CamsModule.load_from_checkpoint(str(ckpt), map_location=dev).eval()
    ds = CamsData("any" if a.smoke else "test", psub=model.hparams.psub, emis=model.hparams.emis, train=False, lag=a.lag)
    doys = a.doys if a.doys else TEST_DOYS
    out = run / a.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / "pred.npy"                                                     # (days, members, leads, NY, NX) ug m-3
    shape = (len(doys), a.members, NL, NY, NX)
    if rank == 0 and not path.exists():
        with open(path, "wb") as f:
            np.lib.format.write_array_header_1_0(f, {"descr": "<f2", "fortran_order": False, "shape": shape})
            f.truncate(f.tell() + int(np.prod(shape)) * 2)
        np.save(out / "doys.npy", np.array(doys))
    while not (out / "doys.npy").exists():
        time.sleep(2)
    time.sleep(2)
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    have = {(int(ds.year[gi]), int(ds.doy[gi]), k) for gi, k in ds.samples}
    grid = flow_grid(model, a.steps, a.refine_tail)
    for n in range(rank, len(doys), world):
        doy, gi = doys[n], I2022 + doys[n]
        if (out / f"done_{doy:03d}").exists() or (2022, doy, 1) not in have:
            continue
        t0 = time.time()
        pred = np.zeros((a.members, NL, NY, NX), np.float16)
        for k in range(1, NL + 1):
            b = collate([ds.item(gi, k, member=m) for m in range(a.members)])
            b = {kk: v.to(dev, non_blocking=True) for kk, v in b.items()}
            gen = torch.Generator(device=dev).manual_seed(gi * 1000 + k)
            x = rewrap(torch.randn(a.members, 1, PY, PX, device=dev, generator=gen))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for s in range(len(grid) - 1):
                    tau = torch.full((a.members,), float(grid[s]), device=dev)
                    x = rewrap(x + model.velocity(x, b, tau).float() * float(grid[s + 1] - grid[s]))
            pred[:, k - 1] = (x[:, 0, :NY, :NX] * SD + MU).cpu().numpy().astype(np.float16)
        blk = memoryview(np.ascontiguousarray(pred).reshape(-1).view(np.uint8))
        fd = os.open(path, os.O_WRONLY)
        try:
            done = 0
            while done < len(blk):
                done += os.pwrite(fd, blk[done:done + (1 << 30)], hdr + n * len(blk) + done)
            os.fsync(fd)
        finally:
            os.close(fd)
        (out / f"done_{doy:03d}").touch()
        print(f"rank {rank}: 2022 day {doy} {time.time() - t0:.0f}s mean {float(pred[:, 0].astype(np.float32).mean()):.2f}", flush=True)


def prep():
    """Aurora's static fields on the experiment grid, and the statistics of the HRES channels (2018-2021, every 15th
    day, steps 0 and 12..120 h)"""
    pk = glob.glob(str(ROOT.parent / "benchmark/aurora/hf/hub/models--microsoft--aurora/snapshots/*/aurora-0.4-air-pollution-static.pickle"))[0]
    with open(pk, "rb") as f:
        s = {k: np.asarray(v, np.float32) for k, v in pickle.load(f).items()}
    cr = lambda a: np.roll(a[15:376], 450, axis=-1)                                # 84 N .. 60 S, lon -180 .. 179.6
    static = np.stack([cr(s["lsm"]), cr(s["z"]) / 9.80665 / 1000.0] +
                      [cr(s[f"static_{n}_log"]) / np.log(10.0) for n in ("ammonia", "co", "nox", "so2")])
    np.savez(STATIC, static=static.astype(np.float32), names=np.array(["lsm", "orography_km", "nh3_log", "co_log", "nox_log", "so2_log"]))
    print("static", static.shape, [f"{float(a.min()):.2f}..{float(a.max()):.2f}" for a in static], flush=True)
    s1 = np.zeros(NWX); s2 = np.zeros(NWX); n = 0
    meta = np.load(G / "meta.npz")
    ds = CamsData.__new__(CamsData); ds.year, ds.doy, ds.hb = meta["year"], meta["doy"], {}
    ds.hmu, ds.hsd = np.zeros((NWX, 1, 1), np.float32), np.ones((NWX, 1, 1), np.float32)
    for gi in range(365, 1826, 15):
        for k in (1, 4, 7, 10):
            x = ds.hres(gi, k).astype(np.float64)
            s1 += x.mean((1, 2)); s2 += (x ** 2).mean((1, 2)); n += 1
    mean = s1 / n; std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    np.savez(HSTATS, mean=mean.astype(np.float32), std=std.astype(np.float32), n=n)
    print(f"HRES statistics over {n} fields: mean {np.round(mean[:5], 2)} ... std {np.round(std[:5], 2)} ...", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "check", "train", "predict"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--shift", type=float, default=2.6)
    ap.add_argument("--shift-mix", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--psub", type=int, default=16)
    ap.add_argument("--emis", action="store_true", help="the 6 anthropogenic emission terms of FLEXGREEN (if final Foehn has them)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--smoke", action="store_true", help="use every day that has a transport (plumbing and timing only)")
    ap.add_argument("--split", default="any")
    ap.add_argument("--run", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="cams12h")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--refine-tail", type=int, default=3)
    ap.add_argument("--doys", type=int, nargs="*", default=None)
    ap.add_argument("--lag", type=int, default=0, help="predict with the HRES run of day d - lag and its transport (weather test)")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep()
    elif a.cmd == "check":
        ds = CamsData(a.split, psub=a.psub, emis=a.emis, train=a.split == "train")
        print(len(ds), "samples")
        b = ds[0]
        print({k: (tuple(v.shape), str(v.dtype), round(float(v.float().min()), 3), round(float(v.float().max()), 3)) for k, v in b.items()})
    elif a.cmd == "train":
        train(a)
    else:
        predict(a)
