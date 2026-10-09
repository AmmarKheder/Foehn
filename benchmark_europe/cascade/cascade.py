#!/usr/bin/env python3
"""Seamless 1 km generative forecasting of PM2.5 over Europe (GHAP), as a two-stage cascade.

Stage 1 (../dejavu, run noise_t31_fpstab): 0.25 deg conditional flow matching with the footprint layer.
Stage 2 (this file): a 1 km conditional flow-matching U-Net that is *translation-equivariant by construction*: no
positional embedding, no tile-frame statistics (per-pixel channel normalisation only), pooling grids anchored to the
image origin. It is trained on random 512 x 512 crops and applied to the whole 4192 x 6992 image by tiles with a halo,
so the map does not depend on the tiling (no seams; the tile is a compute chunk, not a model input).
Conditioning: PM today and yesterday at 1 km, elevation, land; the stage-1 field (training: the true 0.25 deg block
means of the target day, with noise augmentation); PM today, weather of today and of the forecast day (HRES
variables) at 0.25 deg, injected at the 16 km level; and the 1 km footprint: PM today (5 km, whole Europe) gathered
along the back-trajectories of the forecast-period 3-hourly winds, a source-receptor operator over the entire image at
the cost of 32 bilinear gathers, independent of the tiling.
    python cascade.py train [--epochs 40] [--no-footprint] [--reg]       (srun, one node, 8 GCDs)
    evaluation: ../retrain/evaluate.py --cascade runs/<run> --coarse-ckpt ../dejavu/runs/noise_t31_fpstab/final.ckpt
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
sys.path.insert(0, str(EXP / "dejavu"))
sys.path.insert(0, str(EXP / "fullimage"))
from dejavu import DVData, DVModule, attach_hres, emb, norm, MU, SD  # noqa: E402
from build_ghap1km import D as NDAYS, H as FH, W as FW, F as FINE_FILE  # noqa: E402

FINE = EXP / "fullimage" / "data"
CROP, CWIN, HALO = 512, 24, 64   # 1 km crop; 0.25 deg window (24 x 24 covers 512 px + a cell on each side); eval halo
S5, H5, W5 = 5, 838, 1398        # footprint source: PM today at 5 km
FP_STRIDE = 4                    # footprint evaluated every 4 km inside the crop, bilinearly upsampled
DEG_M = 111.0e3                  # metres per degree of latitude
LAT0, LON0 = 72.0, -25.0


def header(path):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); return f.tell()


class DayFile:
    """One day of a (D, H, W) float16 .npy at a time (a 128 GB mapping exceeds the login-node limits)."""

    def __init__(self, path, h, w):
        self.path, self.h, self.w, self.hdr = path, h, w, header(path)

    def __getitem__(self, d):
        return np.memmap(self.path, dtype=np.float16, mode="r", offset=self.hdr + d * self.h * self.w * 2, shape=(self.h, self.w))


# ----------------------------------------------------------------------------------------------------------- geometry
# 1 km pixel (r, c) centre: lat = 72 - 0.01 (r + 0.5), lon = -25 + 0.01 (c + 0.5); 0.25 deg point j: lat = 72 - 0.25 j.
def block_latlon(r0, c0, n, stride):
    """(B, n, n) lat / lon of the centres of the stride x stride blocks of an n*stride crop starting at (r0, c0)."""
    k = torch.arange(n, device=r0.device, dtype=torch.float32) * stride + stride / 2.0
    lat = LAT0 - 0.01 * (r0[:, None].float() + k[None])
    lon = LON0 + 0.01 * (c0[:, None].float() + k[None])
    return lat[:, :, None].expand(-1, -1, n), lon[:, None, :].expand(-1, n, -1)


def to_grid(lat, lon, res, off, n_lat, n_lon):
    """grid_sample coordinates (align_corners=True) of (lat, lon) in a grid whose point j is at LAT0 - res (j + off)."""
    y = ((LAT0 - lat) / res - off) * (2.0 / (n_lat - 1)) - 1
    x = ((lon - LON0) / res - off) * (2.0 / (n_lon - 1)) - 1
    return torch.stack([x, y], -1)


def window_grid(r0, c0, j0, i0, n, stride):
    """(B, n, n, 2) coordinates of the stride-block centres of a crop in its 24 x 24 window of 0.25 deg points."""
    k = torch.arange(n, device=r0.device, dtype=torch.float32) * stride + stride / 2.0
    y = ((r0[:, None].float() + k[None]) / 25.0 - j0[:, None].float()) * (2.0 / (CWIN - 1)) - 1
    x = ((c0[:, None].float() + k[None]) / 25.0 - i0[:, None].float()) * (2.0 / (CWIN - 1)) - 1
    return torch.stack([x[:, None, :].expand(-1, n, -1), y[:, :, None].expand(-1, -1, n)], -1)


def window_origin(r0, c0):
    j0 = int(np.clip(r0 // 25 - 1, 0, 168 - CWIN))
    i0 = int(np.clip(c0 // 25 - 1, 0, 280 - CWIN))
    return j0, i0


def sample_border(x, grid):
    return F.grid_sample(x, grid, mode="bilinear", align_corners=True, padding_mode="border")


# --------------------------------------------------------------------------------------------------------------- data
class CascadeData(DVData):
    """Stage-2 samples: a random 512 x 512 crop at 1 km of a (day i, lead h) pair, with the 0.25 deg conditioning of
    ../dejavu (cond: PM today, weather today, forecast-day block; x1: the true 0.25 deg target; winds, rain,
    stability for the footprint) restricted to the crop's 24 x 24 window."""

    def __init__(self, years, train, crops=4):
        super().__init__(years, "noise", train, t31=True, footprint=True, fp_met="stab")
        self.base = list(self.samples)
        self.samples = self.base * crops
        self.fine = DayFile(FINE_FILE, FH, FW)
        self.src5 = DayFile(ROOT / "data" / "pm5km.npy", H5, W5)
        st = np.load(FINE / "static1km.npz")
        self.elev1, self.land1 = st["elev"], st["land"]          # (elev - 300) / 500, bool
        self.land_c = self.land.astype(np.float32)

    def crop_pos(self, idx):
        rng = np.random if self.train else np.random.default_rng(idx)
        for _ in range(100):
            r0, c0 = int(rng.integers(0, FH - CROP + 1)), int(rng.integers(0, FW - CROP + 1))
            j0, i0 = window_origin(r0, c0)
            if self.land_c[j0 + 1:j0 + CWIN - 1, i0 + 1:i0 + CWIN - 1].mean() >= 0.3:
                return r0, c0, j0, i0
        return r0, c0, j0, i0

    def __getitem__(self, idx):
        it = super().__getitem__(idx)
        i, h = self.samples[idx]
        r0, c0, j0, i0 = self.crop_pos(idx)
        crop = lambda d: np.asarray(self.fine[d][r0:r0 + CROP, c0:c0 + CROP], np.float32)
        today, yday, tgt = crop(i), crop(i - 1), crop(i + h)
        fine = np.stack([norm(today), np.isfinite(today), norm(yday), np.isfinite(yday),
                         self.elev1[r0:r0 + CROP, c0:c0 + CROP], self.land1[r0:r0 + CROP, c0:c0 + CROP]]).astype(np.float32)
        src = np.asarray(self.src5[i], np.float32)
        out = {"fine": torch.from_numpy(fine), "x1": torch.from_numpy(norm(tgt))[None],
               "mask": torch.from_numpy(np.isfinite(tgt).astype(np.float32))[None],
               "coarse": torch.cat([it["x1"], it["cond"][:69]])[:, j0:j0 + CWIN, i0:i0 + CWIN].contiguous(),
               "src": torch.from_numpy(np.stack([norm(src), np.isfinite(src)]).astype(np.float16)),
               "wind": it["wind"], "rain": it["rain"], "stab": it["stab"], "lead": it["lead"],
               "r0": torch.tensor(r0), "c0": torch.tensor(c0), "j0": torch.tensor(j0), "i0": torch.tensor(i0)}
        return out


# -------------------------------------------------------------------------------------------------------------- model
class PixelNorm(nn.Module):
    """LayerNorm over channels at each pixel: no statistics over the crop, so the layer is translation-equivariant."""

    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, cin, cout, edim):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(cin, cout, 3, padding=1), nn.Conv2d(cout, cout, 3, padding=1)
        self.n1, self.n2 = PixelNorm(cout), PixelNorm(cout)
        self.film = nn.Linear(edim, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = F.silu(self.n1(self.c1(x)))
        s, b = self.film(e)[:, :, None, None].chunk(2, 1)
        h = F.silu(self.n2(self.c2(h)) * (1 + s) + b)
        return h + self.skip(x)


class Footprint1km(nn.Module):
    """Every 4 km block of the crop gathers PM today (5 km, whole Europe) along the back-trajectory of the air that
    reaches it at 12 UTC of the forecast day, integrated with the 3-hourly forecast-period winds (learned mix of 10 m
    and 925 hPa) back to 12 UTC of the start day. Path points are weighted by a learned score of (age, stability,
    rain, rain accumulated since) minus a learned washout rate; outputs: the weighted mean (PM, validity), the
    end point (pure advection), three age bins and the accumulated rain. Same operator as the 0.25 deg layer, with
    the whole image as the source: no tile can cut the trajectory."""
    DT = 3 * 3600.0

    def __init__(self, kmax=32):
        super().__init__()
        self.kmax = kmax
        self.mix = nn.Parameter(torch.zeros(2))
        self.score = nn.Sequential(nn.Conv2d(5, 32, 1), nn.SiLU(), nn.Conv2d(32, 1, 1))
        self.lam = nn.Parameter(torch.tensor(-1.0))

    def forward(self, src, wind, rain, stab, lead, lat, lon):
        B, mh, mw = lat.shape[0], lat.shape[1], lat.shape[2]
        dev = lat.device
        src, wind, rain, stab = src.float(), wind.float(), rain.float(), stab.float()
        mw, lam = torch.softmax(self.mix, 0), F.softplus(self.lam)
        hh = lead.round().long()
        s0 = 8 * hh + 4
        bidx = torch.arange(B, device=dev)
        m_run = torch.full((B, 1, mh, mw), -1e4, device=dev)
        s_run = torch.zeros(B, 1, mh, mw, device=dev)
        acc = torch.zeros(B, 2, mh, mw, device=dev)
        cum = torch.zeros(B, 1, mh, mw, device=dev)
        bins, nb, end = torch.zeros(B, 3, mh, mw, device=dev), torch.zeros(B, 3, mh, mw, device=dev), torch.zeros(B, 1, mh, mw, device=dev)
        for k in range(self.kmax + 1):
            valid = (k <= 8 * hh).float()[:, None, None, None]
            st = (s0 - k).clamp(min=0)
            day = torch.div(st, 8, rounding_mode="floor")
            gw = to_grid(lat, lon, 0.5, 0.25, 84, 140)
            smp = sample_border(wind[bidx, st], gw)
            u, v = mw[0] * smp[:, 0] + mw[1] * smp[:, 2], mw[0] * smp[:, 1] + mw[1] * smp[:, 3]
            r = sample_border(rain[bidx, day][:, None], gw)
            sg = sample_border(stab[bidx, day], gw) / 5.0
            g = sample_border(src, to_grid(lat, lon, 0.05, 0.5, H5, W5))
            z = torch.cat([torch.full_like(r, k / self.kmax), sg, r, cum], 1)
            logit = self.score(z) - lam * cum
            logit = torch.where(valid > 0, logit, torch.full_like(logit, -1e4))
            m_new = torch.maximum(m_run, logit)
            a, e = torch.exp(m_run - m_new), torch.exp(logit - m_new) * valid
            acc, s_run, m_run = acc * a + e * g, s_run * a + e, m_new
            b = 0 if k <= 8 else 1 if k <= 16 else 2
            bins[:, b:b + 1] = bins[:, b:b + 1] + g[:, :1] * valid
            nb[:, b:b + 1] = nb[:, b:b + 1] + valid
            end = end + g[:, :1] * (k == 8 * hh).float()[:, None, None, None]
            lat = lat - v * self.DT / DEG_M * valid[:, 0]                                  # back in time: against the wind
            lon = lon - u * self.DT / (DEG_M * torch.cos(torch.deg2rad(lat))) * valid[:, 0]
            cum = cum + r * valid / 8.0
        return torch.cat([acc / s_run.clamp(min=1e-6), end, bins / nb.clamp(min=1), cum], 1)


class FineUNet(nn.Module):
    """4-level fully convolutional U-Net at 1 km (512 -> 32), FiLM-conditioned on flow time, lead time and the
    conditioning-noise level; the 0.25 deg window is injected at the 16 km level."""

    def __init__(self, cin, ccoarse, ch=(48, 96, 192, 384), edim=256):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(192, edim), nn.SiLU(), nn.Linear(edim, edim))
        self.inp = nn.Conv2d(cin, ch[0], 3, padding=1)
        self.down = nn.ModuleList([Block(ch[0], ch[0], edim), Block(ch[0], ch[1], edim), Block(ch[1], ch[2], edim), Block(ch[2], ch[3], edim)])
        self.cproj = nn.Conv2d(ccoarse, ch[3], 1)
        self.mid = Block(ch[3], ch[3], edim)
        self.up = nn.ModuleList([Block(ch[3] * 2, ch[2], edim), Block(ch[2] * 2, ch[1], edim), Block(ch[1] * 2, ch[0], edim), Block(ch[0] * 2, ch[0], edim)])
        self.out = nn.Conv2d(ch[0], 1, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, coarse, grid16, tau, lead, sigc):
        e = self.e(torch.cat([emb(tau), emb(lead / 4.0), emb(sigc * 10.0)], 1))
        h = self.inp(x)
        skips = []
        for blk in self.down:
            h = blk(h, e); skips.append(h)
            h = F.avg_pool2d(h, 2)
        h = h + self.cproj(sample_border(coarse, grid16))
        h = self.mid(h, e)
        for blk, s in zip(self.up, reversed(skips)):
            h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = blk(torch.cat([h, s], 1), e)
        return self.out(h)


class CascadeModule(pl.LightningModule):
    def __init__(self, lr=2e-4, epochs=40, footprint=True, reg=False, cond_noise=0.3):
        super().__init__()
        self.save_hyperparameters()
        self.fp = Footprint1km() if footprint else None
        self.net = FineUNet(7 + (7 if footprint else 0) + 1, 70)  # fine 6 + stage-1 field 1 (+ footprint 7) + x_tau

    def features(self, b, coarse):
        """Conditioning at 1 km (independent of the flow time): fine inputs, the stage-1 field upsampled, footprint."""
        r0, c0, j0, i0 = b["r0"], b["c0"], b["j0"], b["i0"]
        n = b["fine"].shape[-1]
        feats = [b["fine"], sample_border(coarse[:, :1], window_grid(r0, c0, j0, i0, n, 1))]
        if self.fp is not None:
            lat, lon = block_latlon(r0, c0, n // FP_STRIDE, FP_STRIDE)
            with torch.autocast(device_type=lat.device.type, enabled=False):
                f = self.fp(b["src"], b["wind"], b["rain"], b["stab"], b["lead"], lat, lon)
            feats.append(F.interpolate(f, size=(n, n), mode="bilinear", align_corners=False))
        return torch.cat(feats, 1), window_grid(r0, c0, j0, i0, n // 16, 16)

    def velocity(self, x, feats, coarse, grid16, tau, lead, sigc):
        return self.net(torch.cat([feats, x], 1), coarse, grid16, tau, lead, sigc)

    def step(self, b):
        x1, m, lead = b["x1"], b["mask"], b["lead"]
        B = x1.shape[0]
        coarse = b["coarse"].clone()
        sigc = torch.rand(B, device=x1.device) * self.hparams.cond_noise   # conditioning augmentation (cascade robustness)
        coarse[:, :1] = coarse[:, :1] + sigc[:, None, None, None] * torch.randn_like(coarse[:, :1])
        feats, g16 = self.features(b, coarse)
        if self.hparams.reg:
            pred = self.velocity(torch.zeros_like(x1), feats, coarse, g16, torch.ones_like(lead), lead, sigc)
            return ((pred - x1) ** 2 * m).sum() / m.sum().clamp(min=1)
        x0 = torch.randn_like(x1)
        tau = torch.rand(B, device=x1.device)
        xt = (1 - tau[:, None, None, None]) * x0 + tau[:, None, None, None] * x1
        v = self.velocity(xt, feats, coarse, g16, tau, lead, sigc)
        return (((v - (x1 - x0)) ** 2) * m).sum() / m.sum().clamp(min=1)

    def training_step(self, b, _):
        loss = self.step(b); self.log("train/loss", loss, prog_bar=True); return loss

    def validation_step(self, b, _):
        self.log("val/loss", self.step(b), sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.hparams.lr, total_steps=self.trainer.estimated_stepping_batches, pct_start=0.05)
        return [opt], [{"scheduler": sched, "interval": "step"}]

    @torch.no_grad()
    def sample(self, b, coarse, steps, sigc=0.0, generator=None):
        feats, g16 = self.features(b, coarse)
        B, n = b["fine"].shape[0], b["fine"].shape[-1]
        lead = b["lead"]
        sig = torch.full((B,), sigc, device=lead.device)
        if self.hparams.reg:
            return self.velocity(torch.zeros(B, 1, n, n, device=lead.device), feats, coarse, g16, torch.ones_like(lead), lead, sig)
        x = torch.randn(B, 1, n, n, device=lead.device, generator=generator)
        for s in range(steps):
            tau = torch.full((B,), s / steps, device=lead.device)
            x = x + self.velocity(x, feats, coarse, g16, tau, lead, sig) / steps
        return x


# -------------------------------------------------------------------------------------------------------- training
def warm_cache(days, rank, world):
    """Read the training days once, sequentially, so that random crops are then served from the node's page cache."""
    f = DayFile(FINE_FILE, FH, FW)
    mine = days[rank::world]
    for k, d in enumerate(mine):
        np.asarray(f[d]).sum()
        if k % 50 == 0:
            print(f"rank {rank}: warm {k}/{len(mine)}", flush=True)


def train(a):
    pl.seed_everything(a.seed, workers=True)
    tr = CascadeData([2017, 2018, 2019, 2020], train=True, crops=a.crops)
    va = CascadeData([2021], train=False, crops=1)
    va.samples = va.samples[::6]
    if a.limit < 1:
        tr.samples = tr.samples[:int(len(tr.samples) * a.limit)]; va.samples = va.samples[:32]
    rank, world = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    if a.warm:
        days = sorted({d for i, h in tr.samples for d in (i - 1, i, i + h)} | {d for i, h in va.samples for d in (i - 1, i, i + h)})
        warm_cache(days, rank, world)
    nw = int(os.environ.get("SLURM_CPUS_PER_TASK", 7)) - 1
    dl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=nw, persistent_workers=True, drop_last=True, pin_memory=True)
    dv = DataLoader(va, batch_size=a.batch, num_workers=nw, persistent_workers=True, pin_memory=True)
    out = ROOT / "runs" / f"{'reg' if a.reg else 'flow'}{'' if a.footprint else '_nofp'}{'' if a.seed == 42 else f'_s{a.seed}'}{a.tag}"
    model = CascadeModule(lr=a.lr, epochs=a.epochs, footprint=a.footprint, reg=a.reg, cond_noise=a.cond_noise)
    trainer = pl.Trainer(accelerator="gpu", devices=int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)), strategy="ddp",
                         precision="bf16-mixed", max_epochs=a.epochs, gradient_clip_val=1.0, log_every_n_steps=10,
                         default_root_dir=str(out), logger=CSVLogger(str(out), name="logs"), enable_checkpointing=True)
    if trainer.global_rank == 0:
        print(f"{out.name}: train {len(tr)} val {len(va)} params {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    trainer.fit(model, dl, dv)
    trainer.save_checkpoint(out / "final.ckpt")


# ------------------------------------------------------------------------------------------------------- inference
def tile_starts(n, size):
    s = list(range(0, n - size, size))
    return s + [n - size]


class CascadePredictor:
    """Whole-Europe forecast at 1 km for a day: stage 1 sampled at 0.25 deg (M members), stage 2 applied by tiles of
    512 x 512 with a halo of HALO (a multiple of 16: pooling grids stay anchored to the image origin), the halo
    discarded. Returns the ensemble mean in ug/m3 (and keeps the members for the CRPS)."""

    def __init__(self, ckpt, coarse_ckpt, dev, year=2022, weather="hres", members=4, steps1=8, steps2=8, batch=8, cond_noise=0.0):
        self.dev, self.M, self.S1, self.S2, self.batch, self.sigc = dev, members, steps1, steps2, batch, cond_noise
        self.stage1 = DVModule.load_from_checkpoint(coarse_ckpt, map_location="cpu").to(dev).eval()
        self.stage2 = CascadeModule.load_from_checkpoint(ckpt, map_location="cpu").to(dev).eval()
        self.dv = DVData([year], "noise", train=False, t31=True, footprint=True, fp_met="stab")
        if weather == "hres":
            attach_hres(self.dv, True, True)
        self.index = {(i, h): n for n, (i, h) in enumerate(self.dv.samples)}
        self.off = int(np.where(self.dv.year == year)[0][0])   # dejavu index of 1 January of the year
        self.fine = DayFile(FINE_FILE, FH, FW)
        self.src5 = DayFile(ROOT / "data" / "pm5km.npy", H5, W5)
        st = np.load(FINE / "static1km.npz")
        self.static = np.stack([st["elev"].astype(np.float32), st["land"].astype(np.float32)])
        self.gen = torch.Generator(device=dev); self.gen.manual_seed(0)
        self.tiles = [(r, c) for r in tile_starts(FH, CROP) for c in tile_starts(FW, CROP)]
        self.members = None

    def fine_full(self, d):
        a = np.asarray(self.fine[d], np.float32)
        return np.stack([norm(a), np.isfinite(a).astype(np.float32)])

    @torch.no_grad()
    def forecast(self, d, hs):
        """d: day of year; hs: lead times. -> {h: (M, FH, FW) members in ug/m3}"""
        i = self.off + d
        P = HALO
        today, yday = self.fine_full(i), self.fine_full(i - 1)
        full = np.pad(np.concatenate([today, yday, self.static]), ((0, 0), (P, P), (P, P)))     # (6, FH+2P, FW+2P)
        src = np.asarray(self.src5[i], np.float32)
        src = torch.from_numpy(np.stack([norm(src), np.isfinite(src)]).astype(np.float32)).to(self.dev)[None]
        out = {}
        for h in hs:
            if (i, h) not in self.index:
                continue
            it = self.dv[self.index[(i, h)]]
            M = self.M
            cond = it["cond"][None].to(self.dev).expand(M, -1, -1, -1)
            lead = torch.full((M,), float(h), device=self.dev)
            fp = tuple(it[k][None].to(self.dev).expand(M, *([-1] * it[k].dim())) for k in ("wind", "rain", "stab"))
            x0 = torch.randn(M, 1, 168, 280, device=self.dev, generator=self.gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c1 = self.stage1.sample(x0, cond, lead, self.S1, fp)[:, 0].float()             # (M, 168, 280) stage-1 members
            coarse_full = torch.cat([c1[:, None], cond[:, :69]], 1)                            # (M, 70, 168, 280)
            pred = torch.zeros(M, FH, FW, device=self.dev)
            jobs = [(m, r, c) for m in range(M) for (r, c) in self.tiles]
            for k in range(0, len(jobs), self.batch):
                chunk = jobs[k:k + self.batch]
                n = CROP + 2 * P
                b = {"fine": [], "coarse": [], "r0": [], "c0": [], "j0": [], "i0": []}
                for m, r, c in chunk:
                    j0, i0 = window_origin(r - P, c - P)
                    b["fine"].append(torch.from_numpy(full[:, r:r + n, c:c + n]))
                    b["coarse"].append(coarse_full[m, :, j0:j0 + CWIN, i0:i0 + CWIN])
                    b["r0"].append(r - P); b["c0"].append(c - P); b["j0"].append(j0); b["i0"].append(i0)
                B = len(chunk)
                batch = {"fine": torch.stack(b["fine"]).to(self.dev), "coarse": torch.stack(b["coarse"]),
                         "src": src.expand(B, -1, -1, -1), "lead": torch.full((B,), float(h), device=self.dev),
                         **{k: torch.tensor(b[k], device=self.dev) for k in ("r0", "c0", "j0", "i0")},
                         **{k: it[k][None].to(self.dev).expand(B, *([-1] * it[k].dim())) for k in ("wind", "rain", "stab")}}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    y = self.stage2.sample(batch, batch["coarse"], self.S2, self.sigc, self.gen)[:, 0].float()
                for (m, r, c), yy in zip(chunk, y):
                    pred[m, r:r + CROP, c:c + CROP] = yy[P:P + CROP, P:P + CROP]
            out[h] = (pred * SD + MU).cpu().numpy()
        return out


def cascade_predictor(args, dev):
    """Plug-in for ../retrain/evaluate.py: the paper's 1 km scoring (grid, EEA stations, 0.25 deg block means)."""
    sys.path.insert(0, str(EXP.parent / "src"))
    from cranpm.data.europe_dataset_v2 import EuropeDatasetV2
    ds = EuropeDatasetV2([args.year], mode="fixed", horizons=args.horizons, train=False)
    cp = CascadePredictor(sorted(Path(args.ckpt).glob("final.ckpt"))[0] if Path(args.ckpt).is_dir() else args.ckpt,
                          args.coarse_ckpt, dev, year=args.year, weather=args.weather, members=args.members,
                          steps1=args.steps1, steps2=args.steps2, batch=args.batch, cond_noise=args.cond_noise)
    extra = {k: np.zeros(max(args.horizons) + 1) for k in ("crps_n", "crps", "spread2")}

    def predict(d, met_day, hs, g_t, g_p):
        members = cp.forecast(d, hs)
        out = {}
        for h, E in members.items():
            tgt = np.asarray(cp.fine[cp.off + d + h], np.float32)
            m = np.isfinite(tgt)
            T, Em = tgt[m], E[:, m]
            M = Em.shape[0]
            pair = sum(np.abs(Em[a] - Em[b]) for a in range(M) for b in range(M)) / max(M * (M - 1), 1)
            crps = np.abs(Em - T).mean(0) - 0.5 * pair
            extra["crps_n"][h] += m.sum(); extra["crps"][h] += crps.sum(); extra["spread2"][h] += Em.var(0).sum()
            out[h] = E.mean(0).astype(np.float32)
        return out

    predict.extra = extra
    return ds, predict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crops", type=int, default=4, help="random crops per (day, lead) pair and epoch")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cond-noise", type=float, default=0.3)
    ap.add_argument("--no-footprint", dest="footprint", action="store_false")
    ap.add_argument("--reg", action="store_true", help="ablation: deterministic regression with the same network")
    ap.add_argument("--no-warm", dest="warm", action="store_false")
    ap.add_argument("--limit", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    train(a)
