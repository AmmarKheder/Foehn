#!/usr/bin/env python3
"""Deja-vu Flow: probabilistic forecasting by flow matching whose SOURCE is memory, not noise.

The source sample is the successor field of a retrieved past analog (a real atmospheric state with real peaks and
texture); a conditional velocity field transports it to the forecast. Ensembles = different analogs (+ small source
noise). Same network / inputs / training for every variant, only the source changes:
  dejavu : x0 = PM_{j+h} of an analog j (top-8, sampled by similarity) + sigma * z
  noise  : x0 = z ~ N(0, 1)                     (standard conditional flow matching)
  today  : x0 = PM_i + sigma * z                (bridge from persistence)
  reg    : deterministic regression of PM_{i+h} (MSE), no flow
Conditioning (all variants): PM today, weather of today and of the forecast day (37 + 37 channels), elevation, land,
lead time. 0.25 deg Europe (168 x 280). Train 2017-2020, validation 2021, test 2022.
    python dejavu.py train --source dejavu          (srun, one node)
    python dejavu.py eval runs/dejavu --year 2022   (one GPU)
"""
import argparse
import json
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MU, SD = 15.0, 20.0          # PM normalisation (paper convention)
SIGMA = 0.1                  # source noise (normalised units, = 2 ug/m3)
TOPK = 8
T31 = list(range(30)) + [33]   # forecast-day channels available in IFS HRES: 30 ERA5 store channels + precipitation
STATS = Path(__file__).resolve().parents[2] / "src/cranpm/data"
HRES = DATA / "hres"
T_MEAN, T_STD = 260.0, 30.0    # store convention of ERA5 temperature channels (aria constants)


def theta_gaps(t_store):
    """(..., 3, H, W) temperatures at 1000/925/850 hPa in store units -> potential-temperature gaps (925-1000, 850-1000),
    K: a stability measure that forecasts provide (unlike BLH). Large positive gap = inversion, trapped air."""
    t = t_store * T_STD + T_MEAN
    th = t * np.array([1.0, (1000 / 925) ** 0.286, (1000 / 850) ** 0.286], np.float32)[:, None, None]
    return np.stack([th[..., 1, :, :] - th[..., 0, :, :], th[..., 2, :, :] - th[..., 0, :, :]], -3)


def norm(x):
    return np.nan_to_num((x - MU) / SD, nan=0.0).astype(np.float32)


class DVData(Dataset):
    def __init__(self, years, source, train, t31=False, footprint=False, wind_shift=0, blh_persist=False, fp_met="blh"):
        self.t31, self.footprint, self.wind_shift, self.blh_persist, self.fp_met = t31, footprint, wind_shift, blh_persist, fp_met
        if footprint:
            self.wind = np.load(DATA / "wind3h.npy", mmap_mode="r")          # (D, 8, 5, 84, 140) u10 v10 u925 v925 BLH
            ex = np.load(STATS / "europe_v2_extra_stats.npz")
            self.rain_mu, self.rain_sd = float(ex["extra_mean"][3]), float(ex["extra_std"][3])
            st = np.load(STATS / "europe_v2_fixed_stats.npz")
            self.t_mu, self.t_sd = st["era5_mean"][5:8, None, None], st["era5_std"][5:8, None, None]
        self.pm = np.load(DATA / "pm.npy", mmap_mode="r")
        self.met = np.load(DATA / "met.npy", mmap_mode="r")
        meta, an = np.load(DATA / "meta.npz"), np.load(DATA / "analogs.npz")
        self.year, self.elev, self.land = meta["year"], meta["elev"], meta["land"].astype(np.float32)
        self.start, self.dist, valid = an["start"], an["dist"], an["valid"]
        self.source, self.train = source, train
        D = len(self.year)
        self.samples = [(i, h) for i in range(1, D - 4) for h in (1, 2, 3, 4)
                        if np.isin(self.year[i], years) and self.year[i + h] == self.year[i] and valid[i] and valid[i + h]
                        and self.start[i, h - 1, 0] >= 0]

    def __len__(self):
        return len(self.samples)

    def analog_sources(self, i, h, k=None):
        js = self.start[i, h - 1, :TOPK]
        return np.stack([norm(self.pm[j + h]) for j in (js if k is None else js[k:k + 1])])

    def __getitem__(self, idx):
        i, h = self.samples[idx]
        fut = np.asarray(self.met[i + h], np.float32)
        cond = np.concatenate([norm(self.pm[i])[None], np.asarray(self.met[i], np.float32),
                               fut[T31] if self.t31 else fut, self.elev[None], self.land[None]])
        tgt = self.pm[i + h]
        item = {"cond": torch.from_numpy(cond), "x1": torch.from_numpy(norm(tgt))[None],
                "mask": torch.from_numpy(np.isfinite(tgt).astype(np.float32))[None],
                "lead": torch.tensor(float(h)), "idx": torch.tensor(idx)}
        if self.source in ("dvmean", "noise_rc"):
            item["x0"] = torch.from_numpy(self.analog_sources(i, h).mean(0, keepdims=True))  # analog-ensemble mean
        elif self.source == "dejavu":
            d = self.dist[i, h - 1, :TOPK]
            p = np.exp(-(d - d.min()) / max(d.std(), 1e-3)); p /= p.sum()
            k = np.random.choice(TOPK, p=p) if self.train else 0
            item["x0"] = torch.from_numpy(self.analog_sources(i, h, k))
        elif self.source == "today":
            item["x0"] = torch.from_numpy(norm(self.pm[i]))[None]
        else:
            item["x0"] = torch.zeros(1, *tgt.shape)
        if self.footprint:  # 3-hourly winds + daily rain of days i .. i+4 (or of a day 15 days away: control)
            w = i + self.wind_shift if i + self.wind_shift + 4 < len(self.year) else i - self.wind_shift
            wind = np.array(self.wind[w:w + 5]).reshape(40, 5, 84, 140)  # a copy: the memmap is read-only
            if self.blh_persist:  # BLH is not forecast in HRES: use the start day's diurnal cycle for every day
                wind = wind.copy(); wind[8:, 4] = np.tile(wind[:8, 4], (4, 1, 1))
            item["wind"] = torch.from_numpy(wind)
            r = np.asarray(self.met[w:w + 5, 33], np.float32).reshape(5, 84, 2, 140, 2).mean((2, 4))
            item["rain"] = torch.from_numpy((r * self.rain_sd + self.rain_mu).astype(np.float16))  # log(1 + mm/day)
            if self.fp_met == "stab":  # daily stability of days w .. w+4 from t at 1000/925/850 hPa (store units)
                t = np.asarray(self.met[w:w + 5, 5:8], np.float32) * self.t_sd + self.t_mu
                item["stab"] = torch.from_numpy(theta_gaps(t.reshape(5, 3, 84, 2, 140, 2).mean((3, 5))).astype(np.float16))
        return item


def emb(t, dim=64):
    f = torch.exp(torch.linspace(0, math.log(1000), dim // 2, device=t.device))
    a = t[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class Block(nn.Module):
    def __init__(self, cin, cout, edim):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(cin, cout, 3, padding=1), nn.Conv2d(cout, cout, 3, padding=1)
        self.n1, self.n2 = nn.GroupNorm(8, cout), nn.GroupNorm(8, cout)
        self.film = nn.Linear(edim, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = F.silu(self.n1(self.c1(x)))
        s, b = self.film(e)[:, :, None, None].chunk(2, 1)
        h = F.silu(self.n2(self.c2(h)) * (1 + s) + b)
        return h + self.skip(x)


class Footprint(nn.Module):
    """Footprint layer: every cell gathers features along the back-trajectory of the air that reaches it on the
    forecast day. Trajectories are integrated backwards with the 3-hourly forecast-period wind (a learned mix of 10 m
    and 925 hPa) from 12 UTC of day d+h to 12 UTC of the start day d (8h steps of 3 h). Each point of the path gets a
    learned weight from its age, the boundary-layer height and the rain met there, minus a learned washout penalty on
    the rain accumulated since; the weighted sum (online softmax over the path) is added back residually (zero-init).
    For linear advection-deposition with known wind, this is the exact source-receptor (Green's function) operator."""
    DT, CELL = 3 * 3600.0, 55.6e3  # time step (s), 0.5 deg of latitude (m)

    def __init__(self, c, kmax=32, h=84, w=140, met="blh"):
        super().__init__()
        self.kmax, self.met = kmax, met
        self.mix = nn.Parameter(torch.zeros(2))
        self.score = nn.Sequential(nn.Conv2d(5 if met == "stab" else 4, 32, 1), nn.SiLU(), nn.Conv2d(32, 1, 1))
        self.lam = nn.Parameter(torch.tensor(-1.0))
        self.proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        lat = 72.0 - 0.5 * torch.arange(h) - 0.125
        self.register_buffer("coslat", torch.cos(torch.deg2rad(lat))[None, :, None], persistent=False)

    def forward(self, feat, wind, rain, lead, stab=None):
        B, C, H, W = feat.shape
        dev = feat.device
        wind, rain = wind.float(), rain.float()
        gy, gx = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing="ij")
        pos = torch.stack([gx, gy], -1)[None].repeat(B, 1, 1, 1)
        mw = torch.softmax(self.mix, 0)
        lam = F.softplus(self.lam)
        hh = lead.round().long()
        s0 = 8 * hh + 4
        bidx = torch.arange(B, device=dev)
        m_run = torch.full((B, 1, H, W), -1e4, device=dev)
        s_run = torch.zeros(B, 1, H, W, device=dev)
        acc = torch.zeros_like(feat)
        cum = torch.zeros(B, 1, H, W, device=dev)
        for k in range(self.kmax + 1):
            valid = (k <= 8 * hh).float()[:, None, None, None]
            st = (s0 - k).clamp(min=0)
            smp = F.grid_sample(wind[bidx, st], pos, align_corners=True, padding_mode="border")
            u, v = mw[0] * smp[:, 0] + mw[1] * smp[:, 2], mw[0] * smp[:, 1] + mw[1] * smp[:, 3]
            blh = smp[:, 4:5]
            r = F.grid_sample(rain[bidx, torch.div(st, 8, rounding_mode="floor")][:, None], pos, align_corners=True,
                              padding_mode="border")
            g = F.grid_sample(feat, pos, align_corners=True, padding_mode="border")
            if self.met == "stab":
                day = torch.div(st, 8, rounding_mode="floor")
                sg = F.grid_sample(stab.float()[bidx, day], pos, align_corners=True, padding_mode="border") / 5.0
                z = torch.cat([torch.full_like(r, k / self.kmax), sg, r, cum], 1)
            else:
                z = torch.cat([torch.full_like(r, k / self.kmax), torch.log(blh.clamp(min=10.0)) / 7.0 - 1.0, r, cum], 1)
            logit = self.score(z) - lam * cum
            logit = torch.where(valid > 0, logit, torch.full_like(logit, -1e4))
            m_new = torch.maximum(m_run, logit)
            a, e = torch.exp(m_run - m_new), torch.exp(logit - m_new) * valid
            acc, s_run, m_run = acc * a + e * g, s_run * a + e, m_new
            dx = -u * self.DT / (self.CELL * self.coslat) * (2.0 / (W - 1))   # back in time: against the wind
            dy = v * self.DT / self.CELL * (2.0 / (H - 1))                     # rows grow southwards
            pos = pos + torch.stack([dx, dy], -1) * valid[:, 0, :, :, None]
            cum = cum + r * valid / 8.0
        return feat + self.proj(acc / s_run.clamp(min=1e-6))


class UNet(nn.Module):
    """3-level U-Net (168x280 -> 21x35), FiLM-conditioned on flow time and lead time."""

    def __init__(self, cin, ch=(64, 128, 256), edim=256, footprint=False, fp_met="blh"):
        super().__init__()
        self.fp = Footprint(ch[0], met=fp_met) if footprint else None
        self.e = nn.Sequential(nn.Linear(128, edim), nn.SiLU(), nn.Linear(edim, edim))
        self.inp = nn.Conv2d(cin, ch[0], 3, padding=1)
        self.down = nn.ModuleList([Block(ch[0], ch[0], edim), Block(ch[0], ch[1], edim), Block(ch[1], ch[2], edim)])
        self.mid = Block(ch[2], ch[2], edim)
        self.up = nn.ModuleList([Block(ch[2] * 2, ch[1], edim), Block(ch[1] * 2, ch[0], edim), Block(ch[0] * 2, ch[0], edim)])
        self.out = nn.Conv2d(ch[0], 1, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, tau, lead, fp=None):
        e = self.e(torch.cat([emb(tau), emb(lead / 4.0)], 1))
        h = self.inp(x)
        skips = []
        for i, blk in enumerate(self.down):
            h = blk(h, e); skips.append(h)
            h = F.avg_pool2d(h, 2)
            if i == 0 and self.fp is not None:  # footprint at 0.5 deg, where the transport winds live
                h = self.fp(h, fp[0], fp[1], lead, fp[2] if len(fp) > 2 else None)
        h = self.mid(h, e)
        for blk, s in zip(self.up, reversed(skips)):
            h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = blk(torch.cat([h, s], 1), e)
        return self.out(h)


class DVModule(pl.LightningModule):
    def __init__(self, source, lr=2e-4, epochs=60, n_cond=77, sigma=SIGMA, t31=False, footprint=False, wind_shift=0,
                 blh_persist=False, fp_met="blh"):
        super().__init__()
        self.save_hyperparameters()
        self.net = UNet(n_cond + 1 + (source == "noise_rc"), footprint=footprint, fp_met=fp_met)

    def velocity(self, x, cond, tau, lead, fp=None):
        # The source x0 is NOT an input: with x_tau = (1 - tau) x0 + tau x1, knowing x0 would make the target velocity
        # x1 - x0 = (x_tau - x0) / tau computable without learning anything (a leak). Standard CFM: v(x_tau, tau | cond).
        return self.net(torch.cat([cond, x], 1), tau, lead, fp)

    def step(self, b):
        x1, m, cond, lead = b["x1"], b["mask"], b["cond"], b["lead"]
        fp = ((b["wind"], b["rain"]) + ((b["stab"],) if "stab" in b else ())) if self.hparams.get("footprint", False) else None
        if self.hparams.source == "noise_rc":  # retrieval as conditioning, not as source
            cond = torch.cat([cond, b["x0"]], 1)
        if self.hparams.source == "reg":
            pred = self.velocity(torch.zeros_like(x1), cond, torch.ones_like(lead), lead, fp)
            return ((pred - x1) ** 2 * m).sum() / m.sum()
        noise_src = self.hparams.source in ("noise", "noise_rc")
        x0 = torch.randn_like(x1) if noise_src else b["x0"] + self.hparams.get("sigma", SIGMA) * torch.randn_like(x1)
        tau = torch.rand(x1.shape[0], device=x1.device)
        xt = (1 - tau[:, None, None, None]) * x0 + tau[:, None, None, None] * x1
        v = self.velocity(xt, cond, tau, lead, fp)
        return (((v - (x1 - x0)) ** 2) * m).sum() / m.sum()

    def training_step(self, b, _):
        loss = self.step(b); self.log("train/loss", loss, prog_bar=True); return loss

    def validation_step(self, b, _):
        self.log("val/loss", self.step(b), sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.hparams.lr, total_steps=self.trainer.estimated_stepping_batches,
                                                    pct_start=0.05)
        return [opt], [{"scheduler": sched, "interval": "step"}]

    @torch.no_grad()
    def sample(self, x0, cond, lead, steps, fp=None):
        if self.hparams.source == "reg":
            return self.velocity(torch.zeros_like(x0), cond, torch.ones_like(lead), lead, fp)
        x = x0.clone()
        for s in range(steps):
            tau = torch.full((x.shape[0],), s / steps, device=x.device)
            x = x + self.velocity(x, cond, tau, lead, fp) / steps
        return x


def train(a):
    pl.seed_everything(a.seed, workers=True)
    kw = dict(t31=a.t31, footprint=a.footprint, wind_shift=a.wind_shift, blh_persist=a.blh_persist, fp_met=a.fp_met)
    tr = DVData([2017, 2018, 2019, 2020], a.source, train=True, **kw)
    va = DVData([2021], a.source, train=False, **kw)
    va.samples = va.samples[::6]
    nw = int(os.environ.get("SLURM_CPUS_PER_TASK", 7)) - 1
    dl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=nw, persistent_workers=True, drop_last=True, pin_memory=True)
    dv = DataLoader(va, batch_size=a.batch, num_workers=nw, persistent_workers=True, pin_memory=True)
    out = ROOT / "runs" / f"{a.source}{'' if a.sigma == SIGMA else f'_sig{a.sigma}'}{'_t31' if a.t31 else ''}{('_fp' if not a.wind_shift else f'_fpshift{a.wind_shift}') if a.footprint else ''}{'blhp' if a.blh_persist else ''}{'stab' if a.fp_met == 'stab' else ''}{'' if a.seed == 42 else f'_s{a.seed}'}"
    model = DVModule(a.source, lr=a.lr, epochs=a.epochs, sigma=a.sigma, t31=a.t31, n_cond=71 if a.t31 else 77,
                     footprint=a.footprint, wind_shift=a.wind_shift, blh_persist=a.blh_persist, fp_met=a.fp_met)
    trainer = pl.Trainer(accelerator="gpu", devices=int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)), strategy="ddp",
                         precision="bf16-mixed", max_epochs=a.epochs, gradient_clip_val=1.0, log_every_n_steps=10,
                         default_root_dir=str(out), logger=CSVLogger(str(out), name="logs"),
                         limit_train_batches=a.limit, enable_checkpointing=True)
    if trainer.global_rank == 0:
        print(f"{a.source}: train {len(tr)} val {len(va)} params {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    trainer.fit(model, dl, dv)
    trainer.save_checkpoint(out / "final.ckpt")  # collective call: every rank, rank 0 writes


def attach_hres(ds, t31, fpr):
    """Real ECMWF HRES forecast of day i+h, issued at 00 UTC of day i, instead of ERA5 (test year 2022): swaps the
    forecast-day block of cond and, with the footprint, the forecast-period winds, rain and stability."""
    assert t31
    st, ex = np.load(STATS / "europe_v2_fixed_stats.npz"), np.load(STATS / "europe_v2_extra_stats.npz")
    em, es = st["era5_mean"][:, None, None], st["era5_std"][:, None, None]
    table = {}
    for f in sorted(HRES.glob("*.npz")):
        with np.load(f) as m:
            blk, days = m["block"], m["day_of_year"]
        table.update({int(dd): blk[k] for k, dd in enumerate(days)})
    doy = np.load(DATA / "meta.npz")["doy"]
    hwind = {}
    if fpr:
        for f in sorted((DATA / "hres_wind").glob("*.npz")):
            with np.load(f) as m:
                wd, rd, days = m["wind"], m["rain_d0"], m["day_of_year"]
            hwind.update({int(dd): (wd[k], rd[k]) for k, dd in enumerate(days)})
    get = DVData.__getitem__

    def with_hres(self, idx):
        it = get(self, idx)
        i, h = self.samples[idx]
        raw = table[int(doy[i])][h - 1].astype(np.float32)[:, :168, :280]
        sw = int(os.environ.get("WIND_SWAP", 0))
        dw = int(doy[i])
        if sw:  # control: forecast wind (block and footprint) of the forecast issued sw days later (or earlier)
            dw = dw + sw if (dw + sw) in table else dw - sw
            wch = [0, 1] + list(range(10, 20))
            raw = raw.copy(); raw[wch] = table[dw][h - 1].astype(np.float32)[wch, :168, :280]
        if os.environ.get("WIND_FLIP"):  # control: reverse the forecast wind direction (block and footprint), keep speed
            wch = [0, 1] + list(range(10, 20))
            raw = raw.copy(); raw[wch] = -raw[wch]
        blk = np.concatenate([(raw[:30] - em) / es,
                              ((np.log1p(np.maximum(raw[30], 0) * 1000.0) - ex["extra_mean"][3]) / ex["extra_std"][3])[None]])
        it["cond"][38:69] = torch.from_numpy(blk.astype(np.float32))
        if fpr:  # forecast-period winds (u/v 10 m, 925 hPa) and daily rain from the same HRES run
            wtab, rain_d0 = hwind[int(doy[i])]
            fs = int(os.environ.get("FP_WIND_SWAP", 0))  # control: footprint alone driven by the forecast of fs days away
            dfp = (dw + fs if (dw + fs) in hwind else dw - fs) if fs else dw
            it["wind"][:, :4] = torch.from_numpy(hwind[dfp][0].astype(np.float16)) * (-1 if os.environ.get("WIND_FLIP") else 1)
            rain = [np.log1p(np.maximum(rain_d0, 0))]
            for k in range(1, 5):
                tp = table[int(doy[i])][k - 1][30].astype(np.float32)[:168, :280] * 1000.0 if k <= 4 else None
                rain.append(np.log1p(np.maximum(tp.reshape(84, 2, 140, 2).mean((1, 3)), 0)))
            it["rain"] = torch.from_numpy(np.stack(rain).astype(np.float16))
            if "stab" in it:  # days d+1..d+4 stability from the HRES temperatures (store units, 0.25 -> 0.5 deg)
                st_ = it["stab"].numpy().astype(np.float32)
                for k in range(1, 5):
                    tb = table[int(doy[i])][k - 1][5:8].astype(np.float32)[:, :168, :280].reshape(3, 84, 2, 140, 2).mean((2, 4))
                    st_[k] = theta_gaps(tb)
                it["stab"] = torch.from_numpy(st_.astype(np.float16))
        return it
    ds.__class__ = type("DVDataHRES", (DVData,), {"__getitem__": with_hres})


@torch.no_grad()
def evaluate(a):
    run = Path(a.run)
    dev = torch.device("cuda")
    model = DVModule.load_from_checkpoint(run / "final.ckpt", map_location="cpu").to(dev).eval()
    src = model.hparams.source
    sig = model.hparams.get("sigma", SIGMA)
    t31 = model.hparams.get("t31", False)
    fpr = model.hparams.get("footprint", False)
    shift = a.eval_wind_shift if a.eval_wind_shift is not None else model.hparams.get("wind_shift", 0)
    ds = DVData([a.year], src if src != "reg" else "noise", train=False, t31=t31, footprint=fpr, wind_shift=shift,
                blh_persist=model.hparams.get("blh_persist", False), fp_met=model.hparams.get("fp_met", "blh"))
    if a.weather == "hres":
        assert a.year == 2022
        attach_hres(ds, t31, fpr)
    ds.samples = ds.samples[::int(os.environ.get("EVAL_STRIDE", 1))]
    if os.environ.get("EVAL_DOY"):  # restrict to start days in day-of-year ranges, e.g. "72-80,190-205" (events)
        rng_ = [tuple(map(int, r.split("-"))) for r in os.environ["EVAL_DOY"].split(",")]
        doy_ = np.load(DATA / "meta.npz")["doy"]
        ds.samples = [(i, h) for i, h in ds.samples if any(lo <= doy_[i] <= hi for lo, hi in rng_)]
    an = DVData([a.year], "dejavu", train=False)  # for the analog-ensemble reference
    M, land = a.members, torch.from_numpy(ds.land).bool().to(dev)
    stats, maps, records = {}, {}, []   # records: per (start day, lead) scores of the model, for paired bootstrap tests
    doy_all = np.load(DATA / "meta.npz")["doy"]
    rng = torch.Generator(device=dev); rng.manual_seed(0)
    for n, (i, h) in enumerate(ds.samples):
        it = ds[n]
        cond = it["cond"][None].to(dev).expand(M, -1, -1, -1)
        lead = torch.full((M,), float(h), device=dev)
        x1 = it["x1"][0].to(dev); m = (it["mask"][0].to(dev) > 0) & land
        if src == "noise_rc":
            cond = torch.cat([cond, it["x0"][None].to(dev).expand(M, -1, -1, -1)], 1)
        if src == "dejavu":
            x0 = torch.from_numpy(ds.analog_sources(i, h)[:M]).to(dev)[:, None] + sig * torch.randn(M, 1, *x1.shape, device=dev, generator=rng)
        elif src == "dvmean":
            x0 = it["x0"][None].to(dev).expand(M, -1, -1, -1) + sig * torch.randn(M, 1, *x1.shape, device=dev, generator=rng)
        elif src == "today":
            x0 = it["x0"][None].to(dev).expand(M, -1, -1, -1) + sig * torch.randn(M, 1, *x1.shape, device=dev, generator=rng)
        else:
            x0 = torch.randn(M, 1, *x1.shape, device=dev, generator=rng)
        fp = ((it["wind"][None].to(dev).expand(M, -1, -1, -1, -1), it["rain"][None].to(dev).expand(M, -1, -1, -1))
              + ((it["stab"][None].to(dev).expand(M, -1, -1, -1, -1),) if "stab" in it else ())) if fpr else None
        analog = torch.from_numpy(an.analog_sources(i, h)[:M]).to(dev)
        today = it["cond"][0].to(dev)
        outs = {"persistence": today[None], "analog-ensemble": analog}
        for S in a.steps:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs[f"{src} S={S}"] = model.sample(x0, cond, lead, S, fp)[:, 0].float()
            if src == "reg":
                break
        if os.environ.get("SAVE_DOY") and int(doy_all[i]) in [int(v) for v in os.environ["SAVE_DOY"].split(",")]:
            maps[f"d{int(doy_all[i])}_h{h}"] = {"truth": np.where(m.cpu().numpy(), (x1 * SD + MU).cpu().numpy(), np.nan),
                                               **{k: (v * SD + MU).float().cpu().numpy() for k, v in outs.items()}}
        T = x1[m] * SD + MU
        for name, ens in outs.items():
            E = ens[:, m] * SD + MU                                   # (members, pixels)
            mean = E.mean(0)
            crps = (E - T).abs().mean(0) - 0.5 * (E[:, None] - E[None]).abs().sum((0, 1)) / max(E.shape[0] * (E.shape[0] - 1), 1)
            e, f, fp = T > 25, mean > 25, (E > 25).float().mean(0) >= 0.5
            if os.environ.get("SAVE_RECORDS") and name == f"{src} S={os.environ.get('RECORD_S', '4')}":
                records.append([i, int(doy_all[i]), h, m.sum().item(), ((mean - T) ** 2).sum().item(), crps.sum().item(),
                                (e & fp).sum().item(), (e | fp).sum().item(), float(T.mean().item()), float((T > 25).float().mean().item()),
                                (e & f).sum().item(), (e | f).sum().item()])   # last two: CSI of the ensemble mean
            s = stats.setdefault(f"{name}|T+{h}", np.zeros(9))
            s += [m.sum().item(), ((mean - T) ** 2).sum().item(), crps.sum().item(), E.var(0).sum().item() if E.shape[0] > 1 else 0,
                  (e & f).sum().item(), (e | f).sum().item(), (e & fp).sum().item(), (e | fp).sum().item(), 0]
        if n % 400 == 0:
            print(n, len(ds.samples), flush=True)
    res = {}
    for k, s in stats.items():
        n = s[0]
        res[k] = dict(rmse=float(np.sqrt(s[1] / n)), crps=float(s[2] / n), spread=float(np.sqrt(s[3] / n)),
                      csi_mean=float(s[4] / max(s[5], 1)), csi_prob=float(s[6] / max(s[7], 1)))
    tag = ("" if a.weather == "era5" else "_" + a.weather) + ("" if a.eval_wind_shift is None else f"_evalshift{a.eval_wind_shift}") \
        + (f"_{os.environ['EVAL_TAG']}" if os.environ.get("EVAL_TAG") else "")
    (run / f"eval_{a.year}{tag}.json").write_text(json.dumps(res, indent=1))
    if records:  # columns: index, doy, lead, pixels, sse, crps_sum, hits, union (prob. CSI), mean truth, frac > 25
        np.save(run / f"records_{a.year}{tag}.npy", np.array(records, np.float64))
    if maps:
        np.savez_compressed(run / f"maps_{a.year}{tag}.npz", **{f"{k}|{n}": v for k, d in maps.items() for n, v in d.items()})
    for k in sorted(res):
        r = res[k]
        print(f"{k:28s} RMSE {r['rmse']:.2f}  CRPS {r['crps']:.2f}  spread {r['spread']:.2f}  CSI25 mean {r['csi_mean']:.3f} prob {r['csi_prob']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("run", nargs="?")
    ap.add_argument("--source", choices=["dejavu", "dvmean", "noise", "noise_rc", "today", "reg"], default="dejavu")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma", type=float, default=SIGMA)
    ap.add_argument("--t31", action="store_true", help="forecast-day block restricted to the HRES variables (31 ch)")
    ap.add_argument("--footprint", action="store_true", help="footprint layer on forecast-period 3-hourly winds")
    ap.add_argument("--wind-shift", type=int, default=0, help="control: footprint driven by the winds of a day N days away")
    ap.add_argument("--eval-wind-shift", type=int, default=None, help="eval: feed the footprint the winds of a day N days away")
    ap.add_argument("--blh-persist", action="store_true", help="footprint BLH = start-day cycle (BLH is not in HRES)")
    ap.add_argument("--fp-met", choices=["blh", "stab"], default="blh", help="footprint path features: BLH or forecast stability")
    ap.add_argument("--weather", choices=["era5", "hres"], default="era5", help="eval: forecast-day weather source")
    ap.add_argument("--limit", type=float, default=1.0)
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 4, 8])
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)
