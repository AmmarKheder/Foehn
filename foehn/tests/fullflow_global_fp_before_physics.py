#!/usr/bin/env python3
"""Global FOEHN-km with a transport branch: the tile-free 1 km flow-matching network of fullflow_global.py, plus a
footprint layer (the Green's-function operator of the paper) run at 0.5 deg over the window and a 12 deg margin. The
layer integrates back-trajectories through the 3-hourly wind of days d .. d+h and gathers, along each path, the PM2.5 of
day d, the anthropogenic emissions (CAMS-GLOB-ANT, month of day d) and the fire emissions of day d (GFAS; fires of the
forecast days are never used). Its output, with the emissions of the window, is injected at 16 km with the 0.25 deg
weather. Train 2017-2020 (ERA5 weather, perfect prognosis), validation 2021, test 2022.
    python fullflow_global_fp.py train [--epochs E] ...     (srun, 8 GCDs per node, any number of nodes)"""
import argparse
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
sys.path.insert(0, str(ROOT))
from fullflow_global import CH_, CW_, G, GlobalData, Npy  # noqa: E402
from fullflow import FullModule, FullNet, Timer  # noqa: E402
from dejavu import norm, theta_gaps  # noqa: E402

M = 96                                   # margin of the transport context, 0.25 deg cells (24 deg: a week of transport)
CTX_H, CTX_W = CH_ + 2 * M, CW_ + 2 * M  # 360 x 472 at 0.25 deg, 180 x 236 at 0.5 deg
NCTX, NFP = 10, 32                       # context channels (PM, 6 emissions, 3 fire), footprint features


def ctx_index(Hs, Ws, k0, j0, scale=1):
    """rows (clipped at the grid edges) and columns (wrapped around the globe) of the context of the window at (k0, j0)
    (0.25 deg cells); scale = 2 for 0.5 deg arrays."""
    r = np.clip(np.arange(k0 - M, k0 + CH_ + M, scale) // scale, 0, Hs - 1)
    c = (np.arange(j0 - M, j0 + CW_ + M, scale) // scale) % Ws
    return r, c


def ctx_slice(a, k0, j0, scale=1):
    """in-memory array a[..., H, W] -> its context of the window."""
    r, c = ctx_index(a.shape[-2], a.shape[-1], k0, j0, scale)
    return np.asarray(a)[..., r, :][..., c]


def ctx_read(npy, leads, k0, j0, scale=1):
    """context of the window for the planes `leads` of an Npy file: full-width row blocks read directly, cut in memory."""
    r, c = ctx_index(npy.shape[-2], npy.shape[-1], k0, j0, scale)
    r0, r1 = int(r.min()), int(r.max()) + 1
    return npy.rows(leads, r0, r1)[:, r - r0, :][..., c]


class GlobalDataFP(GlobalData):
    def __init__(self, years, train, seed=0, max_lead=7, no_emis=False, wind_shift=0, no_ant=False):
        super().__init__(years, train, seed, max_lead)
        self.no_emis, self.wind_shift = no_emis, wind_shift              # ablations (defaults = the model)
        self.no_ant = no_ant                                             # no anthropogenic inventory (fires kept)
        st = np.load(G / "stats.npz")
        self.t_mu, self.t_sd = st["era5_mean"][5:8, None, None], st["era5_std"][5:8, None, None]
        self.r_mu, self.r_sd = float(st["extra_mean"][3]), float(st["extra_std"][3])
        meta = np.load(G / "meta.npz")
        self.month = np.array([12 * (y - 2017) + int((np.datetime64(f"{y}-01-01") + np.timedelta64(int(d), "D")).astype("datetime64[M]").astype(int) % 12)
                               for y, d in zip(meta["year"], meta["doy"])])
        self.ant, self.fire, self.wind = Npy(G / "emis_ant_fixed.npy"), Npy(G / "fires.npy"), Npy(G / "wind05.npy")
        # emis_ant.npy, used by the runs without "_fix" (2026-09-29), had the anthropogenic emissions shifted by 180 deg

    def item(self, i, h, k0, j0):
        it = super().item(i, h, k0, j0)
        mo, L = int(self.month[i]), self.max_lead                          # days d .. d+L
        D = self.year.shape[0]
        w = i if not self.wind_shift else (i + self.wind_shift if i + self.wind_shift + L < D else i - self.wind_shift)
        ctx = np.concatenate([norm(ctx_read(self.pm, [(i,)], k0, j0).astype(np.float32)),
                              ctx_read(self.ant, [(mo, s) for s in range(6)], k0, j0).astype(np.float32),
                              ctx_read(self.fire, [(i, s) for s in range(3)], k0, j0).astype(np.float32)])
        if self.no_emis:                                                   # ablation: no emission / fire information
            ctx[1:] = 0.0
        if self.no_ant:                                                    # inventories lag reality: only observed fires
            ctx[1:7] = 0.0
        wind = ctx_read(self.wind, [(w + t, s, ch) for t in range(L + 1) for s in range(8) for ch in range(4)], k0, j0, 2)
        wind = wind.reshape(8 * (L + 1), 4, CTX_H // 2, CTX_W // 2)
        rain = ctx_read(self.met, [(w + t, 33) for t in range(L + 1)], k0, j0).astype(np.float32) * self.r_sd + self.r_mu
        t = ctx_read(self.met, [(w + t, c) for t in range(L + 1) for c in (5, 6, 7)], k0, j0).astype(np.float32)
        t = t.reshape(L + 1, 3, CTX_H, CTX_W) * self.t_sd + self.t_mu                                  # store units
        pool = lambda a: a.reshape(*a.shape[:-2], CTX_H // 2, 2, CTX_W // 2, 2).mean((-3, -1))
        lat = 84.0 - 0.25 * np.clip(np.arange(k0 - M, k0 + CH_ + M, 2), 0, 575) - 0.125
        it.update(ctx=torch.from_numpy(np.nan_to_num(ctx).astype(np.float16)),
                  wind=torch.from_numpy(np.nan_to_num(wind.astype(np.float32)).astype(np.float16)),
                  rain=torch.from_numpy(pool(rain).astype(np.float16)),
                  stab=torch.from_numpy(theta_gaps(pool(t)).astype(np.float16)),
                  coslat=torch.from_numpy(np.cos(np.deg2rad(lat)).astype(np.float32)))
        return it


class FootprintG(nn.Module):
    """The footprint layer of ../dejavu (online softmax over back-trajectory points, weights from age, stability,
    rain and the rain met since, zero-initialised residual), with a per-sample latitude (windows anywhere on Earth)."""
    DT, CELL = 3 * 3600.0, 55.6e3

    def __init__(self, c, kmax=32):
        super().__init__()
        self.kmax = kmax
        self.mix = nn.Parameter(torch.zeros(2))
        self.score = nn.Sequential(nn.Conv2d(5, 32, 1), nn.SiLU(), nn.Conv2d(32, 1, 1))
        self.lam = nn.Parameter(torch.tensor(-1.0))
        self.proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, feat, wind, rain, lead, stab, coslat):
        B, C, H, W = feat.shape
        dev = feat.device
        wind, rain, stab = wind.float(), rain.float(), stab.float()
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
        cl = coslat.float().clamp(min=0.05)[:, :, None]                     # (B, H, 1)
        for k in range(self.kmax + 1):
            valid = (k <= 8 * hh).float()[:, None, None, None]
            st = (s0 - k).clamp(min=0)
            smp = F.grid_sample(wind[bidx, st], pos, align_corners=True, padding_mode="border")
            u, v = mw[0] * smp[:, 0] + mw[1] * smp[:, 2], mw[0] * smp[:, 1] + mw[1] * smp[:, 3]
            day = torch.div(st, 8, rounding_mode="floor")
            r = F.grid_sample(rain[bidx, day][:, None], pos, align_corners=True, padding_mode="border")
            g = F.grid_sample(feat, pos, align_corners=True, padding_mode="border")
            sg = F.grid_sample(stab[bidx, day], pos, align_corners=True, padding_mode="border") / 5.0
            z = torch.cat([torch.full_like(r, k / self.kmax), sg, r, cum], 1)
            logit = self.score(z) - lam * cum
            logit = torch.where(valid > 0, logit, torch.full_like(logit, -1e4))
            m_new = torch.maximum(m_run, logit)
            a, e = torch.exp(m_run - m_new), torch.exp(logit - m_new) * valid
            acc, s_run, m_run = acc * a + e * g, s_run * a + e, m_new
            dx = -u * self.DT / (self.CELL * cl) * (2.0 / (W - 1))
            dy = v * self.DT / self.CELL * (2.0 / (H - 1))
            pos = pos + torch.stack([dx, dy], -1) * valid[:, 0, :, :, None]
            cum = cum + r * valid / 8.0
        return feat + self.proj(acc / s_run.clamp(min=1e-6))


class FullNetFP(FullNet):
    def __init__(self, cin, max_lead=7):
        super().__init__(cin, ccoarse=69 + NCTX + NFP)
        self.enc = nn.Sequential(nn.Conv2d(NCTX, NFP, 3, padding=1), nn.SiLU(), nn.Conv2d(NFP, NFP, 3, padding=1))
        self.fp = FootprintG(NFP, kmax=8 * max_lead)

    def forward(self, x, coarse, tau, lead, tr=None):
        f = F.avg_pool2d(self.enc(tr["ctx"].float()), 2)                                 # 0.5 deg features
        f = self.fp(f, tr["wind"], tr["rain"], lead, tr["stab"], tr["coslat"])
        f = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)[:, :, M:M + CH_, M:M + CW_]
        win = tr["ctx"].float()[:, :, M:M + CH_, M:M + CW_]
        return super().forward(x, torch.cat([coarse, win, f], 1), tau, lead)


class FullModuleFP(FullModule):
    def __init__(self, lr=2e-4, epochs=10, reg=False, shift=1.0, shift_mix=0.5, max_lead=7):
        super().__init__(lr=lr, epochs=epochs, reg=reg, shift=shift, shift_mix=shift_mix)
        self.save_hyperparameters()
        self.net = FullNetFP(6 + 1, max_lead)

    def velocity(self, x, b, tau):
        tr = {k: b[k] for k in ("ctx", "wind", "rain", "stab", "coslat")}
        return self.net(torch.cat([b["fine"].float(), x], 1), b["coarse"], tau, b["lead"], tr)


def train(a):
    pl.seed_everything(a.seed, workers=True)
    kw = dict(max_lead=a.max_lead, no_emis=a.no_emis, wind_shift=a.wind_shift, no_ant=a.no_ant)
    tr = GlobalDataFP([2017, 2018, 2019, 2020], train=True, **kw)
    va = GlobalDataFP([2021], train=False, seed=1, **kw)
    keep = list(range(0, len(va.samples), max(1, len(va.samples) // 64)))[:64]
    va.samples, va.fixed = [va.samples[k] for k in keep], va.fixed[keep]
    if a.limit < 1:
        tr.samples = tr.samples[:int(len(tr.samples) * a.limit)]; va.samples, va.fixed = va.samples[:8], va.fixed[:8]
    dl = DataLoader(tr, batch_size=1, shuffle=True, num_workers=a.workers, persistent_workers=True, pin_memory=True)
    dv = DataLoader(va, batch_size=1, num_workers=a.workers, persistent_workers=True, pin_memory=True)
    abl = ("_noemis" if a.no_emis else "") + ("_noant" if a.no_ant else "") + (f"_windshift{a.wind_shift}" if a.wind_shift else "")
    out = ROOT / "runs" / f"fp_flow_shift{a.shift:g}_global_L{a.max_lead}{abl}{'' if a.seed == 42 else f'_s{a.seed}'}{a.tag}"
    model = FullModuleFP(lr=a.lr, epochs=a.epochs, reg=False, shift=a.shift, shift_mix=a.shift_mix, max_lead=a.max_lead)
    trainer = pl.Trainer(accelerator="gpu", devices=int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)),
                         num_nodes=int(os.environ.get("SLURM_NNODES", 1)), strategy="ddp", precision="bf16-mixed",
                         max_epochs=a.epochs, gradient_clip_val=1.0, log_every_n_steps=10, default_root_dir=str(out),
                         logger=CSVLogger(str(out), name="logs"), callbacks=[Timer()])
    if trainer.global_rank == 0:
        print(f"{out.name}: train {len(tr)} val {len(va)} windows {len(tr.cands)} "
              f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    trainer.fit(model, dl, dv, ckpt_path=a.resume)       # resume: optimizer and one-cycle schedule restored
    trainer.save_checkpoint(out / "final.ckpt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "check"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--shift", type=float, default=25.0)
    ap.add_argument("--shift-mix", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-lead", type=int, default=7)
    ap.add_argument("--resume", default=None, help="checkpoint to resume from (e.g. after a time limit)")
    ap.add_argument("--no-emis", action="store_true", help="ablation: emission and fire channels set to zero")
    ap.add_argument("--no-ant", action="store_true", help="anthropogenic inventory channels set to zero, GFAS fires kept")
    ap.add_argument("--wind-shift", type=int, default=0, help="ablation: footprint path weather of a day N days away")
    a = ap.parse_args()
    if a.cmd == "check":             # one sample through the data pipeline (CPU), shapes and ranges
        ds = GlobalDataFP([2017], train=True, max_lead=a.max_lead)
        b = ds[0]
        print({k: (tuple(v.shape), float(v.float().min()), float(v.float().max())) for k, v in b.items()})
    else:
        train(a)
