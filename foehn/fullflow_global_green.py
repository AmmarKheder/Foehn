#!/usr/bin/env python3
"""Global Foehn-km with FLEXGREEN transport: physical transport, learned chemistry (operator splitting).
The transport of the air reaching each 0.5 deg cell is the Green's function of a FLEXPART-like 3-D backward particle
model (flexgreen.py: advection by the 3-D forecast wind, boundary-layer turbulence, dry and wet removal), computed once
outside training; it has no learnable parameter. Transport is linear, so it commutes with any linear mixing of the
transported quantities: a learned encoder makes 32 latent tracers of day d on the 0.5 deg globe (from the PM2.5 and the
GFAS fires of day d) and every receptor cell of the window gathers them at the origins of its particles (bilinear;
weight = survival of the particle when its origin lies in the mixed layer). With the fraction of the air that came from
above the mixed layer, the mean survival, the transported PM2.5 of day d and the fire sensitivity (FLEXPART's emission
sensitivity x GFAS fires of day d), these channels enter the 1 km flow-matching network with the 0.25 deg weather.
Member diversity: each training sample, and each forecast member, uses a random half of the particles.
Fine-tuning (--store green256 --emis --psub 128 --init <ckpt>): 256 particles per receptor, and the anthropogenic emission term of
FLEXGREEN (FLEXPART's sensitivity to surface emissions x CAMS-GLOB-ANT, 6 species) as 6 more channels whose input
weights start at zero, so that the fine-tuning starts from the model of the checkpoint.
weather="hres": the forecast-day weather of the training years is the IFS HRES run of day d (WeatherBench2), as at test
time ("train as you forecast"); "era5": the analysed weather (perfect prognosis, the ablation).
CRPS fine-tuning (--crps-steps K --init <ckpt> --psub 32): the ensemble is trained for calibration. Each sample draws M = 2
members exactly as a forecast does: own noise, own random half of the particles (16 of 32), K Euler steps of the flow on
the quantile grid of the training flow time (tau_grid(K)); the loss is their fair CRPS against GHAP on the valid pixels
(Ferro 2014; unbiased for any M), back-propagated through the K steps (each network call recomputed in the backward pass,
gradient checkpointing). --crps-fm w adds w x the flow-matching loss (member 0's particles). Forecasts then use the same
sampler: eval_global_green.py --steps K --refine-tail 1.
    python fullflow_global_green.py train --weather hres --green hres [--epochs 60] ...   (srun, 8 GCDs per node)"""
import argparse
import os
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "fullimage"))
sys.path.insert(0, str(ROOT.parent / "dejavu"))
from fullflow_global import CH_, CW_, G, GlobalData, Npy  # noqa: E402
from fullflow import FullModule, FullNet, Timer  # noqa: E402
from dejavu import norm  # noqa: E402

NR, NC = 288, 720                       # 0.5 deg receptor grid (84 N .. 60 S)
NG = 32                                 # latent tracers
NWIN = 4                                # window channels: PM2.5 and fires of day d at 0.25 deg
NGF = NG + 2 + 1 + 3                    # gathered latents, aloft fraction, survival, transported PM2.5, fire sensitivity
NE = 6                                  # anthropogenic emission term (flexgreen.py --emis: BC, OC, SO2, NOx, NH3, CO)
GREEN = G / "green"
I2022 = 1826                            # global index of 1 January 2022
FS_SCALE = np.array([1e9, 1.0, 1e9], np.float32)   # fire terms -> ug m-3, J m-3, ug m-3 before log1p


def green_rows(npy, d, h, r0, r1):
    """indices r0:r1 of the third axis of (day d, lead h) of a green store (days, 7, n2, ...): one contiguous read
    (origins: n2 = 288 receptor rows; fire terms: n2 = 3 channels)"""
    per_row = int(np.prod(npy.shape[3:]))
    off = ((d * 7 + (h - 1)) * int(npy.shape[2]) + r0) * per_row
    return npy._read(off, (r1 - r0) * per_row).reshape((r1 - r0,) + tuple(npy.shape[3:]))


def green_block(npy, d, h, r0, r1, c0, nc):
    """particle origins of receptor rows r0:r1 and columns c0 .. c0 + nc - 1 (periodic) of (day d, lead h) of an origins
    store (days, 7, 288, 720, P, 4): per row, one read of the needed columns (two across the date line), instead of
    whole rows (5x less I/O for a window)"""
    ncol, tail = int(npy.shape[3]), tuple(int(v) for v in npy.shape[4:])
    per_col = int(np.prod(tail))
    base = (d * 7 + (h - 1)) * int(npy.shape[2])
    out = np.empty((r1 - r0, nc) + tail, npy.dtype)
    n1 = min(nc, ncol - c0)
    for k, r in enumerate(range(r0, r1)):
        row = (base + r) * ncol
        out[k, :n1] = npy._read((row + c0) * per_col, n1 * per_col).reshape((n1,) + tail)
        if n1 < nc:
            out[k, n1:] = npy._read(row * per_col, (nc - n1) * per_col).reshape((nc - n1,) + tail)
    return out


class GlobalDataGreen(GlobalData):
    def __init__(self, years, train, seed=0, max_lead=7, weather="hres", green="hres", psub=16, store="green", emis=False,
                 emis_store=None):
        super().__init__(years, train, seed, max_lead)
        self.weather, self.src, self.psub, self.emis = weather, green, psub, emis
        root = G / store / green
        meta = np.load(G / "meta.npz")
        self.doy = meta["doy"]
        self.fire = Npy(G / "fires.npy")
        st = np.load(G / "stats.npz")
        self.em, self.es = st["era5_mean"][:, None, None], st["era5_std"][:, None, None]
        self.rm, self.rs = float(st["extra_mean"][3]), float(st["extra_std"][3])
        if weather == "hres":
            self.hb_train = Npy(G / "hres_block_train.npy")    # (1826, 7, 31, 576, 1440): global days 0..1825 (2017-2021)
            self.hb_test = Npy(G / "hres_block.npy")           # (365, 7, 31, 576, 1440): 2022
        ys = sorted({int(y) for y in self.year[[i for i, _ in self.samples]]}) if self.samples else []
        self.org = {y: Npy(root / f"origins_{y}.npy") for y in ys}
        self.fs = {y: Npy(root / f"fire_{y}.npy") for y in ys}
        self.esn = {y: Npy(G / (emis_store or store) / green / f"emis_{y}.npy") for y in ys} if emis else {}   # may come from another store
        done = {y: {int(p.name) for p in (root / f"done_{y}").glob("*")} for y in ys}
        n0 = len(self.samples)
        keep = [k for k, (i, h) in enumerate(self.samples) if int(self.doy[i]) in done[int(self.year[i])]]
        self.samples = [self.samples[k] for k in keep]
        self.fixed = self.fixed[keep]
        if n0 and len(keep) < n0:
            print(f"GlobalDataGreen {years}: {len(keep)} of {n0} samples have their Green's functions", flush=True)

    def hres_block(self, i, h, k0, j0):
        hb, d = (self.hb_train, i) if i < I2022 else (self.hb_test, i - I2022)
        raw = hb.rows([(d, h - 1, c) for c in range(31)], k0, k0 + CH_)[..., j0:j0 + CW_].astype(np.float32)
        return np.concatenate([(raw[:30] - self.em[:30]) / self.es[:30],
                               ((np.log1p(np.maximum(raw[30], 0) * 1000.0) - self.rm) / self.rs)[None]])

    def item(self, i, h, k0, j0, member=None):
        it = super().item(i, h, k0, j0)
        if self.weather == "hres":                                       # forecast-day weather = HRES run of day d
            it["coarse"][-31:] = torch.from_numpy(np.nan_to_num(self.hres_block(i, h, k0, j0)).astype(np.float32))
        y, d = int(self.year[i]), int(self.doy[i])
        pm = self.pm.plane_rows((i,), 0, 576).astype(np.float32)
        fire = self.fire.rows([(i, s) for s in range(3)], 0, 576).astype(np.float32)
        gpm = np.concatenate([np.nan_to_num(norm(pm))[None], np.isfinite(pm)[None].astype(np.float32),
                              np.nan_to_num(fire)])                     # (5, 576, 1440): fires standardised (build_transport.py)
        win = np.concatenate([gpm[:1, k0:k0 + CH_, j0:j0 + CW_], gpm[2:, k0:k0 + CH_, j0:j0 + CW_]])
        r0, c0 = k0 // 2, j0 // 2
        cols = (c0 + np.arange(CW_ // 2)) % NC
        org = green_block(self.org[y], d, h, r0, r0 + CH_ // 2, c0, CW_ // 2)            # (84, 140, P, 4)
        P = org.shape[-2]
        rng = np.random.default_rng(None if self.train else (int(i) * 31 + int(h)) * 101 + (member or 0))
        sub = np.sort(rng.choice(P, self.psub, replace=False))
        fs = green_rows(self.fs[y], d, h, 0, 3)[:, r0:r0 + CH_ // 2][:, :, cols]             # (3, 84, 140)
        it.update(gpm=torch.from_numpy(gpm.astype(np.float16)), win=torch.from_numpy(win.astype(np.float32)),
                  org=torch.from_numpy(np.ascontiguousarray(org[:, :, sub]).astype(np.float32)),
                  fsens=torch.from_numpy(np.log1p(np.maximum(fs, 0) * FS_SCALE[:, None, None]).astype(np.float32)))
        if self.emis:                                                     # ug m-3 added by the emissions since day d
            es = green_rows(self.esn[y], d, h, 0, NE)[:, r0:r0 + CH_ // 2][:, :, cols]           # (6, 84, 140)
            it["esens"] = torch.from_numpy(np.log1p(np.maximum(np.nan_to_num(es), 0)).astype(np.float32))
        return it


def gather(f, org):
    """parameter-free transport: f (B, C, 288, 720) tracers on the 0.5 deg grid, org (B, R, Q, P, 4) particle origins
    (lon, lat, z/h, survival) -> (B, C + 2, R, Q): survival-weighted mean of f over the origins in the mixed layer,
    fraction of the (surviving) air that came from above the mixed layer, mean survival."""
    B, C = f.shape[:2]
    _, R, Q, P, _ = org.shape
    fw = torch.cat([f, f[..., :1]], -1)                                    # periodic: column 720 = column 0
    lon, lat, zr, sv = org.unbind(-1)
    gx = ((lon + 180.0 - 0.125) % 360.0) / 0.5 / NC * 2.0 - 1.0          # cell centres at -180 + 0.5 c + 0.125
    gy = (84.0 - 0.125 - lat) / 0.5 / (NR - 1) * 2.0 - 1.0
    smp = F.grid_sample(fw, torch.stack([gx, gy], -1).view(B, R * Q, P, 2).to(fw.dtype), mode="bilinear",
                        padding_mode="border", align_corners=True)         # (B, C, R*Q, P)
    sv = torch.nan_to_num(sv).clamp(0, 1)
    inm = (zr <= 1.0).to(sv.dtype) * sv
    out = (smp * inm.view(B, 1, R * Q, P).to(smp.dtype)).sum(-1) / inm.view(B, 1, R * Q, P).sum(-1).clamp(min=1e-3).to(smp.dtype)
    aloft = ((zr > 1.0).to(sv.dtype) * sv).sum(-1) / sv.sum(-1).clamp(min=1e-3)
    return torch.cat([out, aloft.view(B, 1, R * Q).to(out.dtype), sv.mean(-1).view(B, 1, R * Q).to(out.dtype)], 1).view(B, C + 2, R, Q)


class FullNetGreen(FullNet):
    def __init__(self, cin, no_transport=False, emis=False, emis_scale=1.0):
        super().__init__(cin, ccoarse=69 + NWIN + NGF + (NE if emis else 0))
        self.no_transport, self.emis, self.emis_scale = no_transport, emis, emis_scale
        self.enc = nn.Sequential(nn.Conv2d(5, NG, 3, padding=1), nn.SiLU(), nn.Conv2d(NG, NG, 3, padding=1))

    def forward(self, x, coarse, tau, lead, tr=None):
        gpm = tr["gpm"].float()
        f = F.avg_pool2d(self.enc(gpm), 2)                                 # (B, 32, 288, 720) latent tracers of day d
        g = gather(f, tr["org"])                                          # (B, 34, 84, 140)
        pm_t = gather(F.avg_pool2d(gpm[:, :1], 2), tr["org"])[:, :1]      # transported PM2.5 of day d (physics only)
        g = torch.cat([g, pm_t, tr["fsens"].to(g.dtype)], 1)
        if self.emis:                                                      # appended last: zero-initialised input weights
            g = torch.cat([g, tr["esens"].to(g.dtype) * self.emis_scale], 1)   # scale k: k x the learning rate of their weights (Adam)
        if self.no_transport:                                              # ablation: same network, no transport inputs
            g = g * 0.0                                                     # keeps the encoder in the graph (DDP)
        g = F.interpolate(g, scale_factor=2, mode="bilinear", align_corners=False)          # (B, 38, 168, 280)
        return super().forward(x, torch.cat([coarse, tr["win"].to(coarse.dtype), g.to(coarse.dtype)], 1), tau, lead)


def fair_crps(xs, y):
    """fair CRPS of the members xs (list of tensors) against y, per pixel: mean |x_i - y| - sum_{i<j} |x_i - x_j| / (M (M - 1))"""
    M = len(xs)
    pair = sum((xs[i] - xs[j]).abs() for i in range(M) for j in range(i + 1, M)) / (M * (M - 1))
    return sum((x - y).abs() for x in xs) / M - pair


class FullModuleGreen(FullModule):
    def __init__(self, lr=2e-4, epochs=10, reg=False, shift=1.0, shift_mix=0.5, max_lead=7, weather="hres", green="hres",
                 no_transport=False, emis=False, store="green", emis_store=None, emis_scale=1.0, crps_steps=0, crps_members=2,
                 crps_fm=0.0, psub_member=16):
        super().__init__(lr=lr, epochs=epochs, reg=reg, shift=shift, shift_mix=shift_mix)
        self.save_hyperparameters()
        self.net = FullNetGreen(6 + 1, no_transport, emis, emis_scale)
        self._t = None

    def velocity(self, x, b, tau):
        tr = {k: b[k] for k in ("gpm", "org", "fsens", "win", "esens") if k in b}
        return self.net(torch.cat([b["fine"].float(), x], 1), b["coarse"], tau, b["lead"], tr)

    def member_batch(self, b):
        """the batch of one ensemble member: a random subset of psub_member of the particles, as member_particles() of a forecast"""
        org = b["org"]
        sub = torch.randperm(org.shape[-2], device=org.device)[:self.hparams.psub_member].sort().values
        return dict(b, org=org[..., sub, :])

    def members(self, b, K, M, grad=True):
        """M members of K Euler steps on tau_grid(K) (the forecast sampler with --refine-tail 1); with grad, every network call
        is recomputed in the backward pass instead of keeping its activations"""
        grid = self.tau_grid(K)
        x1 = b["x1"]; out = []
        for _ in range(M):
            bm = self.member_batch(b)
            x = torch.randn(x1.shape, device=x1.device, dtype=torch.float32)
            for k in range(K):
                tau = torch.full((x1.shape[0],), float(grid[k]), device=x1.device)
                v = checkpoint(self.velocity, x, bm, tau, use_reentrant=False) if grad else self.velocity(x, bm, tau)
                x = x + v.float() * float(grid[k + 1] - grid[k])
            out.append(x)
        return out

    def crps_loss(self, b, grad=True):
        x1, m = b["x1"].float(), b["mask"].float()
        xs = self.members(b, int(self.hparams.crps_steps), int(self.hparams.crps_members), grad)
        return (fair_crps(xs, x1) * m).sum() / m.sum().clamp(min=1)

    def step(self, b):
        if not self.hparams.get("crps_steps", 0):
            return super().step(b)
        loss = self.crps_loss(b)
        if self.hparams.crps_fm > 0:                                       # flow matching with one member's particles
            loss = loss + self.hparams.crps_fm * super().step(self.member_batch(b))
        return loss

    def training_step(self, b, i):
        loss = self.step(b); self.log("train/loss", loss, prog_bar=True)
        if self.hparams.get("crps_steps", 0) and self.global_rank == 0 and i < 12:   # price of a step (smoke test)
            t = time.time()
            if self._t is not None:
                print(f"crps step {i}: {t - self._t:.1f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            self._t = t
        return loss

    def validation_step(self, b, _):
        if not self.hparams.get("crps_steps", 0):
            return super().validation_step(b, _)
        self.log("val/crps", self.crps_loss(b, grad=False), sync_dist=True)
        self.log("val/loss", super().step(self.member_batch(b)), sync_dist=True)


def init_from(model, path):
    """fine-tuning: the weights of a checkpoint; the input channels added since (emission term, appended last to the
    coarse inputs) get zero weights, so that the model starts as the checkpoint"""
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    own = model.state_dict()
    for k, v in sd.items():
        if k in own and own[k].shape != v.shape:
            assert k.endswith("cproj.weight") and own[k].shape[1] > v.shape[1], (k, own[k].shape, v.shape)
            w = torch.zeros_like(own[k]); w[:, :v.shape[1]] = v; sd[k] = w
    model.load_state_dict(sd, strict=True)


def train(a):
    pl.seed_everything(a.seed, workers=True)
    kw = dict(max_lead=a.max_lead, weather=a.weather, green=a.green, psub=a.psub, store=a.store, emis=a.emis, emis_store=a.emis_store)
    tr = GlobalDataGreen([2017, 2018, 2019, 2020], train=True, **kw)
    va = GlobalDataGreen([2021], train=False, seed=1, **kw)
    keep = list(range(0, len(va.samples), max(1, len(va.samples) // 64)))[:64]
    va.samples, va.fixed = [va.samples[k] for k in keep], va.fixed[keep]
    if a.limit < 1:
        tr.samples = tr.samples[:int(len(tr.samples) * a.limit)]; va.samples, va.fixed = va.samples[:8], va.fixed[:8]
    dl = DataLoader(tr, batch_size=1, shuffle=True, num_workers=a.workers, persistent_workers=True, pin_memory=True)
    dv = DataLoader(va, batch_size=1, num_workers=a.workers, persistent_workers=True, pin_memory=True)
    out = ROOT / "runs" / (f"green_flow_shift{a.shift:g}_global_L{a.max_lead}_noant_w{a.weather}_"
                           f"{'notransport' if a.no_transport else 'g' + a.green}"
                           f"{'' if a.seed == 42 else f'_s{a.seed}'}{a.tag}")
    model = FullModuleGreen(lr=a.lr, epochs=a.epochs, reg=False, shift=a.shift, shift_mix=a.shift_mix, max_lead=a.max_lead,
                            weather=a.weather, green=a.green, no_transport=a.no_transport, emis=a.emis, store=a.store,
                            emis_store=a.emis_store, emis_scale=a.emis_scale, crps_steps=a.crps_steps, crps_members=a.crps_members,
                            crps_fm=a.crps_fm, psub_member=a.psub_member)
    assert not a.crps_steps or a.psub >= 2 * a.psub_member, "CRPS fine-tuning: --psub 32, members draw their own 16"

    if a.init:                                       # fine-tuning: weights of a checkpoint, new optimiser and schedule
        init_from(model, a.init)
        print(f"weights from {a.init}", flush=True)
    trainer = pl.Trainer(accelerator="gpu", devices=int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)),
                         num_nodes=int(os.environ.get("SLURM_NNODES", 1)), strategy="ddp", precision="bf16-mixed",
                         max_epochs=a.epochs, gradient_clip_val=1.0, log_every_n_steps=10, default_root_dir=str(out),
                         logger=CSVLogger(str(out), name="logs"),
                         callbacks=[Timer()] + ([ModelCheckpoint(dirpath=str(out / "ckpts"), filename="{epoch:02d}", save_top_k=-1,
                                                                 every_n_epochs=1)] if a.keep_all_ckpt else []))
    if trainer.global_rank == 0:
        print(f"{out.name}: train {len(tr)} val {len(va)} windows {len(tr.cands)} "
              f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    trainer.fit(model, dl, dv, ckpt_path=a.resume)
    trainer.save_checkpoint(out / "final.ckpt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "check"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--shift", type=float, default=25.0)
    ap.add_argument("--shift-mix", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-lead", type=int, default=7)
    ap.add_argument("--weather", choices=["hres", "era5"], default="hres")
    ap.add_argument("--green", choices=["hres", "era5"], default="hres")
    ap.add_argument("--psub", type=int, default=16, help="particles used per sample / member (of the stored 32)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--keep-all-ckpt", action="store_true", help="keep the checkpoint of every epoch (selection on 2021)")
    ap.add_argument("--init", default=None, help="fine-tuning: start from the weights of this checkpoint")
    ap.add_argument("--store", default="green", help="Green's function store (green: 32 particles; green256: 256 + emissions)")
    ap.add_argument("--emis", action="store_true", help="anthropogenic emission term of the store (6 channels)")
    ap.add_argument("--emis-store", default=None, help="store holding emis_<year>.npy if not --store (e.g. green256)")
    ap.add_argument("--emis-scale", type=float, default=1.0, help="scale of the emission inputs (k: k x their weights' learning rate)")
    ap.add_argument("--no-transport", action="store_true", help="ablation: transport inputs set to zero (same network)")
    ap.add_argument("--check-year", type=int, default=2017)
    ap.add_argument("--crps-steps", type=int, default=0, help="CRPS fine-tuning: Euler steps of the sampler (0: flow matching)")
    ap.add_argument("--crps-members", type=int, default=2)
    ap.add_argument("--crps-fm", type=float, default=0.0, help="weight of the flow-matching loss added to the CRPS")
    ap.add_argument("--psub-member", type=int, default=16, help="particles per member (CRPS fine-tuning; forecasts use 16)")
    a = ap.parse_args()
    if a.cmd == "check":
        ds = GlobalDataGreen([a.check_year], train=True, max_lead=a.max_lead, weather=a.weather, green=a.green, psub=a.psub,
                             store=a.store, emis=a.emis, emis_store=a.emis_store)
        b = ds[0]
        print({k: (tuple(v.shape), float(v.float().min()), float(v.float().max())) for k, v in b.items()})
    else:
        train(a)
