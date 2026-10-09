#!/usr/bin/env python3
"""Whole-image 1 km generative forecasting of PM2.5 over Europe (GHAP): the entire 4192 x 6992 map is the input and
the output, in training and at inference. No tiles, no crops, no stitching.

Network: 2 x 2 space-to-depth (1 km -> 2 km, 4 sub-pixels as channels), a 6-level convolutional U-Net (2 -> 64 km) with
global self-attention over the whole of Europe at 64 km (66 x 110 tokens), depth-to-space back to 1 km.
Conditional flow matching from noise to the field of day d + h (FiLM on flow time and lead time).
Conditioning (same information as the baselines): PM today and yesterday at 1 km, elevation, land; PM today, weather
of today and the forecast-day weather (the 31 HRES variables) at 0.25 deg, injected at the 16 km level.
Train 2017-2020 (one full image per GCD), validation 2021, test 2022.
    python fullflow.py train [--epochs 10] [--reg]      (srun, one node, 8 GCDs)
    evaluation: ../retrain/evaluate.py --full runs/<run> --weather hres
"""
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
from pytorch_lightning.loggers import CSVLogger
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "dejavu"))
from dejavu import Block, DVData, attach_hres, emb, norm, MU, SD  # noqa: E402
from build_ghap1km import F as FINE_FILE, H as FH, W as FW  # noqa: E402

PH, PW = 4224, 7040          # image padded to a multiple of 64 km (2 x 2 space-to-depth + 5 poolings)
CH = (64, 96, 128, 192, 256, 256)


def header(path):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); return f.tell()


class DayFile:
    """Day d of the (D, 4192, 6992) float16 GHAP file (one day mapped at a time)."""

    def __init__(self, path):
        self.path, self.hdr = path, header(path)

    def __getitem__(self, d):
        return np.memmap(self.path, dtype=np.float16, mode="r", offset=self.hdr + d * FH * FW * 2, shape=(FH, FW))


class FullData(DVData):
    """One (day i, lead h) sample = the whole Europe image at 1 km + the 0.25 deg conditioning of ../dejavu."""

    def __init__(self, years, train):
        super().__init__(years, "noise", train, t31=True)
        valid = np.load(ROOT.parent / "dejavu" / "data" / "analogs.npz")["valid"]
        self.samples = [(i, h) for i, h in self.samples if valid[i - 1]]   # yesterday's map exists too (GHAP V1 gaps)
        self.fine = DayFile(FINE_FILE)
        st = np.load(ROOT / "data" / "static1km.npz")
        self.static = np.stack([st["elev"], st["land"]]).astype(np.float16)

    @staticmethod
    def pad(a):
        return np.pad(a, ((0, 0), (0, PH - FH), (0, PW - FW)))

    def day(self, d):
        a = np.asarray(self.fine[d], np.float32)
        ok = a > 0                                   # GHAP V1: 0 = sea / no retrieval
        return np.stack([norm(np.where(ok, a, np.nan)), ok]).astype(np.float16)

    def __getitem__(self, idx):
        it = super().__getitem__(idx)
        i, h = self.samples[idx]
        tgt = self.day(i + h)
        return {"fine": torch.from_numpy(self.pad(np.concatenate([self.day(i), self.day(i - 1), self.static]))),
                "x1": torch.from_numpy(self.pad(tgt[:1])), "mask": torch.from_numpy(self.pad(tgt[1:])),
                "coarse": it["cond"][:69].contiguous(), "lead": it["lead"]}


class AttnBlock(nn.Module):
    """Transformer block over the 64 km tokens of the whole of Europe, FiLM-conditioned."""

    def __init__(self, c, heads, edim):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.qkv, self.proj = nn.Linear(c, 3 * c), nn.Linear(c, c)
        self.mlp = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))
        self.film = nn.Linear(edim, 2 * c)
        self.heads = heads

    def forward(self, t, e):
        B, N, C = t.shape
        s, b = self.film(e)[:, None].chunk(2, -1)
        q, k, v = self.qkv(self.n1(t) * (1 + s) + b).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        t = t + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))
        return t + self.mlp(self.n2(t))


class FullNet(nn.Module):
    def __init__(self, cin, ccoarse=69, ch=CH, edim=256, attn_depth=4, heads=8):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(128, edim), nn.SiLU(), nn.Linear(edim, edim))
        self.stem = nn.Conv2d(4 * cin, ch[0], 3, padding=1)
        self.down = nn.ModuleList([Block(ch[k], ch[k + 1], edim) for k in range(len(ch) - 1)])
        self.cproj = nn.Conv2d(ccoarse, ch[3], 1)
        self.pos = nn.Parameter(0.02 * torch.randn(1, ch[-1], PH // 64, PW // 64))
        self.attn = nn.ModuleList([AttnBlock(ch[-1], heads, edim) for _ in range(attn_depth)])
        self.up = nn.ModuleList([Block(2 * ch[k + 1], ch[k], edim) for k in reversed(range(len(ch) - 1))])
        self.out = nn.Conv2d(ch[0], 4, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        # 16 km cell (a, b) centre = 1 km pixel (16 a + 8, 16 b + 8) = 0.25 deg coordinate (16 a + 8) / 25
        ya = (torch.arange(PH // 16) * 16 + 8) / 25.0 * (2.0 / 167) - 1
        xb = (torch.arange(PW // 16) * 16 + 8) / 25.0 * (2.0 / 279) - 1
        self.register_buffer("grid16", torch.stack([xb[None].expand(PH // 16, -1), ya[:, None].expand(-1, PW // 16)], -1)[None],
                             persistent=False)

    def forward(self, x, coarse, tau, lead):
        e = self.e(torch.cat([emb(tau), emb(lead / 4.0)], 1))
        h = self.stem(F.pixel_unshuffle(x, 2))
        skips = []
        for k, blk in enumerate(self.down):
            h = checkpoint(blk, h, e, use_reentrant=False)
            skips.append(h)
            h = F.avg_pool2d(h, 2)
            if k == 2:  # 16 km: the 0.25 deg conditioning
                g = self.grid16.expand(h.shape[0], -1, -1, -1).to(coarse.dtype)
                h = h + self.cproj(F.grid_sample(coarse, g, align_corners=True, padding_mode="border"))
        B, C, Hh, Ww = h.shape
        t = (h + self.pos).flatten(2).transpose(1, 2)
        for blk in self.attn:
            t = blk(t, e)
        h = t.transpose(1, 2).reshape(B, C, Hh, Ww)
        for blk, s in zip(self.up, reversed(skips)):
            h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = checkpoint(blk, torch.cat([h, s], 1), e, use_reentrant=False)
        return F.pixel_shuffle(self.out(h), 2)


class FullModule(pl.LightningModule):
    """shift > 1: resolution-shifted noise schedule (simple diffusion, Hoogeboom et al. 2023; SD3 timestep shift). With
    white noise on 29M pixels the 25 km structure is already visible at small noise levels; with the noise level
    sigma = 1 - tau mapped to s sigma / (1 + (s - 1) sigma), the SNR of the large scales matches that of a 168 x 280
    image for s = 25 (the linear resolution ratio). shift_mix: fraction of training samples drawn from the shifted
    schedule, the rest uniform (keeps the fine-detail end of the path trained). Sampling uses the quantiles of the same
    mixture as its time grid."""

    def __init__(self, lr=2e-4, epochs=10, reg=False, shift=1.0, shift_mix=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.net = FullNet(6 + 1)

    def draw_tau(self, B, device):
        u = torch.rand(B, device=device)
        s = self.hparams.get("shift", 1.0)
        if s == 1.0:
            return u
        sig = 1 - u
        shifted = 1 - s * sig / (1 + (s - 1) * sig)
        return torch.where(torch.rand(B, device=device) < self.hparams.get("shift_mix", 0.5), shifted, u)

    def tau_grid(self, steps):
        """steps + 1 flow times from 0 to 1 at the quantiles of the training distribution of tau."""
        s, mix = self.hparams.get("shift", 1.0), self.hparams.get("shift_mix", 0.5)
        q = np.linspace(0, 1, steps + 1)
        if s == 1.0:
            return q
        t = np.linspace(0, 1, 100001)
        y = 1 - t
        cdf = mix * (1 - y / (s - (s - 1) * y)) + (1 - mix) * t     # P(tau <= t) of the mixture
        return np.interp(q, cdf, t)

    def velocity(self, x, b, tau):
        return self.net(torch.cat([b["fine"].float(), x], 1), b["coarse"], tau, b["lead"])

    def step(self, b):
        x1, m, lead = b["x1"].float(), b["mask"].float(), b["lead"]
        if self.hparams.reg:
            pred = self.velocity(torch.zeros_like(x1), b, torch.ones_like(lead))
            return ((pred - x1) ** 2 * m).sum() / m.sum().clamp(min=1)
        x0 = torch.randn_like(x1)
        tau = self.draw_tau(x1.shape[0], x1.device)
        xt = (1 - tau[:, None, None, None]) * x0 + tau[:, None, None, None] * x1
        return (((self.velocity(xt, b, tau) - (x1 - x0)) ** 2) * m).sum() / m.sum().clamp(min=1)

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
    def sample(self, b, steps, generator=None):
        B, lead = b["fine"].shape[0], b["lead"]
        if self.hparams.reg:
            return self.velocity(torch.zeros(B, 1, PH, PW, device=lead.device), b, torch.ones_like(lead))
        x = torch.randn(B, 1, PH, PW, device=lead.device, generator=generator)
        grid = self.tau_grid(steps)
        for k in range(steps):
            x = x + self.velocity(x, b, torch.full((B,), float(grid[k]), device=lead.device)) * float(grid[k + 1] - grid[k])
        return x


class Timer(pl.Callback):
    """Seconds per training step and peak GPU memory (to price the full training from the smoke test)."""

    def on_train_epoch_start(self, trainer, *_):
        self.t, self.n = time.time(), 0

    def on_train_batch_end(self, trainer, *_):
        self.n += 1
        if trainer.global_rank == 0 and self.n % 50 == 0:   # wall time between steps, data loading included
            print(f"step {trainer.global_step}: {(time.time() - self.t) / self.n:.2f} s/step, peak "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)


def train(a):
    pl.seed_everything(a.seed, workers=True)
    tr, va = FullData([2017, 2018, 2019, 2020], train=True), FullData([2021], train=False)
    va.samples = va.samples[::6]
    if a.limit < 1:
        tr.samples = tr.samples[:int(len(tr.samples) * a.limit)]; va.samples = va.samples[:16]
    dl = DataLoader(tr, batch_size=1, shuffle=True, num_workers=a.workers, persistent_workers=True, pin_memory=True)
    dv = DataLoader(va, batch_size=1, num_workers=a.workers, persistent_workers=True, pin_memory=True)
    out = ROOT / "runs" / (f"{'reg' if a.reg else 'flow'}{f'_shift{a.shift:g}' if a.shift != 1 and not a.reg else ''}"
                           f"{'' if a.seed == 42 else f'_s{a.seed}'}{a.tag}")
    model = FullModule(lr=a.lr, epochs=a.epochs, reg=a.reg, shift=a.shift, shift_mix=a.shift_mix)
    trainer = pl.Trainer(accelerator="gpu", devices=int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)), strategy="ddp",
                         precision="bf16-mixed", max_epochs=a.epochs, gradient_clip_val=1.0, log_every_n_steps=10,
                         default_root_dir=str(out), logger=CSVLogger(str(out), name="logs"), callbacks=[Timer()])
    if trainer.global_rank == 0:
        print(f"{out.name}: train {len(tr)} val {len(va)} params {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    trainer.fit(model, dl, dv)
    trainer.save_checkpoint(out / "final.ckpt")


class FullPredictor:
    """Whole-Europe forecasts at 1 km, sampling ensemble members in bounded GPU batches."""

    def __init__(self, ckpt, dev, year=2022, weather="hres", members=4, steps=8, member_batch=1):
        if min(members, steps, member_batch) < 1:
            raise ValueError("members, steps and member_batch must be positive")
        self.dev, self.M, self.S = torch.device(dev), members, steps
        self.member_batch = member_batch
        self.model = FullModule.load_from_checkpoint(ckpt, map_location="cpu").to(dev).eval()
        self.ds = FullData([year], train=False)
        self.index = {(i, h): n for n, (i, h) in enumerate(self.ds.samples)}
        self.hres = None
        if weather == "hres":  # the 0.25 deg conditioning with the real HRES forecast block, from a plain DVData
            self.hres = DVData([year], "noise", train=False, t31=True)
            attach_hres(self.hres, True, False)
            self.hindex = {(i, h): n for n, (i, h) in enumerate(self.hres.samples)}
        self.off = int(np.where(self.ds.year == year)[0][0])
        self.gen = torch.Generator(device=dev); self.gen.manual_seed(0)

    @torch.no_grad()
    def forecast(self, d, hs):
        i, out = self.off + d, {}
        for h in hs:
            if (i, h) not in self.index:
                continue
            it = self.ds[self.index[(i, h)]]
            if self.hres is not None:
                it["coarse"] = self.hres[self.hindex[(i, h)]]["cond"][:69].contiguous()
            base = {k: it[k][None].to(self.dev) for k in ("fine", "coarse", "lead")}
            members = []
            for start in range(0, self.M, self.member_batch):
                count = min(self.member_batch, self.M - start)
                b = {k: v.expand(count, *([-1] * (v.ndim - 1))) for k, v in base.items()}
                with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                    y = self.model.sample(b, self.S, self.gen)[:, 0, :FH, :FW].float()
                members.append((y * SD + MU).cpu().numpy())
            out[h] = np.concatenate(members, axis=0)
        return out


def full_predictor(args, dev):
    """Plug-in for ../retrain/evaluate.py (1 km grid, EEA stations, 0.25 deg block means) + CRPS of the members."""
    from cranpm.data.europe_dataset_v2 import EuropeDatasetV2
    ds = EuropeDatasetV2([args.year], mode="fixed", horizons=args.horizons, train=False)
    ck = Path(args.ckpt)
    fp = FullPredictor(ck / "final.ckpt" if ck.is_dir() else ck, dev, year=args.year, weather=args.weather,
                       members=args.members, steps=args.steps2, member_batch=args.member_batch)
    extra = {k: np.zeros(max(args.horizons) + 1) for k in ("crps_n", "crps", "spread2")}

    def predict(d, met_day, hs, g_t, g_p):
        out = {}
        for h, E in fp.forecast(d, hs).items():
            tgt = np.asarray(fp.ds.fine[fp.off + d + h], np.float32)
            m = tgt > 0
            T, Em = tgt[m], E[:, m]
            M = Em.shape[0]
            pair = sum(np.abs(Em[a] - Em[b]) for a in range(M) for b in range(M)) / max(M * (M - 1), 1)
            extra["crps_n"][h] += m.sum(); extra["crps"][h] += (np.abs(Em - T).mean(0) - 0.5 * pair).sum()
            extra["spread2"][h] += Em.var(0).sum()
            out[h] = E.mean(0).astype(np.float32)
        return out

    predict.extra = extra
    return ds, predict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--reg", action="store_true", help="ablation: deterministic regression with the same network")
    ap.add_argument("--shift", type=float, default=1.0, help="resolution shift of the noise schedule (1 = none)")
    ap.add_argument("--shift-mix", type=float, default=0.5, help="fraction of training times drawn from the shifted schedule")
    ap.add_argument("--limit", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    train(ap.parse_args())
