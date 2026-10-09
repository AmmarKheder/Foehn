#!/usr/bin/env python3
"""Regional equal-resolution study: every model is trained natively at 1 km on the same window (Paris - Florence,
Lyon - Venice: 49.5-41.82 N, 0-15.36 E, 768 x 1536 px) with the same inputs and the same target.

Inputs (identical for all models):
  1 km : GHAP today, GHAP yesterday (normalised, sea -> 0) + their validity masks, elevation, land;
  0.25 deg (../dejavu conditioning, bilinearly interpolated to 1 km for the baselines, injected at 16 km for the flow):
         PM today, weather today (37), forecast-day weather (the 31 HRES variables; ERA5 in training, HRES at test).
Target: GHAP 1 km of day d + h (h = 1..4). Train 2017-2020, validation 2021, test 2022. No station supervision.
Models: the baselines of ../retrain/train_baselines.py (same architectures, img_size = window) and our whole-image flow
(../fullimage/fullflow.py FullNet, window-sized).
    python regional.py train --model earthformer|climax|simvp|flow|flowreg [--epochs 30]      (srun, one node)
    evaluation: ../retrain/evaluate.py --regional runs/<model> --region --weather hres
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
sys.path.insert(0, str(EXP / "dejavu"))
sys.path.insert(0, str(EXP / "fullimage"))
sys.path.insert(0, str(EXP / "retrain"))
from dejavu import DVData, attach_hres, emb, norm, MU, SD  # noqa: E402
import fullflow  # noqa: E402
from build_ghap1km import F as FINE_FILE, H as FH, W as FW  # noqa: E402

R0, C0, NR, NC = 2250, 2500, 768, 1536     # window in the Europe 1 km grid (rows from 72 N, cols from 25 W)
N_COND = 69                                # PM today + weather today (37) + forecast-day block (31), 0.25 deg


def header(path):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); return f.tell()


PM5 = EXP / "cascade" / "data" / "pm5km.npy"   # PM today at 5 km over the whole of Europe (footprint source)


class RegionData(DVData):
    def __init__(self, years, train, footprint=False):
        super().__init__(years, "noise", train, t31=True, footprint=footprint, fp_met="stab")
        self.fp = footprint
        self.hdr5 = header(PM5) if footprint else None
        valid = np.load(EXP / "dejavu" / "data" / "analogs.npz")["valid"]
        self.samples = [(i, h) for i, h in self.samples if valid[i - 1]]
        self.hdr = header(FINE_FILE)
        st = np.load(EXP / "fullimage" / "data" / "static1km.npz")
        self.static = np.stack([st["elev"][R0:R0 + NR, C0:C0 + NC], st["land"][R0:R0 + NR, C0:C0 + NC]]).astype(np.float32)

    def day(self, d):
        a = np.memmap(FINE_FILE, dtype=np.float16, mode="r", offset=self.hdr + d * FH * FW * 2, shape=(FH, FW))
        a = np.asarray(a[R0:R0 + NR, C0:C0 + NC], np.float32)
        ok = a > 0
        return np.stack([norm(np.where(ok, a, np.nan)), ok]).astype(np.float32)

    def __getitem__(self, idx):
        it = super().__getitem__(idx)
        i, h = self.samples[idx]
        tgt = self.day(i + h)
        out = {"fine": torch.from_numpy(np.concatenate([self.day(i), self.day(i - 1), self.static])),   # (6, NR, NC)
               "coarse": it["cond"][:N_COND].contiguous(), "x1": torch.from_numpy(tgt[:1]),
               "mask": torch.from_numpy(tgt[1:]), "lead": it["lead"]}
        if self.fp:
            src = np.asarray(np.memmap(PM5, dtype=np.float16, mode="r", offset=self.hdr5 + i * 838 * 1398 * 2, shape=(838, 1398)), np.float32)
            out.update(src=torch.from_numpy(np.stack([norm(src), np.isfinite(src)]).astype(np.float16)),
                       wind=it["wind"], rain=it["rain"], stab=it["stab"])
        return out


def coarse_grid(n_r, n_c, stride, device):
    """grid_sample coordinates (align_corners=True, 168 x 280 grid of 0.25 deg points) of the centres of the stride x
    stride blocks of the window: pixel (R0 + r, C0 + c) centre = 0.25 deg coordinate ((R0 + r + 0.5) / 25, ...)."""
    r = (R0 + torch.arange(n_r, device=device) * stride + stride / 2.0) / 25.0 * (2.0 / 167) - 1
    c = (C0 + torch.arange(n_c, device=device) * stride + stride / 2.0) / 25.0 * (2.0 / 279) - 1
    return torch.stack([c[None].expand(n_r, -1), r[:, None].expand(-1, n_c)], -1)[None]


def baseline(name, in_channels):
    import train_baselines as TB
    kw = dict(in_channels=in_channels, img_size=(NR, NC))
    return {"climax": lambda: TB.ClimaXBaselineV2(**kw, patch_size=8, embed_dim=768, depth=8, num_heads=12, mlp_ratio=4.0, drop_rate=0.1, drop_path=0.1),
            "earthformer": lambda: TB.EarthformerBaseline(**kw, patch_size=8, embed_dim=768, depth=8, num_heads=12, mlp_ratio=4.0, drop_rate=0.1,
                                                          drop_path=0.1, window_size=(3, 5)),
            "simvp": lambda: TB.SimVPBaseline(**kw, hidden_channels=256, translator_layers=8, reduction=4, drop_rate=0.1)}[name]()


class RegionalModule(pl.LightningModule):
    """model in {earthformer, climax, simvp}: deterministic baseline (MSE on land, as train_baselines.py);
    flow / flowreg: our whole-window network, flow matching / the same network by regression."""

    def __init__(self, model, lr=None, epochs=30):
        super().__init__()
        self.save_hyperparameters()
        self.fp = None
        if model in ("flow", "flowreg", "flowfp"):
            fullflow.PH, fullflow.PW = NR, NC
            if model == "flowfp":   # 1 km footprint: blocks of 4 km gather PM today (5 km, whole Europe) along the forecast wind
                sys.path.insert(0, str(EXP / "cascade"))
                from cascade import Footprint1km
                self.fp = Footprint1km()
            self.net = fullflow.FullNet(6 + (7 if model == "flowfp" else 0) + 1, ccoarse=N_COND)
            self.net.grid16 = coarse_grid(NR // 16, NC // 16, 16, "cpu")   # window-aware 16 km -> 0.25 deg mapping
        else:
            # era5 slot: the 0.25 deg conditioning at 1 km (69) + yesterday, the two masks and land at 1 km (4);
            # ghap slot: GHAP today at 1 km; elevation slot: 1 km elevation. in_channels counts the GHAP channel.
            self.net = baseline(model, N_COND + 4 + 1)
        self.grid1 = None

    def baseline_inputs(self, b):
        if self.grid1 is None or self.grid1.device != b["fine"].device:
            self.grid1 = coarse_grid(NR, NC, 1, b["fine"].device)
        B = b["fine"].shape[0]
        up = F.grid_sample(b["coarse"].float(), self.grid1.expand(B, -1, -1, -1), align_corners=True, padding_mode="border")
        f = b["fine"].float()
        era5 = torch.cat([up, f[:, 2:3], f[:, 1:2], f[:, 3:4], f[:, 5:6]], 1)
        return era5, f[:, 0:1], f[:, 4], b["lead"].float()

    def fine_inputs(self, b):
        """1 km conditioning: the fine fields, plus the footprint features (flowfp) upsampled from 4 km."""
        f = b["fine"].float()
        if self.fp is None:
            return f
        B, dev = f.shape[0], f.device
        rr = R0 + torch.arange(NR // 4, device=dev, dtype=torch.float32) * 4 + 2.0
        cc = C0 + torch.arange(NC // 4, device=dev, dtype=torch.float32) * 4 + 2.0
        lat = (72.0 - 0.01 * rr)[None, :, None].expand(B, -1, NC // 4).contiguous()
        lon = (-25.0 + 0.01 * cc)[None, None, :].expand(B, NR // 4, -1).contiguous()
        with torch.autocast(device_type=dev.type, enabled=False):
            g = self.fp(b["src"], b["wind"], b["rain"], b["stab"], b["lead"], lat, lon)
        return torch.cat([f, F.interpolate(g, size=(NR, NC), mode="bilinear", align_corners=False)], 1)

    def forward_det(self, b):
        if self.hparams.model == "flowreg":
            x = torch.zeros(b["fine"].shape[0], 1, NR, NC, device=b["fine"].device)
            return self.net(torch.cat([b["fine"].float(), x], 1), b["coarse"].float(), torch.ones_like(b["lead"]), b["lead"])
        return self.net(*self.baseline_inputs(b))

    def step(self, b):
        x1, m = b["x1"].float(), b["mask"].float()
        if self.hparams.model in ("flow", "flowfp"):
            x0 = torch.randn_like(x1)
            tau = torch.rand(x1.shape[0], device=x1.device)
            xt = (1 - tau[:, None, None, None]) * x0 + tau[:, None, None, None] * x1
            v = self.net(torch.cat([self.fine_inputs(b), xt], 1), b["coarse"].float(), tau, b["lead"])
            return ((v - (x1 - x0)) ** 2 * m).sum() / m.sum().clamp(min=1)
        return ((self.forward_det(b).float() - x1) ** 2 * m).sum() / m.sum().clamp(min=1)

    def training_step(self, b, _):
        loss = self.step(b); self.log("train/loss", loss, prog_bar=True); return loss

    def validation_step(self, b, _):
        self.log("val/loss", self.step(b), sync_dist=True)

    def configure_optimizers(self):
        hp = self.hparams
        if hp.model in ("flow", "flowreg"):  # the recipe of ../fullimage/fullflow.py
            lr = hp.lr or 2e-4
            opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=self.trainer.estimated_stepping_batches, pct_start=0.05)
            return [opt], [{"scheduler": sched, "interval": "step"}]
        # the baselines keep their own recipe (../retrain/train_baselines.py): AdamW 5e-5, wd 0.05, warmup + cosine
        opt = torch.optim.AdamW(self.parameters(), lr=hp.lr or 5e-5, weight_decay=0.05)
        warm = max(1, hp.epochs // 12)
        f = lambda e: (e + 1) / warm if e < warm else 0.5 * (1 + np.cos(np.pi * (e - warm) / max(1, hp.epochs - warm)))
        return [opt], [{"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, f), "interval": "epoch"}]

    @torch.no_grad()
    def predict(self, b, members=4, steps=8, generator=None):
        """-> (members, NR, NC) normalised; deterministic models return one member."""
        if self.hparams.model not in ("flow", "flowfp"):
            return self.forward_det(b)[:, 0].float()
        B = b["fine"].shape[0]
        fin = self.fine_inputs(b)
        x = torch.randn(B, 1, NR, NC, device=b["fine"].device, generator=generator)
        for k in range(steps):
            tau = torch.full((B,), k / steps, device=x.device)
            x = x + self.net(torch.cat([fin, x], 1), b["coarse"].float(), tau, b["lead"]) / steps
        return x[:, 0].float()


class Timer(pl.Callback):
    def on_train_epoch_start(self, trainer, *_):
        self.t, self.n = time.time(), 0

    def on_train_batch_end(self, trainer, *_):
        self.n += 1
        if trainer.global_rank == 0 and self.n % 25 == 0:
            print(f"step {trainer.global_step}: {(time.time() - self.t) / self.n:.2f} s/step, peak "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)


def train(a):
    pl.seed_everything(a.seed, workers=True)
    fp = a.model == "flowfp"
    tr, va = RegionData([2017, 2018, 2019, 2020], train=True, footprint=fp), RegionData([2021], train=False, footprint=fp)
    va.samples = va.samples[::6]
    if a.limit < 1:
        tr.samples = tr.samples[:int(len(tr.samples) * a.limit)]; va.samples = va.samples[:16]
    dl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, persistent_workers=True, pin_memory=True, drop_last=True)
    dv = DataLoader(va, batch_size=a.batch, num_workers=a.workers, persistent_workers=True, pin_memory=True)
    out = ROOT / "runs" / f"{a.model}{'' if a.seed == 42 else f'_s{a.seed}'}{a.tag}"
    module = RegionalModule(a.model, lr=a.lr, epochs=a.epochs)
    trainer = pl.Trainer(accelerator="gpu", devices=int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)), strategy="ddp",
                         precision="bf16-mixed", max_epochs=a.epochs, gradient_clip_val=1.0, log_every_n_steps=10,
                         default_root_dir=str(out), logger=CSVLogger(str(out), name="logs"), callbacks=[Timer()])
    if trainer.global_rank == 0:
        print(f"{out.name}: train {len(tr)} val {len(va)} params {sum(p.numel() for p in module.parameters())/1e6:.1f}M", flush=True)
    trainer.fit(module, dl, dv)
    trainer.save_checkpoint(out / "final.ckpt")


def regional_predictor(args, dev):
    """Plug-in for ../retrain/evaluate.py: full-size 1 km maps, the window filled, zeros elsewhere (use --region)."""
    from cranpm.data.europe_dataset_v2 import EuropeDatasetV2
    ck = Path(args.ckpt)
    module = RegionalModule.load_from_checkpoint(ck / "final.ckpt" if ck.is_dir() else ck, map_location="cpu").to(dev).eval()
    fpm = module.hparams.model == "flowfp"
    rd = RegionData([args.year], train=False, footprint=fpm)
    index = {(i, h): n for n, (i, h) in enumerate(rd.samples)}
    hres = None
    if args.weather == "hres":  # the 0.25 deg conditioning (and footprint winds) from the real HRES forecast, plain DVData
        hres = DVData([args.year], "noise", train=False, t31=True, footprint=fpm, fp_met="stab")
        attach_hres(hres, True, fpm)
        hindex = {(i, h): n for n, (i, h) in enumerate(hres.samples)}
    off = int(np.where(rd.year == args.year)[0][0])
    ds = EuropeDatasetV2([args.year], mode="fixed", horizons=args.horizons, train=False)
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    extra = {k: np.zeros(max(args.horizons) + 1) for k in ("crps_n", "crps", "spread2")}
    flow = module.hparams.model in ("flow", "flowfp")
    M = args.members if flow else 1

    def predict(d, met_day, hs, g_t, g_p):
        out = {}
        for h in hs:
            if (off + d, h) not in index:
                continue
            it = rd[index[(off + d, h)]]
            if hres is not None:
                ih = hres[hindex[(off + d, h)]]
                it["coarse"] = ih["cond"][:N_COND].contiguous()
                if fpm:
                    it.update(wind=ih["wind"], rain=ih["rain"], stab=ih["stab"])
            keys = ("fine", "coarse", "lead") + (("src", "wind", "rain", "stab") if fpm else ())
            b = {k: it[k][None].to(dev).expand(M, *([-1] * it[k].dim())) for k in keys}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                E = module.predict(b, steps=args.steps2, generator=gen).float().cpu().numpy() * SD + MU     # (M, NR, NC)
            t = it["x1"][0].numpy() * SD + MU
            m = it["mask"][0].numpy() > 0
            if flow:
                from evaluate import crps_members
                extra["crps_n"][h] += m.sum(); extra["crps"][h] += crps_members(E[:, m], t[m]); extra["spread2"][h] += E[:, m].var(0).sum()
            full = np.zeros((FH, FW), np.float32)
            full[R0:R0 + NR, C0:C0 + NC] = E.mean(0)
            out[h] = full
        return out

    predict.extra = extra
    return ds, predict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--model", required=True, choices=["earthformer", "climax", "simvp", "flow", "flowreg", "flowfp"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None, help="default: 2e-4 (flow), 5e-5 (baselines, their recipe)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    train(ap.parse_args())
