#!/usr/bin/env python3
"""Retrain the paper's deep-learning baselines (ClimaX, Earthformer, ConvLSTM, SimVP, TopoFlow) on the
corrected inputs, so that they are compared with CRAN-PM V1/V2 on identical data.

Same models, grid (0.25 deg, 168x280), loss and optimiser as topoflow_europe/baselines/train_baseline.py
and train_topoflow_baseline.py. Only the inputs change: the corrected 70-channel global input of
EuropeDatasetV2(mode="fixed") (ERA5 standardised once, EAC4 CAMS), plus the GHAP block means on
the ERA5 grid (ghap_025deg/v1) for today (input) and t+h (target). Scored at 1 km by evaluate.py.
    python train_baselines.py --model simvp --epochs 100
--met-day today|target appends one more weather block (30 ERA5 + 7 extra channels: BLH, rain, dewpoint, radiation,
cloud) of the start day (today) or of the forecast day t+h (target, "perfect forecast"): same channels and
network, only the day differs, to measure what the forecast-period weather is worth.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

sys.path.insert(0, "/scratch/project_462001140/ammar/eccv/topoflow_europe")
from baselines.climax_baseline import ClimaXBaseline, ClimaXBaselineV2  # noqa: E402
from baselines.convlstm_baseline import ConvLSTMBaseline  # noqa: E402
from baselines.earthformer_baseline import EarthformerBaseline  # noqa: E402
from baselines.simvp_baseline import SimVPBaseline  # noqa: E402
from baselines.topoflow_baseline import TopoFlowBaseline  # noqa: E402

from cranpm.data.europe_dataset_v2 import (  # noqa: E402
    ELEV_MEAN, ELEV_STD, ERA5_C0, ERA5_R0, GHAP_MEAN, GHAP_STD, EuropeDatasetV2)

ROOT = Path(__file__).resolve().parent
CH, CW = 168, 280
LAND = -GHAP_MEAN / GHAP_STD
N_MET = 37  # extra weather block: 30 ERA5 + 7 extra channels
MODELS = {
    "climax": lambda c: ClimaXBaselineV2(   # V1 (ClimaXBaseline) averaged all input variables: see climax_baseline.py
                                         in_channels=c, img_size=(CH, CW), patch_size=8, embed_dim=768, depth=8, num_heads=12, mlp_ratio=4.0,
                                     drop_rate=0.1, drop_path=0.1),
    "earthformer": lambda c: EarthformerBaseline(in_channels=c, img_size=(CH, CW), patch_size=8, embed_dim=768, depth=8, num_heads=12,
                                               mlp_ratio=4.0, drop_rate=0.1, drop_path=0.1, window_size=(3, 5)),
    "convlstm": lambda c: ConvLSTMBaseline(in_channels=c, img_size=(CH, CW), hidden_channels=(128, 256), kernel_size=3, drop_rate=0.1),
    "simvp": lambda c: SimVPBaseline(in_channels=c, img_size=(CH, CW), hidden_channels=256, translator_layers=8, reduction=4, drop_rate=0.1),
    "topoflow": lambda c: TopoFlowBaseline(in_channels=c, img_size=(CH, CW), patch_size=8, embed_dim=768, depth=8, num_heads=12, mlp_ratio=4.0,
                                         drop_rate=0.1, drop_path=0.1, out_channels=1),
}
UNDERESTIMATE = {"topoflow": 2.0}  # train_topoflow_baseline.py; the others use 1.0 (train_baseline.py)


class CoarseDatasetV2(EuropeDatasetV2):
    """The baselines' 0.25 deg sample: CRAN-PM global input + GHAP block means (sea -> 0, as in the paper)."""
    met_day = "none"

    def met_block(self, y, s):
        """Standardised ERA5 (30) + extra (7) fields of day s, (37, 168, 280); target31 keeps ERA5 + rain only (the
        variables an operational forecast such as IFS HRES provides)."""
        era5, _ = self._era5(y, s)
        ex = self._extra(y, s)
        blk = np.concatenate([era5, ex[3:4] if self.met_day == "target31" else ex])
        return blk[:, :CH, :CW].astype(np.float32)

    def ghap_coarse(self, y, d):
        g = np.asarray(self.ghap025[y][d, 0, ERA5_R0:ERA5_R0 + CH], np.float32)
        g = np.concatenate([g[:, ERA5_C0:], g[:, :CW - (g.shape[1] - ERA5_C0)]], axis=1)
        return ((np.nan_to_num(g, nan=0.0) - GHAP_MEAN) / GHAP_STD).astype(np.float32)

    def global_input(self, y, d):
        era5_t, _ = self._era5(y, d)
        era5_p, _ = self._era5(y, d - 1)
        x = np.concatenate([era5_t, self._cams(y, d), era5_p, self._cams(y, d - 1)])
        return x[:, :CH, :CW].astype(np.float32)

    def __getitem__(self, idx):
        y, d, h = self.samples[idx]
        era5, g_t, tgt = self.global_input(y, d), self.ghap_coarse(y, d), self.ghap_coarse(y, d + h)
        if self.met_day != "none":
            era5 = np.concatenate([era5, self.met_block(y, d if self.met_day == "today" else d + h)])
        elev = ((self.elev_coarse[:CH, :CW] - ELEV_MEAN) / ELEV_STD).astype(np.float32)
        if self.augment and np.random.random() < 0.5:  # horizontal flip, as in the paper's baseline dataset
            era5, g_t, tgt, elev = era5[..., ::-1], g_t[..., ::-1], tgt[..., ::-1], elev[..., ::-1]
        return {"era5": torch.from_numpy(era5.copy()), "ghap_coarse": torch.from_numpy(g_t.copy())[None],
                "elevation_coarse": torch.from_numpy(elev.copy()), "lead_time": torch.tensor(float(h)),
                "target": torch.from_numpy(tgt.copy())[None]}


class BaselineModule(pl.LightningModule):
    def __init__(self, name, lr=5e-5, weight_decay=0.05, warmup=5, epochs=100, met_day="none"):
        super().__init__()
        self.save_hyperparameters()
        self.model = MODELS[name](71 + {"none": 0, "target31": 31}.get(met_day, N_MET))
        self.penalty = UNDERESTIMATE.get(name, 1.0)
        for m in self.model.modules():  # TopoFlow's elevation_scale is in metres, the elevation input is standardised
            if hasattr(m, "elevation_scale"):
                m.elevation_scale = m.elevation_scale / ELEV_STD

    def forward(self, batch):
        return self.model(batch["era5"], batch["ghap_coarse"], batch["elevation_coarse"], batch["lead_time"])

    def _loss(self, pred, target):
        land = (target > LAND) & torch.isfinite(target)
        r = pred[land].float() - target[land].float()
        w = torch.where(r < 0, torch.full_like(r, self.penalty), torch.ones_like(r))
        return (w * r ** 2).sum() / land.sum().clamp(min=1)

    def training_step(self, batch, _):
        loss = self._loss(self(batch), batch["target"])
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        pred, tgt = self(batch).float(), batch["target"].float()
        land = (tgt > LAND) & torch.isfinite(tgt)
        self.log("val/loss", self._loss(pred, tgt), sync_dist=True)
        self.log("val/rmse", ((pred[land] - tgt[land]) ** 2).mean().sqrt() * GHAP_STD, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        hp = self.hparams
        opt = torch.optim.AdamW(self.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

        def lr_lambda(epoch):
            if epoch < hp.warmup:
                return (epoch + 1) / hp.warmup
            return 0.5 * (1 + math.cos(math.pi * (epoch - hp.warmup) / max(1, hp.epochs - hp.warmup)))

        return [opt], [{"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda), "interval": "epoch"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit-train", type=float, default=1.0)
    ap.add_argument("--limit-val", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--met-day", choices=["none", "today", "target", "target31"], default="none")
    args = ap.parse_args()

    pl.seed_everything(args.seed, workers=True)
    extra = args.met_day != "none"
    train_ds = CoarseDatasetV2([2017, 2018, 2019, 2020], mode="fixed", train=True, augment=True, extra=extra)
    val_ds = CoarseDatasetV2([2021], mode="fixed", train=False, day_stride=3, extra=extra)
    train_ds.met_day = val_ds.met_day = args.met_day
    nw = int(os.environ.get("SLURM_CPUS_PER_TASK", 7)) - 1
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=True,
                          persistent_workers=True, drop_last=True, prefetch_factor=4)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True,
                        persistent_workers=True)

    met = "" if args.met_day == "none" else f"_met{args.met_day}"
    out = ROOT / "runs" / f"B_{args.model}{met}{'' if args.seed == 42 else f'_s{args.seed}'}{args.tag}"
    model = BaselineModule(args.model, warmup=args.warmup, epochs=args.epochs, met_day=args.met_day)
    trainer = pl.Trainer(
        accelerator="gpu", devices=int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)),
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)), strategy="ddp_find_unused_parameters_true",
        precision="bf16-mixed", max_epochs=args.epochs, gradient_clip_val=1.0,
        limit_train_batches=args.limit_train, limit_val_batches=args.limit_val, log_every_n_steps=10,
        default_root_dir=str(out), logger=CSVLogger(str(out), name="logs"),
        callbacks=[ModelCheckpoint(dirpath=str(out / "ckpt"), monitor="val/rmse", mode="min", save_top_k=1,
                                   save_last=True, filename="best-{epoch:03d}"), LearningRateMonitor()],
    )
    if trainer.global_rank == 0:
        print(f"{args.model}: met_day={args.met_day} train={len(train_ds)} val={len(val_ds)} "
              f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    trainer.fit(model, train_dl, val_dl, ckpt_path="last" if (out / "ckpt/last.ckpt").exists() else None)


if __name__ == "__main__":
    main()
