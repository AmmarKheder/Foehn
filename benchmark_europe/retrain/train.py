#!/usr/bin/env python3
"""Retrain CRAN-PM on the paper's Europe data. Wind-prior ablation, all on the corrected inputs except V0:
  V0   paper pipeline (input bugs reproduced), paper model
  V1   paper wind ordering of the global tokens, from the physical 10 m wind
  V1r  no wind: tokens kept in raster order (same order embedding)
  V1s  control: wind ordering from the wind of a day 15 days away (the ERA5 input stays the true one)
  V2   upwind attention: the wind biases who attends to whom (models/upwind.py), no reordering
  V2s  control: upwind attention with the wind of a day 15 days away
If the wind prior carries physics, V1 < V1s and V2 < V2s; V1 ~ V1r ~ V1s means the ordering is inert.
Data extension (not in the paper's inputs): V1x / V2x = V1 / V2 + boundary-layer height, precipitation,
dewpoint, solar radiation and cloud cover at t and t-1 (84 global channels instead of 70).

Same model/loss/optimiser settings as configs/europe_multiscale.yaml; only the data mode and the
global-branch attention differ. Usage (one node, 8 GCDs, launched by srun):
    python train.py --variant V1 --epochs 30 [--limit-train 20 --limit-val 5]
"""
import argparse
import copy
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from cranpm.data.europe_dataset_v2 import ELEV_STD, EuropeDatasetV2
from cranpm.models.upwind import MultiScaleTopoFlowUpwind, UpwindAttention
from cranpm.models.wind_order import MultiScaleTopoFlowWind
from cranpm.training.lightning_module import MultiScaleTopoFlowLightning

ROOT = Path(__file__).resolve().parent
GATE_LR = 1e-3
CONFIG = ROOT.parents[1] / "configs/europe_multiscale.yaml"
# name: (data mode, global-branch arch, token order, wind_shift in days, extra meteorology)
VARIANTS = {"V0": ("paper", "reorder", "wind", 0, False),
            "V1": ("fixed", "order", "wind", 0, False), "V1r": ("fixed", "order", "raster", 0, False),
            "V1s": ("fixed", "order", "wind", 15, False),
            "V2": ("fixed", "upwind", None, 0, False), "V2s": ("fixed", "upwind", None, 15, False),
            "V1x": ("fixed", "order", "wind", 0, True), "V2x": ("fixed", "upwind", None, 0, True)}
N_EXTRA = 7


def model_kwargs(mc):
    return dict(
        era5_channels=mc["era5_channels"], global_img_size=tuple(mc["global_img_size"]),
        global_patch_size=mc["global_patch_size"], global_embed_dim=mc["global_embed_dim"],
        global_depth=mc["global_depth"], global_num_heads=mc["global_num_heads"],
        local_channels=mc["local_channels"], local_img_size=tuple(mc["local_img_size"]),
        local_patch_size=mc["local_patch_size"], local_embed_dim=mc["local_embed_dim"],
        local_depth=mc["local_depth"], local_num_heads=mc["local_num_heads"],
        cross_num_heads=mc["cross_num_heads"], cross_layers=mc["cross_layers"],
        decoder_depth=mc["decoder_depth"], out_channels=mc["out_channels"], mlp_ratio=mc["mlp_ratio"],
        drop_rate=mc["drop_rate"], drop_path=mc["drop_path"],
        global_region_h=mc["global_region_h"], global_region_w=mc["global_region_w"])


class RetrainModule(MultiScaleTopoFlowLightning):
    def __init__(self, config):
        super().__init__(config)
        mc = config["model"]
        if mc["arch"] in ("upwind", "order"):
            if mc["arch"] == "upwind":
                self.model = MultiScaleTopoFlowUpwind(**model_kwargs(mc))
            else:
                self.model = MultiScaleTopoFlowWind(shuffle_mode=mc["order"], **model_kwargs(mc))
            n_t = 35 + (N_EXTRA if config["data"].get("extra") else 0)  # [ERA5, CAMS(, extra)] at t, then at t-1
            with torch.no_grad():  # same zero-init of the t-1 block as the paper's module (NOT the forecast block)
                k = mc["global_patch_size"] ** 2
                self.model.global_branch.patch_embed.weight[:, n_t * k:2 * n_t * k] = 0.0
                k = mc["local_patch_size"] ** 2
                self.model.local_branch.patch_embed.weight[:, 4 * k:] = 0.0
        if config["data"]["mode"] == "fixed":
            # TopoFlow attention divides elevation differences by elevation_scale, given in metres (1000 global,
            # 500 local), but both elevation inputs are standardised ((z - 300) / 500): in the paper the terrain
            # bias was ~500x too weak (-0.008 instead of -4 for a 2 km rise). Express the scales in input units.
            for m in self.model.modules():
                if hasattr(m, "elevation_scale"):
                    m.elevation_scale = m.elevation_scale / ELEV_STD

    def configure_optimizers(self):
        """Paper optimiser and schedule, except (fixed mode) for the scalar strengths of the attention priors
        (TopoFlow elevation alpha, cross-attention wind beta, upwind beta): in the paper checkpoint they sit at
        their initial values minus weight decay (1.0 -> 0.993-1.001, 2.0 -> 1.993 after 18 epochs), i.e. at
        lr 5e-5 they never moved. They get lr 1e-3 and no weight decay, so what they learn is readable."""
        if self.config["data"]["mode"] != "fixed":
            return super().configure_optimizers()
        tc = self.config["train"]
        named = list(self.model.named_parameters())
        gates = [p for n, p in named if n.rsplit(".", 1)[-1] in ("alpha", "beta")]
        ids = {id(p) for p in gates}
        opt = torch.optim.AdamW([{"params": [p for _, p in named if id(p) not in ids]},
                                 {"params": gates, "lr": GATE_LR, "weight_decay": 0.0}],
                                lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
        w = tc["warmup_epochs"]
        sched = torch.optim.lr_scheduler.SequentialLR(opt, milestones=[w], schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=w),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"] - w, eta_min=tc.get("min_lr", 1e-6))])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    def on_validation_epoch_end(self):
        for name, m in self.model.named_modules():  # learned strength of every attention prior
            for attr in ("alpha", "beta"):
                p = getattr(m, attr, None)
                if isinstance(p, torch.nn.Parameter):
                    v = F.softplus(p) if isinstance(m, UpwindAttention) else p
                    self.log(f"prior/{name}.{attr}", v.detach().float().mean(), sync_dist=True)

    def forward(self, batch):
        kw = dict(era5=batch["era5"], elevation_coarse=batch["elevation_coarse"], ghap_patch=batch["local_input"],
                  elevation_hires=batch["elevation_hires"], lead_time=batch["lead_time"],
                  patch_center=batch.get("patch_center"), wind_at_patch=batch.get("wind_at_patch"))
        if isinstance(self.model, (MultiScaleTopoFlowUpwind, MultiScaleTopoFlowWind)):
            kw["wind"] = batch["wind"]
        return self.model(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--limit-train", type=float, default=1.0)
    ap.add_argument("--limit-val", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--fcst", action="store_true", help="forecast-day weather block (HRES variables) in the global input")
    ap.add_argument("--no-station", action="store_true", help="drop the EEA station term from the loss (fair benchmark)")
    args = ap.parse_args()

    pl.seed_everything(args.seed, workers=True)
    mode, arch, order, wind_shift, extra = VARIANTS[args.variant]
    cfg = copy.deepcopy(yaml.safe_load(CONFIG.read_text()))
    cfg["model"].update(arch=arch, order=order)
    if extra:
        cfg["model"]["era5_channels"] += 2 * N_EXTRA
    cfg["data"].update(wind_shift=wind_shift, extra=extra, fcst=args.fcst)
    if args.fcst:
        cfg["model"]["era5_channels"] += 31
    if args.no_station:
        cfg["train"]["alpha_station"] = 0.0
    cfg["train"].update(epochs=args.epochs, warmup_epochs=args.warmup)
    cfg["data"]["mode"] = mode

    d = cfg["data"]
    train_ds = EuropeDatasetV2(d["train_years"], mode=mode, horizons=d["horizons"], train=True, augment=d["augment"],
                               hotspot_ratio=d["hotspot_ratio"], hotspot_power=d["hotspot_power"], wind_shift=wind_shift,
                               extra=extra, fcst=args.fcst)
    val_ds = EuropeDatasetV2(d["val_years"], mode=mode, horizons=d["horizons"], train=False, day_stride=3,
                             wind_shift=wind_shift, extra=extra, fcst=args.fcst)
    nw = int(os.environ.get("SLURM_CPUS_PER_TASK", 7)) - 1
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=nw,
                          pin_memory=True, persistent_workers=True, drop_last=True, prefetch_factor=4)
    val_dl = DataLoader(val_ds, batch_size=cfg["train"]["val_batch_size"], shuffle=False, num_workers=nw,
                        pin_memory=True, persistent_workers=True)

    sfx = ("_fcst" if args.fcst else "") + ("_nost" if args.no_station else "")
    out = ROOT / "runs" / f"{args.variant}{sfx}{'' if args.seed == 42 else f'_s{args.seed}'}{args.tag}"
    model = RetrainModule(cfg)
    trainer = pl.Trainer(
        accelerator="gpu", devices=int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)),
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)), strategy="ddp", precision=cfg["train"]["precision"],
        max_epochs=args.epochs, gradient_clip_val=cfg["train"]["gradient_clip_val"],
        limit_train_batches=args.limit_train, limit_val_batches=args.limit_val,
        log_every_n_steps=cfg["train"]["log_every_n_steps"], default_root_dir=str(out),
        # TensorBoard first: the paper's validation_step logs prediction maps through logger.experiment.add_image
        logger=[TensorBoardLogger(str(out), name="tb"), CSVLogger(str(out), name="logs")],
        callbacks=[ModelCheckpoint(dirpath=str(out / "ckpt"), monitor="val/rmse", mode="min", save_top_k=1,
                                   save_last=True, filename="best-{epoch:03d}"),
                   LearningRateMonitor()],
    )
    if trainer.global_rank == 0:
        print(f"{args.variant}: mode={mode} arch={arch} order={order} wind_shift={wind_shift} extra={extra} seed={args.seed} train={len(train_ds)} val={len(val_ds)} "
              f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    trainer.fit(model, train_dl, val_dl, ckpt_path="last" if (out / "ckpt/last.ckpt").exists() else None)


if __name__ == "__main__":
    main()
