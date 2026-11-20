#!/usr/bin/env python3
"""RT 1.2: fine-tuning of Foehn-CAMS (foehn_cams.py: same network, flow-matching loss, shifted schedule, optimiser)
on day directories built from the live inputs (realtime/data/<date>/: cams.npy, hres_B.npy, fires.npy, green/), the
weights initialised from the paper's final.ckpt. Targets: the CAMS analyses of the valid times, read from the day
directories of the following days (00 UTC of d+j in data/<d+j>/cams.npy[1], 12 UTC in data/<d+j+1>/cams.npy[0]).
    srun python finetune.py --first 2026-01-01 --last 2026-08-31 --epochs 6 --lr 5e-5 [--replay 0.3] [--tag x] [--smoke 30]
Output: realtime/runs/rt12<tag>/ (Lightning logs, ckpts/, final.ckpt)."""
import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))                       # global1km
import foehn_cams as FC                                                                 # noqa: E402
from foehn_cams import (CMETA, FS_SCALE, G, GI_FIRST, HSTATS, MU, NC, NE, NL, NR, NW, NX, NY, SD, STATIC, CamsData,   # noqa: E402
                        CamsModule, Npy, Timer, clean_origins, collate, read, wrap_pad)
import common as C                                                                      # noqa: E402


class DayDirData(CamsData):
    """CamsData over real-time day directories: one sample = (day, lead), inputs and targets from data/<date>/."""

    def __init__(self, first, last, psub=16, emis=True, train=True):
        meta = np.load(G / "meta.npz")
        self.year, self.doy = meta["year"], meta["doy"]
        cm = np.load(CMETA)
        self.eps, self.cmu, self.csd = cm["eps"][:, None, None], cm["mean"][:, None, None], cm["std"][:, None, None]
        hs = np.load(HSTATS)
        self.hmu, self.hsd = hs["mean"][:, None, None], hs["std"][:, None, None]
        self.static = np.load(STATIC)["static"].astype(np.float32)
        self.psub, self.emis, self.train, self.lag = psub, emis, train, 0
        self.days, self.samples, self.cache = {}, [], {}
        d = C.parse_date(first); e = C.parse_date(last)
        while d <= e:
            D = C.day_dir(d); ok = (D / "cams.npy").exists() and (D / "hres_B.npy").exists() and (D / "fires.npy").exists() and (D / "green" / f"origins_{d.year}.npy").exists()
            ok = ok and all((C.day_dir(d + dt.timedelta(days=j)) / "cams.npy").exists() for j in range(1, 6))
            if ok:
                self.days[C.gi_of(d)] = d
                self.samples += [(C.gi_of(d), k) for k in range(1, NL + 1)]
            d += dt.timedelta(days=1)

    def day(self, gi):                                                                   # lazily opened, memory-mapped files of one day
        if gi not in self.cache:
            if len(self.cache) > 24:
                self.cache.pop(next(iter(self.cache)))
            d = self.days[gi]; D = C.day_dir(d); g = D / "green"
            self.cache[gi] = {"cams": np.load(D / "cams.npy", mmap_mode="r"), "hres": np.load(D / "hres_B.npy", mmap_mode="r"),
                              "fire": np.load(D / "fires.npy"), "org": Npy(g / f"origins_{d.year}.npy"), "fs": Npy(g / f"fire_{d.year}.npy"),
                              "es": Npy(g / f"emis_{d.year}.npy") if self.emis else None,
                              "tgt": [np.load(C.day_dir(d + dt.timedelta(days=j)) / "cams.npy", mmap_mode="r") for j in range(0, 6)]}
        return self.cache[gi]

    def norm(self, x):
        return (np.log(np.maximum(x.astype(np.float32), 0) + self.eps) - self.cmu) / self.csd

    def item(self, gi, k, member=0):
        y = self.day(gi)
        c0, cm = self.norm(y["cams"][1]), self.norm(y["cams"][0])
        pm0 = (y["cams"][1, 1].astype(np.float32) - MU) / SD
        tgt = (y["tgt"][(k + 1) // 2][1 if k % 2 == 0 else 0, 1].astype(np.float32) - MU) / SD          # analysis of the valid time
        fine = wrap_pad(np.concatenate([c0, cm, self.static]))
        mask = wrap_pad(np.ones((1, NY, NX), np.float32)); mask[..., NX:] = 0
        b = y["hres"]
        st = [b[s].astype(np.float32) for s in (0, 2 * k)]
        st[0][NW - 1] = np.nan
        tp = b[2 * k - 2, NW - 1].astype(np.float32) + b[2 * k - 1, NW - 1].astype(np.float32)
        for a in st:
            a[NW - 1] = np.log1p(np.maximum(a[NW - 1], 0))
        wx = np.nan_to_num((np.concatenate([st[0], st[1], np.log1p(np.maximum(tp, 0))[None]]) - self.hmu) / self.hsd)
        o = y["org"]; P = int(o.shape[4]); per = NR * NC * P * 4
        org = read(o, (0 * NL + k - 1) * per, per, (NR, NC, P, 4))
        rng = np.random.default_rng(None if self.train else (int(gi) * 31 + int(k)) * 101 + int(member))
        sub = np.sort(rng.choice(P, self.psub, replace=False))
        fs = read(y["fs"], (0 * NL + k - 1) * 3 * NR * NC, 3 * NR * NC, (3, NR, NC))
        it = {"fine": torch.from_numpy(fine.astype(np.float16)), "x1": torch.from_numpy(wrap_pad(tgt[None])),
              "mask": torch.from_numpy(mask), "lead": torch.tensor(0.5 * k, dtype=torch.float32),
              "pm0": torch.from_numpy(pm0[None]), "wx": torch.from_numpy(wx.astype(np.float32)),
              "fire": torch.from_numpy(np.nan_to_num(y["fire"].astype(np.float32)).astype(np.float16)),
              "org": torch.from_numpy(clean_origins(org[:, :, sub])),
              "fsens": torch.from_numpy(np.log1p(np.maximum(np.nan_to_num(fs), 0) * FS_SCALE[:, None, None]).astype(np.float32))}
        if self.emis:
            es = read(y["es"], (0 * NL + k - 1) * NE * NR * NC, NE * NR * NC, (NE, NR, NC))
            it["esens"] = torch.from_numpy(np.log1p(np.maximum(np.nan_to_num(es), 0)).astype(np.float32))
        return it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", required=True); ap.add_argument("--last", required=True)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=1); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--replay", type=float, default=0.0, help="fraction of the paper's training samples (2017-2022 stores) mixed in each epoch")
    ap.add_argument("--init", default=str(C.CKPT)); ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", type=int, default=0, help="steps per epoch (speed test)"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    pl.seed_everything(a.seed, workers=True)
    live = DayDirData(a.first, a.last, emis=True, train=True)
    ds = live
    if a.replay > 0:
        old = CamsData("train", psub=16, emis=True, train=True)
        rng = np.random.default_rng(a.seed); n = int(len(old.samples) * a.replay)
        old.samples = [old.samples[i] for i in sorted(rng.choice(len(old.samples), n, replace=False))]
        ds = ConcatDataset([live, old])
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, persistent_workers=True, pin_memory=True, drop_last=True, collate_fn=collate)
    out = C.RT / "runs" / f"rt12{a.tag}"
    model = CamsModule(lr=a.lr, epochs=a.epochs, shift=2.6, shift_mix=0.5, psub=16, emis=True)
    sd = torch.load(a.init, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    trainer = pl.Trainer(accelerator="gpu", devices=int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)), num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
                         strategy="ddp", precision="bf16-mixed", max_epochs=a.epochs, gradient_clip_val=1.0, log_every_n_steps=5,
                         default_root_dir=str(out), logger=CSVLogger(str(out), name="logs"), limit_val_batches=0,
                         limit_train_batches=a.smoke if a.smoke else 1.0,
                         callbacks=[Timer(), ModelCheckpoint(dirpath=str(out / "ckpts"), filename="{epoch:02d}", save_top_k=-1, every_n_epochs=1)])
    if trainer.global_rank == 0:
        print(f"{out.name}: {len(live)} live samples ({len(live) // NL} days {a.first}..{a.last})" + (f" + {len(ds) - len(live)} replay samples" if a.replay else "") +
              f", init {Path(a.init).name} (missing {len(missing)}, unexpected {len(unexpected)}), lr {a.lr}, epochs {a.epochs}, "
              f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    trainer.fit(model, dl)
    trainer.save_checkpoint(out / "final.ckpt")


if __name__ == "__main__":
    main()
