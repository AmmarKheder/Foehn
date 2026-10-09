#!/usr/bin/env python3
"""Global FOEHN-km: the tile-free 1 km flow-matching network of ../fullimage (same network, same resolution-shifted
schedule), trained on the whole Earth. A training sample is a Europe-sized 1 km window (4192 x 6992 px) drawn at random
among the windows of the globe with at least 30 % land, with its aligned 0.25 deg conditioning (168 x 280 cells: PM
today, weather of today, forecast-day weather). Data from build_fine.py / build_coarse.py / build_static.py
(84 N .. 60 S, all longitudes). Train 2017-2020, validation 2021, test 2022.
    python fullflow_global.py prep                       (valid days, once)
    python fullflow_global.py train [--epochs E] ...     (srun, 8 GCDs per node, any number of nodes)"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "fullimage"))
sys.path.insert(0, str(ROOT.parent / "dejavu"))
from fullflow import FH, FW, PH, PW, FullModule, Timer  # noqa: E402
from dejavu import T31, norm  # noqa: E402

G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
GH, GW = 14400, 36000            # global 1 km grid
CH_, CW_ = 168, 280              # 0.25 deg window = FH x FW at 1 km (fine pixel p <-> coarse coordinate p / 25)
# Registration (checked 2026-09-30, do not change): the 0.25 deg values are POINT values at 84 - 0.25 k (ERA5 points;
# pm.npy[k] = mean of fine rows 25 k - 12 .. 25 k + 12), and FullNet samples coarse point k at the fine EDGE coordinate
# 25 k of the window (grid16 = (16 a + 8) / 25, align_corners=True). Hence the fine window of coarse points k0 .. starts
# at fine row 25 k0: sampling pm.npy as the network does matches the 1 km map best at this origin (+-12 px is worse).
STRIDE, MIN_LAND = 4, 0.3
TILE, NTR, NTC = 512, 29, 71     # storage blocks of ghap1km_tiles.npy / static1km_tiles.npy (build_fine_tiles.py)


def header(path):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); return f.tell()


class Npy:
    """Direct reads (os.pread of contiguous byte ranges) from a .npy file: much faster than memmap page faults on Lustre.
    Safe across DataLoader workers (pread does not move a shared file position)."""

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            np.lib.format.read_magic(f)
            self.shape, _, self.dtype = np.lib.format.read_array_header_1_0(f)
            self.hdr = f.tell()
        self.fd = None

    def _read(self, offset, count):
        if self.fd is None:
            self.fd = os.open(self.path, os.O_RDONLY)
        nbytes = count * self.dtype.itemsize
        buf = os.pread(self.fd, nbytes, self.hdr + offset * self.dtype.itemsize)
        assert len(buf) == nbytes, (self.path, offset, count)
        return np.frombuffer(buf, dtype=self.dtype, count=count)

    def plane_rows(self, lead, r0, r1):
        """rows r0:r1 (all columns) of the 2-D plane at leading index `lead` (tuple)."""
        Hh, Ww = self.shape[-2:]
        k = int(np.ravel_multi_index(lead, self.shape[:-2])) if len(self.shape) > 2 else 0
        return self._read((k * Hh + r0) * Ww, (r1 - r0) * Ww).reshape(r1 - r0, Ww)

    def rows(self, leads, r0, r1):
        """stack of plane_rows over a list of leading-index tuples."""
        return np.stack([self.plane_rows(l, r0, r1) for l in leads])

    def __getstate__(self):                      # workers reopen their own descriptor
        d = dict(self.__dict__); d["fd"] = None; return d


def tile_window(npy, k, r0, c0, h=FH, w=FW):
    """Window [r0:r0+h, c0:c0+w] of map k of a (K, 29, 71, 512, 512) block file: one contiguous read per block row."""
    tr0, tr1, tc0, tc1 = r0 // TILE, (r0 + h - 1) // TILE, c0 // TILE, (c0 + w - 1) // TILE
    nb, B = tc1 - tc0 + 1, TILE * TILE
    blk = np.stack([npy._read(((k * NTR + tr) * NTC + tc0) * B, nb * B).reshape(nb, TILE, TILE) for tr in range(tr0, tr1 + 1)])
    blk = blk.transpose(0, 2, 1, 3).reshape((tr1 - tr0 + 1) * TILE, nb * TILE)
    return blk[r0 - tr0 * TILE:r0 - tr0 * TILE + h, c0 - tc0 * TILE:c0 - tc0 * TILE + w]


class GlobalData(Dataset):
    def __init__(self, years, train, seed=0, max_lead=4):
        meta = np.load(G / "meta.npz")
        self.year, land = meta["year"], meta["land"].astype(np.float32)
        valid = np.load(G / "valid.npy")
        D = len(self.year)
        self.max_lead = max_lead
        self.samples = [(i, h) for i in range(1, D - max_lead) for h in range(1, max_lead + 1)
                        if np.isin(self.year[i], years) and self.year[i + h] == self.year[i]
                        and valid[i] and valid[i - 1] and valid[i + h]]
        cs = np.cumsum(np.cumsum(np.pad(land, ((1, 0), (1, 0))), 0), 1)          # land fraction of every window
        frac = lambda k, j: (cs[k + CH_, j + CW_] - cs[k, j + CW_] - cs[k + CH_, j] + cs[k, j]) / (CH_ * CW_)
        self.cands = np.array([(k, j) for k in range(0, land.shape[0] - CH_ + 1, STRIDE)
                               for j in range(0, land.shape[1] - CW_ + 1, STRIDE) if frac(k, j) >= MIN_LAND])
        self.fixed = np.random.default_rng(seed).integers(0, len(self.cands), len(self.samples))
        self.train = train
        self.fine, self.stat = Npy(G / "ghap1km_tiles.npy"), Npy(G / "static1km_tiles.npy")
        self.pm, self.met = Npy(G / "pm.npy"), Npy(G / "met.npy")

    def __len__(self):
        return len(self.samples)

    def fine_day(self, d, r0, c0):
        a = tile_window(self.fine, d, r0, c0).astype(np.float32)
        ok = a > 0
        return np.stack([norm(np.where(ok, a, np.nan)), ok]).astype(np.float16)

    def window(self, idx):
        return self.cands[np.random.randint(len(self.cands)) if self.train else self.fixed[idx]]

    def item(self, i, h, k0, j0):
        r0, c0 = 25 * k0, 25 * j0
        cond = np.concatenate([norm(self.pm.plane_rows((i,), k0, k0 + CH_)[:, j0:j0 + CW_].astype(np.float32))[None],
                               self.met.rows([(i, c) for c in range(37)], k0, k0 + CH_)[..., j0:j0 + CW_].astype(np.float32),
                               self.met.rows([(i + h, c) for c in T31], k0, k0 + CH_)[..., j0:j0 + CW_].astype(np.float32)])
        tgt = self.fine_day(i + h, r0, c0)
        static = np.stack([tile_window(self.stat, s, r0, c0) for s in (0, 1)])
        fine = np.concatenate([self.fine_day(i, r0, c0), self.fine_day(i - 1, r0, c0), static])
        pad = lambda a: np.pad(a, ((0, 0), (0, PH - FH), (0, PW - FW)))
        return {"fine": torch.from_numpy(pad(fine)), "x1": torch.from_numpy(pad(tgt[:1])), "mask": torch.from_numpy(pad(tgt[1:])),
                "coarse": torch.from_numpy(np.nan_to_num(cond)), "lead": torch.tensor(float(h))}

    def __getitem__(self, idx):
        i, h = self.samples[idx]
        k0, j0 = self.window(idx)
        return self.item(i, h, int(k0), int(j0))


def prep():
    pm = np.load(G / "pm.npy", mmap_mode="r")
    land = np.load(G / "meta.npz")["land"]
    valid = np.array([np.isfinite(np.asarray(pm[k])[land]).mean() > 0.2 for k in range(pm.shape[0])])
    np.save(G / "valid.npy", valid)
    print("valid days", int(valid.sum()), "of", len(valid))


def train(a):
    pl.seed_everything(a.seed, workers=True)
    tr = GlobalData([2017, 2018, 2019, 2020], train=True, max_lead=a.max_lead)
    va = GlobalData([2021], train=False, seed=1, max_lead=a.max_lead)
    keep = list(range(0, len(va.samples), max(1, len(va.samples) // 64)))[:64]
    va.samples, va.fixed = [va.samples[k] for k in keep], va.fixed[keep]
    if a.limit < 1:
        tr.samples = tr.samples[:int(len(tr.samples) * a.limit)]; va.samples, va.fixed = va.samples[:8], va.fixed[:8]
    dl = DataLoader(tr, batch_size=1, shuffle=True, num_workers=a.workers, persistent_workers=True, pin_memory=True)
    dv = DataLoader(va, batch_size=1, num_workers=a.workers, persistent_workers=True, pin_memory=True)
    out = ROOT / "runs" / f"flow_shift{a.shift:g}_global_L{a.max_lead}{'' if a.seed == 42 else f'_s{a.seed}'}{a.tag}"
    model = FullModule(lr=a.lr, epochs=a.epochs, reg=False, shift=a.shift, shift_mix=a.shift_mix)
    trainer = pl.Trainer(accelerator="gpu", devices=int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)),
                         num_nodes=int(os.environ.get("SLURM_NNODES", 1)), strategy="ddp", precision="bf16-mixed",
                         max_epochs=a.epochs, gradient_clip_val=1.0, log_every_n_steps=10, default_root_dir=str(out),
                         logger=CSVLogger(str(out), name="logs"), callbacks=[Timer()])
    if trainer.global_rank == 0:
        print(f"{out.name}: train {len(tr)} val {len(va)} windows {len(tr.cands)} "
              f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    trainer.fit(model, dl, dv)
    trainer.save_checkpoint(out / "final.ckpt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--shift", type=float, default=25.0)
    ap.add_argument("--shift-mix", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-lead", type=int, default=4)
    a = ap.parse_args()
    prep() if a.cmd == "prep" else train(a)
