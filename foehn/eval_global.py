#!/usr/bin/env python3
"""Global test (2022, real ECMWF HRES 0.25 deg forecast weather) of the global FOEHN-km.
The globe (84 N .. 60 S) is covered by Europe-sized windows (168 x 280 cells at 0.25 deg, 4192 x 6992 px at 1 km) on a
regular grid with overlaps; every 1 km pixel and every station is scored in the window whose centre is nearest (each
pixel counted once). Forecast = mean of M members (S flow steps).
Outputs (runs/<run>/eval_global/part_<rank>.npz): station forecasts st_pred[N, 365, 4] (NaN where not forecast) and, per
(start day, lead, window), 1 km sums over land pixels with a valid target: n, sum of squared errors, sum of absolute
errors, fair-CRPS sum, exceedance counts (hits, union) of 25 ug/m3.
    python eval_global.py hres_memmap                         (month npz -> hres_2022.npy, once)
    srun ... python eval_global.py run runs/<run> [--members 4 --steps 8 --every 2]"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from fullflow_global import CH_, CW_, FH, FW, G, GlobalData  # noqa: E402
from fullflow import MU, SD, FullModule  # noqa: E402

HRES_NPY = G / "hres_2022.npy"


def hres_memmap():
    """Monthly npz blocks [days, 4, 31, 576, 1440] -> one (365, 4, 31, 576, 1440) float16 file indexed by day of year."""
    shape = (365, 4, 31, 576, 1440)
    with open(HRES_NPY, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": "<f2", "fortran_order": False, "shape": shape})
        hdr = f.tell(); f.truncate(hdr + int(np.prod(shape)) * 2)
    n = int(np.prod(shape[1:]))
    for m in range(1, 13):
        z = np.load(G / "hres" / f"{m:02d}.npz")
        for b, d in zip(z["block"], z["day_of_year"]):
            np.memmap(HRES_NPY, dtype=np.float16, mode="r+", offset=hdr + int(d) * n * 2, shape=shape[1:])[:] = b
        print("month", m, len(z["day_of_year"]), flush=True)


def windows(land, min_land=0.05, stride=(136, 232)):
    """Window origins (k0, j0) on a regular grid covering 576 x 1440, keeping windows with some land."""
    H, W = land.shape
    ks = sorted(set(list(range(0, H - CH_, stride[0])) + [H - CH_]))
    js = sorted(set(list(range(0, W - CW_, stride[1])) + [W - CW_]))
    return [(k, j) for k in ks for j in js if land[k:k + CH_, j:j + CW_].mean() >= min_land]


class Test(GlobalData):
    """2022 samples with the forecast-day block replaced by HRES (same standardisation as ERA5)."""

    def __init__(self):
        super().__init__([2022], train=False)
        st = np.load(G / "stats.npz")
        self.em, self.es = st["era5_mean"][:, None, None], st["era5_std"][:, None, None]
        self.rm, self.rs = float(st["extra_mean"][3]), float(st["extra_std"][3])
        self.hres = np.load(HRES_NPY, mmap_mode="r")
        self.doy = np.load(G / "meta.npz")["doy"]

    def item(self, i, h, k0, j0):
        it = super().item(i, h, k0, j0)
        raw = np.asarray(self.hres[int(self.doy[i]), h - 1, :, k0:k0 + CH_, j0:j0 + CW_], np.float32)
        blk = np.concatenate([(raw[:30] - self.em[:, :, :]) / self.es,
                              ((np.log1p(np.maximum(raw[30], 0) * 1000.0) - self.rm) / self.rs)[None]])
        it["coarse"][-31:] = torch.from_numpy(np.nan_to_num(blk).astype(np.float32))
        return it


def run(a):
    rank, world = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    dev = torch.device("cuda")
    model = FullModule.load_from_checkpoint(Path(a.run) / "final.ckpt", map_location="cpu").to(dev).eval()
    ds = Test()
    land = np.load(G / "meta.npz")["land"]
    win = windows(land)
    cent = np.array([(k + CH_ / 2, j + CW_ / 2) for k, j in win])
    kk, jj = np.meshgrid(np.arange(land.shape[0]) + 0.5, np.arange(land.shape[1]) + 0.5, indexing="ij")
    owner = np.argmin((kk[..., None] - cent[:, 0]) ** 2 + (jj[..., None] - cent[:, 1]) ** 2, -1)   # (576, 1440)
    stz = np.load(G / "stations_2022.npz")
    sr, sc = np.floor((84.0 - stz["lat"]) / 0.01).astype(int), np.floor((stz["lon"] + 180.0) / 0.01).astype(int)
    sown = owner[np.clip(sr // 25, 0, 575), np.clip(sc // 25, 0, 1439)]
    st_pred = np.full((len(sr), 365, 4), np.nan, np.float32)
    doy = ds.doy
    days = sorted({i for i, h in ds.samples})[::a.every]
    mine = days[rank::world]
    recs, have = [], set(ds.samples)
    for i in mine:
        for h in (1, 2, 3, 4):
            if (i, h) not in have:
                continue
            for w, (k0, j0) in enumerate(win):
                b = ds.item(i, h, k0, j0)
                bt = {k: (v[None].to(dev) if torch.is_tensor(v) and v.dim() > 0 else v.reshape(1).to(dev)) for k, v in b.items()}
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    mem = torch.cat([model.sample(bt, a.steps) for _ in range(a.members)], 0)[:, 0, :FH, :FW].float()
                mem = mem * SD + MU
                mean = mem.mean(0).cpu().numpy()
                x1 = (b["x1"][0, :FH, :FW].float() * SD + MU).numpy(); m = b["mask"][0, :FH, :FW].numpy() > 0
                r0, c0 = 25 * k0, 25 * j0
                core = np.repeat(np.repeat(owner[k0:k0 + CH_, j0:j0 + CW_] == w, 25, 0), 25, 1)[:FH, :FW]
                ok = m & core & (np.asarray(ds.arr["land"][r0:r0 + FH, c0:c0 + FW]) > 0)
                e = (mean - x1)[ok]
                M = mem[:, torch.from_numpy(ok).to(dev)]
                t = torch.from_numpy(x1[ok]).to(dev)
                crps = (M - t).abs().mean(0) - (M[:, None] - M[None]).abs().sum((0, 1)) / (2 * a.members * max(a.members - 1, 1))
                ex, fx = x1[ok] > 25, mean[ok] > 25
                recs.append((i, h, w, ok.sum(), (e ** 2).sum(), np.abs(e).sum(), float(crps.sum()), (ex & fx).sum(), (ex | fx).sum()))
                sel = np.where(sown == w)[0]
                rr, cc = sr[sel] - r0, sc[sel] - c0
                inb = (rr >= 0) & (rr < FH) & (cc >= 0) & (cc < FW)
                st_pred[sel[inb], int(doy[i + h]), h - 1] = mean[rr[inb], cc[inb]]
            print(f"rank {rank}: day {i} lead {h} done", flush=True)
    out = Path(a.run) / "eval_global"; out.mkdir(exist_ok=True)
    np.savez(out / f"part_{rank}.npz", st_pred=st_pred, recs=np.array(recs, np.float64), windows=np.array(win))
    print("rank", rank, "done", len(recs), "window-samples")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hres_memmap", "run"])
    ap.add_argument("run", nargs="?")
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    hres_memmap() if a.cmd == "hres_memmap" else run(a)
