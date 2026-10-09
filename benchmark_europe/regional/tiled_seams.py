#!/usr/bin/env python3
"""Seams of a tiled kilometre-scale model: a baseline trained natively at 1 km on the 768 x 1536 window
(runs/<model>, regional.py) is applied tile by tile to the whole of Europe (non-overlapping tiles, the last row and
column anchored to the border), as tiled high-resolution models are used. For each map we report the mean absolute
jump between neighbouring pixels across tile edges divided by the mean jump elsewhere (1 = no seam), for this tiled
map, the GHAP truth and the tile-free whole-image flow on the same lines. Real HRES weather, 2022, lead 1 day.
    python tiled_seams.py <model> [days...]"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regional as R  # noqa: E402
from dejavu import DVData, attach_hres, norm, MU, SD  # noqa: E402

model_name = sys.argv[1]
DAYS = [int(x) for x in sys.argv[2:]] or [40, 100, 200, 300]
H = 1
dev = torch.device("cuda")
module = R.RegionalModule.load_from_checkpoint(Path(__file__).resolve().parent / "runs" / model_name / "final.ckpt",
                                               map_location="cpu").to(dev).eval()
dv = DVData([2022], "noise", train=False, t31=True)
attach_hres(dv, True, False)
doy = np.load(R.EXP / "dejavu/data/meta.npz")["doy"]
index = {(int(doy[i]), h): n for n, (i, h) in enumerate(dv.samples)}
off = int(np.where(dv.year == 2022)[0][0])
hdr = R.header(R.FINE_FILE)
st = np.load(R.EXP / "fullimage/data/static1km.npz")
elev, land = st["elev"].astype(np.float32), st["land"].astype(np.float32)
FH, FW, NR, NC = R.FH, R.FW, R.NR, R.NC
rows = list(range(0, FH - NR, NR)) + [FH - NR]
cols = list(range(0, FW - NC, NC)) + [FW - NC]


def day(d):
    a = np.asarray(np.memmap(R.FINE_FILE, dtype=np.float16, mode="r", offset=hdr + d * FH * FW * 2, shape=(FH, FW)), np.float32)
    ok = a > 0
    return norm(np.where(ok, a, np.nan)), ok.astype(np.float32), a


def grid(r0, c0):
    r = (r0 + torch.arange(NR, device=dev) + 0.5) / 25.0 * (2.0 / 167) - 1
    c = (c0 + torch.arange(NC, device=dev) + 0.5) / 25.0 * (2.0 / 279) - 1
    return torch.stack([c[None].expand(NR, -1), r[:, None].expand(-1, NC)], -1)[None]


def ratio(a, rlines, clines):
    a = np.where(land > 0, a, np.nan)
    dc, dr = np.abs(np.diff(a, axis=1)), np.abs(np.diff(a, axis=0))
    return 0.5 * (np.nanmean(dc[:, [c - 1 for c in clines]]) / np.nanmean(dc) + np.nanmean(dr[[r - 1 for r in rlines]]) / np.nanmean(dr))


rlines, clines = rows[1:], cols[1:]           # tile edges (a pixel row / column where a new tile starts)
out = {"tiled": [], "truth": []}
for d in DAYS:
    i = off + d
    it = dv[index[(d, H)]]
    coarse = it["cond"][:R.N_COND][None].to(dev)
    tn, tm, _ = day(i); yn, ym, _ = day(i - 1)
    _, _, truth = day(i + H)
    pred = np.zeros((FH, FW), np.float32)
    for r0 in rows:
        for c0 in cols:
            sl = (slice(r0, r0 + NR), slice(c0, c0 + NC))
            fine = torch.from_numpy(np.stack([tn[sl], tm[sl], yn[sl], ym[sl], elev[sl], land[sl]])).to(dev)[None]
            up = F.grid_sample(coarse, grid(r0, c0), align_corners=True, padding_mode="border")
            era5 = torch.cat([up, fine[:, 2:3], fine[:, 1:2], fine[:, 3:4], fine[:, 5:6]], 1)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                y = module.net(era5, fine[:, 0:1], fine[:, 4], torch.tensor([float(H)], device=dev)).float()
            pred[sl] = y[0, 0].cpu().numpy() * SD + MU
    out["tiled"].append(ratio(pred, rlines, clines))
    out["truth"].append(ratio(truth, rlines, clines))
    print(f"day {d}: tiled {out['tiled'][-1]:.2f}  truth {out['truth'][-1]:.2f}", flush=True)
    np.save(Path(__file__).resolve().parent / "runs" / model_name / f"tiled_europe_d{d}_h{H}.npy", pred.astype(np.float16))
print(f"{model_name} applied tile by tile: jump across tile edges / elsewhere = x{np.mean(out['tiled']):.2f} "
      f"(truth x{np.mean(out['truth']):.2f}); tile edges rows {rlines} cols {clines}")
