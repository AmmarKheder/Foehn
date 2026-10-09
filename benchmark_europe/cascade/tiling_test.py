#!/usr/bin/env python3
"""Tiling invariance: the same 1 km region forecast from two tilings offset by 256 px must agree (up to the halo's
truncation of the receptive field). Reports the max / RMS difference in ug/m3 on the shared centre. Same noise seed."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade import CROP, HALO, CascadePredictor, window_origin, MU, SD  # noqa: E402

ckpt, coarse_ckpt, d = sys.argv[1], sys.argv[2], int(sys.argv[3])
dev = torch.device("cuda")
cp = CascadePredictor(ckpt, coarse_ckpt, dev, members=1, weather="hres")
i, h, P = cp.off + d, 1, HALO
today, yday = cp.fine_full(i), cp.fine_full(i - 1)
full = np.pad(np.concatenate([today, yday, cp.static]), ((0, 0), (P, P), (P, P)))
src = np.asarray(cp.src5[i], np.float32)
src = torch.from_numpy(np.stack([np.nan_to_num((src - MU) / SD), np.isfinite(src)]).astype(np.float32)).to(dev)[None]
it = cp.dv[cp.index[(i, h)]]
cond = it["cond"][None].to(dev)
fp = tuple(it[k][None].to(dev) for k in ("wind", "rain", "stab"))
g = torch.Generator(device=dev); g.manual_seed(0)
with torch.autocast("cuda", dtype=torch.bfloat16):
    c1 = cp.stage1.sample(torch.randn(1, 1, 168, 280, device=dev, generator=g), cond, torch.tensor([1.0], device=dev), 8, fp)[:, 0].float()
coarse_full = torch.cat([c1[:, None], cond[:, :69]], 1)
r, c = 1600, 3000   # Po valley
n = CROP + 2 * P
outs = []
for dr, dc in ((0, 0), (256, 256), (16, 0)):
    j0, i0 = window_origin(r + dr - P, c + dc - P)
    b = {"fine": torch.from_numpy(full[:, r + dr:r + dr + n, c + dc:c + dc + n])[None].to(dev),
         "coarse": coarse_full[:, :, j0:j0 + 24, i0:i0 + 24], "src": src, "lead": torch.tensor([1.0], device=dev),
         "r0": torch.tensor([r + dr - P], device=dev), "c0": torch.tensor([c + dc - P], device=dev),
         "j0": torch.tensor([j0], device=dev), "i0": torch.tensor([i0], device=dev),
         **{k: it[k][None].to(dev) for k in ("wind", "rain", "stab")}}
    torch.manual_seed(0)
    # the same noise field in image coordinates for every tiling: draw it once on the union and crop
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats, g16 = cp.stage2.features(b, b["coarse"])
        rng = torch.Generator(device=dev); rng.manual_seed(1)
        big = torch.randn(1, 1, n + 256, n + 256, device=dev, generator=rng)
        x = big[:, :, dr:dr + n, dc:dc + n].clone()
        S = 8
        for s in range(S):
            tau = torch.full((1,), s / S, device=dev)
            x = x + cp.stage2.velocity(x, feats, b["coarse"], g16, tau, b["lead"], torch.zeros(1, device=dev)) / S
    outs.append((dr, dc, (x[0, 0].float() * SD + MU).cpu().numpy()))
ref = outs[0][2]
for dr, dc, o in outs[1:]:
    a = ref[P + dr:P + CROP, P + dc:P + CROP]          # region of tiling 0 that is inside tiling (dr, dc)'s centre
    bb = o[P:P + CROP - dr, P:P + CROP - dc]
    diff = a - bb
    print(f"offset ({dr},{dc}): max |diff| {np.abs(diff).max():.4f} ug/m3, RMS {np.sqrt((diff ** 2).mean()):.4f}, field RMS {np.sqrt((a ** 2).mean()):.2f}", flush=True)
