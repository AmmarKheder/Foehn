#!/usr/bin/env python3
"""The fine-tuning model (128-particle store, 6 emission channels with zero input weights) must start exactly as the
checkpoint: same velocity on a 2022 window when both read the same particles (fine-tuning store, 256 particles). Also prints the ranges of the inputs.
    python tests/test_finetune_init.py <ckpt> [store green256] [psub 128]     (one GCD; needs <store>/hres 2022 days)"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from fullflow_global_green import NC, FullModuleGreen, GlobalDataGreen, green_block, green_rows, init_from  # noqa: E402

ck = sys.argv[1]
store = sys.argv[2] if len(sys.argv) > 2 else "green256"
psub = int(sys.argv[3]) if len(sys.argv) > 3 else 128
emis_store = sys.argv[4] if len(sys.argv) > 4 else None
emis_scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
dev = torch.device("cuda")
ds = GlobalDataGreen([2022], train=False, store=store, emis=True, psub=psub, emis_store=emis_store)
print(len(ds), "validation samples with Green's functions")
for (d, h, r0, c0) in ((0, 1, 10, 0), (3, 7, 100, 650), (5, 4, 200, 580), (2, 2, 150, 300)):   # incl. across the date line
    a_ = green_rows(ds.org[2022], d, h, r0, r0 + 84)[:, (c0 + np.arange(140)) % NC]
    b_ = green_block(ds.org[2022], d, h, r0, r0 + 84, c0, 140)
    assert np.array_equal(a_, b_), (d, h, r0, c0)
print("green_block == green_rows[:, cols]")
b = {k: (v[None] if v.dim() > 0 else v.reshape(1)).to(dev) for k, v in ds[0].items()}
for k in ("org", "fsens", "esens"):
    v = b[k].float()
    print(k, tuple(v.shape), "min", float(v.min()), "max", float(v.max()), "mean", float(v.mean()))
old = FullModuleGreen.load_from_checkpoint(ck, map_location="cpu").to(dev).eval()
hp = dict(old.hparams); hp.update(emis=True, store=store, emis_store=emis_store, emis_scale=emis_scale)
new = FullModuleGreen(**hp)
init_from(new, ck)
new = new.to(dev).eval()
x = torch.randn_like(b["x1"].float())
tau = torch.full((1,), 0.5, device=dev)
with torch.no_grad():
    b_old = {k: v for k, v in b.items() if k != "esens"}
    v_old = old.velocity(x, b_old, tau).float()
    v_new = new.velocity(x, b, tau).float()
    b_zero = dict(b); b_zero["esens"] = torch.zeros_like(b["esens"])
    v_zero = new.velocity(x, b_zero, tau).float()
d = (v_new - v_old).abs().max().item()
print(f"max |v_new - v_ckpt| = {d:.3e} (scale {v_old.abs().mean().item():.3e}); with zero emissions {(v_zero - v_old).abs().max().item():.3e}")
assert d < 1e-4 * max(1.0, v_old.abs().max().item()), "the fine-tuning model does not start from the checkpoint"
print("OK: the fine-tuning starts exactly from the checkpoint")
