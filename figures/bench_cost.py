#!/usr/bin/env python3
"""Inference cost for the skill-vs-cost figure: wall time of one forecast (one member, one lead time) and peak GPU memory,
on one AMD MI250X GCD, bfloat16, batch 1, after warm-up; median of 10 runs.
    python bench_cost.py <name>     (foehn, nofp, convlstm, earthformer, simvp, topoflow, climax, fullimage)"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
sys.path.insert(0, str(EXP / "dejavu")); sys.path.insert(0, str(EXP / "retrain")); sys.path.insert(0, str(EXP / "fullimage"))
dev = torch.device("cuda")
name = sys.argv[1]


def timeit(fn, n=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(n):
        t = time.time(); fn(); torch.cuda.synchronize(); ts.append(time.time() - t)
    return float(np.median(ts)), torch.cuda.max_memory_allocated() / 2**30


if name in ("foehn", "nofp"):
    from dejavu import DVData, DVModule
    run = {"foehn": "noise_t31_fpstab", "nofp": "noise_t31"}[name]
    m = DVModule.load_from_checkpoint(EXP / "dejavu/runs" / run / "final.ckpt", map_location="cpu").to(dev).eval()
    ds = DVData([2022], "noise", train=False, t31=True, footprint=m.hparams.get("footprint", False), fp_met=m.hparams.get("fp_met", "blh"))
    it = ds[100]
    cond, lead = it["cond"][None].to(dev), torch.tensor([2.0], device=dev)
    fp = tuple(it[k][None].to(dev) for k in ("wind", "rain", "stab") if k in it) or None
    def fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            m.sample(torch.randn(1, 1, 168, 280, device=dev), cond, lead, 8, fp)
elif name == "fullimage":
    from fullflow import FullModule, PH, PW
    m = FullModule.load_from_checkpoint(EXP / "fullimage/runs/flow/final.ckpt", map_location="cpu").to(dev).eval()
    b = {"fine": torch.randn(1, 6, PH, PW, device=dev), "coarse": torch.randn(1, 69, 168, 280, device=dev), "lead": torch.tensor([2.0], device=dev)}
    def fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            m.sample(b, 8)
else:
    from train_baselines import BaselineModule
    ck = sorted((EXP / "retrain/runs" / f"B_{name}_mettarget31/ckpt").glob("best-*.ckpt"))[0]
    m = BaselineModule.load_from_checkpoint(ck, map_location="cpu").to(dev).eval()
    b = {"era5": torch.randn(1, 101, 168, 280, device=dev), "ghap_coarse": torch.randn(1, 1, 168, 280, device=dev),
         "elevation_coarse": torch.randn(1, 168, 280, device=dev), "lead_time": torch.tensor([2.0], device=dev)}
    def fn():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            m(b)
sec, mem = timeit(fn)
params = sum(p.numel() for p in m.parameters()) / 1e6
(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "data" / f"cost_{name}.json").write_text(json.dumps(dict(seconds=sec, peak_gb=mem, params_m=params)))
print(name, f"{sec:.4f} s per member per forecast, peak {mem:.2f} GB, {params:.1f} M params")
