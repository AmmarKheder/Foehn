#!/usr/bin/env python3
"""(module) Trajectories of the footprint figure, importable. The footprint layer at work (the MOSAIC 'mechanism' figure):
(a) back-trajectories of the air reaching eight cities at 12 UTC on 17 March 2022 (Saharan dust episode, lead 2 days),
    integrated through the real HRES 3-hourly forecast wind exactly as in the layer, over the PM2.5 field of the start
    day that the layer gathers; line width / opacity = learned weight of each path point (FOEHN, seed 42);
(b) learned log-weight versus boundary-layer height (true-wind model vs wrong-wind control, 3 seeds each);
(c) learned log-weight versus stability (realistic HRES model, 3 seeds);
(d) gain of the footprint with 95 % block-bootstrap intervals (lead 2 days): CRPS and CSI of exceedances, on all days,
    documented transported episodes and the 10 % / 5 % most polluted days."""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

from fig_style import COL2, EXP, STYLE, save
STYLE_C = STYLE['foehn']['color']

sys.path.insert(0, str(EXP / "dejavu"))
sys.path.insert(0, str(EXP / "dejavu" / "figs"))
from dejavu import DVData, attach_hres  # noqa: E402
from footprint_physics import load as load_fp, z_of  # noqa: E402

START_DOY, H = 71, 2                       # 13 March 2022 -> arrival 15 March 12 UTC (dust intrusion)
CITIES = {"Madrid": (40.42, -3.70), "Bordeaux": (44.84, -0.58), "Paris": (48.86, 2.35), "Lyon": (45.76, 4.84),
          "Milan": (45.46, 9.19), "Munich": (48.14, 11.58), "Rome": (41.90, 12.50), "Vienna": (48.21, 16.37)}
DT, KMAX = 3 * 3600.0, 32


def bilinear(field, lat, lon):
    """field (..., 84, 140) on 0.5 deg cells (centres 71.875 - 0.5 r, -24.875 + 0.5 c), border-clamped."""
    r = np.clip((71.875 - lat) / 0.5, 0, field.shape[-2] - 1.001); c = np.clip((lon + 24.875) / 0.5, 0, field.shape[-1] - 1.001)
    r0, c0 = int(r), int(c); fr, fc = r - r0, c - c0
    return ((1 - fr) * (1 - fc) * field[..., r0, c0] + (1 - fr) * fc * field[..., r0, c0 + 1]
            + fr * (1 - fc) * field[..., r0 + 1, c0] + fr * fc * field[..., r0 + 1, c0 + 1])


def trajectories():
    ds = DVData([2022], "noise", train=False, t31=True, footprint=True, fp_met="stab")
    attach_hres(ds, True, True)
    doy = np.load(EXP / "dejavu/data/meta.npz")["doy"]
    n = [k for k, (i, h) in enumerate(ds.samples) if doy[i] == START_DOY and h == H][0]
    i = ds.samples[n][0]
    it = ds[n]
    wind, rain, stab = (it[k].float().numpy() for k in ("wind", "rain", "stab"))
    logit_fn, lam, mix = load_fp("noise_t31_fpstab")
    paths = {}
    for name, (la, lo) in CITIES.items():
        pts, logits, cum = [], [], 0.0
        s0 = 8 * H + 4
        for k in range(8 * H + 1):
            st = s0 - k; day = st // 8
            w = bilinear(wind[st], la, lo)
            u, v = mix[0] * w[0] + mix[1] * w[2], mix[0] * w[1] + mix[1] * w[3]
            r = float(bilinear(rain[day], la, lo)); sg = bilinear(stab[day], la, lo)
            logits.append(logit_fn(np.array([[k / KMAX, sg[0] / 5, sg[1] / 5, r, cum]], np.float32))[0])
            pts.append((lo, la))
            la, lo = la - v * DT / 111e3, lo - u * DT / (111e3 * np.cos(np.deg2rad(la)))
            cum += r / 8
        wgt = np.exp(np.array(logits) - max(logits)); wgt /= wgt.sum()
        paths[name] = (np.array(pts), wgt)
    pm = np.asarray(np.load(EXP / "dejavu/data/pm.npy", mmap_mode="r")[i], np.float32)
    return paths, pm


