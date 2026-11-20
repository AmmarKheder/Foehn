"""realtime_1km.blh -- boundary-layer-height substitute (encoder channels 31-33) from 1 deg profiles (see calib_blh.py).

blh_daily_from_B(B): B (4, 53, 181, 360) dl_wb2-format fields at 00/06/12/18 UTC of ONE day (ERA5 era5_B or the IFS
hres_B.npy of RT) -> (3, 576, 1440) float32 log(max(h, 10 m)) daily mean / max / min of FLEXGREEN's mixing height,
bilinear from the 1 deg points (lat 90..-90, lon -180..179) to the 0.25 deg encoder points (84..-59.75, -180..179.75).
Container (torch) only."""
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
G = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
sys.path.insert(0, str(PROJECT / "transport"))
import flexgreen_pm as fg  # noqa: E402  (derive(), static(): FLEXGREEN's meteorology, the T6 transport operator)

D2017 = dt.date(2017, 1, 1)
CALIB = Path(__file__).resolve().parent / "blh_calib.json"


def era5_B_day(d):
    """(4, 53, 181, 360) float32: ERA5 1 deg fields of day d at 00/06/12/18 UTC (Foehn wb2/era5_B_<y>_<m>.npy)"""
    return fg.npy_block(fg.WB / f"era5_B_{d.year}_{d.month:02d}.npy", d.day - 1).astype(np.float32)


def met_row(d):
    """(37, 576, 1440) float16 row of Foehn met.npy (2017-2022)"""
    return fg.npy_block(G / "met.npy", (d - D2017).days)


def up_1deg_to_025(a):
    """(..., 181, 360) on 1 deg points (lat 90..-90, lon -180..179) -> (..., 576, 1440) bilinear, periodic lon"""
    a = np.asarray(a, np.float32)
    lat = 84.0 - 0.25 * np.arange(576)
    lon = -180.0 + 0.25 * np.arange(1440)
    fi = 90.0 - lat
    i0 = np.floor(fi).astype(np.int64)
    wi = (fi - i0).astype(np.float32)
    fj = lon + 180.0
    j0 = np.floor(fj).astype(np.int64) % 360
    wj = (fj - np.floor(fj)).astype(np.float32)
    j1 = (j0 + 1) % 360
    top = a[..., i0, :][..., :, j0] * (1 - wj) + a[..., i0, :][..., :, j1] * wj
    bot = a[..., i0 + 1, :][..., :, j0] * (1 - wj) + a[..., i0 + 1, :][..., :, j1] * wj
    return top * (1 - wi)[:, None] + bot * wi[:, None]


_ST = {}


@torch.no_grad()
def blh_daily_from_B(B, device="cpu"):
    if device not in _ST:
        _ST[device] = fg.static(torch.device(device))
    zs, z0 = _ST[device]
    out = fg.derive(torch.as_tensor(np.asarray(B, np.float32), device=device), zs, z0)
    h = out[:, fg.C_PBL].cpu().numpy()                                      # (4, 181, 360) m, 100..4500
    x = np.stack([np.log(np.maximum(h.mean(0), 10.0)), np.log(np.maximum(h.max(0), 10.0)),
                  np.log(np.maximum(h.min(0), 10.0))])
    return up_1deg_to_025(x)


def blh_channels(B, device="cpu"):
    """encoder channels 31-33 (standardised as met.npy) from 4 instants of 1 deg fields, with the calibration of
    blh_calib.json: (3, 576, 1440) float32"""
    cal = json.loads(CALIB.read_text())
    st = np.load(G / "stats.npz")
    xm, xs = st["extra_mean"][:3], st["extra_std"][:3]
    x = blh_daily_from_B(B, device)
    a, b = np.array(cal["a"], np.float32), np.array(cal["b"], np.float32)
    y = a[:, None, None] + b[:, None, None] * x
    return np.nan_to_num((y - xm[:, None, None]) / xs[:, None, None], nan=0.0).astype(np.float32)
