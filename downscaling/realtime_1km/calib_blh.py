#!/usr/bin/env python3
"""BLH substitute for the real-time encoder channels 31-33 (log ERA5 boundary-layer height: daily mean, max, min).

ECMWF open data has no boundary-layer height (checked on the 2026-10-26 00z index: 43 surface parameters, no 'blh').
Substitute: the mixing height h that FLEXGREEN's derive() computes from the 1 deg profiles (critical bulk Richardson
number 0.25, FLEXPART 'richardson' with the thermal excess), at the 4 instants 00/06/12/18 UTC of the day, then the
daily mean / max / min, log(max(., 10 m)), bilinear 1 deg -> 0.25 deg, and a per-channel affine map a + b x fitted
HERE against ERA5's own BLH (Foehn met.npy channels 30-32, i.e. encoder channels 31-33) on training-period days (the same derive() run on ERA5 1 deg fields, wb2/era5_B, so the
fit sees exactly the operator used in real time; only the driving model differs: ERA5 here, IFS open data live).
No station, no LGHAP, no 2023-2024 day is used.

Fit days: every 35th day from 2019-08-01 (training split A: 2019-07-11 .. 2022-12-31), alternately assigned to the
fit set and to a check set (out-of-sample numbers). Scores in the standardised units the encoder sees ((x - mean) /
std with Foehn's stats.npz): RMSE of the substitute vs RMSE of the 'mean' fallback (channel = 0), and correlations.
Writes realtime_1km/blh_calib.json.  Container (torch CPU):  python -I calib_blh.py"""
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from blh import G, blh_daily_from_B, era5_B_day, met_row  # noqa: E402


def main():
    torch.set_num_threads(16)
    st = np.load(G / "stats.npz")
    xm, xs = st["extra_mean"][:3], st["extra_std"][:3]
    days = [dt.date(2019, 8, 1) + dt.timedelta(35 * k) for k in range(40)]
    days = [d for d in days if d <= dt.date(2022, 12, 20)]
    lat = 84.0 - 0.25 * np.arange(576)
    w = np.repeat(np.cos(np.deg2rad(lat))[:, None], 1440, 1)
    land = np.load(G / "meta.npz")["land"].astype(bool)
    rows = {"fit": [], "check": []}
    t0 = time.time()
    for k, d in enumerate(days):
        B = era5_B_day(d)                                                     # (4, 53, 181, 360) 00/06/12/18 UTC
        x = blh_daily_from_B(B)                                               # (3, 576, 1440) log(max(h, 10))
        y = met_row(d)[30:33].astype(np.float32) * xs[:, None, None] + xm[:, None, None]   # ERA5 log BLH (met.npy)
        rows["fit" if k % 2 == 0 else "check"].append((d, x, y))
        print(f"{d}: derived/ERA5 log-BLH means {[round(float(v), 2) for v in x.mean((1, 2))]} / "
              f"{[round(float(v), 2) for v in y.mean((1, 2))]} ({time.time() - t0:.0f}s)", flush=True)
    res = {"fit_days": [str(r[0]) for r in rows["fit"]], "check_days": [str(r[0]) for r in rows["check"]],
           "channels": ["log_blh_mean", "log_blh_max", "log_blh_min"], "a": [], "b": [], "check": {}}
    for c in range(3):
        X = np.concatenate([r[1][c].ravel() for r in rows["fit"]])
        Y = np.concatenate([r[2][c].ravel() for r in rows["fit"]])
        W = np.tile(w.ravel(), len(rows["fit"]))
        mx, my = np.average(X, weights=W), np.average(Y, weights=W)
        b = float(np.average((X - mx) * (Y - my), weights=W) / np.average((X - mx) ** 2, weights=W))
        a = float(my - b * mx)
        res["a"].append(a)
        res["b"].append(b)
        Xc = np.stack([r[1][c] for r in rows["check"]])
        Yc = np.stack([r[2][c] for r in rows["check"]])
        Wc = np.broadcast_to(w, Xc.shape)
        Lc = np.broadcast_to(land, Xc.shape)
        sub = ((a + b * Xc) - xm[c]) / xs[c]
        tru = (Yc - xm[c]) / xs[c]
        rmse = lambda e_, m_=None: float(np.sqrt(np.average(e_ ** 2, weights=Wc if m_ is None else Wc * m_)))
        corr = lambda p, q, m_=None: float(np.corrcoef(p[m_] if m_ is not None else p.ravel(),
                                                       q[m_] if m_ is not None else q.ravel())[0, 1])
        # anomaly correlation: the day-to-day part (each cell's mean over the check days removed)
        an = lambda v: v - v.mean(0, keepdims=True)
        res["check"][res["channels"][c]] = {
            "rmse_std_units_substitute": rmse(sub - tru), "rmse_std_units_mean_fallback": rmse(tru),
            "rmse_std_units_substitute_land": rmse(sub - tru, Lc), "rmse_std_units_mean_fallback_land": rmse(tru, Lc),
            "corr_all": corr(sub, tru), "corr_land": corr(sub, tru, Lc),
            "anomaly_corr_all": corr(an(sub), an(tru)), "bias_std_units": float(np.average(sub - tru, weights=Wc))}
        print(res["channels"][c], f"a={a:.3f} b={b:.3f}", json.dumps(res["check"][res["channels"][c]]), flush=True)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    res["note"] = ("derive() of transport/flexgreen_pm.py (= Foehn flexgreen.py) on 1 deg fields at 00/06/12/18 UTC; "
                   "ERA5 BLH daily stats are over 24 hourly values")
    (HERE / "blh_calib.json").write_text(json.dumps(res, indent=1))
    print(f"written {HERE / 'blh_calib.json'}", flush=True)


if __name__ == "__main__":
    main()
