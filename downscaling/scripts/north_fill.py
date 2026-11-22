#!/usr/bin/env python3
"""North of the LGHAP domain (60 N): there is no 1 km truth, so no CD-FFM. Fill = CAMS (or RT, its emulator) corrected
by a boreal quantile mapping (BQM): per calendar month, the quantiles of log1p CAMS day-0 at 0.25 deg land cells are mapped
onto those of the LGHAP cell means of the same cells and days, fitted in the boreal analogue band 50-60 N (training days
2019-07-11 .. 2022-12-31 only; lghap_clim/cellmean_train_A.npz = cell means of log1p(max(LGHAP, 1))). A raw fill
leaves a step at 60 N.
Transfer test (does a map fitted further south hold further north?): fit on 50-56 N with 2019-07 .. 2021, score on
56-60 N in 2022 (daily cell values): raw CAMS, a monthly mean-ratio correction, BQM.
Writes data/processed/global/north_bqm.npz (month, quantile levels, q_cams, q_lghap in log1p) + logs/north_fill.json.
    python north_fill.py"""
import datetime as dt
import json
from pathlib import Path

import numpy as np

P = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
G = P / "data/processed/global"
D2017 = dt.date(2017, 1, 1)
QL = np.linspace(0.005, 0.995, 199)


def read_row(path, i):
    import os
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = reader(f)
        off = f.tell()
    dtype = np.dtype(dtype)
    n = int(np.prod(shape[1:], dtype=np.int64))
    fd = os.open(path, os.O_RDONLY)
    try:
        buf = os.pread(fd, n * dtype.itemsize, off + i * n * dtype.itemsize)
    finally:
        os.close(fd)
    return np.frombuffer(buf, dtype).reshape(shape[1:])


def fit_bqm(cams, lghap, months):
    """per month (with the neighbouring months pooled): quantiles of log1p CAMS and of log1p LGHAP"""
    qc, ql = np.zeros((12, QL.size)), np.zeros((12, QL.size))
    for m in range(1, 13):
        sel = np.isin(months, [(m - 2) % 12 + 1, m, m % 12 + 1])
        a, b = cams[sel].ravel(), lghap[sel].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        qc[m - 1] = np.quantile(np.log1p(a[ok]), QL)
        ql[m - 1] = np.quantile(b[ok], QL)                  # already log1p (cell mean of log1p LGHAP)
    return qc, ql


def apply_bqm(x, month, qc, ql):
    """x (ug m-3, any shape) of a calendar month -> mapped ug m-3 (constant offset beyond the end quantiles)"""
    lx = np.log1p(np.maximum(x, 0))
    c, l = qc[month - 1], ql[month - 1]
    y = np.interp(lx, c, l)
    y = np.where(lx < c[0], l[0] + (lx - c[0]), y)
    y = np.where(lx > c[-1], l[-1] + (lx - c[-1]), y)
    return np.expm1(y)


def scores(pred, truth):
    ok = np.isfinite(pred) & np.isfinite(truth)
    p, t = pred[ok], truth[ok]
    return dict(rmse=float(np.sqrt(np.mean((p - t) ** 2))), bias=float(np.mean(p - t)), r=float(np.corrcoef(p, t)[0, 1]),
                mean_truth=float(t.mean()), n=int(ok.sum()))


def main():
    cm = np.load(G / "lghap_clim/cellmean_train_A.npz")
    cells, dates, lg = cm["cells"].astype(np.int64), cm["dates"].astype("datetime64[D]").astype(object), cm["mean"]
    lat = 84.0 - 0.25 * (cells // 1440)
    band = (lat >= 50) & (lat <= 60)
    cells_b, lat_b = cells[band], lat[band]
    lg_b = lg[:, band].astype(np.float64)                   # log1p space
    cams_path = G / "coarse/cams_pm25_025.npy"
    cams_b = np.stack([read_row(cams_path, (d - D2017).days).reshape(-1)[cells_b].astype(np.float64) for d in dates])
    months = np.array([d.month for d in dates])
    years = np.array([d.year for d in dates])
    print(f"boreal band 50-60 N: {band.sum():,} land cells x {len(dates)} training days", flush=True)
    # transfer test: fit 50-56 N, 2019-07 .. 2021; score 56-60 N, 2022
    fit_c, test_c = lat_b < 56, lat_b >= 56
    fit_d, test_d = years <= 2021, years == 2022
    qc, ql = fit_bqm(cams_b[fit_d][:, fit_c], lg_b[fit_d][:, fit_c], months[fit_d])
    ratio = np.array([np.nanmean(np.expm1(lg_b[fit_d & (months == m)][:, fit_c])) / np.nanmean(cams_b[fit_d & (months == m)][:, fit_c])
                      for m in range(1, 13)])
    ct, lt, mt = cams_b[test_d][:, test_c], np.expm1(lg_b[test_d][:, test_c]), months[test_d]
    bqm_t = np.stack([apply_bqm(ct[k], mt[k], qc, ql) for k in range(len(mt))])
    res = {"transfer_test": {"fit": "50-56 N, 2019-07-11..2021-12-31", "score": "56-60 N, 2022 (daily 0.25 deg land cells)",
                             "raw_cams": scores(ct, lt), "monthly_ratio": scores(ct * ratio[mt - 1][:, None], lt),
                             "bqm": scores(bqm_t, lt)}}
    for k, v in res["transfer_test"].items():
        if isinstance(v, dict):
            print(f"  {k:14s} RMSE {v['rmse']:.2f} bias {v['bias']:+.2f} R {v['r']:.3f} (truth mean {v['mean_truth']:.2f})")
    # final map: whole band, every training day
    qc, ql = fit_bqm(cams_b, lg_b, months)
    np.savez(G / "north_bqm.npz", levels=QL, q_cams=qc, q_lghap=ql,
             note=np.array("log1p quantiles per calendar month (neighbours pooled); fit: land cells 50-60 N, training days "
                           "2019-07-11..2022-12-31; apply north of the LGHAP domain with scripts/north_fill.apply_bqm"))
    res["final_fit"] = {"band": "50-60 N land cells", "cells": int(band.sum()), "days": len(dates),
                        "file": str(G / "north_bqm.npz")}
    (P / "logs/north_fill.json").write_text(json.dumps(res, indent=1))
    print("written", G / "north_bqm.npz", flush=True)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------------------------
# chosen fill (2026-10-28, from the LGHAP transfer test only; stations never used): boreal analogue climatology.
# Transfer test (fit 50-56 N 2019-07..2021, score 56-60 N 2022, daily 0.25 deg cells, truth mean 8.16 ug m-3):
#   raw CAMS RMSE 10.70 (bias -4.30, R 0.03) | BQM 11.63 (-0.49, 0.07) | monthly ratio 17.86 | one monthly climatology
#   6.03 (+1.15, 0.31) | monthly climatology per 60 deg longitude sector 5.96 (+2.56, 0.62).
# CAMS carries no day-to-day signal of LGHAP there (R 0.03): the fill is the per-month, per-60-deg-sector mean of log1p
# LGHAP over the land cells of the edge band 56-60 N (all training days), i.e. a climatology, labelled as such.
# ---------------------------------------------------------------------------------------------------------------
SECTOR_DEG = 60


def fit_north_clim():
    cm = np.load(G / "lghap_clim/cellmean_train_A.npz")
    cells, lg = cm["cells"].astype(np.int64), cm["mean"]
    months = np.array([d.month for d in cm["dates"].astype("datetime64[D]").astype(object)])
    lat, lon = 84.0 - 0.25 * (cells // 1440), -180.0 + 0.25 * (cells % 1440)
    edge = (lat >= 56) & (lat <= 60)
    sec = ((lon[edge] + 180) // SECTOR_DEG).astype(int)
    ns = 360 // SECTOR_DEG
    clim = np.full((12, ns), np.nan)
    for m in range(1, 13):
        X = lg[months == m][:, edge].astype(np.float64)
        for s in range(ns):
            if (sec == s).any():
                clim[m - 1, s] = np.nanmean(X[:, sec == s])
    clim = np.where(np.isfinite(clim), clim, np.nanmean(clim, 1, keepdims=True))
    np.savez(G / "north_clim.npz", log1p_mean=clim, sector_deg=SECTOR_DEG,
             note=np.array("per calendar month x 60 deg longitude sector (from 180 W): mean of log1p(max(LGHAP,1)) over "
                           "land cells 56-60 N, training days 2019-07-11..2022-12-31; fill north of the LGHAP domain"))
    return clim


def north_clim_value(month, lon, clim=None):
    """fill value (ug m-3) north of the LGHAP domain for a calendar month at longitudes lon"""
    clim = clim if clim is not None else np.load(G / "north_clim.npz")["log1p_mean"]
    s = ((np.asarray(lon) + 180) // SECTOR_DEG).astype(int) % clim.shape[1]
    return np.expm1(clim[month - 1][s])


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "clim":
    c = fit_north_clim()
    print(np.round(np.expm1(c), 1))


def fit_north_clim_smooth(sector_deg=10, smooth_deg=30):
    """v2 for display (2026-10-28, smoother northern edge): the same edge-band climatology on 10 deg
    sectors, circularly smoothed in longitude (Gaussian, sigma smooth_deg) -> no sector steps; a function of (month, lon)"""
    cm = np.load(G / "lghap_clim/cellmean_train_A.npz")
    cells, lg = cm["cells"].astype(np.int64), cm["mean"]
    months = np.array([d.month for d in cm["dates"].astype("datetime64[D]").astype(object)])
    lat, lon = 84.0 - 0.25 * (cells // 1440), -180.0 + 0.25 * (cells % 1440)
    edge = (lat >= 56) & (lat <= 60)
    ns = 360 // sector_deg
    sec = ((lon[edge] + 180) // sector_deg).astype(int)
    S, N = np.zeros((12, ns)), np.zeros((12, ns))
    for m in range(1, 13):
        X = lg[months == m][:, edge].astype(np.float64)
        for s in range(ns):
            v = X[:, sec == s]
            S[m - 1, s], N[m - 1, s] = np.nansum(v), np.isfinite(v).sum()
    centres = -180 + sector_deg * (np.arange(ns) + 0.5)
    d = (centres[:, None] - centres[None, :] + 180) % 360 - 180
    W = np.exp(-0.5 * (d / smooth_deg) ** 2)
    clim = (S @ W) / np.maximum(N @ W, 1)                     # count-weighted circular smoothing
    np.savez(G / "north_clim_smooth.npz", log1p_mean=clim, sector_deg=sector_deg, smooth_deg=smooth_deg,
             note=np.array("per calendar month x 10 deg sector centres (from 180 W), Gaussian-smoothed in longitude "
                           "(sigma 30 deg, count-weighted): mean log1p(max(LGHAP,1)), land cells 56-60 N, training days; "
                           "evaluate with north_clim_smooth_value (linear interpolation in longitude)"))
    return clim


def north_clim_smooth_value(month, lon, clim=None):
    z = np.load(G / "north_clim_smooth.npz")
    clim = clim if clim is not None else z["log1p_mean"]
    sd = int(z["sector_deg"])
    centres = -180 + sd * (np.arange(clim.shape[1]) + 0.5)
    x = np.concatenate([centres - 360, centres, centres + 360])
    y = np.tile(clim[month - 1], 3)
    return np.expm1(np.interp(np.asarray(lon, np.float64), x, y))
