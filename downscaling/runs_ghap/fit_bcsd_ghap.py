#!/usr/bin/env python3
"""GHAP version (PROTOCOL.md A2(d); paths and split G = 2017-2020; NaN cell means = cells without a valid pixel that day, ignored) of the B2 BCSD baseline of PROTOCOL.md section 3, fitted on the 1,270 split-A training days ONLY (2019-07-11..2022-12-31).

For each 0.25 deg cell c (the 194,261 cells of grid/cell_id.npy) and calendar month m: a 21-quantile map (probability
levels 0, 0.05, ..., 1) from x = log1p CAMS day-0 PM2.5 at the cell's 0.25 deg point (coarse/cams_pm25_025.npy, the
cell value) to y = the cell mean of log1p(max(LGHAP, 1)) over the cell's mask pixels (lghap_clim/cellmean_train_A.npz,
exact daily means from the full fields), using that month's training days. Linear between quantiles, constant offset
(slope 1) beyond the end quantiles. Disaggregation (applied at evaluation, apply_bcsd below):
    y_p(t) = mu_p(t) - mubar_c(t) + QM_c,m(t)(x_c(t)),  mubar_c(t) = sum_k abar_ck b_k(doy(t)),
abar_ck = mean over the cell's mask pixels of the K = 2 harmonic coefficients of mu_p (lghap_clim/clim_mean_A.npy),
i.e. mubar_c(t) is exactly the cell mean of mu_p(t). Point forecast in ug m-3: expm1(y_p). B2d (dressing) is fitted on
2023 later (PROTOCOL.md 3), not here. The standard Wood et al. (2004) form adopted in PROTOCOL.md Amendment A1(d)
(R1-F3), y_p = mu_p(t) + I[QM_c(x_c) - mubar_c(t)](p) with I = bilinear interpolation over the 0.25 deg cell centres
(periodic in longitude), uses the same fitted arrays (qx, qy, clim_cell_coef): only the disaggregation differs.
Ties: the x quantiles get + 1e-6 x level index so np.interp sees a strictly increasing abscissa.
No sealed day, no 2023/2024 truth and no station is read (dates asserted inside the training split).

  python fit_bcsd.py fit     -> data/processed/global/lghap_clim/bcsd_A.npz + bcsd_A.json (counts, in-sample checks)
  python fit_bcsd.py check   -> independent recompute (sort + linear interpolation) on a few cells and months
World venv (numpy): /projappl/project_462001804/envs/world/bin/python, partition small."""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
G = ROOT / "data/processed/global"
GH = ROOT / "data/processed/ghap"
OUT = GH / "ghap_clim/bcsd_G.npz"
LEVELS = np.linspace(0.0, 1.0, 21)
D2017 = dt.date(2017, 1, 1)
TRAIN = (dt.date(2017, 1, 1), dt.date(2020, 12, 31))
EPS_TIE = 1e-6


def read_rows(path, rows):
    """rows of a C-order .npy (no memory map, login-node safe): (len(rows),) + row shape"""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        rd = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(f)
        off = f.tell()
        assert not fortran
        dtype = np.dtype(dtype)
        rs = tuple(shape[1:])
        nb = int(np.prod(rs)) * dtype.itemsize
        out = np.empty((len(rows),) + rs, dtype)
        for i, r in enumerate(rows):
            f.seek(off + int(r) * nb)
            out[i] = np.frombuffer(f.read(nb), dtype).reshape(rs)
    return out


def harmonic_basis(doy):
    ang = 2 * np.pi * np.asarray(doy, np.float64) / 365.0
    return np.stack([np.ones_like(ang), np.cos(ang), np.sin(ang), np.cos(2 * ang), np.sin(2 * ang)], -1)


def qmap(x, qx, qy):
    """quantile map of one cell-month: linear inside [qx0, qx20], slope-1 offsets outside"""
    y = np.interp(x, qx, qy)
    y = np.where(x < qx[0], qy[0] + (x - qx[0]), y)
    return np.where(x > qx[-1], qy[-1] + (x - qx[-1]), y)


def apply_bcsd(bcsd, cell_pos, month, x_cell, mu_p, doy):
    """BCSD log1p prediction at pixels: bcsd = np.load(bcsd_A.npz); cell_pos (n,) positions of the pixels' cells in
    bcsd['cells']; month 1..12; x_cell (n_cells,) log1p CAMS of the day at every cell; mu_p (n,) climatology mean of
    the pixels on that day; doy 0-based -> y_p (n,) log1p ug m-3"""
    qx, qy = bcsd["qx"][:, month - 1], bcsd["qy"][:, month - 1]
    u = np.unique(cell_pos)
    qm = np.empty(bcsd["qx"].shape[0], np.float64)
    for c in u:
        qm[c] = qmap(x_cell[c], qx[c], qy[c])
    mubar = bcsd["clim_cell_coef"] @ harmonic_basis(doy)
    return mu_p - mubar[cell_pos] + qm[cell_pos]


def cmd_fit(a):
    t0 = time.time()
    cm = np.load(GH / "ghap_clim/cellmean_train_G.npz")
    cells = cm["cells"].astype(np.int64)
    dates = [dt.date.fromisoformat(str(d)) for d in cm["dates"].astype("datetime64[D]").astype(str)]
    assert len(dates) == json.loads((GH / "manifest.json").read_text())["split_G"]["n_train_days"] and dates[0] >= TRAIN[0] and dates[-1] <= TRAIN[1]
    assert all(TRAIN[0] <= d <= TRAIN[1] for d in dates)  # listed training days only (A2(a))
    y = cm["mean"].astype(np.float64)                                          # (1270, nc) log1p LGHAP cell means
    cid = np.load(GH / "grid/cell_id.npy").astype(np.int64)
    ucells, inv, cnt = np.unique(cid, return_inverse=True, return_counts=True)
    assert np.array_equal(ucells, cells), "cellmean_train_A cells != unique(cell_id)"
    rows = [(d - D2017).days for d in dates]
    cams = read_rows(G / "coarse/cams_pm25_025.npy", rows)                    # (1270, 576, 1440) f16
    W = cams.shape[2]
    x = np.log1p(np.maximum(cams.reshape(len(rows), -1)[:, cells].astype(np.float64), 0.0))
    del cams
    print(f"inputs: {len(dates)} training days x {cells.size} cells ({time.time() - t0:.0f} s)", flush=True)
    months = np.array([d.month for d in dates])
    nc = cells.size
    qx = np.zeros((nc, 12, LEVELS.size), np.float32)
    qy = np.zeros((nc, 12, LEVELS.size), np.float32)
    ndays = np.zeros(12, np.int32)
    for m in range(1, 13):
        sel = months == m
        ndays[m - 1] = sel.sum()
        qx_m = np.quantile(x[sel], LEVELS, axis=0).T + EPS_TIE * np.arange(LEVELS.size)[None]
        qx[:, m - 1] = np.maximum.accumulate(qx_m, axis=1)
        qy[:, m - 1] = np.nanquantile(y[sel], LEVELS, axis=0).T
    # cell mean of the harmonic mean coefficients (mubar_c(t) = exact cell mean of mu_p(t))
    clim = np.load(GH / "ghap_clim/clim_mean_G.npy")                         # (Nm, 5) float32
    coef = np.zeros((nc, 5), np.float64)
    for k in range(5):
        coef[:, k] = np.bincount(inv, weights=clim[:, k].astype(np.float64), minlength=nc) / cnt
    del clim
    # in-sample diagnostics (training days only): cell-mean log1p RMSE of raw CAMS vs BCSD
    pred = np.empty_like(x)
    for m in range(1, 13):
        sel = np.nonzero(months == m)[0]
        xs = x[sel]
        q0, q1 = qx[:, m - 1].astype(np.float64), qy[:, m - 1].astype(np.float64)
        for j in range(nc):
            pred[sel, j] = qmap(xs[:, j], q0[j], q1[j])
    rmse_raw = float(np.sqrt(np.nanmean((x - y) ** 2)))
    rmse_bcsd = float(np.sqrt(np.nanmean((pred - y) ** 2)))
    np.savez(OUT, cells=cells.astype(np.int32), levels=LEVELS, qx=qx, qy=qy, clim_cell_coef=coef,
             n_pixels_per_cell=cnt.astype(np.int32), ndays_per_month=ndays, grid_W=W,
             train_first_last=np.array([str(TRAIN[0]), str(TRAIN[1])]))
    info = {"file": str(OUT), "n_cells": int(nc), "training_days": len(dates), "train": [str(TRAIN[0]), str(TRAIN[1])],
            "days_per_month": ndays.tolist(), "levels": LEVELS.size, "x": "log1p CAMS day-0 at the cell's 0.25 deg point",
            "y": "cell mean of log1p(max(LGHAP, 1)) (cellmean_train_A.npz)",
            "in_sample_training_days_cellmean_log1p_rmse": {"raw_cams": rmse_raw, "bcsd": rmse_bcsd},
            "qx_strictly_increasing": bool((np.diff(qx, axis=2) > 0).all()),
            "qy_non_decreasing": bool((np.diff(qy, axis=2) >= 0).all()), "seconds": time.time() - t0}
    (GH / "ghap_clim/bcsd_G.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1), flush=True)


def cmd_check(a):
    """independent recompute on a few cells x months: sorted sample, linear-interpolated order statistics (numpy's
    default quantile definition, h = (n - 1) p), then the map at 3 test values"""
    b = np.load(OUT)
    cm = np.load(GH / "ghap_clim/cellmean_train_G.npz")
    dates = [dt.date.fromisoformat(str(d)) for d in cm["dates"].astype("datetime64[D]").astype(str)]
    months = np.array([d.month for d in dates])
    rng = np.random.default_rng(3)
    cells = b["cells"].astype(np.int64)
    pick = rng.choice(cells.size, 5, replace=False)
    rows = [(d - D2017).days for d in dates]
    cams = read_rows(G / "coarse/cams_pm25_025.npy", rows).reshape(len(rows), -1)
    errs = []
    for j in pick:
        xs_all = np.log1p(np.maximum(cams[:, cells[j]].astype(np.float64), 0.0))
        ys_all = cm["mean"][:, j].astype(np.float64)
        for m in rng.choice(np.arange(1, 13), 2, replace=False):
            sel = months == m

            def oq(v):
                v = np.sort(v)
                h = (v.size - 1) * LEVELS
                lo = np.floor(h).astype(int)
                hi = np.minimum(lo + 1, v.size - 1)
                return v[lo] + (h - lo) * (v[hi] - v[lo])

            qx_ref = oq(xs_all[sel]) + EPS_TIE * np.arange(LEVELS.size)
            qy_ref = oq(ys_all[sel])
            errs.append(float(np.abs(qx_ref - b["qx"][j, m - 1]).max()))
            errs.append(float(np.abs(qy_ref - b["qy"][j, m - 1]).max()))
            t = [qx_ref[0] - 0.5, qx_ref[-1] + 0.3]
            ref = [qy_ref[0] - 0.5, qy_ref[-1] + 0.3]
            if qx_ref[4] - qx_ref[3] > 1e-3:                    # interior point (skipped on near-ties)
                t.append(0.5 * (qx_ref[3] + qx_ref[4]))
                ref.append(0.5 * (qy_ref[3] + qy_ref[4]))
            mine = qmap(np.array(t), b["qx"][j, m - 1].astype(np.float64), b["qy"][j, m - 1].astype(np.float64))
            errs.append(float(np.abs(mine - np.array(ref)).max()))
    res = {"max_abs_err": max(errs), "n_checks": len(errs), "ok": max(errs) < 1e-4}
    (GH / "ghap_clim/bcsd_G_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "check"])
    a = ap.parse_args()
    {"fit": cmd_fit, "check": cmd_check}[a.cmd](a)
