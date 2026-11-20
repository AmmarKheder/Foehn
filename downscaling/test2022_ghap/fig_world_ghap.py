#!/usr/bin/env python3
"""World maps (Robinson) of the GHAP 2022 single pass (PROTOCOL.md A2(d)), in the style of final2024/fig_world_results.py
(PM2.5 classes, white ocean, land without data #F0F0F0, bold letters, RMSE | R | bias boxes, no titles). Reads saved
files only (test2022_ghap/results/accum_2022.npz, preds_st, st_truth_2022.npz); the system is not changed.
  fig 1  test2022_ghap/results/fig_world_2022_ghap.png: (A) monitors, 2022 mean; (B) GHAP V1 2022; (C) CAMS 0.4 deg day 0;
         (D) CD-FFM 1 km (GHAP system; outside its mask, i.e. permanent ice and sea: CAMS, lighter).
         C, D: daily RMSE, R and bias at the monitors (pixel-days, monitors in one pixel averaged).
  fig 2  test2022_ghap/results/fig_world_stations_delta_2022.png: per pixel-station RMSE change, CD-FFM 1 km - CAMS 0.4 deg
    python fig_world_ghap.py"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

P = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
sys.path.insert(0, str(P / "final2024"))
sys.path.insert(0, str(P / "coupling"))
import fig_world_results as F  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

O = P / "test2022_ghap"
RES = O / "results"
GH = P / "data/processed/ghap"
Y0, ND = dt.date(2022, 1, 1), 365


def sub_latlon():
    flat = np.load(GH / "grid/ghap_mask.npy")[::8].astype(np.int64)
    return 83.995 - 0.01 * (flat // 36000), -179.995 + 0.01 * (flat % 36000)


def st_latlon():
    pos = np.load(O / "st_pix_2022.npz")["pos"]
    flat = np.asarray(np.load(GH / "grid/ghap_mask.npy", mmap_mode="r")[pos], np.int64)
    return 83.995 - 0.01 * (flat // 36000), -179.995 + 0.01 * (flat % 36000)


def cams_mean_2022():
    path = P / "data/processed/global/coarse/cams_pm25_025.npy"
    r0 = (Y0 - dt.date(2017, 1, 1)).days
    acc = np.zeros((576, 1440), np.float64)
    import common
    for k in range(ND):
        acc += common.read_row(path, r0 + k).astype(np.float32)
    return np.where(F.land_mask_025(), acc / ND, np.nan)


def preds(arm):
    out = []
    for k in range(ND):
        d = Y0 + dt.timedelta(k)
        m = np.load(O / f"preds_st_mon/{arm}/{d}.npz")["members"].astype(np.float32)
        out.append(np.maximum(m, 0).mean(0))
    return np.stack(out, 1)


def fig1():
    z = np.load(RES / "accum_2022.npz")
    lat, lon = sub_latlon()
    nok = np.maximum(z["n_ok"].astype(np.float64), 1)
    g_ghap = F.binned(lat, lon, np.where(z["n_ok"] > 0, z["y"].astype(np.float64) / nok, np.nan))
    g_cdffm = F.binned(lat, lon, z["mean_cdffm"].astype(np.float64) / ND)
    cams = cams_mean_2022()
    T, _ = F.st_pixel_days(O / "st_truth_2022.npz", 2022)
    s_lat, s_lon = st_latlon()
    pc, pm = preds("cams_bilinear"), preds("cdffm")
    S = {"cams": F.stats(pc, T), "cdffm": F.stats(pm, T)}
    if (O / "preds_st_mon/ghap").exists():
        S["ghap"] = F.stats(preds("ghap"), T)
    st_mean = np.nanmean(np.where(np.isfinite(T).sum(1, keepdims=True) >= 30, T, np.nan), 1)
    print("station stats (pixel-days):", json.dumps(S), flush=True)
    fig = plt.figure(figsize=(7.2, 4.6))
    ext = [-180, 180, -90, 90]
    ext025 = [-180.125, 179.875, -59.875, 84.125]
    for k in range(4):
        ax = fig.add_axes([0.005 + 0.5 * (k % 2), 0.56 - 0.44 * (k // 2), 0.49, 0.38], projection=F.ROB)
        F.base_map(ax)
        if k == 0:
            o = np.argsort(np.nan_to_num(st_mean))
            ax.scatter(s_lon[o], s_lat[o], c=st_mean[o], s=0.6, lw=0, cmap=F.CMAP, norm=F.NORM, transform=F.PC, zorder=5)
        elif k == 1:
            ax.imshow(g_ghap, origin="upper", extent=ext, transform=F.PC, cmap=F.CMAP, norm=F.NORM, interpolation="nearest",
                      regrid_shape=(1800, 3600), zorder=1)
            if "ghap" in S:
                F.box(ax, S["ghap"], "GHAP V1 (fitted to these monitors)")
        elif k == 2:
            ax.imshow(cams, origin="upper", extent=ext025, transform=F.PC, cmap=F.CMAP, norm=F.NORM,
                      interpolation="nearest", regrid_shape=(900, 1800), zorder=1)
            F.box(ax, S["cams"], "CAMS 0.4°")
        else:
            ax.imshow(cams, origin="upper", extent=ext025, transform=F.PC, cmap=F.CMAP, norm=F.NORM, alpha=0.45,
                      interpolation="nearest", regrid_shape=(900, 1800), zorder=1)
            ax.imshow(g_cdffm, origin="upper", extent=ext, transform=F.PC, cmap=F.CMAP, norm=F.NORM, interpolation="nearest",
                      regrid_shape=(1800, 3600), zorder=2)
            F.box(ax, S["cdffm"], "CD-FFM 1 km (GHAP-trained)")
        ax.text(0.01, 0.98, "ABCD"[k], transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    cax = fig.add_axes([0.2, 0.06, 0.6, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=F.NORM, cmap=F.CMAP), cax=cax, orientation="horizontal", extend="max",
                      ticks=F.PM_BOUNDS)
    cb.set_label("PM$_{2.5}$, 2022 mean (µg m$^{-3}$)", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5)
    out = RES / "fig_world_2022_ghap.png"
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.03)
    return S, out


def fig2():
    T, _ = F.st_pixel_days(O / "st_truth_2022.npz", 2022)
    d = F.per_station_rmse(preds("cdffm"), T) - F.per_station_rmse(preds("cams_bilinear"), T)
    fig = plt.figure(figsize=(3.6, 2.35))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-10, vmax=10)
    ax = fig.add_axes([0.005, 0.17, 0.99, 0.78], projection=F.ROB)
    F.base_map(ax)
    la, lo = st_latlon()
    o = np.argsort(np.abs(np.nan_to_num(d)))
    ax.scatter(lo[o], la[o], c=d[o], s=0.8, lw=0, cmap="RdBu_r", norm=norm, transform=F.PC, zorder=5)
    ok = np.isfinite(d)
    ax.text(0.02, 0.03, f"CD-FFM 1 km − CAMS 0.4°, 2022\n1 km better at {np.mean(d[ok] < 0):.0%} of {ok.sum():,} pixel-stations",
            transform=ax.transAxes, fontsize=4.8, va="bottom", zorder=6,
            bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#999999", lw=0.4))
    cax = fig.add_axes([0.2, 0.1, 0.6, 0.035])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), cax=cax, orientation="horizontal", extend="both")
    cb.set_label("RMSE change at the monitor, 1 km − 0.4° (µg m$^{-3}$)", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5)
    out = RES / "fig_world_stations_delta_2022.png"
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    print("share better", float(np.mean(d[ok] < 0)), out, flush=True)


if __name__ == "__main__":
    print(fig1())
    fig2()
