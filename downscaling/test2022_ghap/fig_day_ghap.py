#!/usr/bin/env python3
"""One day of the GHAP 2022 pass (style of final2024/fig_day.py): world maps (Robinson) of (A) monitors, (B) GHAP V1,
(C) CAMS 0.4 deg day 0, (D) CD-FFM 1 km ensemble mean (GHAP system; outside its mask = permanent ice: CAMS, lighter),
with the day's RMSE | R | bias at the monitors on C and D; then 1 km zooms (rows: GHAP, CAMS, CD-FFM) over Delhi, the Po
Valley and a northern city (Yakutsk by default). Reads saved files only (test2022_ghap/fields, preds_st, st_truth_2022);
GHAP 2022 of that day from ghap1km_global.npy, logged in logs/UNSEALING.md; the system is not changed (A2(d)).
    python fig_day_ghap.py 2022-11-18 [--north 62.03,129.73,Yakutsk]"""
import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np

P = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
sys.path.insert(0, str(P / "final2024"))
sys.path.insert(0, str(P / "coupling"))
import fig_world_results as F  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import common  # noqa: E402

O = P / "test2022_ghap"
RES = O / "results"
GH = P / "data/processed/ghap"
SRC = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km/ghap1km_global.npy")
NR, NC = 14400, 36000
EXT = [-180.0, 180.0, -60.0, 84.0]                                  # lattice edges (row 0 centre 83.995)


def to_grid(vals):
    flat = np.load(GH / "grid/ghap_mask.npy").astype(np.int64)
    g = np.full(NR * NC, np.nan, np.float32)
    g[flat] = vals
    return g.reshape(NR, NC)


def ghap_day(d):
    with open(SRC, "rb") as f:
        ver = np.lib.format.read_magic(f)
        np.lib.format.read_array_header_1_0(f) if ver == (1, 0) else np.lib.format.read_array_header_2_0(f)
        hdr = f.tell()
    a = np.empty(NR * NC, np.float16)
    mv = memoryview(a.view(np.uint8))
    fd = os.open(SRC, os.O_RDONLY)
    off, pos = hdr + (d - dt.date(2017, 1, 1)).days * NR * NC * 2, 0
    while pos < len(mv):
        pos += os.preadv(fd, [mv[pos:]], off + pos)
    os.close(fd)
    g = a.astype(np.float32).reshape(NR, NC)
    g[~(np.isfinite(g) & (g > 0))] = np.nan
    return g


def coarsen(g, k):
    H, W = g.shape[0] // k * k, g.shape[1] // k * k
    with np.errstate(invalid="ignore"):
        return np.nanmean(g[:H, :W].reshape(H // k, k, W // k, k), axis=(1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--north", default="62.03,129.73,Yakutsk")
    a = ap.parse_args()
    d = dt.date.fromisoformat(a.date)
    t = (d - dt.date(2022, 1, 1)).days
    nla, nlo, nname = a.north.split(",")
    boxes = [(26.4, 31.0, 74.9, 79.5, "Delhi", 28.61, 77.21), (43.6, 46.4, 7.0, 13.0, "Po Valley", 45.46, 9.19),
             (float(nla) - 1.6, float(nla) + 1.6, float(nlo) - 3.4, float(nlo) + 3.4, nname, float(nla), float(nlo))]
    model = to_grid(np.load(O / f"fields/cdffm/{d}.npz")["ensmean"].astype(np.float32))
    truth = ghap_day(d)
    with open(P / "logs/UNSEALING.md", "a") as f:
        f.write(f"- {dt.datetime.now():%Y-%m-%d %H:%M:%S} (Helsinki), [ghap] fig_day_ghap.py read the GHAP 2022 field of {d} "
                f"(visualisation of the frozen pass; system unchanged)\n")
    cams = common.read_row(P / "data/processed/global/coarse/cams_pm25_025.npy", (d - dt.date(2017, 1, 1)).days)
    cams = np.where(F.land_mask_025(), cams.astype(np.float32), np.nan)
    zt = np.load(O / "st_truth_2022.npz")
    n = int(zt["pix"].max()) + 1
    s, c = np.zeros(n), np.zeros(n)
    v = zt["daily"][:, t]
    ok = np.isfinite(v)
    np.add.at(s, zt["pix"][ok], v[ok])
    np.add.at(c, zt["pix"][ok], 1)
    with np.errstate(invalid="ignore"):
        st = s / c
    pos = np.load(O / "st_pix_2022.npz")["pos"]
    flat = np.asarray(np.load(GH / "grid/ghap_mask.npy", mmap_mode="r")[pos], np.int64)
    s_lat, s_lon = 83.995 - 0.01 * (flat // NC), -179.995 + 0.01 * (flat % NC)
    arm = {k: np.maximum(np.load(O / f"preds_st_mon/{k}/{d}.npz")["members"].astype(np.float32), 0).mean(0)
           for k in ("cams_bilinear", "cdffm")}
    S = {k: F.stats(v_, st) for k, v_ in arm.items()}
    print(d, S, flush=True)
    ext025 = [-180.125, 179.875, -59.875, 84.125]
    m5, t5 = coarsen(model, 5), coarsen(truth, 5)
    fig = plt.figure(figsize=(7.2, 9.6))
    for k in range(4):
        ax = fig.add_axes([0.005 + 0.5 * (k % 2), 0.765 - 0.215 * (k // 2), 0.49, 0.205], projection=F.ROB)
        F.base_map(ax)
        if k == 0:
            o = np.argsort(np.nan_to_num(st))
            ax.scatter(s_lon[o], s_lat[o], c=st[o], s=0.7, lw=0, cmap=F.CMAP, norm=F.NORM, transform=F.PC, zorder=5)
        elif k == 1:
            ax.imshow(t5, origin="upper", extent=EXT, transform=F.PC, cmap=F.CMAP, norm=F.NORM, interpolation="nearest",
                      regrid_shape=(1500, 3000), zorder=2)
        elif k == 2:
            ax.imshow(cams, origin="upper", extent=ext025, transform=F.PC, cmap=F.CMAP, norm=F.NORM, interpolation="nearest",
                      regrid_shape=(900, 1800), zorder=2)
            F.box(ax, S["cams_bilinear"], "CAMS 0.4°")
        else:
            ax.imshow(cams, origin="upper", extent=ext025, transform=F.PC, cmap=F.CMAP, norm=F.NORM, alpha=0.45,
                      interpolation="nearest", regrid_shape=(900, 1800), zorder=1.8)
            ax.imshow(m5, origin="upper", extent=EXT, transform=F.PC, cmap=F.CMAP, norm=F.NORM, interpolation="nearest",
                      regrid_shape=(1500, 3000), zorder=2)
            F.box(ax, S["cdffm"], "CD-FFM 1 km")
        if k > 0:
            for la0, la1, lo0, lo1, *_ in boxes:
                ax.plot([lo0, lo1, lo1, lo0, lo0], [la0, la0, la1, la1, la0], transform=F.PC, color="black", lw=0.4, zorder=6)
        ax.text(0.01, 0.98, "ABCD"[k], transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    letters = iter("EFGHIJKLM")
    for bi, (la0, la1, lo0, lo1, name, cla, clo) in enumerate(boxes):
        r0, r1 = int((84.0 - la1) / 0.01), int((84.0 - la0) / 0.01)
        c0, c1 = int((lo0 + 180.0) / 0.01), int((lo1 + 180.0) / 0.01)
        rc0, rc1 = int((84.125 - la1) / 0.25), int((84.125 - la0) / 0.25) + 1
        cc0, cc1 = int((lo0 + 180.125) / 0.25), int((lo1 + 180.125) / 0.25) + 1
        zooms = [(truth[r0:r1, c0:c1], [lo0, lo1, la0, la1], "GHAP V1, 1 km"),
                 (cams[rc0:rc1, cc0:cc1], [-180.125 + 0.25 * cc0, -180.125 + 0.25 * cc1, 84.125 - 0.25 * rc1,
                                           84.125 - 0.25 * rc0], "CAMS 0.4°"),
                 (model[r0:r1, c0:c1], [lo0, lo1, la0, la1], "CD-FFM, 1 km")]
        for k, (img, ext, lab) in enumerate(zooms):
            ax = fig.add_axes([0.01 + 0.33 * k, 0.335 - 0.135 * bi, 0.31, 0.125], projection=F.PC)
            ax.set_extent([lo0, lo1, la0, la1], crs=F.PC)
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="#555555", linewidth=0.3, zorder=4)
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="#777777", linewidth=0.3, zorder=4)
            ax.imshow(img, origin="upper", extent=ext, transform=F.PC, cmap=F.CMAP, norm=F.NORM, interpolation="nearest", zorder=2)
            ax.plot(clo, cla, marker="o", ms=2.2, mfc="none", mec="black", mew=0.5, transform=F.PC, zorder=5)
            ax.text(0.02, 0.03, f"{lab}  ({name})", transform=ax.transAxes, fontsize=4.8, va="bottom", zorder=6,
                    bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="#999999", lw=0.4))
            ax.text(0.01, 0.98, next(letters), transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", color="white")
    cax = fig.add_axes([0.2, 0.045, 0.6, 0.010])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=F.NORM, cmap=F.CMAP), cax=cax, orientation="horizontal", extend="max",
                      ticks=F.PM_BOUNDS)
    cb.set_label(f"PM$_{{2.5}}$, {d:%d %B %Y} (µg m$^{{-3}}$)", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5)
    out = RES / f"fig_day_ghap_{d}.png"
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.03)
    print(out, flush=True)


if __name__ == "__main__":
    main()
