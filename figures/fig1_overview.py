#!/usr/bin/env python3
"""Fig. 1 of the Nature paper: one forecast of the whole planet at 1 km. a, the forecast (ensemble mean) issued on
day d for day d+1 over the globe, with the four regions of b. The day is chosen by a rule fixed in advance: the day of
2022 with the highest population-weighted PM2.5 (GHAP). b, the same forecast in four regions next to the operational
CAMS forecast (40 km) and the GHAP reference (1 km, not a forecast); dots: station measurements of that day (same
colours). Numbers: RMSE over the whole of 2022 (lead time 1 day) at the stations of the region, identical
station-days for the three rows.
    python fig1_overview.py <run> [start doy] [lead]"""
import os
import sys
from pathlib import Path

import numpy as np
import zarr

from nature_style import DOUBLE, letter, plt, pm_cmap, save, world_axes, PM_BOUNDS

RUN = sys.argv[1] if len(sys.argv) > 1 else "fp_flow_shift25_global_L7_fix"
D0 = int(sys.argv[2]) if len(sys.argv) > 2 else 19
H = int(sys.argv[3]) if len(sys.argv) > 3 else 1
DT = D0 + H
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
E = Path("/scratch/project_462001140/ammar/eccv/cran-pm/experiments/global1km/runs") / RUN / "eval_global"
date = (np.datetime64("2022-01-01") + np.timedelta64(DT, "D")).astype("datetime64[D]")
ours = np.load(E / f"map1km_d{D0}_h{H}.npy", mmap_mode="r")


def tiles(name, k):
    with open(G / name, "rb") as f:
        np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    n = 29 * 71 * 512 * 512
    fd = os.open(G / name, os.O_RDONLY)
    a = np.frombuffer(os.pread(fd, n * 2, hdr + k * n * 2), np.float16).reshape(29, 71, 512, 512)
    os.close(fd)
    return a.transpose(0, 2, 1, 3).reshape(29 * 512, 71 * 512)[:14400, :36000]


ghap = tiles("ghap1km_tiles.npy", 1826 + DT)
land = tiles("static1km_tiles.npy", 1) > 0
cams = np.asarray(zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")[D0, 0, H], np.float32)
st = np.load(G / "stations_2022.npz")
obs = st["daily"][:, DT]; okst = np.isfinite(obs)
slat, slon, sobs = st["lat"][okst], st["lon"][okst], obs[okst]
cmap, norm = pm_cmap()
# yearly RMSE (lead 1 day) at every station: this work, CAMS, GHAP on identical station-days
import glob
Pp = [np.load(p) for p in sorted(glob.glob(str(E / "part_*.npz")))]
kk = np.concatenate([p["keys"] for p in Pp]); ee = np.concatenate([p["st_ens"] for p in Pp]); gg = np.concatenate([p["st_ghap"] for p in Pp])
assert ee.shape[1] == len(st["lat"]), "station file differs from the evaluation"
czarr = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
la_all, lo_all = st["lat"], st["lon"]
fi = (90.0 - la_all) / 0.4; fj = (lo_all % 360.0) / 0.4
i0 = np.clip(np.floor(fi).astype(int), 0, 449); j0 = np.floor(fj).astype(int) % 900; j1 = (j0 + 1) % 900; wi, wj = fi - i0, fj - np.floor(fj)
SSE = np.zeros((len(la_all), 3)); NSD = np.zeros(len(la_all))
for n in np.where(kk[:, 1] == 1)[0]:
    d, t = int(kk[n, 0]), int(kk[n, 2])
    c = np.asarray(czarr[d, 0, 1], np.float32)
    cs = (1 - wi) * ((1 - wj) * c[i0, j0] + wj * c[i0, j1]) + wi * ((1 - wj) * c[i0 + 1, j0] + wj * c[i0 + 1, j1])
    X = np.stack([ee[n].astype(np.float32).mean(-1), cs, gg[n].astype(np.float32)]); o = st["daily"][:, t]
    m = np.isfinite(o) & np.all(np.isfinite(X), 0)
    SSE[m] += ((X[:, m] - o[m]) ** 2).T; NSD[m] += 1
print("yearly station RMSE computed on", int(NSD.sum()), "station-days")
CENTRES = {"North China Plain": (38.3, 116.6, 700), "Delhi and the Ganges plain": (28.7, 77.6, 700),
           "Po Valley and the Alps": (45.35, 10.0, 600), "Los Angeles basin": (34.05, -117.9, 300)}
ASPECT = 1.5                                                       # width / height of every zoom (km)
REG = {}
for nm, (la, lo, wkm) in CENTRES.items():
    dla = wkm / ASPECT / 111.2 / 2; dlo = wkm / (111.2 * np.cos(np.deg2rad(la))) / 2
    REG[nm] = (la + dla, la - dla, lo - dlo, lo + dlo, wkm)

fig = plt.figure(figsize=(DOUBLE, 6.9))
# a: the globe at 1 km (ensemble mean), regions of b
import cartopy.crs as ccrs
ax = world_axes(fig, [0.01, 0.585, 0.98, 0.41], bg=False)
import cartopy.feature as cfeature
ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f2f5f8", edgecolor="none", zorder=0)
ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e4e4e4", edgecolor="none", zorder=0)
m = np.asarray(ours, np.float32); m[~land] = np.nan
ax.imshow(m, origin="upper", extent=(-180, 180, -60, 84), transform=ccrs.PlateCarree(), cmap=cmap, norm=norm,
          regrid_shape=4400, zorder=1, interpolation="nearest")
del m
for k, (nm, (n, s, w, e, _)) in enumerate(REG.items()):
    t = np.linspace(0, 1, 40)
    bx = np.concatenate([w + (e - w) * t, np.full(40, e), e - (e - w) * t, np.full(40, w)])
    by = np.concatenate([np.full(40, n), n - (n - s) * t, np.full(40, s), s + (n - s) * t])
    ax.plot(bx, by, transform=ccrs.PlateCarree(), color="#111111", lw=0.7, zorder=6)
    ax.text(e + 1.2, n + 0.5, str(k + 1), transform=ccrs.PlateCarree(), fontsize=6.5, fontweight="bold", va="center", zorder=7,
            bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="#111111", lw=0.5))
dd = lambda d: (np.datetime64("2022-01-01") + np.timedelta64(d, "D")).astype("datetime64[D]").item()
ax.text(0.0, 0.985, f"Forecast for {dd(DT):%d %B %Y}, issued on {dd(D0):%d %B}\n1 km over the whole Earth (ensemble mean)",
        transform=ax.transAxes, fontsize=6.3, va="top", ha="left", linespacing=1.2)
letter(ax, "a", x=-0.005, y=0.985)
cax = fig.add_axes([0.3, 0.578, 0.4, 0.009])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max", spacing="uniform", ticks=PM_BOUNDS)
cb.ax.set_xticklabels([f"{t:g}" for t in PM_BOUNDS]); cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1)
cb.outline.set_linewidth(0.4); cb.set_label("Daily mean PM$_{2.5}$ (µg m$^{-3}$); 15 = WHO 24-h guideline, 25-75 = interim targets", fontsize=5.8, labelpad=1)

# b: four regions x (CAMS 40 km, this work 1 km, GHAP 1 km)
rows = (("CAMS forecast (40 km)", "cams"), ("Foehn (1 km)", "ours"), ("GHAP, target day (1 km)", "ghap"))
cw = 0.228; ch = cw * DOUBLE / 6.9 / ASPECT; x0 = 0.045; y0 = 0.555 - 0.02 - ch
for j, (nm, (n, s, w, e, wkm)) in enumerate(REG.items()):
    r0, r1 = int((84 - n) * 100), int((84 - s) * 100); c0, c1 = int((w + 180) * 100), int((e + 180) * 100)
    sel = (slat <= n) & (slat >= s) & (slon >= w) & (slon <= e)
    for i, (rt, key) in enumerate(rows):
        a = fig.add_axes([x0 + j * 0.2385, y0 - i * (ch + 0.012), cw, ch]); a.set_facecolor("#f2f5f8")
        if key == "cams":
            ci0, ci1 = int((90 - n) / 0.4), int((90 - s) / 0.4) + 1; cj0, cj1 = int((w % 360) / 0.4), int((e % 360) / 0.4) + 1
            lw0 = cj0 * 0.4 - 0.2; lw0 = lw0 - 360.0 if lw0 > 180 else lw0          # back to -180 .. 180 like the panel
            a.imshow(cams[ci0:ci1, cj0:cj1], extent=(lw0, lw0 + (cj1 - cj0) * 0.4, 90 - ci1 * 0.4 + 0.2, 90 - ci0 * 0.4 + 0.2),
                     cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
            fld = lambda la, lo: cams[np.clip(np.round((90 - la) / 0.4).astype(int), 0, 450), np.round((lo % 360) / 0.4).astype(int) % 900]
        else:
            src = ours if key == "ours" else ghap
            v = np.asarray(src[r0:r1, c0:c1], np.float32); v[~land[r0:r1, c0:c1]] = np.nan
            if key == "ghap":
                v[v <= 0] = np.nan
            a.imshow(v, extent=(w, e, s, n), cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
            fld = lambda la, lo, v=v: v[np.clip(((n - la) * 100).astype(int), 0, v.shape[0] - 1), np.clip(((lo - w) * 100).astype(int), 0, v.shape[1] - 1)]
        a.scatter(slon[sel], slat[sel], c=sobs[sel], cmap=cmap, norm=norm, s=6, linewidths=0.35, edgecolors="#111111", zorder=5)
        inbox = (la_all <= n) & (la_all >= s) & (lo_all >= w) & (lo_all <= e) & (NSD > 0)
        if inbox.sum() >= 3:
            rm = np.sqrt(SSE[inbox, {"ours": 0, "cams": 1, "ghap": 2}[key]].sum() / NSD[inbox].sum())
            a.text(0.03, 0.04, f"2022 RMSE {rm:.1f} µg m$^{{-3}}$" + (f"  ({int(inbox.sum())} stations)" if i == 0 else ""),
                   transform=a.transAxes, fontsize=5.2, zorder=8, fontweight="bold" if key == "ours" else "normal",
                   bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.85))
        a.set_xlim(w, e); a.set_ylim(s, n); a.set_aspect(1 / np.cos(np.deg2rad(0.5 * (n + s))), adjustable="box")
        if i == 2:                                                 # scale bar
            L = {700: 200, 600: 100, 300: 50}[wkm]; dlo = L / (111.2 * np.cos(np.deg2rad(0.5 * (n + s))))
            xb, yb = e - 0.06 * (e - w) - dlo, s + 0.08 * (n - s)
            a.plot([xb, xb + dlo], [yb, yb], color="#111111", lw=1.0, zorder=9, solid_capstyle="butt")
            a.text(xb + dlo / 2, yb + 0.03 * (n - s), f"{L} km", ha="center", va="bottom", fontsize=5.2, zorder=9,
                   bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7))
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color("#333333")
        if i == 0:
            a.set_title(f"{j + 1}  {nm}", fontsize=6.5, pad=2, fontweight="bold")
        if j == 0:
            a.text(-0.05, 0.5, rt, transform=a.transAxes, rotation=90, ha="right", va="center", fontsize=6)
        if i == 0 and j == 0:
            letter(a, "b", x=-0.12, y=1.04)
save(fig, f"fig1_overview_{RUN}_{date}_h{H}")
