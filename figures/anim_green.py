#!/usr/bin/env python3
"""Animation of the 3-D Green's functions (FLEXGREEN): where the air breathed in ten cities on a target day came from,
followed backwards for 7 days through the 3-D wind of the HRES run issued 7 days before (as the transport of a 7-day
forecast). Same physics as flexgreen.py (advection by u, v, w; boundary-layer turbulence; dry and wet removal); the
particles are coloured by their height above ground. Output: paper/nature/figs/anim_green_<date>.gif
    python anim_green.py [target_doy 24] [particles 150]       (container, CPU or GPU)"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "global1km"))
import flexgreen as fg  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402

from nature_style import OUT, plt, ryb  # noqa: E402

T = int(sys.argv[1]) if len(sys.argv) > 1 else 24
P = int(sys.argv[2]) if len(sys.argv) > 2 else 250
H = 7
D0 = T - H                                                     # run issued 7 days before the target day
CITIES = {"Beijing": (39.90, 116.40), "Seoul": (37.57, 126.98), "Delhi": (28.61, 77.21), "Dhaka": (23.81, 90.41),
          "Tehran": (35.69, 51.39), "Cairo": (30.04, 31.24), "Paris": (48.86, 2.35), "Lagos": (6.52, 3.38),
          "Los Angeles": (34.05, -118.25), "São Paulo": (-23.55, -46.63)}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = torch.Generator(device=dev); gen.manual_seed(7)
zs, z0 = fg.static(dev)
B = torch.tensor(fg.load_run("hres", 2022, D0), device=dev)
fld = fg.Field(fg.derive(B, zs, z0)); del B
lon = torch.cat([torch.full((P,), lo) for la, lo in CITIES.values()]).to(dev) + (torch.rand(P * len(CITIES), generator=gen, device=dev) - .5) * .3
lat = torch.cat([torch.full((P,), la) for la, lo in CITIES.values()]).to(dev) + (torch.rand(P * len(CITIES), generator=gen, device=dev) - .5) * .3
z = torch.rand(P * len(CITIES), generator=gen, device=dev) * 100.0
t_rel = 24.0 * H + 24.0 * torch.rand(P * len(CITIES), generator=gen, device=dev)   # release over the target day
ts = t_rel / 6.0
surv = torch.ones_like(lon)
frames = [(lon.cpu().numpy().copy(), lat.cpu().numpy().copy(), z.cpu().numpy().copy(), surv.cpu().numpy().copy())]
n = int(round(24 * H * 3600 / fg.DT))
for k in range(n):                                             # the loop of flexgreen.backward, recording every 3 h
    lon, lat, z, s = fg.advect(fld, lon, lat, z, ts, -fg.DT)
    ts = ts - fg.DT / (6 * 3600.0)
    h_, ust, invL, wst = s[fg.C_PBL], s[fg.C_UST], s[fg.C_IL], s[fg.C_WST]
    z, lon, lat = fg.turbulence(z, lon, lat, h_, ust, invL, wst, gen, kbar=s[fg.C_KBAR])
    lon = ((lon + 180) % 360) - 180
    surv = surv * torch.where(z < 2 * fg.HREF, math.exp(-fg.VDEP * fg.DT / (2 * fg.HREF)), 1.0)
    pr = s[fg.C_PR].clamp(min=0)
    surv = surv * torch.where((pr > 0.01) & (z < s[fg.C_CTOP]), torch.exp(-fg.WET_A * pr ** fg.WET_B * fg.DT), torch.ones_like(pr))
    if (k + 1) % 12 == 0:
        frames.append((lon.cpu().numpy().copy(), lat.cpu().numpy().copy(), z.cpu().numpy().copy(), surv.cpu().numpy().copy()))
print(len(frames), "frames")

ZB = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 9]
cmap = ryb(len(ZB) - 1); norm = BoundaryNorm(ZB, cmap.N)
target = (np.datetime64("2022-01-01") + np.timedelta64(T, "D")).astype("datetime64[D]").astype(object)
fig = plt.figure(figsize=(7.2, 4.3), dpi=150)
ax = fig.add_axes([0.01, 0.12, 0.98, 0.8], projection=ccrs.Robinson())
ax.set_global()
ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#eef2f6", edgecolor="none")
ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#e2e2e2", edgecolor="none")
ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.3, edgecolor="#777777")
ax.spines["geo"].set_linewidth(0.5)
city = np.repeat(np.arange(len(CITIES)), P)
for (name, (la, lo)) in CITIES.items():
    ax.plot(lo, la, "o", ms=3.5, mfc="white", mec="#111111", mew=0.7, transform=ccrs.PlateCarree(), zorder=6)
    ax.annotate(name, (lo, la), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), xytext=(3, 3), textcoords="offset points",
                fontsize=5.5, zorder=7, path_effects=[pe.withStroke(linewidth=1.3, foreground="white")])
NTR = 6
trails = [ax.scatter([], [], s=1, c=[], cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), linewidths=0, zorder=3) for _ in range(NTR)]
head = ax.scatter([], [], s=3, c=[], cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), linewidths=0, zorder=5)
title = fig.text(0.5, 0.95, "", ha="center", fontsize=8)
cax = fig.add_axes([0.3, 0.085, 0.4, 0.02])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", spacing="uniform", ticks=ZB)
cb.ax.set_xticklabels([f"{v:g}" for v in ZB]); cb.ax.tick_params(labelsize=5.5, length=1.5)
cb.set_label("Particle height above ground (km)", fontsize=6, labelpad=1)
fig.text(0.99, 0.005, "FLEXGREEN: backward particles in the 3-D HRES wind (turbulence, removal), the transport operator of Foehn",
         ha="right", fontsize=5, color="#666666")


def draw(i):
    lo, la, zz, sv = frames[i]
    head.set_offsets(np.c_[lo, la]); head.set_array(zz / 1000.0); head.set_sizes(3 + 7 * sv)
    for j, tr in enumerate(trails):                             # comet tails: the six previous positions, fading
        k = max(i - (j + 1), 0)
        tr.set_offsets(np.c_[frames[k][0], frames[k][1]]); tr.set_array(frames[k][2] / 1000.0)
        tr.set_alpha(0.32 * (1 - j / NTR)); tr.set_sizes([1.2])
    hours = 3 * i
    aloft = np.mean(zz > 2000)
    title.set_text(f"Where did the air breathed on {target:%-d %B %Y} come from?   {hours // 24} d {hours % 24:02d} h earlier"
                   f"   ·   {100 * aloft:.0f}% of the particles above 2 km")
    return [head, title] + trails


anim = FuncAnimation(fig, draw, frames=len(frames), interval=150, blit=False)
out = OUT / f"anim_green_{target:%Y-%m-%d}.gif"
anim.save(out, writer=PillowWriter(fps=7), dpi=130)
print("saved", out)
