#!/usr/bin/env python3
"""Architecture of Foehn (Foehn-Green): physical 3-D transport (FLEXGREEN, no learned parameter) feeding learned latent
tracers into a 1 km flow-matching generator. All thumbnails are real data of the forecast issued on 22 January 2022 for
25 January (East Asia): HRES 3-D wind, FLEXGREEN trajectories, global map of day d, latent tracers of the trained
encoder, gathered transport maps, 1 km inputs, two 1 km members and the exceedance probability (16 members).
    python fig_arch.py         (data: figdata_arch.py)"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

from nature_style import DOUBLE, PM_BOUNDS, STYLE, plt, pm_cmap, ryb, save

Z = np.load("data/arch_thumbs.npz", allow_pickle=True)
fig = plt.figure(figsize=(DOUBLE, 4.25))
T = fig.transFigure
cmap_pm, norm_pm = pm_cmap()
C_PHYS, E_PHYS = "#d6e6f4", "#3b75af"           # physics, no learned parameter
C_LEARN, E_LEARN = "#fde0c5", "#d0661b"         # learned
C_GATH, E_GATH = "#fff2b3", "#c79a00"           # parameter-free operator
C_GEN, E_GEN = "#f6d3cf", STYLE["ours"]["color"]  # generative sampling


def box(x, y, w, h, title, lines=(), fc=C_LEARN, ec=E_LEARN, fs=6.2):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", transform=T, fc=fc, ec=ec, lw=0.9))
    fig.text(x + w / 2, y + h - 0.018, title, ha="center", va="top", fontsize=fs, fontweight="bold", color="#111111")
    for k, l in enumerate(lines):
        fig.text(x + w / 2, y + h - 0.05 - k * 0.026, l, ha="center", va="top", fontsize=5.2, color="#222222")


def arrow(x0, y0, x1, y1, color="#333333", lw=0.8, style="-|>", conn="arc3", ls="-"):
    fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=T, arrowstyle=style, mutation_scale=7, lw=lw, color=color,
                                   connectionstyle=conn, linestyle=ls, shrinkA=0, shrinkB=0))


def thumb(rect, img, cmap="viridis", norm=None, title=None, below=None, frame="#555555"):
    ax = fig.add_axes(rect)
    ax.imshow(np.ma.masked_invalid(np.asarray(img, np.float32)), cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_edgecolor(frame); sp.set_visible(True)
    if title:
        ax.set_title(title, fontsize=5.2, pad=1.5)
    if below:
        ax.text(0.5, -0.06, below, transform=ax.transAxes, ha="center", va="top", fontsize=5.0)
    return ax


def stack(x, y, w, h, imgs, cmaps, norms=None, dx=0.009, dy=0.016):
    axs = []
    for k in range(len(imgs) - 1, -1, -1):                  # back to front
        axs.append(thumb([x + k * dx, y + k * dy, w, h], imgs[k], cmaps[k], None if norms is None else norms[k]))
    return axs


def wind_img(u, v):
    return np.hypot(u, v)


# ------------------------------------------------------------------ band titles
fig.text(0.005, 0.975, "Physical transport (no learned parameter)", fontsize=6.6, fontweight="bold", color=E_PHYS, va="top")
fig.text(0.005, 0.655, "Learned latent tracers, transported by the physics", fontsize=6.6, fontweight="bold", color=E_LEARN, va="top")
fig.text(0.005, 0.345, "Generator at 1 km (flow matching)", fontsize=6.6, fontweight="bold", color=E_GEN, va="top")

# ------------------------------------------------------------------ row 1: transport
stack(0.012, 0.735, 0.105, 0.155, [wind_img(Z["w10_u"], Z["w10_v"]), wind_img(Z["w850_u"], Z["w850_v"]), wind_img(Z["w500_u"], Z["w500_v"])],
      ["Blues", "Blues", "Blues"], [Normalize(0, 15), Normalize(0, 30), Normalize(0, 60)])
fig.text(0.075, 0.705, "3-D forecast wind (HRES run of day d)\nu, v, w · 10 m + 8 levels · 6-hourly", ha="center", va="top", fontsize=5.2)
arrow(0.142, 0.83, 0.165, 0.83)
box(0.167, 0.725, 0.135, 0.205, "FLEXGREEN", ("backward particles, 32 per cell", "3-D advection by u, v, w", "turbulence in the mixed layer",
                                             "dry and wet removal", "(FLEXPART physics, PyTorch)"), C_PHYS, E_PHYS)
arrow(0.305, 0.83, 0.33, 0.83)
ax = fig.add_axes([0.335, 0.70, 0.165, 0.235], projection=ccrs.PlateCarree())
ax.set_extent([95, 145, 15, 55], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#eeeeee", edgecolor="none")
ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.3, edgecolor="#777777")
tr = Z["traj"]
zb = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 9]
zc, zn = ryb(len(zb) - 1), BoundaryNorm(zb, len(zb) - 1)
for p in range(tr.shape[0]):
    pts = tr[p, :, :2]
    seg = np.stack([pts[:-1], pts[1:]], 1)
    lc = LineCollection(seg, cmap=zc, norm=zn, lw=0.7, transform=ccrs.PlateCarree())
    lc.set_array(tr[p, 1:, 2] / 1000.0); ax.add_collection(lc)
    ax.plot(pts[-1, 0], pts[-1, 1], "o", ms=1.6, color=zc(zn(tr[p, -1, 2] / 1000.0)), transform=ccrs.PlateCarree(), zorder=5)
for (la, lo) in Z["city_ll"]:
    ax.plot(lo, la, "o", ms=2.6, mfc="white", mec="#111111", mew=0.5, transform=ccrs.PlateCarree(), zorder=6)
ax.set_title("Green's functions: where the air of 25 Jan was on 22 Jan", fontsize=5.2, pad=1.5)
ax.spines["geo"].set_linewidth(0.5)
cax = fig.add_axes([0.337, 0.674, 0.072, 0.009])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=zn, cmap=zc), cax=cax, orientation="horizontal", ticks=[0, 1, 2, 4, 9])
cb.ax.tick_params(labelsize=5, length=1, pad=0.5); cb.outline.set_linewidth(0.3)
cb.set_label("particle height (km)", fontsize=5, labelpad=0.5)

# ------------------------------------------------------------------ row 2: latent tracers
thumb([0.012, 0.435, 0.105, 0.155], Z["pm_global"], cmap_pm, norm_pm, title="Global map of day d (PM2.5, fires)")
arrow(0.122, 0.512, 0.145, 0.512)
box(0.147, 0.44, 0.085, 0.15, "Encoder", ("2 conv 3×3", "5 → 32 channels", "learned"), C_LEARN, E_LEARN)
arrow(0.235, 0.512, 0.255, 0.512)
lat = Z["latent_ea"]
stack(0.258, 0.44, 0.075, 0.12, [lat[k] for k in range(4)], ["RdBu_r"] * 4,
      [Normalize(-np.nanpercentile(np.abs(lat[k]), 99), np.nanpercentile(np.abs(lat[k]), 99)) for k in range(4)], dx=0.007, dy=0.012)
fig.text(0.31, 0.425, "32 latent tracers of day d", ha="center", va="top", fontsize=5.2)
arrow(0.368, 0.512, 0.39, 0.512)
arrow(0.418, 0.70, 0.418, 0.598, color=E_PHYS, lw=1.0)                            # particle origins come down from row 1
fig.text(0.424, 0.64, "origins, height,\nsurvival", fontsize=5, color=E_PHYS, va="center")
box(0.392, 0.44, 0.115, 0.155, "Gather", ("read the tracers where", "each cell's air came from", "weights: survival × surface",
                                         "no learned parameter"), C_GATH, E_GATH)
arrow(0.51, 0.512, 0.53, 0.512)
vl = float(np.nanpercentile(np.abs(Z["tr_latent"]), 99))
trm = [Z["tr_latent"], Z["tr_pm"], Z["tr_aloft"], Z["tr_surv"]]                 # front: a transported latent tracer
stack(0.533, 0.44, 0.075, 0.12, trm, ["RdBu_r", "magma_r", "Purples", "Greens"], [Normalize(-vl, vl), None, None, None],
      dx=0.007, dy=0.012)
fig.text(0.562, 0.428, "44 transport maps =", ha="center", va="top", fontsize=5.2, fontweight="bold")
fig.text(0.562, 0.411, "32 transported tracers\n+ PM2.5 of d, aloft, survival\n+ 3 fire + 6 emission terms", ha="center", va="top",
         fontsize=5.0, linespacing=1.15)

# ------------------------------------------------------------------ row 3: generator
stack(0.012, 0.125, 0.105, 0.14, [Z["t2m_d"], Z["pm_ea"], wind_img(Z["w850_u"], Z["w850_v"])], ["RdYlBu_r", "magma_r", "Blues"],
      [None, None, None])
fig.text(0.066, 0.105, "0.25° weather + PM2.5\n(day d + forecast d+h, 69 ch)", ha="center", va="top", fontsize=5.0)
stack(0.15, 0.17, 0.085, 0.09, [Z["ghap_d"], Z["ghap_dm1"], Z["elev"]], [cmap_pm, cmap_pm, "gist_earth"], [norm_pm, norm_pm, None], dx=0.006, dy=0.01)
fig.text(0.198, 0.155, "1 km: PM2.5 of d, d−1,\nelevation, land", ha="center", va="top", fontsize=5.0)
thumb([0.15, 0.03, 0.085, 0.06], Z["noise"], "Greys", None)
fig.text(0.193, 0.094, "noise (one per member)", fontsize=5, ha="center", va="bottom")
# conditioning concatenation
CX, CY, CR = 0.262, 0.30, 0.011
fig.add_artist(Ellipse((CX, CY), CR * 2, CR * 2 * 7.2 / 4.25, transform=T, fc="white", ec="#333333", lw=0.8, zorder=10))
fig.add_artist(plt.Line2D([CX - CR * 0.6, CX + CR * 0.6], [CY, CY], transform=T, color="#333333", lw=0.8, zorder=11))
fig.add_artist(plt.Line2D([CX, CX], [CY - CR * 1.0, CY + CR * 1.0], transform=T, color="#333333", lw=0.8, zorder=11))
arrow(0.13, 0.272, CX - CR, CY - 0.004, conn="arc3,rad=-0.15")
arrow(0.497, 0.366, CX + 0.006, CY + 0.016, conn="arc3,rad=0.12")
fig.text(CX + 0.015, CY - 0.012, "concat. 117 ch", fontsize=5, ha="left", va="top")
# U-Net: descending blocks, Transformer, ascending blocks
xs = np.linspace(0.30, 0.47, 4)
hs = [0.17, 0.14, 0.115, 0.095]
labels = ["2 km", "4 km", "8 km", "16 km"]
ybase = 0.07
for k, (x, hh, lab) in enumerate(zip(xs, hs, labels)):
    fig.add_artist(Rectangle((x, ybase + (0.17 - hh) / 2 + 0.02), 0.03, hh, transform=T, fc=C_LEARN, ec=E_LEARN, lw=0.7))
    fig.text(x + 0.015, ybase + 0.005, lab, fontsize=5, ha="center")
    if k < 3:
        arrow(x + 0.031, ybase + 0.105, xs[k + 1] - 0.001, ybase + 0.105)
arrow(0.24, 0.215, 0.299, 0.175)                                                   # 1 km inputs + noise into the U-Net
arrow(0.237, 0.06, 0.299, 0.13)
arrow(CX + CR, CY, xs[3] + 0.015, ybase + (0.17 + hs[3]) / 2 + 0.023, conn="arc3,rad=-0.2", color=E_GATH, lw=1.0)   # onto the 16 km block
fig.text(0.40, 0.322, "injected at 16 km (1×1 conv, added)", fontsize=5, color="#7a5c00", ha="center")
box(0.515, 0.085, 0.105, 0.15, "Transformer", ("4 blocks at 64 km", "global attention over", "7,260 tokens", "(continental context)"),
    C_LEARN, E_LEARN, fs=6)
arrow(xs[3] + 0.031, ybase + 0.105, 0.514, ybase + 0.105)
xu = np.linspace(0.64, 0.74, 3)
for k, x in enumerate(xu):
    hh = [0.095, 0.13, 0.17][k]
    fig.add_artist(Rectangle((x, ybase + (0.17 - hh) / 2 + 0.02), 0.025, hh, transform=T, fc=C_LEARN, ec=E_LEARN, lw=0.7))
arrow(0.621, ybase + 0.105, xu[0] - 0.001, ybase + 0.105)
for k in range(2):
    arrow(xu[k] + 0.026, ybase + 0.105, xu[k + 1] - 0.001, ybase + 0.105)
fig.text(0.69, ybase - 0.005, "decoder (skip connections) → 1 km", fontsize=5, ha="center")
fig.text(0.59, 0.268, "lead h, flow time τ → FiLM in every block", fontsize=5, color="#555555", ha="center")
# flow ODE and output
box(0.786, 0.12, 0.091, 0.13, "Flow ODE", ("x ← x + v$_\\theta$ Δτ", "8–12 steps", "1 noise = 1 member"), C_GEN, E_GEN, fs=6)
arrow(0.767, ybase + 0.105, 0.785, ybase + 0.105)
fig.text(0.776, ybase + 0.12, "v$_\\theta$", fontsize=5.2, ha="center")
for (xa, ya), (xb, yb) in (((0.832, 0.118), (0.832, 0.012)), ((0.832, 0.012), (0.258, 0.012))):      # loop under the blocks
    fig.add_artist(plt.Line2D([xa, xb], [ya, yb], transform=T, color=E_GEN, lw=0.7, ls=(0, (2, 1.5))))
arrow(0.258, 0.012, 0.237, 0.035, color=E_GEN, lw=0.7, ls=(0, (2, 1.5)))
fig.text(0.55, 0.018, "repeated from τ = 0 (noise) to τ = 1 (forecast)", fontsize=5, color=E_GEN, ha="center", va="bottom")
thumb([0.885, 0.215, 0.11, 0.105], Z["member0"], cmap_pm, norm_pm, title="Member 1, 1 km")
thumb([0.885, 0.07, 0.11, 0.105], Z["member1"], cmap_pm, norm_pm, title="Member 2, 1 km")
arrow(0.878, 0.195, 0.884, 0.25)
arrow(0.878, 0.165, 0.884, 0.12)
fig.text(0.94, 0.035, "25 Jan 2022, 3 days ahead", fontsize=5, ha="center", va="top")

# ------------------------------------------------------------------ inset: probability and the two equations
fig.add_artist(FancyBboxPatch((0.64, 0.345), 0.355, 0.625, boxstyle="round,pad=0.004,rounding_size=0.01", transform=T, fc="none",
                              ec="#777777", lw=0.6, ls=(0, (3, 2))))
fig.text(0.65, 0.955, "Two equations", fontsize=6.2, fontweight="bold", va="top")
fig.text(0.65, 0.915, "Transport (gather), for each cell c and its particles p:", fontsize=5.2, va="top")
fig.text(0.665, 0.875, r"$g_c = \frac{\sum_p w_p\, f(\mathbf{x}_p)}{\sum_p w_p}$,   $w_p = s_p\,\mathbb{1}[z_p \leq h_{\max}]$", fontsize=6.5, va="top")
fig.text(0.65, 0.81, "x$_p$: origin, s$_p$: survival, z$_p$: height; f: learned tracers", fontsize=5, va="top", color="#444444")
fig.text(0.65, 0.775, "Flow matching (training):", fontsize=5.2, va="top")
fig.text(0.665, 0.74, r"$x_\tau = (1-\tau)\,x_0 + \tau\, x_1$,   $\mathcal{L} = \|\, v_\theta(x_\tau,\tau,\mathrm{cond}) - (x_1 - x_0)\,\|^2$", fontsize=6.3, va="top")
fig.text(0.65, 0.69, "x$_0$: noise, x$_1$: GHAP of day d+h\nmembers: own noise, own half of the particles", fontsize=5, va="top", color="#444444")
PB = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
thumb([0.665, 0.457, 0.15, 0.175], Z["p75"], ryb(7), BoundaryNorm(PB, 7), title="P(PM2.5 > 75 µg m$^{-3}$), 16 members")
thumb([0.83, 0.457, 0.15, 0.175], Z["ghap_target"], cmap_pm, norm_pm, title="Observed (GHAP), 25 Jan")
for rect, sm, ticks, lab, sp in (([0.69, 0.437, 0.1, 0.007], plt.cm.ScalarMappable(norm=BoundaryNorm(PB, 7), cmap=ryb(7)), [0, 0.2, 0.4, 0.6, 0.8, 1], None,
                                  "proportional"),
                                 ([0.85, 0.437, 0.095, 0.007], plt.cm.ScalarMappable(norm=norm_pm, cmap=cmap_pm), [0, 25, 75, 250], "µg m$^{-3}$",
                                  "uniform")):
    cb = fig.colorbar(sm, cax=fig.add_axes(rect), orientation="horizontal", ticks=ticks, spacing=sp)
    cb.ax.tick_params(labelsize=5, length=1, pad=0.5); cb.outline.set_linewidth(0.3)
    cb.ax.set_xticklabels([f"{t:g}" for t in ticks])
    if lab:
        fig.text(rect[0] + rect[2] + 0.004, rect[1] + rect[3] / 2, lab, fontsize=5, va="center")
# legend of colours
for k, (fc, ec, lab) in enumerate(((C_PHYS, E_PHYS, "physics, no learned parameter"), (C_LEARN, E_LEARN, "learned (14.2 M parameters)"),
                                   (C_GATH, E_GATH, "parameter-free operator"), (C_GEN, E_GEN, "generative sampling"))):
    fig.add_artist(FancyBboxPatch((0.655 + (k % 2) * 0.17, 0.383 - (k // 2) * 0.026), 0.014, 0.014, boxstyle="round,pad=0.001",
                                  transform=T, fc=fc, ec=ec, lw=0.6))
    fig.text(0.674 + (k % 2) * 0.17, 0.390 - (k // 2) * 0.026, lab, fontsize=5, va="center")
save(fig, "fig_architecture_foehn")
