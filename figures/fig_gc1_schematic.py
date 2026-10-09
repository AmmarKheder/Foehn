#!/usr/bin/env python3
"""Fig. 1 (GenCast style): how Foehn produces a forecast, and one sample of the whole Earth.
a, schematic. Computed transport (blue): FLEXGREEN follows particles released on the target day backwards in the forecast
   wind and returns where the air was on the start day (origins, heights, survival), without learned parameters; shown: the
   origins on 22 January 2022 of the air reaching Beijing on the 25th. Learned tracers (orange): an encoder turns the maps of
   the start day into 32 tracers, read at the particle origins. Generator (green): conditioned on the 1 km fields of the start
   day, the forecast weather and the 44 transport maps, a network carries a noise field to a 1 km forecast in 12 flow steps;
   each noise field gives one ensemble member (three members of the forecast issued on 20 January for 21 January 2022 shown).
b, one member of that forecast over all land at 1 km.
    python fig_gc1_schematic.py"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from fig_exposure_zoom import ghap_box
import final_model as FM
from nature_style import DOUBLE, HERE, MM, OUT, PM_BOUNDS, plt, pm_cmap, save

WITH_GLOBE = False

RUN = HERE.parent / "global1km" / "runs" / "green_flow_shift25_global_L7_noant_whres_ghres_ftemis12"
BLUE, ORANGE, GREEN = ("#E4EDF8", "#1F5AA6"), ("#F2F2F2", "#5B5B5B"), ("#F7F7F7", "#111111")   # the paper's 3 colours: Foehn blue, greys, black
G = "/scratch/project_462001140/ammar/eccv/data/world/global1km"


def box(fig, x, y, w, h, colors, title):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", transform=fig.transFigure,
                                      fc=colors[0], ec=colors[1], lw=0.9, zorder=-5))
    fig.text(x + 0.008, y + h - 0.012, title, fontsize=6.5, fontweight="bold", color=colors[1], va="top")


def arrow(fig, p0, p1, color="#333333", rad=0.0):
    fig.patches.append(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7, lw=0.9,
                                       color=color, connectionstyle=f"arc3,rad={rad}", zorder=10))


def thumb(fig, rect, img, cmap, norm, title=None, ext=None):
    ax = fig.add_axes(rect, zorder=5)
    ax.imshow(img, cmap=cmap, norm=norm, interpolation="nearest", extent=ext, origin="upper")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_linewidth(0.4); s_.set_edgecolor("#555555")
    if title:
        ax.set_title(title, fontsize=5.5, pad=1.5)
    return ax


def main():
    cmap, norm = pm_cmap()
    W, H = DOUBLE, 150 * MM
    fig = plt.figure(figsize=(W, H))
    if WITH_GLOBE:
        fig.text(0.004, 0.992, "a", fontsize=8, fontweight="bold", va="top")
    # inputs: PM2.5 of the start day at 1 km (GHAP, 20 January), Beijing region
    c = np.load(FM.CASE_DIR / "crop0_d19_h1.npz"); bx = c["box"]; (r0, r1), (c0, c1) = c["rows"], c["cols"]
    sub = (41.5, 38.5, 114.5, 118.5)                                       # Beijing area of the East Asia crops
    a0, a1 = int(round((bx[0] - sub[0]) * 100)), int(round((bx[0] - sub[1]) * 100))
    b0, b1 = int(round((sub[2] - bx[2]) * 100)), int(round((sub[3] - bx[2]) * 100))
    r0, r1, c0, c1 = r0 + a0, r0 + a1, c0 + b0, c0 + b1; bx = np.array(sub)
    x0 = ghap_box(1826 + 19, int(r0), int(r1), int(c0), int(c1)); x0 = np.where(x0 > 0, x0, np.nan)
    ext = (bx[2], bx[3], bx[1], bx[0])
    fig.text(0.012, 0.935, "Start day d (1 km)", fontsize=6, fontweight="bold")
    thumb(fig, [0.012, 0.735, 0.115, 0.19], x0, cmap, norm, ext=ext)
    fig.text(0.012, 0.715, "PM$_{2.5}$ of d and d−1,\nforecast weather of d+h,\nfires and emissions", fontsize=5.3, va="top")
    # computed transport
    box(fig, 0.15, 0.665, 0.26, 0.315, BLUE, "Computed transport, no learned parameter")
    z = np.load(HERE / "data" / "transport_fig.npz"); o = z["bj_org_3"]; lon, lat, zr, sv = o.T; keep = sv > 0.05
    ax = fig.add_axes([0.16, 0.705, 0.13, 0.225], projection=ccrs.PlateCarree(), zorder=5)
    ax.set_extent((92, 128, 33, 56), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#F4F2EE", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.3, edgecolor="#888888")
    for msk, col in ((keep & (zr <= 1), "#111111"), (keep & (zr > 1), "#4F8FD6")):
        ax.scatter(lon[msk], lat[msk], s=0.4, color=col, alpha=0.6, linewidths=0, transform=ccrs.PlateCarree(), rasterized=True)
    ax.plot(116.4, 39.9, "*", ms=6, color="white", mec="#111111", mew=0.4, transform=ccrs.PlateCarree())
    ax.spines["geo"].set_linewidth(0.4)
    fig.text(0.297, 0.915, "FLEXGREEN", fontsize=6, fontweight="bold", color=BLUE[1], va="top")
    fig.text(0.297, 0.885, "particles released\non the target day,\nfollowed backwards\nin the 3-D forecast\nwind: origin $\\mathbf{x}_p$,\n"
             "height $z_p$, survival $s_p$", fontsize=5.2, va="top")
    fig.text(0.16, 0.682, "air reaching Beijing on 25 Jan: black, from the\nmixed layer on 22 Jan; blue, from above it", fontsize=4.8, va="bottom",
             color="#444444")
    # learned tracers
    box(fig, 0.15, 0.36, 0.26, 0.285, ORANGE, "Learned tracers")
    fig.text(0.162, 0.6, "encoder: 32 tracers $f = E_\\phi$(PM$_{2.5}$, fires of d)", fontsize=5.4, va="top")
    fig.text(0.162, 0.54, "each receptor $c$ reads them at the origins of its air:", fontsize=5.4, va="top")
    fig.text(0.175, 0.505, "$g_c = \\sum_p w_p\\, f(\\mathbf{x}_p) \\,/\\, \\sum_p w_p,\\;\\; w_p = s_p\\,\\mathbb{1}[z_p \\leq 1]$", fontsize=6.3, va="top")
    fig.text(0.162, 0.445, "+ advected PM$_{2.5}$,\nair from aloft, survival,\nfire and emission terms:\n44 transport maps",
             fontsize=5.4, va="top")
    am = z["aloft_map"][2][56:102, 544:616]                                       # East Asia, share of air from aloft, 3 days ahead
    land = np.load(f"{G}/meta.npz")["land"].astype(bool).reshape(288, 2, 720, 2).any((1, 3))[56:102, 544:616]
    thumb(fig, [0.305, 0.375, 0.095, 0.115], np.where(land, am, np.nan), "Blues", plt.Normalize(0.2, 0.9), title="air from aloft, 3 d")
    # generator
    box(fig, 0.43, 0.36, 0.40, 0.62, GREEN, "Generator: flow matching at 1 km")
    rng = np.random.default_rng(3); noise = rng.standard_normal((150, 200))
    m = c["members"][:, a0:a1, b0:b1].astype(np.float32)
    thumb(fig, [0.445, 0.70, 0.095, 0.16], noise, "Greys", None, title="noise $x_0$")
    nz = np.random.default_rng(3).standard_normal(m.shape[1:])                     # path of flow matching: (1 - tau) x0 + tau x1
    mid = 20 * (0.5 * nz + 0.5 * (m[0] - 15) / 20) + 15
    thumb(fig, [0.585, 0.70, 0.095, 0.16], mid, cmap, norm, title="$x_{\\tau}$, $\\tau$ = 0.5", ext=ext)
    thumb(fig, [0.725, 0.70, 0.095, 0.16], m[0], cmap, norm, title="sample $x_1$", ext=ext)
    arrow(fig, (0.542, 0.78), (0.583, 0.78)); arrow(fig, (0.682, 0.78), (0.723, 0.78))
    fig.text(0.55, 0.875, f"$x \\leftarrow x + v_\\theta(x, \\tau, h, c)\\,\\Delta\\tau$, {FM.M['nfe']} steps", fontsize=6, va="bottom")
    fig.text(0.445, 0.665, "network $v_\\theta$: U-Net on a 4,192 × 6,992 km window at 1 km,\nglobal attention at 64 km, conditioned "
             "on the start day, the\nforecast weather and the transport maps $c$", fontsize=5.2, va="top")
    fig.text(0.445, 0.575, "Ensemble: one noise field per member", fontsize=5.8, fontweight="bold", va="top", color=GREEN[1])
    for k in range(3):
        thumb(fig, [0.445 + 0.128 * k, 0.385, 0.115, 0.17], m[k + 1], cmap, norm, title=f"member {k + 1}", ext=ext)
    # arrows between blocks
    arrow(fig, (0.128, 0.83), (0.15, 0.83)); arrow(fig, (0.28, 0.665), (0.28, 0.645), color=BLUE[1])
    arrow(fig, (0.41, 0.50), (0.43, 0.50), color=ORANGE[1])
    # output
    fig.text(0.845, 0.94, "Output", fontsize=6.5, fontweight="bold", va="top")
    fig.text(0.845, 0.915, "ensembles of daily\nPM$_{2.5}$ at 1 km over\nall land, 1-7 days\nahead; 21 windows\nshare one noise field\n"
             "per member, so the\nglobe has no seams", fontsize=5.3, va="top")
    cax = fig.add_axes([0.85, 0.40, 0.012, 0.25])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="max", spacing="uniform", ticks=PM_BOUNDS)
    cb.ax.set_yticklabels([f"{t:g}" for t in PM_BOUNDS]); cb.ax.tick_params(labelsize=4.8, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
    cb.set_label("PM$_{2.5}$ (µg m$^{-3}$)", fontsize=5.3, labelpad=1)
    if not WITH_GLOBE:                                                         # the globe is shown in Fig. 2
        for ext_ in ("pdf", "png"):
            fig.savefig(OUT / f"fig_gc1_schematic.{ext_}", bbox_inches=fig.bbox_inches.from_bounds(0, 0.345 * H, W, 0.655 * H))
        print("saved schematic only"); return
    # b: one member of the global forecast
    fig.text(0.004, 0.34, "b", fontsize=8, fontweight="bold", va="top")
    ab = fig.add_axes([0.03, 0.0, 0.94, 0.335], projection=ccrs.Robinson())
    ab.set_extent([-180, 180, -58, 84], crs=ccrs.PlateCarree()); ab.spines["geo"].set_visible(False)
    ab.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#ECECEC", edgecolor="none", zorder=0)
    g = np.load(FM.JAN21_DIR / "map1km_d19_h1_member0.npy").astype(np.float32)
    from fig1_teaser_v6 import tiles
    g[~(tiles("static1km_tiles.npy", 1) > 0)] = np.nan
    ab.imshow(g, origin="upper", extent=(-180, 180, -60, 84), transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, regrid_shape=4400,
              interpolation="nearest", zorder=1)
    ab.text(0.0, 0.02, "One member of the forecast for 21 January 2022, issued on 20 January, 1 km", transform=ab.transAxes, fontsize=6)
    save(fig, "fig_gc1_schematic")


if __name__ == "__main__":
    main()
