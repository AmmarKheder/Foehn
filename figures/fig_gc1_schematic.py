#!/usr/bin/env python3
"""Fig. 1: how Foehn makes a forecast (redrawn from the overview sketch paper/nature/figs/slide2_preview.pdf).
a, computed transport, no learned parameter: the 3-D wind of the forecast (850 hPa at 00 UTC of the start day, 20 January 2022,
   line-integral convolution, brightness = speed) drives FLEXGREEN, which follows particles released over the target day
   backwards and adds up the anthropogenic emissions (CAMS-GLOB-ANT, black carbon shown) and the fires that the air takes
   up on its way (the emission terms of the transport maps); shown: the 5-day back-trajectories of the air reaching 18
   cities on 25 January (colour: height above ground,
   blue near the ground, pink aloft; dots: origins on the start day).
b, learned tracers: an encoder E_phi turns the start day (PM2.5 at 0.25 deg, fires) into tracers f, read at the origins of the
   air. Tracers:
   air of every receptor c: g_c = sum_p w_p f(x_p) / sum_p w_p, w_p = s_p 1[z_p <= 1].
c, generator, conditional flow matching at 1 km: conditioned on the start day at 1 km (and at 0.25 deg with its fires, not drawn),
   the forecast weather and the transport maps
   and the transport maps (three of them shown, lead 5: advected PM2.5, i.e. the same reading with f = PM2.5; share of the air
   from above the mixed layer; survival), the network v_theta carries a noise field x0 to a forecast x1 in 4 Euler steps of
   flow-matching time tau (red), for every lead time (blue). Discs: member 0 of the forecast issued on 20 January 2022 for 21,
   22 and 27 January; intermediate states drawn as (1 - tau) x0 + tau x1, the path the flow is trained on, at the flow times
   of the sampler's 4 Euler steps (0.034, 0.167, 0.534).
    python fig_gc1_schematic.py        (data/arch_fields.npz: figdata_arch_fields.py; data/arch_overview.npz: figdata_arch2.py)"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
import os

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from scipy.ndimage import map_coordinates

from nature_style import HERE, MM, PM_BOUNDS, plt, pm_cmap, save

CLON, CLAT = 58.0, 24.0                                   # the polluted half of the planet in January: Africa to China
W_MM, H_MM = 183.0, 136.0
BLUE, BLUE_BG, BLUE_ED = "#1F5AA6", "#EEF4FB", "#B3CBE8"
GREY, GREY_BG, GREY_ED = "#454B54", "#F4F4F5", "#CDD0D5"
DARK, DARK_BG, DARK_ED = "#16181D", "#F8F8F9", "#C4C7CD"
RED, PINK, INK, MUTED = "#C8102E", "#D0306C", "#1B1E23", "#5F6873"
ROWS = (1, 2, 7)                                          # lead days shown in the disc grid (as in the sketch)
TAUS = (0.0, 0.0338, 0.1667, 0.5338, 1.0)                 # flow times of the 4 Euler steps of the sampler: quantiles of the
                                                          # training mixture of tau (shift 25, mix 0.5; FullModule.tau_grid(4))
PCM, PNORM = pm_cmap()
LAND_NODATA = "#EEF0F2"
EMIS_TERM = (os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km/green256/hres/emis_2022.npy")   # flexgreen.py --emis
EMIS_CM = LinearSegmentedColormap.from_list("emis", ["#F1EFF6", "#CBC4E1", "#9A8EC4", "#6A51A3", "#3F007D"])   # emissions: purple


def R(x, y, w, h):
    return [x / W_MM, y / H_MM, w / W_MM, h / H_MM]


def mm_axes(fig, z):
    ax = fig.add_axes([0, 0, 1, 1], zorder=z)
    ax.set_xlim(0, W_MM); ax.set_ylim(0, H_MM); ax.axis("off"); ax.patch.set_alpha(0)
    return ax


def panel(bg, x, y, w, h, top, bottom, ec):
    """rounded panel: thin edge only, transparent inside (no fill, no shadow; top and bottom are kept for the call sites)"""
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.8", fc="none", ec=ec, lw=0.55, zorder=0.5))


def arrow(ax, p0, p1, color=INK, rad=0.0, lw=0.9, ms=7, z=30, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z, capstyle="round",
                                 joinstyle="round"))


def text(ax, x, y, s, size=5.6, color=INK, weight="normal", ha="center", va="center", style="normal", z=40, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, style=style, zorder=z, **kw)


def balance(ax, cx, cy, s=1.0, color=GREY, z=36):
    """the scales of the sketch (vector): base, column, beam with its pivot, two pans on strings"""
    ax.add_patch(Polygon([(cx - 3.0 * s, cy - 5.4 * s), (cx + 3.0 * s, cy - 5.4 * s), (cx + 1.8 * s, cy - 4.5 * s),
                          (cx - 1.8 * s, cy - 4.5 * s)], closed=True, fc=color, ec=color, lw=0.4, joinstyle="round", zorder=z))
    ax.add_patch(Rectangle((cx - 0.36 * s, cy - 4.55 * s), 0.72 * s, 8.1 * s, fc=color, ec="none", zorder=z))
    ax.add_patch(FancyBboxPatch((cx - 5.7 * s, cy + 3.05 * s), 11.4 * s, 0.6 * s, boxstyle="round,pad=0,rounding_size=0.28",
                                fc=color, ec="none", zorder=z + 0.1))
    ax.add_patch(Circle((cx, cy + 3.95 * s), 0.78 * s, fc=color, ec="white", lw=0.35, zorder=z + 0.2))
    for sx in (-1, 1):
        px, top, py = cx + sx * 5.0 * s, cy + 3.3 * s, cy - 1.2 * s
        for dx in (-2.15, 2.15):
            ax.plot([px, px + dx * s], [top, py], color=color, lw=0.5, zorder=z, solid_capstyle="round")
        ax.add_patch(Circle((px, top), 0.32 * s, fc=color, ec="none", zorder=z + 0.2))
        th = np.linspace(np.pi, 2 * np.pi, 40)
        ax.add_patch(Polygon(np.column_stack([px + 2.45 * s * np.cos(th), py + 1.05 * s * np.sin(th)]), closed=True, fc=color,
                             ec=color, lw=0.4, joinstyle="round", zorder=z))


def pm_scale(ax, x, y, w, h, z=45):
    """segmented PM2.5 colour scale in mm coordinates (classes of nature_style.PM_BOUNDS, open last class)"""
    b = list(PM_BOUNDS)
    mids = [0.5 * (b[k] + b[k + 1]) for k in range(len(b) - 1)] + [b[-1] * 1.2]
    n = len(mids); gap = 0.32; sw = (w - gap * (n - 1)) / n
    for k, v in enumerate(mids):
        x0 = x + k * (sw + gap); col = PCM(PNORM(v))
        if k < n - 1:
            ax.add_patch(FancyBboxPatch((x0, y), sw, h, boxstyle="round,pad=0,rounding_size=0.45", fc=col, ec="none", zorder=z))
        else:                                                                    # open class: a pointed segment
            ax.add_patch(Polygon([(x0, y), (x0 + sw * 0.55, y), (x0 + sw, y + h / 2), (x0 + sw * 0.55, y + h), (x0, y + h)],
                                 closed=True, fc=col, ec=col, lw=0.2, joinstyle="round", zorder=z))
        if k > 0:
            xt = x0 - gap / 2
            ax.text(xt, y - 1.15, f"{b[k]:g}", fontsize=5.0, color=INK, ha="center", va="top", zorder=z)
    ax.text(x, y - 1.15, "0", fontsize=5.0, color=INK, ha="center", va="top", zorder=z)
    ax.text(x, y + h + 1.3, "PM$_{2.5}$", fontsize=5.8, color=INK, fontweight="bold", ha="left", va="bottom", zorder=z)
    ax.text(x + 6.6, y + h + 1.3, "(µg m$^{-3}$)", fontsize=5.4, color=MUTED, ha="left", va="bottom", zorder=z)


def gear(ax, cx, cy, r_out, r_in, n, color, z=35):
    """a gear of n teeth (the sketch's symbol of the physical transport), vector"""
    P = 2 * np.pi / n; base, top = 0.56 * P, 0.34 * P; pts = []
    for k in range(n):
        t0 = k * P
        for r_, a_ in ((r_in, t0 - base / 2), (r_out, t0 - top / 2), (r_out, t0 + top / 2), (r_in, t0 + base / 2)):
            pts.append((cx + r_ * np.cos(a_), cy + r_ * np.sin(a_)))
        for a_ in np.linspace(t0 + base / 2, t0 + P - base / 2, 6)[1:-1]:
            pts.append((cx + r_in * np.cos(a_), cy + r_in * np.sin(a_)))
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec=color, lw=0.3, joinstyle="round", zorder=z))


# ---------------------------------------------------------------- globes
def geo_globe(fig, cx, cy, d, face="white", z=10):
    ax = fig.add_axes(R(cx - d / 2, cy - d / 2, d, d), projection=ccrs.Orthographic(CLON, CLAT), zorder=z)
    ax.set_global(); ax.set_facecolor(face)
    ax.spines["geo"].set_linewidth(0.5); ax.spines["geo"].set_edgecolor("#8794A2")
    return ax


def shade(ax, dark=0.06, strength=0.20, light=0.12, n=600):
    """limb darkening and a soft highlight (upper left), drawn in the projection's own coordinates"""
    xl, yl = ax.projection.x_limits, ax.projection.y_limits
    g = np.linspace(-1, 1, n, dtype=np.float32); X, Y = np.meshgrid(g, g); rr = np.hypot(X, Y)
    a = np.zeros((n, n, 4), np.float32); a[..., :3] = dark; a[..., 3] = strength * np.clip(rr, 0, 1) ** 4; a[rr > 1, 3] = 0
    ax.imshow(a, extent=(xl[0], xl[1], yl[0], yl[1]), transform=ax.projection, origin="lower", interpolation="bilinear", zorder=20)
    if light:
        b = np.zeros((n, n, 4), np.float32); b[..., :3] = 1.0
        b[..., 3] = light * np.exp(-((X + 0.40) ** 2 + (Y - 0.45) ** 2) / 0.16); b[rr > 1, 3] = 0
        ax.imshow(b, extent=(xl[0], xl[1], yl[0], yl[1]), transform=ax.projection, origin="lower", interpolation="bilinear", zorder=21)


def halo(bg, cx, cy, d, color, z=2):
    for k, al in ((0.508, 0.22), (0.522, 0.11), (0.538, 0.05)):
        bg.add_patch(Circle((cx, cy), k * d, fc="none", ec=color, lw=1.1, alpha=al, zorder=z))


def field_on(ax, a, lat0, dlat, lon0, dlon, cmap, norm, regrid=1400, z=5, alpha=None):
    ny, nx = a.shape
    ax.imshow(np.ma.masked_invalid(a), extent=(lon0, lon0 + nx * dlon, lat0 - ny * dlat, lat0), origin="upper",
              transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, interpolation="nearest", regrid_shape=regrid, zorder=z,
              alpha=alpha)


def ortho_ll(n):
    """latitude, longitude (deg) of an n x n orthographic image centred on (CLON, CLAT), row 0 at the top; rho = radius"""
    g = (np.arange(n) + 0.5) / n * 2 - 1
    X, Y = np.meshgrid(g, -g)
    rho = np.hypot(X, Y); c = np.arcsin(np.clip(rho, 0, 1))
    p0, l0 = np.deg2rad(CLAT), np.deg2rad(CLON)
    with np.errstate(all="ignore"):
        lat = np.arcsin(np.cos(c) * np.sin(p0) + np.where(rho > 0, Y * np.sin(c) * np.cos(p0) / rho, 0.0))
        lon = l0 + np.arctan2(X * np.sin(c), rho * np.cos(c) * np.cos(p0) - Y * np.sin(c) * np.sin(p0))
    return np.rad2deg(lat), (np.rad2deg(lon) + 180.0) % 360.0 - 180.0, rho


def sample(a, lat, lon, lat0, dlat, lon0, dlon):
    i = np.floor((lat0 - lat) / dlat).astype(int); j = np.floor((lon - lon0) / dlon).astype(int) % a.shape[1]
    ok = (i >= 0) & (i < a.shape[0])
    out = np.full(lat.shape, np.nan, np.float32); out[ok] = a[i[ok], j[ok]]
    return out


def finish_disc(rgb, rho, n, shade_k=0.16, light=0.10):
    rgb = rgb * (1 - shade_k * np.clip(rho, 0, 1)[..., None] ** 4)
    hl = light * np.exp(-(((np.arange(n) + .5) / n * 2 - 1)[None, :] + 0.40) ** 2 / 0.16
                       - ((-((np.arange(n) + .5) / n * 2 - 1))[:, None] - 0.45) ** 2 / 0.16)
    rgb = rgb + (1 - rgb) * hl[..., None]
    alpha = np.clip((1 - rho) * n / 2.2, 0, 1)
    return np.dstack([np.clip(rgb, 0, 1), alpha]).astype(np.float32)


# ---------------------------------------------------------------- wind: line-integral convolution
def lic(u, v, n=40, step=0.6, seed=4):
    ny, nx = 721, 1440
    lat = 90 - 0.25 * np.arange(ny); lon = -180 + 0.25 * np.arange(nx)
    Rr, Cc = np.meshgrid(90 - lat, lon + 180, indexing="ij")
    uu = map_coordinates(u, [Rr, Cc], order=1, mode="grid-wrap"); vv = map_coordinates(v, [Rr, Cc], order=1, mode="grid-wrap")
    sp = np.hypot(uu, vv); du, dv = uu / (sp + 1e-6), vv / (sp + 1e-6)
    noise = np.random.default_rng(seed).random((ny, nx)).astype(np.float32)
    acc, wsum = noise.copy(), np.ones_like(noise)
    for sgn in (1.0, -1.0):
        y = np.repeat(np.arange(ny, dtype=np.float32)[:, None], nx, 1); x = np.repeat(np.arange(nx, dtype=np.float32)[None, :], ny, 0)
        for k in range(1, n + 1):
            au = map_coordinates(du, [y, x], order=1, mode="grid-wrap"); av = map_coordinates(dv, [y, x], order=1, mode="grid-wrap")
            cl = np.cos(np.deg2rad(90.0 - 0.25 * y)).clip(0.2)
            x = (x + sgn * step * 4 * au / cl) % nx; y = np.clip(y - sgn * step * 4 * av, 0, ny - 1)
            w = 0.5 * (1 + np.cos(np.pi * k / (n + 1)))
            acc += w * map_coordinates(noise, [y, x], order=1, mode="grid-wrap"); wsum += w
    L = acc / wsum
    L = np.clip(0.5 + 0.5 * (L - L.mean()) / (2.2 * L.std()), 0, 1)
    return L, sp


# ---------------------------------------------------------------- the figure
def main():
    Fd = dict(np.load(HERE / "data" / (f"arch_fields_{os.environ['FOEHN_DESIGN']}.npz" if os.environ.get("FOEHN_DESIGN") else "arch_fields.npz")))
    T = dict(np.load(HERE / "data" / "arch_overview.npz"))
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))
    bg = mm_axes(fig, -10); ov = mm_axes(fig, 60); dg = mm_axes(fig, 40)
    YA = 81.5                                                                     # bottom of the top band
    panel(bg, 1.5, YA, 98.0, 53.0, "#F5F9FE", "#E3EDF9", "#BBD0EA")
    panel(bg, 101.5, YA, 80.0, 53.0, "#F9F9FA", "#EBECEF", "#D3D6DB")
    panel(bg, 1.5, 1.5, 180.0, 78.0, "#FCFCFD", "#EFF0F3", "#CFD2D8")
    bg.set_xlim(0, W_MM); bg.set_ylim(0, H_MM)
    ty = YA + 50.0
    text(ov, 4.5, ty, "a", 8, INK, "bold", ha="left"); text(ov, 8.4, ty, "Computed transport", 6.6, BLUE, "bold", ha="left")
    text(ov, 33.4, ty, "·  no learned parameters", 6.2, BLUE, ha="left")
    text(ov, 104.5, ty, "b", 8, INK, "bold", ha="left"); text(ov, 108.4, ty, "Learned tracers", 6.6, GREY, "bold", ha="left")
    text(ov, 4.5, 76.5, "c", 8, INK, "bold", ha="left"); text(ov, 8.4, 76.5, "Generator", 6.6, DARK, "bold", ha="left")
    text(ov, 20.9, 76.5, "·  conditional flow matching at 1 km", 6.2, DARK, ha="left")
    gyA = YA + 22.5                                                               # globe centre line of the top band
    labA = YA + 2.6

    # ---- a1: the 3-D forecast wind (850 hPa, LIC)
    L, sp = lic(Fd["u850"], Fd["v850"])
    s_ = np.clip(sp / 20.0, 0, 1)
    b = np.clip(0.13 + 1.0 * L ** 2.0 * (0.40 + 0.60 * s_ ** 0.85), 0, 1)
    c0_, c1_, c2_ = np.array([0.02, 0.06, 0.13]), np.array([0.20, 0.42, 0.66]), np.array([0.93, 0.97, 1.0])
    t_ = b[..., None]
    wind = np.where(t_ < 0.5, c0_ * (1 - 2 * t_) + c1_ * 2 * t_, c1_ * (2 - 2 * t_) + c2_ * (2 * t_ - 1))
    wcx, wd = 21.5, 36.0
    halo(bg, wcx, gyA, wd, BLUE)
    ax = geo_globe(fig, wcx, gyA, wd, face="#061224")
    ax.imshow(wind, extent=(-180, 180, -90.125, 90.125), origin="upper", transform=ccrs.PlateCarree(), regrid_shape=1600,
              interpolation="bilinear", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.3, edgecolor="#A8C2DD", alpha=0.7, zorder=4)
    shade(ax, dark=0.0, strength=0.40, light=0.07)
    text(ov, wcx, labA, "3-D forecast wind", 5.8, INK)

    # ---- FLEXGREEN
    bx, by = 49.8, gyA                                                          # FLEXGREEN: a physics engine (gear) run backwards (rewind)
    gear(ov, bx, by + 2.2, 5.3, 4.25, 10, BLUE)
    ov.add_patch(Circle((bx, by + 2.2), 2.55, fc="white", ec="none", zorder=36))
    for tip in (bx - 1.2, bx + 0.02):
        ov.add_patch(Polygon([(tip, by + 2.2), (tip + 1.18, by + 3.08), (tip + 1.18, by + 1.32)], closed=True, fc=BLUE, ec=BLUE,
                             lw=0.25, joinstyle="round", zorder=37))
    text(ov, bx, by - 4.7, "FLEXGREEN", 6.0, BLUE, "bold")
    text(ov, bx, by - 8.9, "particles followed\nbackwards in time", 5.0, MUTED, style="italic", linespacing=1.15)
    arrow(ov, (wcx + wd / 2 + 1.0, by + 2.2), (bx - 5.9, by + 2.2), BLUE)

    # ---- a2: back-trajectories of the air reaching 18 cities
    tcx, td = 78.0, 36.0
    halo(bg, tcx, gyA, td, BLUE)
    ax = geo_globe(fig, tcx, gyA, td, face="white")
    ax.add_feature(cfeature.LAND.with_scale("110m"), fc="#EFEEEA", ec="none", zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.25, edgecolor="#A7B1BC", zorder=2)
    lon, lat, z = T["lon"], T["lat"], T["z"] / 1000.0
    pr = ax.projection.transform_points(ccrs.PlateCarree(), lon.ravel().astype(np.float64), lat.ravel().astype(np.float64))
    X, Y = pr[:, 0].reshape(lon.shape), pr[:, 1].reshape(lon.shape)
    nh, npart = lon.shape
    hcm = LinearSegmentedColormap.from_list("h", ["#0D3366", "#1F5AA6", "#7FA8D8", "#EBA8C6", PINK], N=256)
    hn = Normalize(0, 3.5)
    seg = np.stack([np.stack([X[:-1], Y[:-1]], -1), np.stack([X[1:], Y[1:]], -1)], 2)
    zc = 0.5 * (z[:-1] + z[1:]); age = np.repeat((np.arange(nh - 1) / (nh - 2))[:, None], npart, 1)
    ok = np.isfinite(seg).all((2, 3)) & (np.abs(seg[:, :, 1, 0] - seg[:, :, 0, 0]) < 1.5e6)
    rgba = hcm(hn(zc)); rgba[..., 3] = 0.80 - 0.50 * age
    ax.add_collection(LineCollection(seg[ok], colors=rgba[ok], linewidths=0.30, zorder=5, rasterized=True, capstyle="round"))
    org = np.isfinite(X[-1])
    ax.scatter(X[-1][org], Y[-1][org], s=0.5, c=hcm(hn(z[-1][org])), linewidths=0, zorder=6, rasterized=True)
    cp = ax.projection.transform_points(ccrs.PlateCarree(), T["clon"].astype(np.float64), T["clat"].astype(np.float64))
    vis = np.isfinite(cp[:, 0])
    ax.scatter(cp[vis, 0], cp[vis, 1], s=2.6, color="white", edgecolors=INK, linewidths=0.45, zorder=8)
    shade(ax, strength=0.16, light=0.10)
    text(ov, tcx, labA, "Backward trajectories, 5 days", 5.8, INK)
    arrow(ov, (bx + 5.9, by + 2.2), (tcx - td / 2 - 1.0, by + 2.2), BLUE)
    for k, (col, lab) in enumerate(((hcm(0.12), "near the ground"), (hcm(0.95), "aloft"))):   # colour key of the heights
        xk = 69.0 + 20.0 * k; yk = gyA + td / 2 + 2.6
        ov.plot([xk, xk + 3.0], [yk, yk], color=col, lw=1.4, solid_capstyle="round", zorder=40)
        text(ov, xk + 3.8, yk, lab, 5.0, MUTED, ha="left")

    # ---- b: PM2.5 and fires at 0.25 deg -> encoder -> read at the origins
    pcx, pd_ = 158.5, 35.0
    halo(bg, pcx, gyA, pd_, "#8C939C")
    ax = geo_globe(fig, pcx, gyA, pd_, face="white")
    ax.add_feature(cfeature.LAND.with_scale("110m"), fc=LAND_NODATA, ec="none", zorder=1)
    field_on(ax, Fd["pm025"], 84.0, 0.25, -180.0, 0.25, PCM, PNORM, regrid=1000, z=3)
    fr = Fd["fires"] > 3.6
    fi, fj = np.where(fr)
    flat, flon = 84.0 - 0.25 * fi - 0.125, -180.0 + 0.25 * fj + 0.125
    fp = ax.projection.transform_points(ccrs.PlateCarree(), flon.astype(np.float64), flat.astype(np.float64))
    ax.scatter(fp[:, 0], fp[:, 1], s=0.10, color="#4A0B0B", linewidths=0, zorder=6, rasterized=True, alpha=0.55)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.22, edgecolor="#7D8792", zorder=5)
    shade(ax, strength=0.16, light=0.10)
    text(ov, pcx, labA, "PM$_{2.5}$ and fires, 0.25°", 5.8, INK)
    ex, ey = 130.0, gyA
    ov.add_patch(Polygon([(ex + 6.2, ey + 8.0), (ex + 6.2, ey - 8.0), (ex - 5.8, ey - 4.0), (ex - 5.8, ey + 4.0)], closed=True,
                         fc="#3A4049", ec="#2A2F36", lw=0.6, joinstyle="round", zorder=35))
    text(ov, ex + 0.2, ey + 1.0, "Encoder", 5.7, "white", "bold")
    text(ov, ex + 0.2, ey - 2.3, "$E_\\phi$", 6.0, "#D6DAE0")
    arrow(ov, (pcx - pd_ / 2 - 1.0, ey), (ex + 6.5, ey), GREY)
    gx, gr = 112.6, 7.4                                                         # the balance of the sketch: tracers read at the origins
    balance(ov, gx, gyA + 0.2)
    arrow(ov, (ex - 6.1, ey), (gx + gr + 0.4, gyA + 0.3), GREY)
    text(ov, gx, gyA + 10.6, "tracers read at the\norigins of the air", 5.2, MUTED, style="italic", linespacing=1.15)
    arrow(ov, (tcx + td / 2 + 1.0, gyA), (gx - gr - 0.4, gyA + 0.3), BLUE)

    # ---- c: start day at 1 km, with a 1 km lens
    scx, scy, sd = 21.5, 44.0, 34.0
    halo(bg, scx, scy, sd, "#8C939C")
    ax = geo_globe(fig, scx, scy, sd, face="white")
    ax.add_feature(cfeature.LAND.with_scale("110m"), fc=LAND_NODATA, ec="none", zorder=1)
    field_on(ax, Fd["pm005"].astype(np.float32), 84.0, 0.05, -180.0, 0.05, PCM, PNORM, regrid=1900, z=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.2, edgecolor="#7D8792", zorder=5)
    shade(ax, strength=0.16, light=0.10)
    text(ov, scx, scy + sd / 2 + 3.4, "PM$_{2.5}$ at 1 km, start day", 5.8, INK)
    lb = Fd["lens_box"]
    dl = ax.projection.transform_point(0.5 * (lb[2] + lb[3]), 0.5 * (lb[0] + lb[1]), ccrs.PlateCarree())
    xl_ = ax.projection.x_limits
    lx0, ly0 = scx + dl[0] / xl_[1] * sd / 2, scy + dl[1] / xl_[1] * sd / 2
    lcx, lcy, lr = 33.0, 13.5, 7.2
    for sgn in (1, -1):
        ang = np.arctan2(lcy - ly0, lcx - lx0) + sgn * np.pi / 2
        ov.plot([lx0 + 0.9 * np.cos(ang), lcx + (lr + 0.5) * np.cos(ang)], [ly0 + 0.9 * np.sin(ang), lcy + (lr + 0.5) * np.sin(ang)],
                color="#3D4450", lw=0.45, zorder=36)
    ov.add_patch(Circle((lx0, ly0), 0.9, fc="none", ec=INK, lw=0.6, zorder=37))
    la = fig.add_axes(R(lcx - lr, lcy - lr, 2 * lr, 2 * lr), zorder=62)
    im = la.imshow(Fd["lens"].astype(np.float32), cmap=PCM, norm=PNORM, interpolation="nearest")
    la.set_xticks([]); la.set_yticks([]); la.patch.set_visible(False)
    im.set_clip_path(Circle((0.5, 0.5), 0.5, transform=la.transAxes))
    for s2 in la.spines.values():
        s2.set_visible(False)
    ov.add_patch(Circle((lcx, lcy), lr, fc="none", ec="white", lw=1.5, zorder=63))
    ov.add_patch(Circle((lcx, lcy), lr + 0.5, fc="none", ec=INK, lw=0.5, zorder=63))
    text(ov, lcx - lr - 1.0, lcy - lr + 1.6, "1 km", 5.2, INK, "bold", ha="right")
    text(ov, lcx - lr - 1.0, lcy - lr - 1.1, "Delhi", 5.0, MUTED, ha="right", style="italic")

    # ---- forecast weather (c) and the emissions (a) (orthographic renders)
    n = 440
    lat_, lon_, rho = ortho_ll(n)
    Lw = np.nan_to_num(sample(L, lat_, lon_, 90.125, 0.25, -180.125, 0.25))
    Sw = np.nan_to_num(sample(s_, lat_, lon_, 90.125, 0.25, -180.125, 0.25))
    bw = np.clip(0.08 + 1.1 * Lw ** 2.0 * (0.35 + 0.65 * Sw), 0, 1)
    t2c = map_coordinates(Fd["t2"], [90.0 - lat_, (lon_ + 180.0) % 360.0], order=1, mode="grid-wrap")   # 2 m temperature (deg C)
    tcm = LinearSegmentedColormap.from_list("t2", ["#24185C", "#2D5AA6", "#2F98B6", "#62C2A2", "#CBE98F", "#FBE07A"])
    t_rgb = tcm(Normalize(-32.0, 30.0)(t2c))[..., :3]
    w_rgb = t_rgb * (1 - 0.62 * bw[..., None]) + 0.62 * bw[..., None]                  # white wind streaks
    lm_ = sample(Fd["land025"].astype(np.float32), lat_, lon_, 84.0, 0.25, -180.0, 0.25) > 0.5
    edge = (lm_ != np.roll(lm_, 1, 0)) | (lm_ != np.roll(lm_, 1, 1))
    w_rgb[edge] = w_rgb[edge] * 0.45
    wimg = finish_disc(w_rgb, rho, n)
    land_ = sample(Fd["land025"].astype(np.float32), lat_, lon_, 84.0, 0.25, -180.0, 0.25) > 0.5
    # a: the anthropogenic emissions (CAMS-GLOB-ANT, black carbon, January 2022), which FLEXGREEN adds up along the paths of the
    # air (sensitivity to surface emissions): 6 of the 44 transport maps, no learned parameter (the encoder reads PM2.5 and fires)
    em = Fd["emis"]; lo_, hi_ = np.nanpercentile(em[Fd["land025"]], [30, 99.5])
    ev = np.clip((np.nan_to_num(sample(em, lat_, lon_, 84.0, 0.25, -180.0, 0.25), nan=lo_) - lo_) / (hi_ - lo_), 0, 1)
    e_rgb = np.where(land_[..., None], EMIS_CM(ev ** 0.8)[..., :3], 1.0)
    eimg = finish_disc(e_rgb, rho, n)
    ecx, ecy, ed = 49.8, gyA + 17.6, 10.5                                       # a: FLEXGREEN adds up what the air takes up
    halo(bg, ecx, ecy, ed, BLUE)
    dg.imshow(eimg, extent=(ecx - ed / 2, ecx + ed / 2, ecy - ed / 2, ecy + ed / 2), origin="upper", interpolation="bilinear", zorder=5)
    dg.add_patch(Circle((ecx, ecy), ed / 2, fc="none", ec="#8794A2", lw=0.45, zorder=6))
    text(ov, ecx - ed / 2 - 1.2, ecy, "anthropogenic\nemissions", 5.3, INK, ha="right", linespacing=1.05)
    arrow(ov, (ecx, ecy - ed / 2 - 0.5), (ecx, gyA + 2.2 + 5.6), BLUE, ms=5.5)                    # into the gear
    for (cx_, cy_, img, lab) in ((49.5, 57.5, wimg, "forecast weather"),):
        d_ = 14.0
        halo(bg, cx_, cy_, d_, "#8C939C")
        dg.imshow(img, extent=(cx_ - d_ / 2, cx_ + d_ / 2, cy_ - d_ / 2, cy_ + d_ / 2), origin="upper", interpolation="bilinear", zorder=5)
        dg.add_patch(Circle((cx_, cy_), d_ / 2, fc="none", ec="#8794A2", lw=0.45, zorder=6))
        text(ov, cx_, cy_ - d_ / 2 - 2.3, lab, 5.3, INK)

    # ---- transport maps (from a and b), above the network
    tm = Fd["tmaps"]
    r0, r1, c0, c1 = int((84 - 62) / 0.5), int((84 + 8) / 0.5), int((-22 + 180) / 0.5), int((128 + 180) / 0.5)
    cards = [(tm[2][r0:r1, c0:c1], LinearSegmentedColormap.from_list("sv", ["#FFFFFF", "#A9B8C9", "#2E3D52"]), Normalize(0.0, 0.8), "survival"),
             (tm[1][r0:r1, c0:c1], LinearSegmentedColormap.from_list("al", ["#FFFFFF", "#F2BCD3", PINK]), Normalize(0.0, 1.0), "air from aloft"),
             (tm[0][r0:r1, c0:c1], PCM, PNORM, "advected PM$_{2.5}$")]
    cw, ch_ = 19.0, 9.0
    xs0, ys0 = 61.0, 63.0
    for k, (img, cm_, nm_, lab) in enumerate(cards):
        x0_, y0_ = xs0 + 2.2 * k, ys0 - 2.9 * k
        ov.add_patch(Rectangle((x0_ + 0.45, y0_ - 0.45), cw, ch_, fc="#000000", alpha=0.10, ec="none", zorder=40 + 2 * k))
        ca = fig.add_axes(R(x0_, y0_, cw, ch_), zorder=61 + k)
        ca.imshow(img, cmap=cm_, norm=nm_, interpolation="nearest", aspect="auto")
        ca.set_xticks([]); ca.set_yticks([]); ca.set_facecolor("white")
        for s2 in ca.spines.values():
            s2.set_linewidth(0.45); s2.set_edgecolor("#59626E")
        text(ov, x0_ + cw + 1.3, y0_ + ch_ - 1.5, lab, 5.0, MUTED, ha="left", z=70)
    text(ov, xs0, ys0 + ch_ + 2.0, "Transport maps", 5.6, INK, "bold", ha="left")
    text(ov, xs0 + 15.6, ys0 + ch_ + 2.0, "(3 of 44)", 5.0, MUTED, ha="left")
    arrow(ov, (gx, gyA - 6.2), (xs0 + cw + 6.0, ys0 + ch_ + 1.2), GREY, rad=-0.25)

    # ---- the network
    nx_, ny_, nw, nh_ = 73.5, 37.0, 21.0, 16.0
    nbox = FancyBboxPatch((nx_ - nw / 2, ny_ - nh_ / 2), nw, nh_, boxstyle="round,pad=0,rounding_size=2.0", fc="none",
                          ec="black", lw=0.6, zorder=35.2)
    ov.add_patch(nbox)
    g_ = np.linspace(0, 1, 128)[:, None, None]
    ng = np.repeat(np.array([0.20, 0.22, 0.26])[None, None] * (1 - g_) + np.array([0.06, 0.07, 0.08])[None, None] * g_, 2, 1)
    nim = ov.imshow(ng, extent=(nx_ - nw / 2, nx_ + nw / 2, ny_ - nh_ / 2, ny_ + nh_ / 2), origin="upper", aspect="auto",
                    interpolation="bilinear", zorder=35)
    nim.set_clip_path(nbox)
    ov.set_xlim(0, W_MM); ov.set_ylim(0, H_MM)
    ub = ny_ + 3.6                                                               # U-Net pictogram: encoder, bottleneck, decoder
    hs = (3.0, 2.3, 1.6, 1.1, 1.6, 2.3, 3.0); dips = (0.0, 0.45, 0.9, 1.25, 0.9, 0.45, 0.0)
    for k, (hk, dk) in enumerate(zip(hs, dips)):
        ov.add_patch(Rectangle((nx_ + (k - 3) * 1.55 - 0.45, ub - dk - hk / 2), 0.9, hk, fc="#AAB3BF", ec="none", zorder=36))
    for k in range(3):
        xa_, xb_ = nx_ + (k - 3) * 1.55, nx_ + (3 - k) * 1.55
        ov.plot([xa_, xb_], [ub + 1.9 - 0.25 * k, ub + 1.9 - 0.25 * k], color="#6F7884", lw=0.35, zorder=36)
        for xx_ in (xa_, xb_):
            ov.plot([xx_, xx_], [ub + 1.9 - 0.25 * k, ub - dips[k] + hs[k] / 2], color="#6F7884", lw=0.35, zorder=36)
    text(ov, nx_, ny_ - 3.2, "Conditional\nflow matching", 6.0, "white", "bold", linespacing=1.1)
    ov.add_patch(FancyArrowPatch((nx_ + 5.5, ny_ - nh_ / 2 - 0.4), (nx_ - 5.5, ny_ - nh_ / 2 - 0.4), arrowstyle="-|>", mutation_scale=6,
                                 lw=0.9, color=RED, connectionstyle="arc3,rad=-1.2", zorder=36))
    text(ov, nx_, ny_ - nh_ / 2 - 8.2, "Iterative sampling", 5.2, RED, style="italic")
    ys_ = 41.5                                                                   # between the two small globes, never over them
    arrow(ov, (scx + np.sqrt((sd / 2) ** 2 - (ys_ - scy) ** 2) + 0.9, ys_), (nx_ - nw / 2 - 0.4, ys_), INK)
    arrow(ov, (49.5 + 6.3, 57.5 - 6.3), (nx_ - nw / 2 + 3.2, ny_ + nh_ / 2 + 0.4), INK, rad=-0.10)   # clear of the globe
    arrow(ov, (xs0 + 2.2 * 2 + cw / 2, ys0 - 2.9 * 2 - 0.6), (nx_, ny_ + nh_ / 2 + 0.4), INK)

    # ---- the disc grid (geometry of the sketch): flow-matching time (red, down to the right) x lead time (blue, up to the right)
    if "members" in Fd:
        mem = Fd["members"].astype(np.float32)
    else:
        print("WARNING: no forecasts yet, start day used for every lead (layout test)")
        mem = np.repeat(Fd["pm005"][None, ::2, ::2].astype(np.float32), 7, 0)
    nD = 420
    lat_, lon_, rho = ortho_ll(nD)
    D, SX, ROT = 15.6, 0.64, -20.0                                               # disc height, width / height, tilt (deg)
    O = np.array([94.0, 36.5]); EX = np.array([12.6, -2.5]); EY = np.array([7.3, 6.5])
    fo = np.array([0.0, -8.4])                                                   # the dashed floor passes under the discs
    fl = [O + fo - 0.75 * EX - 0.55 * EY, O + fo + 4.75 * EX - 0.55 * EY, O + fo + 4.75 * EX + 2.65 * EY, O + fo - 0.75 * EX + 2.65 * EY]
    dg.add_patch(Polygon(fl, closed=True, fc="#DCE2EA", ec="none", alpha=0.45, zorder=0.2))
    dg.add_patch(Polygon(fl, closed=True, fc="none", ec="#C3CAD4", lw=0.4, alpha=0.8, zorder=0.25))
    for i in range(len(ROWS)):
        a_, b_ = O + i * EY + fo - 0.6 * EX, O + i * EY + fo + 4.6 * EX
        dg.plot([a_[0], b_[0]], [a_[1], b_[1]], color="#C3C8CF", lw=0.45, ls=(0, (2.2, 2.2)), zorder=0.5)
    for j in range(len(TAUS)):
        a_, b_ = O + j * EX + fo - 0.5 * EY, O + j * EX + fo + 2.5 * EY
        dg.plot([a_[0], b_[0]], [a_[1], b_[1]], color="#C3C8CF", lw=0.45, ls=(0, (2.2, 2.2)), zorder=0.5)
    rng = np.random.default_rng(5)
    for i in reversed(range(len(ROWS))):                                         # back rows first, then left to right
        f1 = sample(mem[ROWS[i] - 1], lat_, lon_, 84.0, 0.1, -180.0, 0.1)
        land_d = np.isfinite(f1)
        eps = rng.standard_normal(f1.shape).astype(np.float32)
        grey = np.clip(0.56 + 0.20 * eps, 0.08, 0.97)[..., None]
        pm_rgb = np.where(land_d[..., None], PCM(PNORM(np.nan_to_num(f1)))[..., :3], 1.0)
        for j, tau in enumerate(TAUS):
            img = finish_disc((1 - tau) * grey + tau * pm_rgb, rho, nD, shade_k=0.10, light=0.04)
            c = O + j * EX + i * EY
            zz = 2 + 10 * (len(ROWS) - i) + j * 0.5
            tr = Affine2D().scale(SX * D, D).rotate_deg(ROT).translate(*c) + dg.transData
            sh = Ellipse((0.06, -0.03), 1.0, 1.0, fc="#1A2230", alpha=0.13, ec="none", zorder=zz - 0.2)
            sh.set_transform(Affine2D().scale(SX * D, D).rotate_deg(ROT).translate(c[0] + 0.7, c[1] - 0.5) + dg.transData)
            dg.add_patch(sh)
            im = dg.imshow(img, extent=(-0.5, 0.5, -0.5, 0.5), origin="upper", interpolation="bilinear", zorder=zz)
            im.set_transform(tr)
            rim = Circle((0, 0), 0.5, fc="none", ec="#111111" if tau == 1 else "#8A929C", lw=1.0 if tau == 1 else 0.35,
                         zorder=zz + 0.1)
            rim.set_transform(tr); dg.add_patch(rim)
    arrow(ov, (nx_ + nw / 2 + 0.6, ny_ + 0.5), (O[0] - 7.4, O[1] - 0.6), INK)
    # generative sampling step (flow-matching time), along the front edge
    ang_x = np.degrees(np.arctan2(EX[1], EX[0])); ang_y = np.degrees(np.arctan2(EY[1], EY[0]))
    ux, uy = EX / np.hypot(*EX), EY / np.hypot(*EY)
    nrm = np.array([ux[1], -ux[0]])                                              # points down, away from the discs
    a0 = O + fo + 2.6 * nrm - 0.55 * EX; a1 = O + fo + 2.6 * nrm + 4.75 * EX
    arrow(ov, tuple(a0), tuple(a1), INK, lw=0.8, ms=6.5)
    for j in range(len(TAUS)):
        p_ = O + fo + 2.6 * nrm + j * EX
        ov.plot([p_[0] - 0.55 * nrm[0], p_[0] + 0.55 * nrm[0]], [p_[1] - 0.55 * nrm[1], p_[1] + 0.55 * nrm[1]], color=INK, lw=0.6, zorder=31)
    for j, lab in ((0, "$x_0 \\sim p_{\\mathrm{noise}}$"), (2, "$x_\\tau$"), (4, "$x_1 \\sim p_{\\mathrm{data}}$")):
        p_ = O + fo + 5.4 * nrm + j * EX
        text(ov, p_[0], p_[1], lab, 6.0, INK, rotation=ang_x)
    p_ = O + fo + 9.0 * nrm + 2.0 * EX
    text(ov, p_[0], p_[1], "Generative sampling step (flow-matching time)", 5.8, RED, rotation=ang_x)
    # forecast lead time (physical time), along the right edge
    q0 = O + fo + 4 * EX + 0.95 * EX + 1.0 * nrm * 0 - 0.25 * EY; q1 = q0 + 2.75 * EY
    arrow(ov, tuple(q0), tuple(q1), BLUE, lw=0.8, ms=6.5)
    nq = np.array([uy[1], -uy[0]])                                               # to the lower right of the lead axis
    for i, dd in enumerate(ROWS):
        p_ = q0 + (i + 0.25) * EY
        ov.plot([p_[0] - 0.55 * nq[0], p_[0] + 0.55 * nq[0]], [p_[1] - 0.55 * nq[1], p_[1] + 0.55 * nq[1]], color=BLUE, lw=0.6, zorder=31)
        t_ = p_ + 2.6 * nq
        text(ov, t_[0], t_[1], f"Day {dd}", 5.4, BLUE, "bold", rotation=0)
    t_ = q0 + 1.75 * EY + 2.6 * nq
    text(ov, t_[0], t_[1], "…", 6.0, BLUE, "bold")
    t_ = q0 + 1.32 * EY + 6.0 * nq
    text(ov, t_[0], t_[1], "Forecast lead time (physical time)", 5.5, BLUE, rotation=ang_y)

    # ---- PM2.5 colour scale: one rounded segment per class, values at the class limits, the last class open (> 250)
    pm_scale(ov, 125.5, 68.6, 51.0, 2.5)
    fig.patch.set_alpha(0.0)                                                      # transparent page: only the panel edges
    save(fig, "fig_gc1_schematic")


if __name__ == "__main__":
    main()
