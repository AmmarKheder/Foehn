#!/usr/bin/env python3
"""Fig. 1 of the Nature Communications version: how Foehn makes a forecast (a-c), and how it runs every day (d).

a-c  Panels a, b, c of the Nature figure, unchanged. They are not redrawn here: fig_gc1_schematic.main() itself is
     called, with its page replaced by the upper sub-figure of a taller page and its save() switched off, so the three
     panels come out pixel for pixel as in the original, at the same distance from the top-left corner of the page
     (check_overlay.py proves it against a render of the original and against the published PNG).
d    Real time, the forecast issued on 11 November 2026 (lead day 1 = 11 November). From left to right:
     the live inputs of the issue day (PM2.5 of the CAMS analysis of 00 UTC at 0.4 deg, land only; 850 hPa wind at step 0
     of the IFS 00 UTC run as a line-integral convolution; GFAS fires of 10 November, the cells drawn in panel b);
     the design of a-c at 0.4 deg (the symbols of the three panels under a brace); its eight members (daily mean of lead
     day 1, 0.4 deg cells, within 8 deg of the lens centre); the second generative model, a functional flow matching that
     draws each member at 1 km (pictogram: its encoder, then a function read at any point); the mean of the eight 1 km
     fields of lead day 1 on a globe (every 5th pixel), with a 4 x 4 deg lens around Lahore at 1 km (400 x 400 pixels)
     and, beside it, the same window at 0.4 deg (10 x 10 cells, mean of the eight members).
     Time strip: the clock of that issue day (UTC), to scale between 10:15 and 11:03.

    PY=/projappl/project_462001804/envs/world/bin/python      (world venv; -B: nothing is written beside the originals)
    unset PYTHONPATH; MPLCONFIGDIR=<writable dir> $PY -B fig_gc1_schematic_rt.py            # .pdf and .png (300 dpi)
    ... fig_gc1_schematic_rt.py --prepare        # only (re)build data/rt_fields_2026-11-11.npz from the real-time outputs
    ... fig_gc1_schematic_rt.py --full FILE.png  # whole page at 300 dpi, no tight crop
    ... fig_gc1_schematic_rt.py --band FILE.png  # band d alone (layout work)
    ... check_overlay.py                         # the overlay check, into check/

Reads only; writes only beside this file. The real-time fields are copied once into data/rt_fields_<day>.npz, because
the folders they come from are cleaned by the daily chains when these run."""
import argparse
import json
import math
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/figures")
sys.path.insert(0, str(FIG))

import nature_style                                            # noqa: E402  rcParams, fonts, Natural Earth cache

nature_style.OUT = HERE                                        # second barrier: the original save() can never reach the Nature folder

# The submitted Fig. 1 embeds the Liberation Sans 2.1.5 of FIG/fonts (it was rendered in the container, which has no
# other copy). On the login nodes an older system copy (1.07.4, /usr/share/fonts) comes first in matplotlib's list and
# would be used instead: other glyph widths, so other pixels. Keep only the bundled files (in memory; nothing is written).
from matplotlib import font_manager as _fm                     # noqa: E402

_BUNDLED = str((FIG / "fonts").resolve())
_fm.fontManager.ttflist = [f for f in _fm.fontManager.ttflist
                           if f.name != "Liberation Sans" or str(Path(f.fname).resolve()).startswith(_BUNDLED)]
assert sum(f.name == "Liberation Sans" for f in _fm.fontManager.ttflist) == 4, "the four Liberation Sans files of FIG/fonts are expected"

import cartopy.crs as ccrs                                     # noqa: E402
import cartopy.feature as cfeature                             # noqa: E402
import numpy as np                                             # noqa: E402
from matplotlib.font_manager import FontProperties             # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle   # noqa: E402
from matplotlib.path import Path as MPath                      # noqa: E402
from matplotlib.textpath import TextPath                       # noqa: E402
from nature_style import MM, plt                               # noqa: E402

import fig_gc1_schematic as F                                  # noqa: E402  the Nature figure: helpers, colours, main()

# ------------------------------------------------------------------ the issue day and its sources
DAY, LEAD, CITY = "2026-11-11", 1, "Lahore"
RT = Path("/scratch/project_462001804/ammar/eccv/cran-pm/experiments/global1km/realtime")              # 0.4 deg chain
O1K = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution/realtime_1km/out")       # 1 km chain
G = Path("/scratch/project_462001804/ammar/eccv/data/world/global1km")
TAG, TILE, RES, LON0 = "pmG_final_P3T3", 1000, 0.01, -179.995
CACHE = HERE / "data" / f"rt_fields_{DAY}.npz"
LENS = (33.4, 29.4, 72.2, 76.2)                               # N, S, W, E: exactly 10 x 10 cells of 0.4 deg = 400 x 400 pixels
LENS_C = (0.5 * (LENS[2] + LENS[3]), 0.5 * (LENS[0] + LENS[1]))   # lon, lat of its centre: 74.2 E, 31.4 N (Lahore 74.34 E, 31.55 N)
CAP = 8.0                                                      # angular radius (deg) of the member discs, same centre
MEMBERS = tuple(range(8))                                      # members of the stack, front to back
# clock of the issue day, UTC: RT/data/2026-11-11/journal.txt ("fetch done"), RT/logs/<job>_2026-11-11_23638481.out
# (start, "predict rc=0"), O1K/../logs/o1k_run_23638955.out (start) and run.json (written 10:56:40), o1k_web_23638956.out (end)
CLOCK = dict(inputs="10:15:06", ens0="10:16:08", ens1="10:33:20", km0="10:36:26", km1="10:56:40", online="11:03:39")

# height added under a-c: 192 pt = 67.73 mm. A whole (and even) number of points and of pixels (800 at 300 dpi, 1600 at
# 600 dpi), so that a, b, c fall on the same pixels as in the original, and on the same points in the PDF (matplotlib
# places grouped images on whole points).
DY = 192 / 72 * 25.4
W = F.W_MM
INK, MUTED, BLUE, GREY, DARK = F.INK, F.MUTED, F.BLUE, F.GREY, F.DARK


# ------------------------------------------------------------------ loaders of the real-time fields (facts/figures.md, Appendix A)
def tile_index(day, tag=TAG):
    """tile name -> part number, and the latitude of lattice row 0 (-59.995 for the GHAP-trained system)"""
    idx, lat0 = {}, None
    for p in sorted((O1K / day / tag).glob("part*.json")):
        m = json.loads(p.read_text()); lat0 = float(m["lattice"]["lat_row0"])
        idx.update({t: int(m["rank"]) for t in m["tiles"]})
    return idx, lat0


def window(day, L, product, la0, la1, lo0, lo1, tag=TAG):
    """north-up float32 window of one product of lead day L (NaN outside the mask) and its extent (W, E, S, N)"""
    idx, lat0 = tile_index(day, tag)
    s0, w0 = lat0 - RES / 2, LON0 - RES / 2
    i0, i1 = int(math.floor((la0 - s0) / RES + 1e-6)), int(math.ceil((la1 - s0) / RES - 1e-6))
    j0, j1 = int(math.floor((lo0 - w0) / RES + 1e-6)), int(math.ceil((lo1 - w0) / RES - 1e-6))
    out = np.full((i1 - i0, j1 - j0), np.nan, np.float32)
    files = {}
    for r in range(i0 // TILE, (i1 - 1) // TILE + 1):
        for c in range(j0 // TILE, (j1 - 1) // TILE + 1):
            t = f"r{r:02d}c{c:02d}"
            if t not in idx:
                continue
            if idx[t] not in files:
                files[idx[t]] = np.load(O1K / day / tag / f"lead{L}_part{idx[t]}.npz")
            tile = files[idx[t]][f"{product}/{t}"][::-1].astype(np.float32)
            a0, a1, b0, b1 = max(i0, r * TILE), min(i1, (r + 1) * TILE), max(j0, c * TILE), min(j1, (c + 1) * TILE)
            out[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = tile[a0 - r * TILE:a1 - r * TILE, b0 - c * TILE:b1 - c * TILE]
    return out[::-1], (w0 + j0 * RES, w0 + j1 * RES, s0 + i0 * RES, s0 + i1 * RES)


def world(day, L, product, k=5, tag=TAG):
    """world mosaic from every k-th 1 km pixel: (15000 / k, 36000 / k), row 0 at 90 N"""
    n = TILE // k
    img = np.full((15 * n, 36 * n), np.nan, np.float32)
    for f in sorted((O1K / day / tag).glob(f"lead{L}_part*.npz")):
        z = np.load(f)
        for key in z.files:
            if key.startswith(product + "/"):
                r, c = int(key[-5:-3]), int(key[-2:])
                img[(14 - r) * n:(15 - r) * n, c * n:(c + 1) * n] = z[key][k // 2::k, k // 2::k]
    return img


def daily04(day, L):
    """daily mean of lead day L of the eight 0.4 deg members: trapezoid of the instants 24(L-1), 24L-12, 24L h, the 0 h
    instant being the CAMS analysis. (8, 361, 900), cell centres at latitude 84 - 0.4 i, longitude -180 + 0.4 j"""
    pred = np.load(RT / "out" / day / "pred.npy").astype(np.float32)           # (8, 10, 361, 900): 12, 24, ..., 120 h
    an = np.load(RT / "data" / day / "cams.npy")[1, 1].astype(np.float32)      # PM2.5 of the analysis of 00 UTC
    inst = lambda h: an[None] if h == 0 else pred[:, h // 12 - 1]
    return np.maximum(0.25 * inst(24 * (L - 1)) + 0.5 * inst(24 * L - 12) + 0.25 * inst(24 * L), 0.0)


def prepare():
    """copy what band d shows from the real-time outputs into one file of this directory"""
    import gc_style
    hb = np.load(RT / "data" / DAY / "hres_B.npy", mmap_mode="r")              # (33, 53, 181, 360): step 0 = 00 UTC
    lens, ext = window(DAY, LEAD, "mean", LENS[1], LENS[0], LENS[2], LENS[3])
    assert lens.shape == (400, 400) and np.allclose(ext, (LENS[2], LENS[3], LENS[1], LENS[0])), (lens.shape, ext)
    meta = dict(day=DAY, lead_day=LEAD, tag=TAG, lens_NSWE=LENS, gfas_day=json.loads((RT / "out" / DAY / "meta.json").read_text())["inputs"]["gfas_day"],
                sources=dict(an04="RT/data/<day>/cams.npy[1, 1]", mem04="daily04(): RT/out/<day>/pred.npy + analysis",
                             fires="RT/data/<day>/fires.npy[0]", wind="RT/data/<day>/hres_B.npy[0, 2] and [0, 10]",
                             w005="O1K/out/<day>/<tag>/lead1_part*.npz, product mean, every 5th pixel",
                             lens1km="same files, 400 x 400 pixels", land04="gc_style.land_mask_04()", land025="G/meta.npz land"))
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        CACHE, an04=np.load(RT / "data" / DAY / "cams.npy")[1, 1].astype(np.float32), mem04=daily04(DAY, LEAD).astype(np.float16),
        fires=np.load(RT / "data" / DAY / "fires.npy")[0].astype(np.float16),
        u850=np.asarray(hb[0, 2], np.float32), v850=np.asarray(hb[0, 10], np.float32),
        w005=world(DAY, LEAD, "mean").astype(np.float16), lens1km=lens.astype(np.float16),
        land04=gc_style.land_mask_04(), land025=np.load(G / "meta.npz")["land"].astype(bool), meta=json.dumps(meta))
    print("wrote", CACHE, f"{CACHE.stat().st_size / 1e6:.1f} MB")


def rt_fields():
    if not CACHE.exists():
        prepare()
    return dict(np.load(CACHE))


# ------------------------------------------------------------------ small tools of band d
def tw(s, size, weight="normal", style="normal"):
    """width (mm) of a label, from the font itself (no renderer needed)"""
    return TextPath((0, 0), s, size=size, prop=FontProperties(family="Liberation Sans", weight=weight, style=style)).get_extents().width * 25.4 / 72


def cap_ll(n, lon0, lat0, rad):
    """latitude and longitude (deg) of an n x n orthographic view of the cap of angular radius rad around (lon0, lat0)"""
    g = (np.arange(n) + 0.5) / n * 2 - 1
    X, Y = np.meshgrid(g, -g)
    rho = np.hypot(X, Y); k = np.sin(np.deg2rad(rad)); x, y = X * k, Y * k
    cc = np.sqrt(np.clip(1 - x * x - y * y, 0, 1)); p0 = np.deg2rad(lat0)
    lat = np.arcsin(np.clip(cc * np.sin(p0) + y * np.cos(p0), -1, 1))
    lon = lon0 + np.rad2deg(np.arctan2(x, cc * np.cos(p0) - y * np.sin(p0)))
    return np.rad2deg(lat), (lon + 180.0) % 360.0 - 180.0, rho


def ortho_xy(lon, lat):
    """position on the unit disc of the globes of the figure (orthographic, centre F.CLON, F.CLAT), and visibility"""
    la, lo, p0 = np.deg2rad(lat), np.deg2rad(np.asarray(lon) - F.CLON), np.deg2rad(F.CLAT)
    x = np.cos(la) * np.sin(lo); y = np.cos(p0) * np.sin(la) - np.sin(p0) * np.cos(la) * np.cos(lo)
    return x, y, np.sin(p0) * np.sin(la) + np.cos(p0) * np.cos(la) * np.cos(lo) > 0


def disc(dg, bg, img, cx, cy, d, halo=None, rim="#8794A2", lw=0.45, z=5):
    if halo:
        F.halo(bg, cx, cy, d, halo)
    dg.imshow(img, extent=(cx - d / 2, cx + d / 2, cy - d / 2, cy + d / 2), origin="upper", interpolation="bilinear", zorder=z)
    dg.add_patch(Circle((cx, cy), d / 2, fc="none", ec=rim, lw=lw, zorder=z + 0.1))


def brace(ax, p0, p1, w, color="#59626E", lw=0.6, z=36):
    """curly brace from p0 to p1 (a vertical or a horizontal segment), its tip w mm to the right of / below the middle"""
    (x0, y0), (x1, y1) = p0, p1
    if x0 == x1:                                               # vertical, tip to the right
        ym, h, q = 0.5 * (y0 + y1), w / 2, min(2.4, 0.22 * abs(y1 - y0)) * np.sign(y1 - y0)
        V = [(x0, y1), (x0 + h, y1), (x0 + h, y1 - q), (x0 + h, ym + q), (x0 + h, ym), (x0 + w, ym), (x0 + h, ym), (x0 + h, ym - q),
             (x0 + h, y0 + q), (x0 + h, y0), (x0, y0)]
    else:                                                      # horizontal, tip below
        xm, h, q = 0.5 * (x0 + x1), w / 2, min(2.4, 0.22 * abs(x1 - x0))
        V = [(x0, y0), (x0, y0 - h), (x0 + q, y0 - h), (xm - q, y0 - h), (xm, y0 - h), (xm, y0 - w), (xm, y0 - h), (xm + q, y0 - h),
             (x1 - q, y0 - h), (x1, y0 - h), (x1, y0)]
    C = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.CURVE3, MPath.CURVE3,
         MPath.LINETO, MPath.CURVE3, MPath.CURVE3]
    ax.add_patch(PathPatch(MPath(V, C), fc="none", ec=color, lw=lw, capstyle="round", joinstyle="round", zorder=z))


def mini_gear(ax, cx, cy, r):
    """the FLEXGREEN symbol of panel a (gear, hub, rewind), at radius r"""
    s = r / 5.3
    F.gear(ax, cx, cy, r, 4.25 * s, 10, BLUE)
    ax.add_patch(Circle((cx, cy), 2.55 * s, fc="white", ec="none", zorder=36))
    for tip in (cx - 1.2 * s, cx + 0.02 * s):
        ax.add_patch(Polygon([(tip, cy), (tip + 1.18 * s, cy + 0.88 * s), (tip + 1.18 * s, cy - 0.88 * s)], closed=True, fc=BLUE, ec=BLUE,
                             lw=0.15, joinstyle="round", zorder=37))


def unet(ax, cx, ub, s, bars="#AAB3BF", links="#6F7884", lw=0.35, z=36):
    """the U-Net pictogram of the generator box of panel c (encoder, bottleneck, decoder, skip connections), scaled by s"""
    hs = (3.0, 2.3, 1.6, 1.1, 1.6, 2.3, 3.0); dips = (0.0, 0.45, 0.9, 1.25, 0.9, 0.45, 0.0)
    for k, (hk, dk) in enumerate(zip(hs, dips)):
        ax.add_patch(Rectangle((cx + ((k - 3) * 1.55 - 0.45) * s, ub - (dk + hk / 2) * s), 0.9 * s, hk * s, fc=bars, ec="none", zorder=z))
    if links:
        for k in range(3):
            xa_, xb_, yy = cx + (k - 3) * 1.55 * s, cx + (3 - k) * 1.55 * s, ub + (1.9 - 0.25 * k) * s
            ax.plot([xa_, xb_], [yy, yy], color=links, lw=lw, zorder=z)
            for xx_ in (xa_, xb_):
                ax.plot([xx_, xx_], [yy, ub + (hs[k] / 2 - dips[k]) * s], color=links, lw=lw, zorder=z)


def dark_box(ax, cx, cy, w, h, r=2.0, lw=0.6, z=35):
    """the dark box of the generator of panel c"""
    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc="none", ec="black", lw=lw, zorder=z + 0.2)
    ax.add_patch(box)
    g_ = np.linspace(0, 1, 128)[:, None, None]
    ng = np.repeat(np.array([0.20, 0.22, 0.26])[None, None] * (1 - g_) + np.array([0.06, 0.07, 0.08])[None, None] * g_, 2, 1)
    im = ax.imshow(ng, extent=(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2), origin="upper", aspect="auto", interpolation="bilinear", zorder=z)
    im.set_clip_path(box)
    ax.set_xlim(0, W); ax.set_ylim(0, DY)


def lens(sub, ov, a, cx, cy, r):
    """a round lens as in panel c: the array a, one square per value, white gap, thin dark ring"""
    la = sub.add_axes(F.R(cx - r, cy - r, 2 * r, 2 * r), zorder=62)
    im = la.imshow(a, cmap=F.PCM, norm=F.PNORM, interpolation="nearest")
    la.set_xticks([]); la.set_yticks([]); la.patch.set_visible(False)
    im.set_clip_path(Circle((0.5, 0.5), 0.5, transform=la.transAxes))
    for s2 in la.spines.values():
        s2.set_visible(False)
    ov.add_patch(Circle((cx, cy), r, fc="none", ec="white", lw=1.5, zorder=63))
    ov.add_patch(Circle((cx, cy), r + 0.5, fc="none", ec=INK, lw=0.5, zorder=63))


def secs(hms):
    h, m, s = (int(v) for v in hms.split(":"))
    return 3600 * h + 60 * m + s


# ------------------------------------------------------------------ band d
def band_d(sub, D):
    """the real-time band, in mm from the lower-left corner of its own sub-figure (183 x DY mm)"""
    bg = F.mm_axes(sub, -10); ov = F.mm_axes(sub, 60); dg = F.mm_axes(sub, 40)
    top = DY - 0.5                                                              # 2 mm under the edge of panel c, as between a and c
    F.panel(bg, 1.5, 1.5, 180.0, top - 1.5, "#FCFCFD", "#EFF0F3", "#CFD2D8")
    bg.set_xlim(0, W); bg.set_ylim(0, DY)
    ty = top - 3.0                                                              # title line: letter and title as in a, b, c
    F.text(ov, 4.5, ty, "d", 8, INK, "bold", ha="left"); F.text(ov, 8.4, ty, "Real time", 6.6, DARK, "bold", ha="left")
    gap = 20.9 - 8.4 - tw("Generator", 6.6, "bold")                             # the gap of panel c between title and subtitle
    F.text(ov, 8.4 + tw("Real time", 6.6, "bold") + gap, ty, "·  forecast issued on 11 November 2026", 6.2, DARK, ha="left")
    yc = 40.4                                                                   # centre line of the chain

    # ---- (1) the live inputs of the issue day: three discs, one line each
    n = 440
    lat_, lon_, rho = F.ortho_ll(n)
    land = F.sample(D["land025"].astype(np.float32), lat_, lon_, 84.0, 0.25, -180.0, 0.25) > 0.5
    edge = (land != np.roll(land, 1, 0)) | (land != np.roll(land, 1, 1))
    an = np.where(D["land04"], D["an04"], np.nan)                               # PM2.5 of the analysis, land only (white ocean)
    v = F.sample(an, lat_, lon_, 84.2, 0.4, -180.2, 0.4)
    a_img = F.finish_disc(np.where(np.isfinite(v)[..., None], F.PCM(F.PNORM(np.nan_to_num(v)))[..., :3], 1.0), rho, n)
    L, sp = F.lic(D["u850"], D["v850"])                                         # as the wind globe of panel a
    Lw = np.nan_to_num(F.sample(L, lat_, lon_, 90.125, 0.25, -180.125, 0.25))
    Sw = np.nan_to_num(F.sample(np.clip(sp / 20.0, 0, 1), lat_, lon_, 90.125, 0.25, -180.125, 0.25))
    t_ = np.clip(0.13 + 1.0 * Lw ** 2.0 * (0.40 + 0.60 * Sw ** 0.85), 0, 1)[..., None]
    c0_, c1_, c2_ = np.array([0.02, 0.06, 0.13]), np.array([0.20, 0.42, 0.66]), np.array([0.93, 0.97, 1.0])
    w_rgb = np.where(t_ < 0.5, c0_ * (1 - 2 * t_) + c1_ * 2 * t_, c1_ * (2 - 2 * t_) + c2_ * (2 * t_ - 1))
    w_rgb[edge] = 0.45 * w_rgb[edge] + 0.55 * np.array([0.66, 0.76, 0.87])      # coastline, as on the wind globe of a
    w_img = F.finish_disc(w_rgb, rho, n, shade_k=0.40, light=0.07)
    f_rgb = np.where(land[..., None], np.array([0.933, 0.941, 0.949]), 1.0)     # land without data (#EEF0F2), white ocean
    f_rgb[edge & land] = np.array([0.78, 0.80, 0.83])
    f_img = F.finish_disc(f_rgb, rho, n)
    ix, idd, pitch = 11.2, 12.4, 13.8
    rows = ((yc + pitch, a_img, "#8C939C", "composition analysis, 00 UTC, 0.4°"),
            (yc, w_img, BLUE, "forecast wind, 00 UTC run"),
            (yc - pitch, f_img, "#8C939C", "fires of the day before"))
    xe = 0.0
    for k, (cy_, img, hc, lab) in enumerate(rows):                              # one z level per image: in a PDF, images that
        disc(dg, bg, img, ix, cy_, idd, halo=hc, z=5.0 + 0.3 * k)               # follow each other are merged and set on whole points
        F.text(ov, ix + idd / 2 + 2.0, cy_, lab, 5.3, INK, ha="left")
        xe = max(xe, ix + idd / 2 + 2.0 + tw(lab, 5.3))
    fi, fj = np.where(D["fires"].astype(np.float32) > 3.6)                      # the fire cells, as in panel b
    fx, fy, vis = ortho_xy(-180.0 + 0.25 * fj + 0.125, 84.0 - 0.25 * fi - 0.125)
    vis &= np.hypot(fx, fy) < 0.985                                             # no dot on the rim itself
    dg.scatter(ix + fx[vis] * idd / 2, yc - pitch + fy[vis] * idd / 2, s=0.16, color="#4A0B0B", linewidths=0, alpha=0.75, zorder=5.65,
               rasterized=True)
    bx = xe + 1.4
    brace(ov, (bx, yc - pitch - 4.8), (bx, yc + pitch + 4.8), 1.8)

    # ---- (2) the design of a-c at 0.4 deg: the symbols of the three panels, one after the other, under a brace
    gr, bs, nhw, step = 3.0, 0.46, 3.3, 3.2                                      # gear radius, balance scale, half box, gap
    bhw = 7.45 * bs                                                              # half width of the balance (pans included)
    gx = bx + 1.8 + 6.4 + gr                                                     # gear (a), balance (b), network (c)
    mx = gx + gr + step + bhw
    nx0 = mx + bhw + step + nhw
    mini_gear(ov, gx, yc, gr)
    F.balance(ov, mx, yc + 0.3, bs)
    nb = FancyBboxPatch((nx0 - nhw, yc - 2.7), 2 * nhw, 5.4, boxstyle="round,pad=0,rounding_size=0.9", fc="#1F2329", ec="black", lw=0.4, zorder=35)
    ov.add_patch(nb)
    unet(ov, nx0, yc + 0.25, 0.5, links=None)
    F.arrow(ov, (bx + 1.8 + 1.0, yc), (gx - gr - 0.7, yc), INK)
    F.arrow(ov, (gx + gr + 0.6, yc), (mx - bhw - 0.6, yc), INK, lw=0.7, ms=4.6)
    F.arrow(ov, (mx + bhw + 0.6, yc), (nx0 - nhw - 0.6, yc), INK, lw=0.7, ms=4.6)
    brace(ov, (gx - gr, yc - 4.6), (nx0 + nhw, yc - 4.6), 1.6)
    F.text(ov, 0.5 * (gx - gr + nx0 + nhw), yc - 8.7, "$\\bf{a}$–$\\bf{c}$ at 0.4°", 5.8, INK)

    # ---- (3) the 0.4 deg ensemble: four of the eight members, the cells visible
    nD = 720
    lat_c, lon_c, rho_c = cap_ll(nD, LENS_C[0], LENS_C[1], CAP)
    sd, sx = 15.0, nx0 + nhw + 16.6                                               # disc diameter, centre of the stack
    nm = len(MEMBERS)
    offs = [((k - (nm - 1) / 2) * 7.2 / (nm - 1), (k - (nm - 1) / 2) * 6.0 / (nm - 1)) for k in range(nm)]
    for k in reversed(range(nm)):                                                # back discs first
        m = np.where(D["land04"], D["mem04"][MEMBERS[k]].astype(np.float32), np.nan)
        v = F.sample(m, lat_c, lon_c, 84.2, 0.4, -180.2, 0.4)
        img = F.finish_disc(np.where(np.isfinite(v)[..., None], F.PCM(F.PNORM(np.nan_to_num(v)))[..., :3], 1.0), rho_c, nD,
                            shade_k=0.10, light=0.04)
        cx_, cy_ = sx + offs[k][0], yc + offs[k][1]; zz = 6 + 1.0 * (nm - 1 - k)
        dg.add_patch(Ellipse((cx_ + 0.6, cy_ - 0.5), sd, sd, fc="#1A2230", alpha=0.13 if k in (0, nm - 1) else 0.05, ec="none", zorder=zz - 0.2))
        disc(dg, bg, img, cx_, cy_, sd, rim="#8A929C", lw=0.35, z=zz)
    F.text(ov, sx, yc + offs[-1][1] + sd / 2 + 3.0, "8 members, 0.4°", 5.8, INK)
    xs0 = sx + offs[0][0] - math.sqrt((sd / 2) ** 2 - offs[0][1] ** 2)           # left edge of the front disc on the centre line
    xs1 = sx + offs[-1][0] + math.sqrt((sd / 2) ** 2 - offs[-1][1] ** 2)         # right edge of the back disc on the centre line
    F.arrow(ov, (nx0 + nhw + 0.8, yc), (xs0 - 0.9, yc), INK)

    # ---- (4) the downscaler: a box in the style of the generator of panel c
    nw, nh_ = 22.0, 18.0
    fx_ = xs1 + 5.4 + nw / 2
    dark_box(ov, fx_, yc, nw, nh_)
    ub = yc + 4.4
    unet(ov, fx_ - 4.2, ub, 0.8)                                                 # encoder on the coarse globe ...
    xx = np.linspace(0, 1, 60); cx0, cw_ = fx_ + 1.9, 6.6                        # ... then a function, read at any point
    fy_ = lambda u: ub - 0.25 + 1.25 * np.sin(2 * np.pi * (0.95 * u + 0.08)) * (1 - 0.25 * u)
    ov.plot(cx0 + cw_ * xx, fy_(xx), color="#AAB3BF", lw=0.7, solid_capstyle="round", zorder=36)
    for u in (0.04, 0.27, 0.5, 0.73, 0.96):
        ov.add_patch(Circle((cx0 + cw_ * u, fy_(u)), 0.36, fc="white", ec="#1B1E23", lw=0.25, zorder=37))
    F.text(ov, fx_, yc - 1.7, "Functional\nflow matching", 6.0, "white", "bold", linespacing=1.1)
    F.text(ov, fx_, yc - 6.5, "each member at 1 km", 5.0, "#D6DAE0")
    F.arrow(ov, (xs1 + 0.9, yc), (fx_ - nw / 2 - 0.4, yc), INK)

    # ---- (5) the 1 km forecast on a globe, a 1 km lens over the city and the same window at 0.4 deg beside it
    gd = 32.0
    gcx = fx_ + nw / 2 + 6.4 + gd / 2
    F.halo(bg, gcx, yc, gd, "#8C939C")
    ax = F.geo_globe(sub, gcx, yc, gd, face="white")
    ax.add_feature(cfeature.LAND.with_scale("110m"), fc=F.LAND_NODATA, ec="none", zorder=1)
    F.field_on(ax, D["w005"].astype(np.float32), 90.0, 0.05, -180.0, 0.05, F.PCM, F.PNORM, regrid=1900, z=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.2, edgecolor="#7D8792", zorder=5)
    F.shade(ax, strength=0.16, light=0.10)
    a_, b_ = "PM$_{2.5}$ at 1 km,", "lead days 1–5"
    wa, wb = tw(a_, 5.8), tw(b_, 5.4, "bold")
    yl_ = yc + gd / 2 + 3.3
    F.text(ov, gcx - (wa + 0.9 + wb) / 2, yl_, a_, 5.8, INK, ha="left")
    F.text(ov, gcx - (wa + 0.9 + wb) / 2 + wa + 0.9, yl_, b_, 5.4, BLUE, "bold", ha="left")
    F.arrow(ov, (fx_ + nw / 2 + 0.6, yc), (gcx - gd / 2 - 1.0, yc), INK)
    dl = ax.projection.transform_point(LENS_C[0], LENS_C[1], ccrs.PlateCarree())
    xl_ = ax.projection.x_limits
    lx0, ly0 = gcx + dl[0] / xl_[1] * gd / 2, yc + dl[1] / xl_[1] * gd / 2      # the city on the globe
    lr = 7.2                                                                     # as the lens of panel c
    lcx, lcy = gcx - 15.4, yc - 20.1                                             # the 1 km lens, clear of the globe and its halo
    for sgn in (1, -1):
        ang = np.arctan2(lcy - ly0, lcx - lx0) + sgn * np.pi / 2
        ov.plot([lx0 + 0.9 * np.cos(ang), lcx + (lr + 0.5) * np.cos(ang)], [ly0 + 0.9 * np.sin(ang), lcy + (lr + 0.5) * np.sin(ang)],
                color="#3D4450", lw=0.45, zorder=36)
    ov.add_patch(Circle((lx0, ly0), 0.9, fc="none", ec=INK, lw=0.6, zorder=37))
    lens(sub, ov, D["lens1km"].astype(np.float32), lcx, lcy, lr)
    i0, j0 = int(round((84.2 - LENS[0]) / 0.4)), int(round((LENS[2] + 180.2) / 0.4))
    coarse = D["mem04"].astype(np.float32).mean(0)[i0:i0 + 10, j0:j0 + 10]       # the same window: 10 x 10 cells, mean of the members
    ccx = lcx - 2 * lr - 1.4
    lens(sub, ov, coarse, ccx, lcy, lr)
    F.text(ov, lcx + lr + 2.3, lcy - 2.2, "1 km", 5.2, INK, "bold", ha="left")
    F.text(ov, lcx + lr + 2.3, lcy - 4.9, CITY, 5.0, MUTED, ha="left", style="italic")
    F.text(ov, ccx - lr - 1.5, lcy - 2.2, "0.4°", 5.2, INK, "bold", ha="right")
    F.text(ov, ccx - lr - 1.5, lcy - 4.9, "same window", 5.0, MUTED, ha="right", style="italic")

    # ---- (6) the clock of the issue day (UTC), to scale from the last input to the maps
    ys, x0, x1 = 6.9, 54.0, 172.0
    t0, t1 = secs(CLOCK["inputs"]), secs(CLOCK["online"])
    X = lambda hms: x0 + (secs(hms) - t0) / (t1 - t0) * (x1 - x0)
    xa = 27.0                                                                    # 00 UTC, before the break of the axis
    ov.plot([xa, x0 - 9.0], [ys, ys], color="#8A929C", lw=0.5, ls=(0, (0.6, 2.2)), dash_capstyle="round", zorder=30)
    for dx in (-8.2, -7.2):                                                      # the break
        ov.plot([x0 + dx - 0.45, x0 + dx + 0.45], [ys - 0.9, ys + 0.9], color="#8A929C", lw=0.5, zorder=30)
    ov.plot([x0 - 6.4, x1], [ys, ys], color="#8A929C", lw=0.5, zorder=30)
    for xt, num, word in ((xa, "00 UTC", "analysis time"), (x0, CLOCK["inputs"][:5] + " UTC", "inputs complete"),
                          (x1, CLOCK["online"][:5] + " UTC", "maps online")):
        ov.plot([xt, xt], [ys - 0.95, ys + 0.95], color=INK, lw=0.7, solid_capstyle="round", zorder=32)
        F.text(ov, xt, ys + 2.45, num, 5.4, INK, "bold"); F.text(ov, xt, ys - 2.55, word, 5.0, MUTED)
    for a0, a1, word in ((CLOCK["ens0"], CLOCK["ens1"], "0.4° ensemble"), (CLOCK["km0"], CLOCK["km1"], "1 km fields")):
        ov.add_patch(FancyBboxPatch((X(a0), ys - 0.55), X(a1) - X(a0), 1.1, boxstyle="round,pad=0,rounding_size=0.55", fc=GREY, ec="none", zorder=31))
        xm = 0.5 * (X(a0) + X(a1))
        F.text(ov, xm, ys + 2.45, f"{round((secs(a1) - secs(a0)) / 60):d} min", 5.4, INK, "bold"); F.text(ov, xm, ys - 2.55, word, 5.0, MUTED)
    th = np.deg2rad(np.linspace(15, -285, 60)); rr = 1.2; ex, ey = 5.85, ys      # "every day": an arrow that turns with the clock
    ov.add_patch(FancyArrowPatch(path=MPath(np.column_stack([ex + rr * np.cos(th), ey + rr * np.sin(th)])), arrowstyle="-|>", mutation_scale=4.4,
                                 lw=0.75, color=INK, capstyle="round", joinstyle="round", zorder=32))
    F.text(ov, 8.4, ys, "Every day", 5.6, INK, "bold", ha="left")
    return dict(lens_on_globe=(lx0, ly0), chain_y=yc)


# ------------------------------------------------------------------ the figure
def build():
    D = rt_fields()
    fig = plt.figure(figsize=(W * MM, (F.H_MM + DY) * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[F.H_MM, DY], left=0, right=1, bottom=0, top=1, hspace=0)
    abc, d = fig.add_subfigure(gs[0]), fig.add_subfigure(gs[1])
    keep = F.plt, F.save, F.H_MM
    F.plt = types.SimpleNamespace(figure=lambda *a, **k: abc)                    # the page of main() is the upper sub-figure
    F.save = lambda *a, **k: None                                                # and main() writes nothing
    try:
        F.main()                                                                 # panels a, b, c: the original code, untouched
        F.plt = keep[0]
        F.H_MM = DY                                                              # R() and mm_axes() now work in the 183 x DY mm band
        band_d(d, D)
    finally:
        F.plt, F.save, F.H_MM = keep
    fig.patch.set_alpha(0.0)                                                     # transparent page, as the original
    return fig


def build_band():
    """band d alone on a 183 x DY mm page (layout work only)"""
    fig = plt.figure(figsize=(W * MM, DY * MM))
    keep = F.H_MM
    F.H_MM = DY
    try:
        band_d(fig, rt_fields())
    finally:
        F.H_MM = keep
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--full", default=None)
    ap.add_argument("--band", default=None)
    a = ap.parse_args()
    if a.prepare:
        return prepare()
    if a.band:
        build_band().savefig(a.band, dpi=300)
        return print("saved", a.band)
    fig = build()
    if a.full:
        with plt.rc_context({"savefig.bbox": "standard"}):
            fig.savefig(a.full, dpi=300)
        return print("saved", a.full)
    fig.savefig(HERE / "fig_gc1_schematic_rt.pdf")                               # rcParams of nature_style: tight, 600 dpi rasters, TrueType
    fig.savefig(HERE / "fig_gc1_schematic_rt.png", dpi=300)
    print("saved", HERE / "fig_gc1_schematic_rt")


if __name__ == "__main__":
    main()
