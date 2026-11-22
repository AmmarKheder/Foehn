#!/usr/bin/env python3
"""Fig. 1 of the Nature Communications version, second version of the lower band: how Foehn makes a forecast (a-c), how
it runs every day (d) and how each coarse member becomes a 1 km field (e).

a-c  Panels a, b, c of the Nature figure, unchanged. They are not redrawn here: fig_gc1_schematic.main() itself is
     called, with its page replaced by the upper sub-figure of a taller page and its save() switched off, so the three
     panels come out pixel for pixel as in the original, at the same distance from the top-left corner of the page
     (check_overlay.py proves it against a render of the original and against the published PNG).
d    Real time, the forecast issued on 11 November 2026 (lead day 1 = 11 November), read from top to bottom: the live
     inputs of the issue day (PM2.5 of the CAMS analysis of 00 UTC at 0.4 deg, land only; 850 hPa wind at step 0 of the
     IFS 00 UTC run as a line-integral convolution; GFAS fires of 10 November, the cells drawn in panel b); the design
     of a-c at 0.4 deg (the symbols of the three panels); its eight members (daily mean of lead day 1, 0.4 deg cells,
     within 8 deg of the lens centre).
e    Downscaling to 1 km (functional flow matching), for member 1 of that forecast, read from left to right:
     what enters   the member (the disc in front of the stack of d), the forecast weather of the valid day (2 m
                   temperature, mean of 00, 06, 12, 18 UTC of the IFS run, with the 850 hPa wind of 00 UTC), three of
                   the six one-day transport maps of that member (survival, share of the air from aloft, advected
                   PM2.5: channels 43, 42, 41 of the encoder; land cells only, white ocean) and, at 1 km, the
                   climatology of the 1 km estimates of 2017-2020 for that day of the year (expm1 of mu_p(doy), the
                   two-harmonic fit of log1p PM2.5 that the model uses to standardise its target; its annual mean, its
                   departure from the cell mean, its seasonal amplitude and two spreads are five of the six
                   descriptors each pixel reads; drawn with the colours of the PM2.5 classes interpolated,
                   nature_style.pm_cmap_smooth, so that the 1 km detail shows) and, behind it, the elevation (the
                   sixth descriptor);
     the mechanism the encoder (trapezoid of panel b) reads the coarse fields on the whole globe; its output over the
                   window (first principal component of the 128 features, one square per 0.25 deg point) and the
                   1 km maps carry the same ring, at the pixel of Lahore: a pixel reads the encoding at its own
                   position together with its own 1 km descriptors; the pointwise flow then carries a noise value
                   to PM2.5 in ten Euler steps;
     the strip     the sampler's own states of that member over the window after 0, 4, 7 and 10 steps. Each state x is
                   drawn as a mix of two renderings of the same numbers: the grey of the noise discs of panel c
                   (0.56 + 0.20 x) and the PM2.5 colour of expm1(mu_p + sigma_p x), in the proportions 1 - t and t,
                   t = s / 10 being the flow time (the rule of the discs of panel c). Step 0 is the noise itself (the
                   fitted prior, five wavelength bands from 1 to 3,000 km), step 10 is the member (lens);
     what comes out the member at 1 km over the window (lens, 400 x 400 pixels), the same window of the coarse member
                   (10 x 10 cells of 0.4 deg), and the globe of the 1 km forecast of that day (mean of the eight
                   operational 1 km fields, every 5th pixel).
     The eight operational 1 km members are not stored. The member drawn here was made again with the frozen model
     from the archived inputs (prep_downscaler_fields.py, CPU): same network, prior, sampler and seeds, another
     random stream. OUTPUT_LENS = "mean" draws the stored operational mean in the lens instead.
Time strip: the clock of that issue day (UTC), to scale between 10:15 and 11:03, as in version 1.

    PY=$FOEHN_ENVS/world/bin/python      (world venv; -B: nothing is written beside the originals)
    unset PYTHONPATH; MPLCONFIGDIR=<writable dir> $PY -B fig_gc1_schematic_rt2.py           # .pdf and .png (300 dpi)
    ... fig_gc1_schematic_rt2.py --prepare        # only (re)build data/rt_fields_2026-11-11.npz from the real-time outputs
    ... fig_gc1_schematic_rt2.py --full FILE.png  # whole page at 300 dpi, no tight crop
    ... fig_gc1_schematic_rt2.py --band FILE.png  # the lower band alone (layout work)
    ... check_overlay.py                          # the overlay check, into check/

Reads only; writes only beside this file (data/lic_<day>.npz is a cache of the line-integral convolution of the wind)."""
import argparse
import json
import math
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGOUT = HERE.parents[1] / "rebuild" / f"figs_{os.environ['FOEHN_DESIGN']}" if os.environ.get("FOEHN_DESIGN") else HERE   # A16 rebuild
FIG = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/figures"))
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
from matplotlib.colors import LinearSegmentedColormap, Normalize   # noqa: E402
from matplotlib.font_manager import FontProperties             # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle   # noqa: E402
from matplotlib.path import Path as MPath                      # noqa: E402
from matplotlib.textpath import TextPath                       # noqa: E402
from nature_style import MM, plt, pm_cmap_smooth               # noqa: E402
from scipy.ndimage import map_coordinates                      # noqa: E402

import fig_gc1_schematic as F                                  # noqa: E402  the Nature figure: helpers, colours, main()

# ------------------------------------------------------------------ the issue day and its sources
DAY, LEAD, CITY = "2026-11-11", 1, "Lahore"
CITY_LL = (74.34, 31.55)                                       # lon, lat of Lahore: the pixel that carries the ring in panel e
RT = Path((os.environ["FOEHN_WORK"] + "/eccv/cran-pm/experiments/global1km/realtime"))              # 0.4 deg chain
O1K = Path((os.environ["FOEHN_WORK"] + "/global_model_forecasting_air_pollution/realtime_1km/out"))       # 1 km chain
G = Path((os.environ["FOEHN_WORK"] + "/eccv/data/world/global1km"))
TAG, TILE, RES, LON0 = "pmG_final_P3T3", 1000, 0.01, -179.995
CACHE = HERE / "data" / f"rt_fields_{DAY}.npz"                 # copied from figures_new/fig1/data (version 1)
DS_CACHE = HERE / "data" / f"ds_fields_{DAY}.npz"              # prep_downscaler_fields.py
LIC_CACHE = HERE / "data" / f"lic_{DAY}.npz"
LENS = (33.4, 29.4, 72.2, 76.2)                               # N, S, W, E: exactly 10 x 10 cells of 0.4 deg = 400 x 400 pixels
LENS_C = (0.5 * (LENS[2] + LENS[3]), 0.5 * (LENS[0] + LENS[1]))   # lon, lat of its centre: 74.2 E, 31.4 N (Lahore 74.34 E, 31.55 N)
CAP = 8.0                                                      # angular radius (deg) of the member discs, same centre
MEMBERS = tuple(range(8))                                      # members of the stack, front to back
MEMBER = 0                                                     # the member followed in panel e (member 1, in front of the stack)
STEPS = (0, 4, 7, 10)                                          # sampling steps drawn in the strip of panel e (of 10)
OUTPUT_LENS = "member"                                         # "member": the member made again here; "mean": the stored operational mean
# clock of the issue day, UTC: RT/data/2026-11-11/journal.txt ("fetch done"), RT/logs/<job>_2026-11-11_23638481.out
# (start, "predict rc=0"), O1K/../logs/o1k_run_23638955.out (start) and run.json (written 10:56:40), o1k_web_23638956.out (end)
CLOCK = dict(inputs="10:15:06", ens0="10:16:08", ens1="10:33:20", km0="10:36:26", km1="10:56:40", online="11:03:39")

# height added under a-c: 246 pt = 86.78 mm. A whole (and even) number of points and of pixels (1025 at 300 dpi, 2050 at
# 600 dpi), so that a, b, c fall on the same pixels as in the original. In the PDF there is one more condition:
# matplotlib merges the two small discs of a and c (emissions, forecast weather) into one image whose height in pixels
# is truncated from a floating-point number that depends on the height of the page. With 192 pt (version 1) and 246 pt
# that image has its 1,803 rows, as in the submitted PDF; with 222, 228, 234 and 240 pt it has 1,802 and the two discs
# move by 0.04 mm (tested, check/scan_band_height.txt). The page is 136 + 86.78 = 222.78 mm high (limit 225).
DY = 246 / 72 * 25.4
W = F.W_MM
INK, MUTED, BLUE, GREY, DARK, RED = F.INK, F.MUTED, F.BLUE, F.GREY, F.DARK, F.RED
YB = 12.3                                                      # lower edge of panels d and e; the time strip runs under it
XD = (1.5, 53.5)                                               # panel d, left and right edge (mm)
XE = (55.5, 181.5)                                             # panel e
FEAT_CM = LinearSegmentedColormap.from_list("feat", ["#F4F6F8", "#C3CCD6", "#77838F", "#2F3741"])   # the encoding: greys of panel b
ELEV_CM = LinearSegmentedColormap.from_list("elev", ["#F6F3EA", "#DDD2B8", "#B39F7C", "#7A6448", "#4A3B2A"])


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
    """copy what the band shows of the real-time outputs into one file of this directory (as version 1 did)"""
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


def ds_fields():
    assert DS_CACHE.exists(), f"{DS_CACHE} is missing: run prep_downscaler_fields.py (see its header)"
    return dict(np.load(DS_CACHE))


def lic_cached(D):
    """line-integral convolution of the 850 hPa wind (F.lic, 14 s), kept beside the data: same numbers at every run"""
    if not LIC_CACHE.exists():
        L, sp = F.lic(D["u850"], D["v850"])
        np.savez_compressed(LIC_CACHE, L=L.astype(np.float32), sp=sp.astype(np.float32))
    z = np.load(LIC_CACHE)
    return z["L"], z["sp"]


# ------------------------------------------------------------------ small tools of the band
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
    """a disc image with its rim. One z level per image: in a PDF, images that follow each other in one axes are merged
    and set on whole points; the rim drawn between two images keeps them apart"""
    if halo:
        F.halo(bg, cx, cy, d, halo)
    dg.imshow(img, extent=(cx - d / 2, cx + d / 2, cy - d / 2, cy + d / 2), origin="upper", interpolation="bilinear", zorder=z)
    dg.add_patch(Circle((cx, cy), d / 2, fc="none", ec=rim, lw=lw, zorder=z + 0.1))


def window_disc(dg, rgb, cx, cy, d, z, rim="#8A929C", lw=0.35):
    """a square field shown through a round window, with the soft shadow of the discs of panel c"""
    dg.add_patch(Ellipse((cx + 0.6, cy - 0.5), d, d, fc="#1A2230", alpha=0.13, ec="none", zorder=z - 0.2))
    im = dg.imshow(rgb, extent=(cx - d / 2, cx + d / 2, cy - d / 2, cy + d / 2), origin="upper", interpolation="antialiased", zorder=z)
    im.set_clip_path(Circle((cx, cy), d / 2, transform=dg.transData))
    dg.add_patch(Circle((cx, cy), d / 2, fc="none", ec=rim, lw=lw, zorder=z + 0.1))


def card(sub, ov, img, x, y, w, h, z, cmap=None, norm=None):
    """a card as the transport cards of panel c: own axes, thin frame, soft shadow, one square per value"""
    ov.add_patch(Rectangle((x + 0.45, y - 0.45), w, h, fc="#000000", alpha=0.10, ec="none", zorder=40))
    ca = sub.add_axes(F.R(x, y, w, h), zorder=z)
    ca.imshow(img, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
    ca.set_xticks([]); ca.set_yticks([]); ca.set_facecolor("white")
    for s2 in ca.spines.values():
        s2.set_linewidth(0.45); s2.set_edgecolor("#59626E")
    return ca


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


def bars(ax, cx, ub, s, hs, dips, color="#AAB3BF", z=36):
    """the bars of the network pictogram of panel c (heights hs, lowered by dips), scaled by s"""
    k0 = (len(hs) - 1) / 2
    for k, (hk, dk) in enumerate(zip(hs, dips)):
        ax.add_patch(Rectangle((cx + ((k - k0) * 1.55 - 0.45) * s, ub - (dk + hk / 2) * s), 0.9 * s, hk * s, fc=color, ec="none", zorder=z))


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


def ring(ax, x, y, r=0.85, z=5):
    """the mark of the pixel that is read: a dark ring on a white one, legible on any field"""
    ax.add_patch(Circle((x, y), r, fc="none", ec="white", lw=1.7, zorder=z))
    ax.add_patch(Circle((x, y), r, fc="none", ec=INK, lw=0.65, zorder=z + 0.1))


def secs(hms):
    h, m, s = (int(v) for v in hms.split(":"))
    return 3600 * h + 60 * m + s


# ------------------------------------------------------------------ the fields of panel e, as images
def e_images(D, Z):
    """the pictures of panel e from the two data files (all 400 x 400 over the lens window unless said otherwise)"""
    o = {}
    mu, sg = Z["mu_doy"].astype(np.float32), Z["sigma"].astype(np.float32)
    st = Z["states"].astype(np.float32)                                         # (11, 400, 400): the state after each step
    for s in STEPS:
        x, t = st[s], s / (len(st) - 1)
        grey = np.clip(0.56 + 0.20 * x, 0.08, 0.97)[..., None]                  # the grey of the noise discs of panel c
        pm = np.clip(np.expm1(mu + sg * x), 0.0, 1000.0)                        # what this state is worth in ug m-3
        rgb = (1 - t) * grey + t * F.PCM(F.PNORM(np.nan_to_num(pm)))[..., :3]
        rgb[~np.isfinite(x)] = 0.93                                             # the 5 pixels of the window outside the mask
        o[f"state{s}"] = np.clip(rgb, 0, 1).astype(np.float32)
    o["member"] = Z["members"][MEMBER].astype(np.float32)
    assert np.nanmax(np.abs(np.clip(np.expm1(mu + sg * st[-1]), 0, 1000) - o["member"])) < 0.5, "the last state is not the member"
    o["mean_op"] = D["lens1km"].astype(np.float32)
    i0, j0 = int(round((84.2 - LENS[0]) / 0.4)), int(round((LENS[2] + 180.2) / 0.4))
    o["coarse"] = D["mem04"][MEMBER].astype(np.float32)[i0:i0 + 10, j0:j0 + 10]  # the same window: 10 x 10 cells of member 1
    o["clim"] = np.expm1(mu)                                                    # climatology of the 1 km estimates for this day of the year
    el = Z["static"][5].astype(np.float32)                                      # elevation (m)
    o["elev"] = ELEV_CM(Normalize(np.log10(120.0), np.log10(5000.0))(np.log10(np.clip(np.nan_to_num(el, nan=120.0), 120.0, 5000.0))))[..., :3]
    # the encoding over the window: principal components of the 128 features of the 42 x 44 points stored around it
    f = Z["feat"].astype(np.float32)
    X = f.reshape(f.shape[0], -1).T; X = X - X.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    pcs = (U[:, :3] * S[:3]).T.reshape(3, *f.shape[1:])
    for k in range(3):                                                          # a fixed sign: PC1 rises with the coarse PM2.5
        ref = np.log1p(Z["cams025"]).ravel() if k == 0 else np.sign(pcs[k].ravel()[np.abs(pcs[k]).argmax()]) * pcs[k].ravel()
        if np.corrcoef(pcs[k].ravel(), ref)[0, 1] < 0:
            pcs[k] *= -1
    fr0, fc0 = (int(v) for v in Z["feat_rc0"])
    lat1 = LENS[0] - RES / 2 - RES * np.arange(400); lon1 = LENS[2] + RES / 2 + RES * np.arange(400)
    ri = np.rint((84.0 - lat1) / 0.25).astype(int) - fr0; ci = np.rint((lon1 + 180.0) / 0.25).astype(int) - fc0
    o["feat"] = pcs[:, ri][:, :, ci]                                            # one square per 0.25 deg point
    o["feat_share"] = (S[:3] ** 2 / (S ** 2).sum()).tolist()
    # the three transport maps of member 1 (0.5 deg; cells without particles are left white), the crop of the cards of panel c
    r0, r1, c0, c1 = int((84 - 62) / 0.5), int((84 + 8) / 0.5), int((-22 + 180) / 0.5), int((128 + 180) / 0.5)
    t6 = [Z[k].astype(np.float32) for k in ("t6_pm", "t6_aloft", "t6_surv")]
    void = (t6[0] == 0) & (t6[1] == 0) & (t6[2] == 0)                            # no particle released there
    void |= D["land025"].reshape(288, 2, 720, 2).mean((1, 3)) == 0              # white ocean, as every map of the figure
    o["t6"] = [np.where(void, np.nan, a)[r0:r1, c0:c1] for a in t6]
    return o


# ------------------------------------------------------------------ the lower band: d, e and the clock
def band(sub, D, Z):
    """panels d and e and the time strip, in mm from the lower-left corner of their own sub-figure (183 x DY mm)"""
    bg = F.mm_axes(sub, -10); ov = F.mm_axes(sub, 60); dg = F.mm_axes(sub, 40); tp = F.mm_axes(sub, 70)
    top = DY - 0.5                                                              # 2 mm under the edge of panel c, as between a and c
    for x0_, x1_ in (XD, XE):
        F.panel(bg, x0_, YB, x1_ - x0_, top - YB, "#FCFCFD", "#EFF0F3", "#CFD2D8")
    bg.set_xlim(0, W); bg.set_ylim(0, DY)
    ty = top - 3.0                                                              # title line: letters and titles as in a, b, c
    gap = 20.9 - 8.4 - tw("Generator", 6.6, "bold")                             # the gap of panel c between title and subtitle
    for x0_, let, title, subt in ((XD[0], "d", "Real time", "·  issued 11 November 2026"),
                                  (XE[0], "e", "Downscaling to 1 km", "·  functional flow matching")):
        F.text(ov, x0_ + 3.0, ty, let, 8, INK, "bold", ha="left"); F.text(ov, x0_ + 6.9, ty, title, 6.6, DARK, "bold", ha="left")
        F.text(ov, x0_ + 6.9 + tw(title, 6.6, "bold") + gap, ty, subt, 6.2, DARK, ha="left")
    E = e_images(D, Z)

    # ================================================================== d: inputs of the day -> a-c at 0.4 deg -> eight members
    xd = 0.5 * (XD[0] + XD[1])                                                  # the chain runs down the middle of the panel
    n = 440
    lat_, lon_, rho = F.ortho_ll(n)
    land = F.sample(D["land025"].astype(np.float32), lat_, lon_, 84.0, 0.25, -180.0, 0.25) > 0.5
    edge = (land != np.roll(land, 1, 0)) | (land != np.roll(land, 1, 1))
    an = np.where(D["land04"], D["an04"], np.nan)                               # PM2.5 of the analysis, land only (white ocean)
    v = F.sample(an, lat_, lon_, 84.2, 0.4, -180.2, 0.4)
    a_img = F.finish_disc(np.where(np.isfinite(v)[..., None], F.PCM(F.PNORM(np.nan_to_num(v)))[..., :3], 1.0), rho, n)
    L, sp = lic_cached(D)                                                       # as the wind globe of panel a
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
    idd, iy, pitch = 10.0, top - 12.7, 16.0
    ins = ((xd - pitch, a_img, "#8C939C", "composition\nanalysis, 0.4°"), (xd, w_img, BLUE, "forecast\nwind"),
           (xd + pitch, f_img, "#8C939C", "fires of the\nday before"))
    for k, (cx_, img, hc, lab) in enumerate(ins):
        disc(dg, bg, img, cx_, iy, idd, halo=hc, z=5.0 + 0.3 * k)
        F.text(ov, cx_, iy - idd / 2 - 3.1, lab, 5.2, INK, linespacing=1.1)
    fi, fj = np.where(D["fires"].astype(np.float32) > 3.6)                      # the fire cells, as in panel b
    fx, fy, vis = ortho_xy(-180.0 + 0.25 * fj + 0.125, 84.0 - 0.25 * fi - 0.125)
    vis &= np.hypot(fx, fy) < 0.985                                             # no dot on the rim itself
    dg.scatter(xd + pitch + fx[vis] * idd / 2, iy + fy[vis] * idd / 2, s=0.10, color="#4A0B0B", linewidths=0, alpha=0.75, zorder=5.65,
               rasterized=True)
    yb1 = iy - idd / 2 - 6.7
    brace(ov, (xd - pitch - idd / 2 - 1.2, yb1), (xd + pitch + idd / 2 + 1.2, yb1), 1.5)
    yg = yb1 - 10.0                                                             # the design of a-c at 0.4 deg: the three symbols
    F.arrow(ov, (xd, yb1 - 2.1), (xd, yg + 4.2), INK, lw=0.8, ms=5.5)
    gr, bs, nhw, step = 2.9, 0.44, 3.1, 3.0                                      # gear radius, balance scale, half box, gap
    bhw = 7.45 * bs                                                              # half width of the balance (pans included)
    gx = xd - (2 * gr + 2 * step + 2 * bhw + 2 * nhw) / 2 + gr                   # gear (a), balance (b), network (c)
    mx = gx + gr + step + bhw
    nx0 = mx + bhw + step + nhw
    mini_gear(ov, gx, yg, gr)
    F.balance(ov, mx, yg + 0.3, bs)
    ov.add_patch(FancyBboxPatch((nx0 - nhw, yg - 2.55), 2 * nhw, 5.1, boxstyle="round,pad=0,rounding_size=0.9", fc="#1F2329", ec="black", lw=0.4, zorder=35))
    bars(ov, nx0, yg + 0.25, 0.47, (3.0, 2.3, 1.6, 1.1, 1.6, 2.3, 3.0), (0.0, 0.45, 0.9, 1.25, 0.9, 0.45, 0.0))
    F.arrow(ov, (gx + gr + 0.55, yg), (mx - bhw - 0.55, yg), INK, lw=0.7, ms=4.4)
    F.arrow(ov, (mx + bhw + 0.55, yg), (nx0 - nhw - 0.55, yg), INK, lw=0.7, ms=4.4)
    brace(ov, (gx - gr, yg - 4.3), (nx0 + nhw, yg - 4.3), 1.5)
    F.text(ov, xd, yg - 8.1, "$\\bf{a}$–$\\bf{c}$ at 0.4°", 5.8, INK)

    nD = 720                                                                    # the 0.4 deg ensemble: eight discs, the cells visible
    lat_c, lon_c, rho_c = cap_ll(nD, LENS_C[0], LENS_C[1], CAP)

    def member_img(m):
        a = np.where(D["land04"], D["mem04"][m].astype(np.float32), np.nan)
        v_ = F.sample(a, lat_c, lon_c, 84.2, 0.4, -180.2, 0.4)
        return F.finish_disc(np.where(np.isfinite(v_)[..., None], F.PCM(F.PNORM(np.nan_to_num(v_)))[..., :3], 1.0), rho_c, nD,
                             shade_k=0.10, light=0.04)

    sd, nm = 13.0, len(MEMBERS)
    offs = [((k - (nm - 1) / 2) * 6.2 / (nm - 1), (k - (nm - 1) / 2) * 5.2 / (nm - 1)) for k in range(nm)]
    yc = yg - 14.4 - offs[-1][1] - sd / 2                                        # centre line of the stack, and of the coarse lane of e
    F.arrow(ov, (xd, yg - 10.0), (xd, yc + offs[-1][1] + sd / 2 + 0.3), INK, lw=0.8, ms=5.5)
    for k in reversed(range(nm)):                                                # back discs first
        cx_, cy_ = xd + offs[k][0], yc + offs[k][1]; zz = 6 + 1.0 * (nm - 1 - k)
        dg.add_patch(Ellipse((cx_ + 0.6, cy_ - 0.5), sd, sd, fc="#1A2230", alpha=0.13 if k in (0, nm - 1) else 0.05, ec="none", zorder=zz - 0.2))
        disc(dg, bg, member_img(MEMBERS[k]), cx_, cy_, sd, rim="#8A929C", lw=0.35, z=zz)
    F.text(ov, xd, yc + offs[0][1] - sd / 2 - 2.6, "8 members, 0.4°", 5.8, INK)
    xs1 = xd + offs[-1][0] + math.sqrt((sd / 2) ** 2 - offs[-1][1] ** 2)         # right edge of the back disc on the centre line

    # ================================================================== e: one member -> 1 km
    ws, lgap = 12.5, 7.0                                                         # side of the two windows that are read, gap between them
    yk = yc + ws + lgap                                                          # centre line of the 1 km maps
    ym = 0.5 * (yc + yk)                                                         # centre line of the flow, the strip and the 1 km lens
    # ---- what enters: one member (cells visible) ...
    mcx, md = XE[0] + 9.5, 12.6
    dg.add_patch(Ellipse((mcx + 0.6, yc - 0.5), md, md, fc="#1A2230", alpha=0.13, ec="none", zorder=14.8))
    disc(dg, bg, member_img(MEMBER), mcx, yc, md, rim="#8A929C", lw=0.35, z=15)
    F.text(ov, mcx, yc - md / 2 - 3.2, "one member,\n0.4°", 5.3, INK, linespacing=1.1)
    F.arrow(ov, (xs1 + 0.9, yc), (mcx - md / 2 - 0.8, yc), INK)                  # from the stack of d
    ex0, ex1, eh0, eh1 = mcx + md / 2 + 3.9, mcx + md / 2 + 14.9, 7.0, 3.9       # the encoder: trapezoid of panel b, wide side to the inputs
    F.arrow(ov, (mcx + md / 2 + 0.7, yc), (ex0 - 0.4, yc), INK, lw=0.8, ms=5.0)
    ov.add_patch(Polygon([(ex0, yc + eh0), (ex0, yc - eh0), (ex1, yc - eh1), (ex1, yc + eh1)], closed=True, fc="#3A4049", ec="#2A2F36", lw=0.6,
                         joinstyle="round", zorder=35))
    F.text(ov, 0.5 * (ex0 + ex1) - 0.1, yc, "Encoder", 5.0, "white", "bold")
    etop = lambda x: yc + eh0 - (x - ex0) / (ex1 - ex0) * (eh0 - eh1)            # upper edge of the trapezoid
    # ---- ... the forecast weather (as the disc of panel c) ...
    lw_ = np.clip(0.08 + 1.1 * Lw ** 2.0 * (0.35 + 0.65 * Sw), 0, 1)[..., None]
    t2c = map_coordinates(Z["t2m"].astype(np.float32), [90.0 - lat_, (lon_ + 180.0) % 360.0], order=1, mode="grid-wrap")
    tcm = LinearSegmentedColormap.from_list("t2", ["#24185C", "#2D5AA6", "#2F98B6", "#62C2A2", "#CBE98F", "#FBE07A"])
    x_rgb = tcm(Normalize(-32.0, 30.0)(t2c))[..., :3] * (1 - 0.62 * lw_) + 0.62 * lw_
    x_rgb[edge] = x_rgb[edge] * 0.45
    wcx, wcy, wd = mcx - 0.4, yc + 19.0, 10.4
    disc(dg, bg, F.finish_disc(x_rgb, rho, n), wcx, wcy, wd, halo="#8C939C", z=16)
    F.text(ov, wcx, wcy + wd / 2 + 3.2, "forecast\nweather", 5.3, INK, linespacing=1.1)
    a_ = np.deg2rad(-46.0)
    F.arrow(ov, (wcx + (wd / 2 + 0.6) * np.cos(a_), wcy + (wd / 2 + 0.6) * np.sin(a_)), (ex0 + 1.5, etop(ex0 + 1.5) + 0.4), INK, lw=0.8, ms=5.5)
    # ---- ... and the one-day transport maps (cards of panel c)
    cw, ch, cdx, cdy = 15.0, 7.0, 1.8, 2.4
    xs0, ys0 = ex0 - 1.8, yc + 35.4                                               # lower-left corner of the back card
    cards = ((E["t6"][2], LinearSegmentedColormap.from_list("sv", ["#FFFFFF", "#A9B8C9", "#2E3D52"]), Normalize(0.0, 0.8), "survival"),
             (E["t6"][1], LinearSegmentedColormap.from_list("al", ["#FFFFFF", "#F2BCD3", F.PINK]), Normalize(0.0, 1.0), "air from aloft"),
             (E["t6"][0], F.PCM, F.PNORM, "advected PM$_{2.5}$"))
    for k, (img, cm_, nm_, lab) in enumerate(cards):
        x0_, y0_ = xs0 + cdx * k, ys0 - cdy * k
        card(sub, ov, img, x0_, y0_, cw, ch, 61 + k, cm_, nm_)
        F.text(ov, x0_ + cw + 1.2, y0_ + ch - 1.25, lab, 5.0, MUTED, ha="left", z=70)
    mini_gear(ov, xs0 - 2.5, ys0 + ch + 2.0, 1.55)                               # computed by FLEXGREEN (panel a), one day back
    F.text(ov, xs0, ys0 + ch + 2.0, "Transport maps", 5.6, INK, "bold", ha="left")
    F.text(ov, xs0 + 15.6, ys0 + ch + 2.0, "(3 of 6, one day)", 5.0, MUTED, ha="left")
    xa_ = ex0 + 4.7
    F.arrow(ov, (xs0 + 2 * cdx + cw / 2 - 2.6, ys0 - 2 * cdy - 0.6), (xa_, etop(xa_) + 0.4), INK, lw=0.8, ms=5.5)
    # ---- the encoding (coarse cells) and, above it, the 1 km maps: the same pixel carries a ring in both
    wx0 = ex1 + 1.3
    yf0, yk0 = yc - ws / 2, yk - ws / 2
    fn = Normalize(*np.percentile(E["feat"][0], [2, 98]))
    card(sub, ov, E["feat"][0], wx0, yf0, ws, ws, 63, FEAT_CM, fn)
    card(sub, ov, E["elev"], wx0 + 1.5, yk0 + 1.5, ws, ws, 64)                   # elevation, behind
    card(sub, ov, E["clim"], wx0, yk0, ws, ws, 65, *pm_cmap_smooth())            # climatology of the 1 km estimates; the colours of the
                                                                                 # classes, interpolated, so that the 1 km detail shows
    px = wx0 + (CITY_LL[0] - LENS[2]) / (LENS[3] - LENS[2]) * ws                 # the pixel that is read (Lahore)
    pf, pk = yf0 + (CITY_LL[1] - LENS[1]) / (LENS[0] - LENS[1]) * ws, yk0 + (CITY_LL[1] - LENS[1]) / (LENS[0] - LENS[1]) * ws
    tp.plot([px, px], [pf + 0.85, pk - 0.85], color="white", lw=1.5, solid_capstyle="butt", zorder=4)
    tp.plot([px, px], [pf + 0.85, pk - 0.85], color=INK, lw=0.6, solid_capstyle="butt", zorder=4.5)
    ring(tp, px, pf); ring(tp, px, pk)
    F.text(ov, px - 1.1, 0.5 * (yf0 + ws + yk0), "read at\neach pixel", 5.0, MUTED, ha="right", style="italic", linespacing=1.1)
    F.text(ov, 0.5 * (ex0 + ex1), yc - eh0 - 3.3, "whole globe", 5.0, MUTED, style="italic")
    F.text(ov, wx0 + ws / 2 + 1.0, yc - eh0 - 3.3, "encoding, 0.25°", 5.0, MUTED, style="italic")
    bx = wx0 + ws + 2.9
    brace(ov, (bx, yf0 + 1.0), (bx, yk0 + ws - 1.0), 1.5)
    F.text(ov, bx + 2.7, yk0 + ws + 0.7, "1 km maps", 5.4, INK, "bold", ha="left")
    F.text(ov, bx + 2.7, yk0 + ws - 1.8, "climatology,", 5.0, MUTED, ha="left")
    F.text(ov, bx + 2.7, yk0 + ws - 4.2, "elevation", 5.0, MUTED, ha="left")
    # ---- the pointwise flow: the dark box of panel c, with its red sampling arrow
    fw, fh = 13.0, 14.0
    fcx = bx + 1.5 + 3.5 + fw / 2
    F.arrow(ov, (bx + 2.0, ym), (fcx - fw / 2 - 0.4, ym), INK, lw=0.8, ms=5.0)
    dark_box(ov, fcx, ym, fw, fh, r=1.8)
    bars(ov, fcx, ym + 3.5, 0.9, (2.6, 2.6, 2.6, 2.6), (0.0, 0.0, 0.0, 0.0))     # four equal layers, one value in, one value out
    for sgn in (-1, 1):
        ov.plot([fcx + sgn * 3.0, fcx + sgn * 3.95], [ym + 3.5, ym + 3.5], color="#6F7884", lw=0.35, zorder=36)
        ov.add_patch(Circle((fcx + sgn * 4.4, ym + 3.5), 0.42, fc="white", ec="#1B1E23", lw=0.25, zorder=37))
    F.text(ov, fcx, ym - 2.5, "Pointwise\nflow", 5.6, "white", "bold", linespacing=1.1)
    ov.add_patch(FancyArrowPatch((fcx + 4.4, ym - fh / 2 - 0.4), (fcx - 4.4, ym - fh / 2 - 0.4), arrowstyle="-|>", mutation_scale=5.5,
                                 lw=0.9, color=RED, connectionstyle="arc3,rad=-1.2", zorder=36))
    # ---- the strip: the state of the member after 0, 3, 6 and 10 steps; the last one is the 1 km lens
    lr = 6.6
    lcx = XE[1] - 3.0 - lr - 0.5
    sdd, sgap = 11.5, 1.6
    sx0 = fcx + fw / 2 + 4.6 + sdd / 2
    sxs = [sx0 + k * (sdd + sgap) for k in range(len(STEPS) - 1)]
    assert sxs[-1] + sdd / 2 + 0.9 < lcx - lr - 0.5, "the strip runs into the lens"
    F.arrow(ov, (fcx + fw / 2 + 0.6, ym), (sx0 - sdd / 2 - 0.7, ym), INK, lw=0.8, ms=5.0)
    for k, (s, cx_) in enumerate(zip(STEPS[:-1], sxs)):
        window_disc(dg, E[f"state{s}"], cx_, ym, sdd, 17 + k)
    ya = ym - lr - 0.5 - 2.0
    F.arrow(ov, (sxs[0] - sdd / 2, ya), (lcx - lr - 0.9, ya), INK, lw=0.8, ms=5.5)
    for cx_ in sxs:
        ov.plot([cx_, cx_], [ya - 0.55, ya + 0.55], color=INK, lw=0.6, zorder=31)
    F.text(ov, sxs[0] + 0.9, ya - 2.8, "correlated noise", 5.2, INK)
    F.text(ov, 0.5 * (sxs[1] + sxs[2]) + 2.2, ya - 2.8, "10 sampling steps", 5.6, RED)
    # ---- what comes out: the member at 1 km (lens), the same window of the coarse member, the globe of the 1 km forecast
    gd, gcx = 23.0, XE[1] - 16.5
    gcy = top - 9.9 - 0.538 * gd
    F.halo(bg, gcx, gcy, gd, "#8C939C")
    ax = F.geo_globe(sub, gcx, gcy, gd, face="white")
    ax.add_feature(cfeature.LAND.with_scale("110m"), fc=F.LAND_NODATA, ec="none", zorder=1)
    F.field_on(ax, D["w005"].astype(np.float32), 90.0, 0.05, -180.0, 0.05, F.PCM, F.PNORM, regrid=1500, z=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.2, edgecolor="#7D8792", zorder=5)
    F.shade(ax, strength=0.16, light=0.10)
    F.text(ov, gcx, gcy + 0.538 * gd + 4.1, "PM$_{2.5}$ at 1 km", 5.8, INK)
    F.text(ov, gcx, gcy + 0.538 * gd + 1.5, "lead days 1–5, 8 members", 5.4, BLUE, "bold")
    dl = ax.projection.transform_point(LENS_C[0], LENS_C[1], ccrs.PlateCarree())
    xl_ = ax.projection.x_limits
    lx0, ly0 = gcx + dl[0] / xl_[1] * gd / 2, gcy + dl[1] / xl_[1] * gd / 2      # the city on the globe
    for sgn in (1, -1):
        ang = np.arctan2(ym - ly0, lcx - lx0) + sgn * np.pi / 2
        ov.plot([lx0 + 0.7 * np.cos(ang), lcx + (lr + 0.5) * np.cos(ang)], [ly0 + 0.7 * np.sin(ang), ym + (lr + 0.5) * np.sin(ang)],
                color="#3D4450", lw=0.45, zorder=36)
    ov.add_patch(Circle((lx0, ly0), 0.7, fc="none", ec=INK, lw=0.6, zorder=37))
    lens(sub, ov, E["member"] if OUTPUT_LENS == "member" else E["mean_op"], lcx, ym, lr)
    ring(tp, lcx - lr + (CITY_LL[0] - LENS[2]) / (LENS[3] - LENS[2]) * 2 * lr, ym - lr + (CITY_LL[1] - LENS[1]) / (LENS[0] - LENS[1]) * 2 * lr, r=0.8)
    F.text(ov, lcx, ya - 0.4, "1 km", 5.2, INK, "bold")
    F.text(ov, lcx, ya - 2.9, CITY, 5.0, MUTED, style="italic")
    cr = 6.2                                                                     # the same window of the coarse member, 10 x 10 cells
    ccx, ccy = sxs[2] - 11.0, ym + sdd / 2 + 2.7 + cr + 0.5
    lens(sub, ov, E["coarse"], ccx, ccy, cr)
    F.text(ov, ccx - cr - 1.5, ccy + 1.3, "0.4°", 5.2, INK, "bold", ha="right")
    F.text(ov, ccx - cr - 1.5, ccy - 1.3, "same window", 5.0, MUTED, ha="right", style="italic")

    # ================================================================== the clock of the issue day (UTC), under both panels
    ys, x0, x1 = 5.7, 54.0, 172.0
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
    for a_x in (bg, ov, dg, tp):
        a_x.set_xlim(0, W); a_x.set_ylim(0, DY)
    return dict(lanes=dict(coarse=yc, km=yk, flow=ym), pixel=(px, pf, pk), lens_on_globe=(lx0, ly0), feat_share=E["feat_share"])


# ------------------------------------------------------------------ the figure
def build():
    D, Z = rt_fields(), ds_fields()
    fig = plt.figure(figsize=(W * MM, (F.H_MM + DY) * MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[F.H_MM, DY], left=0, right=1, bottom=0, top=1, hspace=0)
    abc, de = fig.add_subfigure(gs[0]), fig.add_subfigure(gs[1])
    keep = F.plt, F.save, F.H_MM
    F.plt = types.SimpleNamespace(figure=lambda *a, **k: abc)                    # the page of main() is the upper sub-figure
    F.save = lambda *a, **k: None                                                # and main() writes nothing
    try:
        F.main()                                                                 # panels a, b, c: the original code, untouched
        F.plt = keep[0]
        F.H_MM = DY                                                              # R() and mm_axes() now work in the 183 x DY mm band
        info = band(de, D, Z)
    finally:
        F.plt, F.save, F.H_MM = keep
    fig.patch.set_alpha(0.0)                                                     # transparent page, as the original
    fig.band_info = info
    return fig


def build_band():
    """the lower band alone on a 183 x DY mm page (layout work only)"""
    fig = plt.figure(figsize=(W * MM, DY * MM))
    keep = F.H_MM
    F.H_MM = DY
    try:
        fig.band_info = band(fig, rt_fields(), ds_fields())
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
        fig = build_band()
        with plt.rc_context({"savefig.bbox": "standard"}):
            fig.savefig(a.band, dpi=300, facecolor="white")
        return print("saved", a.band, json.dumps(fig.band_info))
    fig = build()
    if a.full:
        with plt.rc_context({"savefig.bbox": "standard"}):
            fig.savefig(a.full, dpi=300)
        return print("saved", a.full)
    fig.savefig(FIGOUT / "fig_gc1_schematic_rt2.pdf")                            # rcParams of nature_style: tight, 600 dpi rasters, TrueType
    fig.savefig(FIGOUT / "fig_gc1_schematic_rt2.png", dpi=300)
    print("saved", FIGOUT / "fig_gc1_schematic_rt2", json.dumps(fig.band_info))


if __name__ == "__main__":
    main()
