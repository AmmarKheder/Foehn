"""Common style of the GenCast-like figures of the Foehn paper: one colour per system everywhere, scorecard colour map
(blue: Foehn better), helpers for two-tier curves (score above, relative difference below) and clean axes."""
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

from nature_style import plt  # noqa: F401  (rcParams: Arial-like, Nature sizes)

FOEHN, CAMS, AURORA, PERS, GHAP = "#1F5AA6", "#111111", "#D55E00", "#9A9A9A", "#5B5B5B"
CLIM = PERS                                                                     # climatology: grey dashed, every figure
SYS = {"Foehn": (FOEHN, "-", 1.6), "CAMS": (CAMS, "-", 1.0), "Aurora": (AURORA, "-", 1.0), "Persistence": (PERS, (0, (3, 2)), 0.9)}
# scorecard: dark blue = Foehn 30% better ... white ... dark red = Foehn 30% worse (GenCast: +-20%)
SC_BOUNDS = [-30, -20, -15, -10, -5, -2, 2, 5, 10, 15, 20, 30]
SC_COLORS = ["#08306B", "#08519C", "#2171B5", "#4292C6", "#9ECAE1", "#F7F7F7", "#FCBBA1", "#FB6A4A", "#EF3B2C", "#CB181D", "#67000D"]


def sc_cmap():
    cm = LinearSegmentedColormap.from_list("sc", SC_COLORS, N=len(SC_COLORS))
    cm.set_under(SC_COLORS[0]); cm.set_over(SC_COLORS[-1])
    return cm, BoundaryNorm(SC_BOUNDS, len(SC_COLORS))


def clean(ax, grid=True):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(labelsize=5.5, length=2, width=0.5)
    if grid:
        ax.grid(lw=0.3, color="#E3E3E3", zorder=0)
        ax.set_axisbelow(True)


def label(fig, x, y, s):
    fig.text(x, y, s, fontsize=8, fontweight="bold", va="bottom", ha="left")


def block_bootstrap(a, b, B=7, n=10000, seed=0):
    """paired moving-block bootstrap of the RMSE ratio - 1 from per-day (sse, count) arrays a, b (days, 2), NaN days dropped"""
    ok = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0]) & (a[:, 1] > 0)
    a, b = a[ok], b[ok]; nd = len(a)
    rng = np.random.default_rng(seed)
    idx = (rng.integers(0, nd - B + 1, size=(n, -(-nd // B)))[:, :, None] + np.arange(B)).reshape(n, -1)[:, :nd]
    r = np.sqrt(a[idx, 0].sum(1) / a[idx, 1].sum(1)) / np.sqrt(b[idx, 0].sum(1) / b[idx, 1].sum(1)) - 1
    pt = np.sqrt(a[:, 0].sum() / a[:, 1].sum()) / np.sqrt(b[:, 0].sum() / b[:, 1].sum()) - 1
    return 100 * pt, 100 * np.percentile(r, 2.5), 100 * np.percentile(r, 97.5)


# ---- one map style for every figure (Aurora-like): white ocean, light-grey land where there are no data, thin coastlines,
# the same PM2.5 palette everywhere (cream -> red -> dark purple, nature_style.pm_cmap), pale diverging scorecards.
LAND_BG = "#F0F0F0"
DIV = ["#2B3A9B", "#5D6FC4", "#9AA6DD", "#D4D9F1", "#FFFFFF", "#F6D2CD", "#EC9C92", "#D9584F", "#B0171F"]


def div_cmap(vmax=40):
    from matplotlib.colors import TwoSlopeNorm
    cm = LinearSegmentedColormap.from_list("div", DIV, N=256)
    return cm, TwoSlopeNorm(0, -vmax, vmax)


def map_axes(fig, rect, proj="robinson", extent=(-180, 180, -60, 84), land=True):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    pr = ccrs.Robinson() if proj == "robinson" else ccrs.PlateCarree()
    ax = fig.add_axes(rect, projection=pr)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    if land:
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=LAND_BG, edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.25, edgecolor="#6E6E6E", zorder=4)
    ax.spines["geo"].set_linewidth(0.4); ax.spines["geo"].set_edgecolor("#9A9A9A")
    ax._geo_extent = tuple(float(v) for v in extent)                             # read by world_coords / geo_ticks
    return ax


def land_mask_04():
    """0.4 deg land mask on the 84N..60S x 180W.. grid (361, 900), from the 0.25 deg land mask of the 1 km data"""
    m = np.load("/scratch/project_462001140/ammar/eccv/data/world/global1km/meta.npz")["land"].astype(bool)   # (576, 1440) 0.25 deg
    li = np.clip(np.round((0.4 * np.arange(361)) / 0.25).astype(int), 0, m.shape[0] - 1)
    lj = np.clip(np.round((0.4 * np.arange(900)) / 0.25).astype(int), 0, m.shape[1] - 1)
    return m[np.ix_(li, lj)]


# ---- geographic coordinates on every map, as in Nature's map figures: 60°N ... 0° ... 60°S beside the world maps,
# 180° 120°W 60°W 0° 60°E 120°E 180° under them, a light grey graticule under the data; degree ticks on the regional maps
GRAT = "#CFCFCF"                                                                # graticule
COORD = "#454545"                                                               # coordinate labels


def fmt_lon(v):
    v = (float(v) + 180.0) % 360.0 - 180.0
    if abs(v) < 1e-6:
        return "0°"
    if abs(abs(v) - 180.0) < 1e-6:
        return "180°"
    return f"{abs(v):g}°{'E' if v > 0 else 'W'}"


def fmt_lat(v):
    v = float(v)
    return "0°" if abs(v) < 1e-6 else f"{abs(v):g}°{'N' if v > 0 else 'S'}"


def nice_ticks(lo, hi, nmax=3, margin=0.07):
    """round-degree ticks inside [lo, hi], kept away from the edges, at most nmax (at least two when the range allows)"""
    lo, hi = sorted((float(lo), float(hi))); m = margin * (hi - lo); one = None
    for s in (0.5, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90):
        t = np.arange(np.ceil((lo + m) / s) * s, hi - m + 1e-9, s)
        if len(t) <= nmax:
            if len(t) >= 2:
                return t
            if one is None and len(t) == 1:
                one = t
    return one if one is not None else np.array([])


def world_coords(ax, left=True, right=True, bottom=True, size=5.0, grid=True, lats=(60, 30, 0, -30, -60),
                 lons=(-180, -120, -60, 0, 60, 120, 180), zorder=0.6):
    """coordinates of a world map made by map_axes (Robinson): graticule every 30° of latitude and 60° of longitude in
    light grey under the data (zorder 0.6: above the land shading, below the fields), latitudes beside the left and right
    edges of the map, longitudes under its bottom edge. Small multiples: labels on the outer panels only."""
    import cartopy.crs as ccrs
    from matplotlib.transforms import offset_copy
    pc = ccrs.PlateCarree(); pr = ax.projection
    W, E, S, N = getattr(ax, "_geo_extent", (-180.0, 180.0, -60.0, 84.0))
    if grid:
        yy = np.linspace(S, N, 300); xx = np.linspace(W, E, 600)
        for lo in lons:
            if W < lo < E:
                ax.plot(np.full_like(yy, lo), yy, transform=pc, color=GRAT, lw=0.3, zorder=zorder, solid_capstyle="butt")
        for la in lats:
            if S < la < N:
                ax.plot(xx, np.full_like(xx, la), transform=pc, color=GRAT, lw=0.3, zorder=zorder, solid_capstyle="butt")
    fig = ax.figure; kw = dict(fontsize=size, color=COORD, clip_on=False, zorder=9)
    tl = offset_copy(ax.transData, fig=fig, x=-1.6, y=0.0, units="points")
    tr = offset_copy(ax.transData, fig=fig, x=1.6, y=0.0, units="points")
    tb = offset_copy(ax.transData, fig=fig, x=0.0, y=-3.2, units="points")      # clear of a latitude label on the bottom edge
    for la in lats:
        if S - 1e-6 <= la <= N + 1e-6:
            xl, yl = pr.transform_point(W, la, pc); xr, _ = pr.transform_point(E, la, pc)
            if left:
                ax.text(xl, yl, fmt_lat(la), transform=tl, ha="right", va="center", **kw)
            if right:
                ax.text(xr, yl, fmt_lat(la), transform=tr, ha="left", va="center", **kw)
    if bottom:
        for lo in lons:
            if W - 1e-6 <= lo <= E + 1e-6:
                xb, yb = pr.transform_point(lo, S, pc)
                ax.text(xb, yb, fmt_lon(lo), transform=tb, ha="center", va="top", **kw)
    return ax


def geo_ticks(ax, xt=None, yt=None, left=True, bottom=True, right=False, top=False, size=5.0, nmax=None):
    """degree coordinates of a regional map (a cartopy PlateCarree map or plain axes in longitude / latitude): ticks
    outside on every panel, labels only where asked (the outer panels of panels sharing one extent). Without xt / yt the
    ticks are round degrees, about one per 9 mm across and 8 mm up the drawn panel (nmax: an int or (nx, ny) instead)"""
    from matplotlib.ticker import FuncFormatter
    if hasattr(ax, "projection"):
        import cartopy.crs as ccrs
        W, E, S, N = getattr(ax, "_geo_extent", None) or ax.get_extent(crs=ccrs.PlateCarree())
    else:
        (W, E), (S, N) = ax.get_xlim(), ax.get_ylim()
    if nmax is None:
        ax.apply_aspect(); bb = ax.get_position(); fw, fh = ax.figure.get_size_inches() * 25.4
        nx, ny = int(np.clip(bb.width * fw // 9, 2, 5)), int(np.clip(bb.height * fh // 8, 2, 5))
    else:
        nx, ny = (nmax, nmax) if np.isscalar(nmax) else nmax
    xt = nice_ticks(W, E, nx) if xt is None else np.asarray(xt, float)
    yt = nice_ticks(S, N, ny) if yt is None else np.asarray(yt, float)
    if hasattr(ax, "projection"):
        ax.set_xticks(xt, crs=ccrs.PlateCarree()); ax.set_yticks(yt, crs=ccrs.PlateCarree())
    else:
        ax.set_xticks(xt); ax.set_yticks(yt)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: fmt_lon(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: fmt_lat(v)))
    ax.tick_params(which="major", direction="out", length=1.8, width=0.4, pad=1.0, labelsize=size, color="#555555",
                   labelcolor=COORD)
    ax.tick_params(axis="x", bottom=True, top=top, labelbottom=bottom, labeltop=top)
    ax.tick_params(axis="y", left=True, right=right, labelleft=left, labelright=right)
    return xt, yt
