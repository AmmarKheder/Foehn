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
    return ax


def land_mask_04():
    """0.4 deg land mask on the 84N..60S x 180W.. grid (361, 900), from the 0.25 deg land mask of the 1 km data"""
    m = np.load("/scratch/project_462001140/ammar/eccv/data/world/global1km/meta.npz")["land"].astype(bool)   # (576, 1440) 0.25 deg
    li = np.clip(np.round((0.4 * np.arange(361)) / 0.25).astype(int), 0, m.shape[0] - 1)
    lj = np.clip(np.round((0.4 * np.arange(900)) / 0.25).astype(int), 0, m.shape[1] - 1)
    return m[np.ix_(li, lj)]
