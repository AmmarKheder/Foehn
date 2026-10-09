"""Nature figure style of the global 1 km paper: Arial-metric sans-serif (Liberation Sans, SIL OFL), 6-7 pt text,
89 / 183 mm widths, open axes, Okabe-Ito colours (colour-blind safe; ours = vermillion as the hero colour), PM2.5 bins
aligned on the WHO 2021 24-h guideline (15) and interim targets (25, 37.5, 50, 75), Robinson maps on grey shaded relief."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap

HERE = Path(__file__).resolve().parent
try:                                                           # Natural Earth cache for every figure (~/.local is dead)
    import cartopy
    cartopy.config["data_dir"] = str(HERE / "cartopy_data")
except ImportError:
    pass
for f in sorted((HERE / "fonts").glob("LiberationSans-*.ttf")):
    font_manager.fontManager.addfont(str(f))
MM = 1 / 25.4
SINGLE, DOUBLE = 89 * MM, 183 * MM
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold", "mathtext.default": "regular",
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "axes.titlepad": 3, "axes.labelpad": 2,
    "axes.edgecolor": "#222222", "axes.labelcolor": "#111111", "xtick.color": "#222222", "ytick.color": "#222222",
    "text.color": "#111111", "lines.linewidth": 1.2, "lines.markersize": 3, "legend.frameon": False,
    "legend.handlelength": 1.6, "legend.handletextpad": 0.5, "legend.columnspacing": 1.2, "legend.borderaxespad": 0.2,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.01, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "image.interpolation": "nearest",
})
OUT = HERE.parents[1] / "paper" / "nature" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# curves (MOSAIC-like palette): this work brick red (ensemble mean) / salmon (one member), Aurora copper, CAMS dark grey
# with squares (IFS-style), climatology black dashes; other AI baselines olive, mustard, purple, steel blue.
COLOR = {"ours": "#1F5AA6", "member": "#9DB9E0", "cams": "#111111", "aurora": "#D55E00", "climatology": "#111111",
         "ghap": "#8c8c8c", "persistence": "#111111",
         "olive": "#8C9A4B", "mustard": "#DDB14F", "purple": "#7A66B5", "steel": "#4C7DB8", "teal": "#3E8E86"}
STYLE = {
    "ours": dict(label="Foehn (1 km, probabilistic)", color=COLOR["ours"], ls="-", marker="o", lw=1.0, ms=3.0, z=10),
    "member": dict(label="Foehn (one member)", color=COLOR["member"], ls="-", marker="o", lw=0.9, ms=2.6, z=9),
    "cams": dict(label="CAMS (operational, 40 km)", color=COLOR["cams"], ls="-", marker="s", lw=0.8, ms=2.4, z=6),
    "aurora": dict(label="Aurora (AI, 40 km)", color=COLOR["aurora"], ls="-", marker="o", lw=0.8, ms=2.4, z=5),
    "persistence": dict(label="Persistence (station, day 0)", color=COLOR["persistence"], ls=(0, (3, 2)), marker="", lw=0.9, ms=0, z=3),
    "climatology": dict(label="Climatology (2017–2020, same calendar days)", color=COLOR["climatology"], ls=(0, (2.2, 1.6)), marker="", lw=0.9, ms=0, z=3),
    "ghap": dict(label="GHAP, target day (not a forecast)", color=COLOR["ghap"], ls=(0, (1, 1.5)), marker="", lw=1.0, ms=0, z=2),
    "earthformer": dict(label="Earthformer", color=COLOR["steel"], ls="-", marker="o", lw=0.75, ms=2.2, z=4),
    "convlstm": dict(label="ConvLSTM", color=COLOR["olive"], ls="-", marker="o", lw=0.75, ms=2.2, z=4),
    "simvp": dict(label="SimVP", color=COLOR["mustard"], ls="-", marker="o", lw=0.75, ms=2.2, z=4),
    "climax": dict(label="ClimaX", color=COLOR["purple"], ls="-", marker="o", lw=0.75, ms=2.2, z=4),
    "topoflow": dict(label="TopoFlow", color=COLOR["teal"], ls="-", marker="^", lw=0.75, ms=2.2, z=4),
}
NET_NAMES = {0: "Europe (EEA)", 1: "China (CNEMC)", 3: "USA (EPA)", 4: "Korea (AirKorea)", 5: "Japan (NIES)",
             6: "India (CPCB)", 2: "Other (OpenAQ)"}

PM_BOUNDS = [0, 5, 10, 15, 25, 37.5, 50, 75, 100, 150, 250]


def ryb(n, lo=0.02, hi=0.98):
    """n discrete steps of RdYlBu_r (blue = low / better, pale yellow = middle, red = high / worse): skill and
    difference maps."""
    return ListedColormap(plt.get_cmap("RdYlBu_r")(np.linspace(lo, hi, n)))


PM_COLORS_CREAM = ["#fffbe9", "#fde8a7", "#fcc566", "#f89d3c", "#ec6a2a", "#d33a24", "#ab1a2e", "#7e1239", "#55103f", "#2e0a3b"]
# the paper's PM2.5 palette (user, 2026-10-08): RdYlBu_r, blue = clean, yellow = middle, red = polluted, 10 classes
PM_COLORS = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"]
PM_OVER = "#5c0019"


def pm_cmap():
    """concentration maps: RdYlBu_r classes on PM_BOUNDS (the same colours in every map of the paper)."""
    cmap = ListedColormap(PM_COLORS)
    cmap.set_over(PM_OVER); cmap.set_bad((0, 0, 0, 0))
    return cmap, BoundaryNorm(PM_BOUNDS, cmap.N)


def pm_cmap_smooth():
    """the same PM2.5 colours as pm_cmap, interpolated: a value gets exactly the colour of its class at the class centre
    (on the class scale of PM_BOUNDS), so smooth maps and class maps of the paper read the same."""
    from matplotlib.colors import FuncNorm, LinearSegmentedColormap
    idx = np.arange(len(PM_BOUNDS), dtype=float)
    fwd = lambda v: np.interp(v, PM_BOUNDS, idx)
    inv = lambda t: np.interp(t, idx, PM_BOUNDS)
    cmap = LinearSegmentedColormap.from_list("pm_smooth", PM_COLORS, N=256)
    cmap.set_over(PM_OVER); cmap.set_under(PM_COLORS[0]); cmap.set_bad((0, 0, 0, 0))
    return cmap, FuncNorm((fwd, inv), vmin=inv(0.5), vmax=inv(len(PM_COLORS) - 0.5))


def diff_cmap(bounds):
    """diverging, blue = this work better (negative difference), pale yellow centre bin."""
    n = len(bounds) - 1
    cmap = ryb(n, 0.04, 0.96)
    return cmap, BoundaryNorm(bounds, n)


def letter(ax, s, x=-0.02, y=1.0, **kw):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right", **kw)


def relief():
    return np.load(HERE / "basemap" / "gray_relief_5400x2700.npy")


def world_axes(fig, rect, bg=True, extent=None):
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    cartopy.config["data_dir"] = str(HERE / "cartopy_data")    # Natural Earth cache (~/.local/share/cartopy is a dead link)
    ax = fig.add_axes(rect, projection=ccrs.Robinson())
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    if bg:
        r = relief().astype(np.float32) / 255.0
        r = 0.80 + 0.20 * r                                     # light relief: land detail without competing with data
        ax.imshow(r, origin="upper", extent=(-180, 180, -90, 90), transform=ccrs.PlateCarree(), cmap="gray", vmin=0, vmax=1,
                  zorder=0, regrid_shape=1200)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.25, edgecolor="#555555", zorder=4)
    ax.spines["geo"].set_linewidth(0.5); ax.spines["geo"].set_edgecolor("#333333")
    return ax


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    print("saved", OUT / name)
