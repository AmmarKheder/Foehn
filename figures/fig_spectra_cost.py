#!/usr/bin/env python3
"""The MOSAIC 'spectra + cost' figure.
(a) error power spectrum of the forecasts relative to the GHAP anomaly power at 0.25 deg (central Europe, 61 start days
    of 2022, lead 3 days): below 1 = skill at that wavelength, above 1 = structure in the wrong place;
(b) the same at 1 km on the regional window (Paris - Florence, the 4 saved days x 4 leads): models trained natively at
    1 km on the window vs continental 0.25 deg models interpolated to 1 km;
(c) skill (1 km RMSE, mean over leads, Europe 2022) versus inference time (one member, one lead, one MI250X GCD);
    marker area = peak GPU memory; filled = probabilistic, open = deterministic."""
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from fig_style import COL2, EXP, STYLE, load, save

DATA = Path(__file__).resolve().parent / "data"


# ------------------------------------------------------------------------------------------------ error spectra
# E(k) = power of (forecast - GHAP) / power of the GHAP anomaly, radially averaged. E < 1: skill at that wavelength;
# E = 1: no skill (a forecast with no structure at that scale); E > 1: structure in the wrong place (worse than none).
# Point forecasts: the ensemble mean for FOEHN, the deterministic output otherwise.
def radial(a, win, kr, bins):
    p = np.abs(np.fft.fft2(a * win)) ** 2
    idx = np.digitize(kr.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < len(bins) - 1)
    return np.bincount(idx[ok], p.ravel()[ok], len(bins) - 1) / np.maximum(np.bincount(idx[ok], minlength=len(bins) - 1), 1)


def setup(nr, nc, nbins):
    win = np.outer(np.hanning(nr), np.hanning(nc))
    ky, kx = np.meshgrid(np.fft.fftfreq(nr), np.fft.fftfreq(nc), indexing="ij")
    kr = np.sqrt(kx ** 2 + ky ** 2)
    bins = np.logspace(np.log10(1.0 / max(nr, nc)), np.log10(0.5), nbins)
    return win, kr, bins, np.sqrt(bins[1:] * bins[:-1])


def err_ratio(pairs, ok, win, kr, bins):
    ae = at = 0
    for f, t in pairs:
        m = ok & np.isfinite(t) & (t > 0)
        ae = ae + radial(np.where(m, f - t, 0.0), win, kr, bins)
        at = at + radial(np.where(m, t - t[m].mean(), 0.0), win, kr, bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ae / at


# (a) 0.25 deg, central Europe, 61 start days of 2022, lead 3 days (index 1 of the saved leads 1 and 3)
R0, C0, NR, NC, LEAD_I = 48, 104, 64, 128, 1
def npz(name):
    z = np.load(DATA / f"{name}.npz"); return {int(d): i for i, d in enumerate(z["days"])}, z["pred"].astype(np.float32)
tdays, truth = npz("truth")
land25 = np.isfinite(truth[0, 0, 0, R0:R0 + NR, C0:C0 + NC])
win25, kr25, bins25, kc25 = setup(NR, NC, 22)


def spec25(name):
    days, pred = npz(name)
    pairs = [(pred[i, LEAD_I, :, R0:R0 + NR, C0:C0 + NC].mean(0), truth[tdays[d], LEAD_I, 0, R0:R0 + NR, C0:C0 + NC])
             for d, i in days.items() if d in tdays]
    return err_ratio(pairs, land25, win25, kr25, bins25)


# (b) 1 km regional window, saved maps of the regional evaluations (4 days x 4 leads)
WR0, WC0, WNR, WNC = 2250, 2500, 768, 1536
st = np.load(EXP / "fullimage/data/static1km.npz")
land1 = st["land"][WR0:WR0 + WNR, WC0:WC0 + WNC] > 0
win1, kr1, bins1, kc1 = setup(WNR, WNC, 24)
FINE = EXP / "fullimage/data/ghap1km.npy"
with open(FINE, "rb") as f:
    np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); HDR = f.tell()
OFF22 = 365 * 4 + 1 + 365


def truth1(d):
    a = np.memmap(FINE, dtype=np.float16, mode="r", offset=HDR + (OFF22 + d) * 4192 * 6992 * 2, shape=(4192, 6992))
    return np.asarray(a[WR0:WR0 + WNR, WC0:WC0 + WNC], np.float32)


def spec1(evaldir):
    pairs = []
    for p in sorted(glob.glob(str(evaldir / "part_*.npz"))):
        z = np.load(p)
        for k in z.files:
            if k.startswith("pred_"):
                d, h = (int(x[1:]) for x in k[5:].split("_"))
                pairs.append((z[k][WR0:WR0 + WNR, WC0:WC0 + WNC].astype(np.float32), truth1(d + h)))
    return err_ratio(pairs, land1, win1, kr1, bins1) if pairs else None


def plot(ax, wl, r, **kw):
    ok = np.isfinite(r)
    ax.plot(wl[ok], r[ok], **kw)


def noskill(ax):
    ax.axhline(1, color="#222222", ls="--", lw=0.7)
    ax.text(0.02, 1.03, "no skill", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=5.8, color="#444444")


import os
SC = (lambda t: t) if os.environ.get("NATURE") else (lambda t: t.upper())   # plain labels in the paper, small caps for MOSAIC
fig = plt.figure(figsize=(COL2, 3.05))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.08], hspace=0.55, wspace=0.28, bottom=0.3, top=0.94)
axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
LEG = dict(loc="upper left", ncol=2, fontsize=4.9, handlelength=1.8, labelspacing=0.15, columnspacing=1.0, borderpad=0.3,
           frameon=False, handletextpad=0.4)

wl25 = 27.8 / kc25
noskill(axa)
for name in ("topoflow", "earthformer", "simvp", "convlstm", "nofp", "foehn"):
    st_ = STYLE[name]
    plot(axa, wl25, spec25(name), color=st_["color"], ls=st_["ls"], lw=2.0 if name == "foehn" else 1.2, label=SC(st_["label"]),
         zorder=st_["z"])
axa.set_xscale("log"); axa.invert_xaxis(); axa.set_ylim(0, 2.45); axa.set_yticks([0, 0.5, 1.0, 1.5])
axa.set_xticks([1000, 300, 100]); axa.set_xticklabels(["1000", "300", "100"])
axa.set_ylabel("Error / signal power"); axa.set_title("(a) Error by scale, 0.25$^\\circ$, 3 days ahead")
axa.legend(**LEG)

noskill(axb)
REG = EXP / "regional/runs"
for evaldir, lab, col, ls, lw in [(EXP / "retrain/runs/B_convlstm_mettarget31/eval_region", "ConvLSTM (interp.)", STYLE["convlstm"]["color"], "--", 1.2),
                                  (REG / "simvp/eval_region", "SimVP (1 km)", STYLE["simvp"]["color"], "-", 1.2),
                                  (REG / "earthformer/eval_region", "Earthformer (1 km)", STYLE["earthformer"]["color"], "-", 1.2),
                                  (EXP / "dejavu/runs/noise_t31_fpstab/eval_region", "Foehn (interp.)", STYLE["foehn"]["color"], "--", 1.4),
                                  (REG / "flow/eval_region", "Foehn (1 km window)", STYLE["foehn"]["color"], "-", 2.0)]:
    r = spec1(evaldir)
    if r is not None:
        plot(axb, 1.0 / kc1, r, color=col, ls=ls, lw=lw, label=SC(lab))
axb.set_xscale("log"); axb.invert_xaxis(); axb.set_ylim(0, 2.45); axb.set_yticks([0, 0.5, 1.0, 1.5])
axb.set_xticks([1000, 100, 10, 2]); axb.set_xticklabels(["1000", "100", "10", "2"])
axb.set_xlabel("Wavelength (km)"); axb.set_ylabel("Error / signal power")
axb.set_title("(b) Error by scale, 1 km window")
axb.legend(**LEG)

# (c) skill vs cost, MOSAIC layout
pts = {}
for name in ("convlstm", "earthformer", "simvp", "climax", "topoflow", "foehn", "aurora"):
    cf = DATA / f"cost_{name}.json"
    if cf.exists():
        c = json.loads(cf.read_text())
        mean, _, n = load(name, "rmse_1km")
        if n:
            pts[name] = (c["seconds"], float(np.mean(mean)), c["peak_gb"])
area = lambda gb: 18 + 11 * gb
line = lambda x: 5.52 + 0.195 * (np.log10(x) + 2.52)
x = np.logspace(np.log10(2e-3), np.log10(60), 50)
from matplotlib.patches import Rectangle


def draw(ax):
    ax.plot(x, line(x), color="#c8c8c8", ls=(0, (1.2, 1.2)), lw=3.0, zorder=1)
    for name, (sec, rmse, gb) in pts.items():
        st_ = STYLE[name]
        if name == "foehn":
            ax.scatter(sec, rmse, s=area(gb) + 55, facecolor="none", edgecolor="#111111", lw=0.8, zorder=4)
        ax.scatter(sec, rmse, s=area(gb), color=st_["color"], edgecolor="#333333", lw=0.5, alpha=0.9, zorder=3)


draw(axc)
axc.text(3.0, 6.27, "deterministic", rotation=7, color="#a8a8a8", fontsize=7, ha="center")
axc.text(3.0, 5.86, "probabilistic", rotation=7, color="#a8a8a8", fontsize=7, ha="center")
if "foehn" in pts and "aurora" in pts:
    (s1, r1, _), (s2, r2, _) = pts["foehn"], pts["aurora"]
    axc.annotate("", xy=(s2 * 0.8, r2 - 0.15), xytext=(s1 * 1.3, r1 + 0.1), arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.4))
    axc.text(np.sqrt(s1 * s2) * 1.25, (r1 + r2) / 2 - 0.12, f"{s2 / s1:.0f}× faster,\n{100 * (1 - r1 / r2):.0f}% lower RMSE",
             fontsize=6.2, color="#444444", ha="left")
axc.set_xscale("log"); axc.set_xlim(2e-3, 60); axc.set_ylim(5.2, 8.9)
axc.set_xlabel("Inference time (seconds per member per forecast)"); axc.set_ylabel("RMSE at 1 km (4-day avg., µg m$^{-3}$)")
axc.set_title("(c) Forecast skill vs. inference speed")
ZX, ZY = (4.5e-3, 0.4), (5.36, 6.32)                         # zoom on the learned models
axc.add_patch(Rectangle((ZX[0], ZY[0]), ZX[1] - ZX[0], ZY[1] - ZY[0], fill=False, ec="#777777", lw=0.7, ls="--", zorder=2))
ins = axc.inset_axes([0.06, 0.52, 0.5, 0.42])
draw(ins)
ins.set_xscale("log"); ins.set_xlim(*ZX); ins.set_ylim(*ZY)
ins.tick_params(labelsize=5.5, length=1.5, pad=1)
for sp in ins.spines.values():
    sp.set_edgecolor("#777777"); sp.set_linewidth(0.7)
ORDER_C = [m for m in ("foehn", "convlstm", "earthformer", "simvp", "topoflow", "climax", "aurora") if m in pts]
mh = [axc.scatter([], [], s=34, color=STYLE[m]["color"], edgecolor="#333333", lw=0.5) for m in ORDER_C]
fig.legend(mh, [SC(STYLE[m]["label"]) for m in ORDER_C], loc="lower center", bbox_to_anchor=(0.5, 0.12), ncol=len(ORDER_C),
           frameon=False, fontsize=6.0, columnspacing=1.0, handletextpad=0.2, borderaxespad=0)
from matplotlib.lines import Line2D
bh = [Line2D([], [], ls="none")] + [axc.scatter([], [], s=area(gb), color="#a0a0a0", edgecolor="#555555", lw=0.5) for gb in (0.5, 5, 20)]
fig.legend(bh, ["Peak GPU memory in (c):", "~0.5 GB", "~5 GB", "~20 GB"], loc="lower center", bbox_to_anchor=(0.5, 0.004),
           ncol=4, frameon=False, fontsize=6.0, columnspacing=1.6, handletextpad=0.9, borderaxespad=0, handlelength=1.2,
           borderpad=1.1)
save(fig, "fig_spectra_cost")
