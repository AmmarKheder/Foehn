#!/usr/bin/env python3
"""The footprint layer at work (the MOSAIC 'mechanism' figure):
(a) back-trajectories of the air reaching eight cities at 12 UTC on 17 March 2022 (Saharan dust episode, lead 2 days),
    integrated through the real HRES 3-hourly forecast wind exactly as in the layer, over the PM2.5 field of the start
    day that the layer gathers; line width / opacity = learned weight of each path point (FOEHN, seed 42);
(b) learned log-weight versus boundary-layer height (true-wind model vs wrong-wind control, 3 seeds each);
(c) learned log-weight versus stability (realistic HRES model, 3 seeds);
(d) gain of the footprint with 95 % block-bootstrap intervals (lead 2 days): CRPS and CSI of exceedances, on all days,
    documented transported episodes and the 10 % / 5 % most polluted days."""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

from fig_style import COL2, EXP, NATURE, STYLE, save
STYLE_C = STYLE['foehn']['color']

sys.path.insert(0, str(EXP / "dejavu"))
sys.path.insert(0, str(EXP / "dejavu" / "figs"))
from dejavu import DVData, attach_hres  # noqa: E402
from footprint_physics import load as load_fp, z_of  # noqa: E402

START_DOY, H = 71, 2                       # 13 March 2022 -> arrival 15 March 12 UTC (dust intrusion)
CITIES = {"Madrid": (40.42, -3.70), "Bordeaux": (44.84, -0.58), "Paris": (48.86, 2.35), "Lyon": (45.76, 4.84),
          "Milan": (45.46, 9.19), "Munich": (48.14, 11.58), "Rome": (41.90, 12.50), "Vienna": (48.21, 16.37)}
DT, KMAX = 3 * 3600.0, 32


def bilinear(field, lat, lon):
    """field (..., 84, 140) on 0.5 deg cells (centres 71.875 - 0.5 r, -24.875 + 0.5 c), border-clamped."""
    r = np.clip((71.875 - lat) / 0.5, 0, field.shape[-2] - 1.001); c = np.clip((lon + 24.875) / 0.5, 0, field.shape[-1] - 1.001)
    r0, c0 = int(r), int(c); fr, fc = r - r0, c - c0
    return ((1 - fr) * (1 - fc) * field[..., r0, c0] + (1 - fr) * fc * field[..., r0, c0 + 1]
            + fr * (1 - fc) * field[..., r0 + 1, c0] + fr * fc * field[..., r0 + 1, c0 + 1])


def trajectories():
    ds = DVData([2022], "noise", train=False, t31=True, footprint=True, fp_met="stab")
    attach_hres(ds, True, True)
    doy = np.load(EXP / "dejavu/data/meta.npz")["doy"]
    n = [k for k, (i, h) in enumerate(ds.samples) if doy[i] == START_DOY and h == H][0]
    i = ds.samples[n][0]
    it = ds[n]
    wind, rain, stab = (it[k].float().numpy() for k in ("wind", "rain", "stab"))
    logit_fn, lam, mix = load_fp("noise_t31_fpstab")
    paths = {}
    for name, (la, lo) in CITIES.items():
        pts, logits, cum = [], [], 0.0
        s0 = 8 * H + 4
        for k in range(8 * H + 1):
            st = s0 - k; day = st // 8
            w = bilinear(wind[st], la, lo)
            u, v = mix[0] * w[0] + mix[1] * w[2], mix[0] * w[1] + mix[1] * w[3]
            r = float(bilinear(rain[day], la, lo)); sg = bilinear(stab[day], la, lo)
            logits.append(logit_fn(np.array([[k / KMAX, sg[0] / 5, sg[1] / 5, r, cum]], np.float32))[0])
            pts.append((lo, la))
            la, lo = la - v * DT / 111e3, lo - u * DT / (111e3 * np.cos(np.deg2rad(la)))
            cum += r / 8
        wgt = np.exp(np.array(logits) - max(logits)); wgt /= wgt.sum()
        paths[name] = (np.array(pts), wgt)
    pm = np.asarray(np.load(EXP / "dejavu/data/pm.npy", mmap_mode="r")[i], np.float32)
    return paths, pm


def bootstrap_rows(lead=2):
    rows = {}
    for line in (EXP / "dejavu/significance_2022.txt").read_text().splitlines():
        m = re.match(r"(.+?)\s+T\+(\d)\s+(\d+) \| (.+)", line)
        if not m or int(m.group(2)) != lead:
            continue
        cells = re.findall(r"([+-]\d+\.\d)\s+\[\s*([+-]\d+\.\d),\s*([+-]\d+\.\d)\]", m.group(4))
        rows[m.group(1).strip()] = [tuple(map(float, c)) for c in cells]   # RMSE, CRPS, CSI: (point, lo, hi)
    return rows


paths, pm = trajectories()
fig = plt.figure(figsize=(COL2, 2.9))
gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1, 1], height_ratios=[1, 0.85], wspace=0.42, hspace=0.75)
ax = fig.add_subplot(gs[:, 0])
land = np.load(EXP / "dejavu/data/meta.npz")["land"]
ext = [-25, -25 + 0.25 * 280, 72 - 0.25 * 168, 72]
from matplotlib.colors import LinearSegmentedColormap as _LSC
from nature_style import PM_COLORS as PMC
im = ax.imshow(np.where(land, pm, np.nan), extent=ext, cmap=_LSC.from_list("pmc", PMC, N=256), vmin=0, vmax=50, interpolation="nearest")
ax.contour(np.linspace(ext[0], ext[1], 280), np.linspace(ext[3], ext[2], 168), land.astype(float), [0.5],
           colors="#52514e", linewidths=0.3)
for name, (p, w) in paths.items():
    ax.plot(p[:, 0], p[:, 1], color="#0b0b0b", lw=0.9, alpha=0.9, solid_capstyle="round", zorder=4)
    ax.plot(p[4::4, 0], p[4::4, 1], "o", ms=1.6, color="#0b0b0b", zorder=4)          # every 12 h back in time
    ax.plot(*p[0], "o", ms=3.4, mfc="white", mec="#0b0b0b", mew=0.7, zorder=5)
    off = {"Paris": (-24, 4), "Munich": (-26, 4), "Bordeaux": (-30, -8), "Lyon": (2, 5), "Vienna": (-8, 5), "Milan": (3, -8)}.get(name, (3, 2))
    ax.annotate(name, p[0], xytext=off, textcoords="offset points", fontsize=5.5, color="#0b0b0b", zorder=6)
ax.set_xlim(-14, 22); ax.set_ylim(28.2, 55); ax.grid(False); ax.set_aspect(1.35)
ax.set_title("(a) Back-trajectories, arrival 15 Mar 2022", loc="left")
ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.16, aspect=35)
cb.set_label("PM$_{2.5}$ on the start day, 13 Mar (µg m$^{-3}$)", fontsize=6); cb.ax.tick_params(labelsize=5.5)

blh = np.linspace(50, 2500, 60)
gap = np.linspace(-1, 14, 61)
axb, axc = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
for run, col, ls, lab in (("noise_fp", STYLE_C, "-", "true wind"), ("noise_fpshift15", "#8a8985", "--", "wrong wind")):
    for s, sfx in enumerate(("", "_s43", "_s44")):
        f, _, _ = load_fp(run + sfx)
        b = f(z_of("blh", np.full_like(blh, 24.0), blh=blh))
        axb.plot(blh, b - b.mean(), color=col, ls=ls, lw=1.0, label=lab if s == 0 else None)
for s, sfx in enumerate(("", "_s43", "_s44")):
    f, _, _ = load_fp("noise_t31_fpstab" + sfx)
    g = f(z_of("stab", np.full_like(gap, 24.0), gap2=gap, gap1=gap * 0.4))
    axc.plot(gap, g - g.mean(), color=STYLE_C, lw=1.0)
axb.set(xlabel="Boundary-layer height (m)", ylabel="Learned log-weight")
axb.set_title("(b) Mixing depth", loc="left"); axb.legend(loc="lower right", handlelength=1.8)
axc.set(xlabel="θ$_{850}$ − θ$_{1000}$ (K)", ylabel="Learned log-weight")
axc.set_title("(c) Stability (HRES model)", loc="left")

axd = fig.add_subplot(gs[1, 1:])
rows = bootstrap_rows(2)
sets = [("all days", "All days"), ("transported episodes", "Transported\nepisodes"), ("top 10 % extent", "Top 10 %\ndays"),
        ("top 5 % extent", "Top 5 %\ndays")]
x = np.arange(len(sets))
for col_idx, mk, mfc, lab, off in ((1, "o", STYLE_C, "CRPS (lower is better)", -0.12), (2, "s", "white", "CSI > 25 µg m$^{-3}$ (higher is better)", 0.12)):
    pts = np.array([rows[k][col_idx] for k, _ in sets])
    axd.errorbar(x + off, pts[:, 0], yerr=[pts[:, 0] - pts[:, 1], pts[:, 2] - pts[:, 0]], fmt=mk, ms=3.2, color=STYLE_C,
                 mfc=mfc, mec=STYLE_C, elinewidth=0.9, capsize=1.5, label=lab)
axd.axhline(0, color="#52514e", lw=0.6)
axd.set_xticks(x); axd.set_xticklabels([s for _, s in sets])
axd.set_ylabel("Change (%)" if NATURE else "Change with footprint (%)")
axd.set_title("(d) Effect of the transport layer, 2 days ahead (95% block-bootstrap intervals)" if NATURE else
              "(d) Effect of the footprint, T+2 (95 % block-bootstrap CI)", loc="left")
axd.set_ylim(-9, 16)
axd.legend(loc="upper left", fontsize=5.5, handlelength=1, ncol=2)
save(fig, "fig_footprint")
