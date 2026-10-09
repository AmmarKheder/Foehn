#!/usr/bin/env python3
"""Spectral fidelity (the MOSAIC spectrum figure): radially averaged power spectrum of each forecast divided by that of
GHAP, over a land-dominated box of central Europe (64 x 128 cells at 0.25 deg, 60-44 N, 1-33 E), 61 start days of 2022,
real HRES weather. A ratio of 1 means the forecast has the observed variability at that scale; below 1 the field is
smoothed. FOEHN: individual ensemble members (solid) and the ensemble mean (dotted)."""
import numpy as np
import matplotlib.pyplot as plt

from fig_style import COL2, STYLE, save
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
R0, C0, NR, NC = 48, 104, 64, 128
WIN = np.outer(np.hanning(NR), np.hanning(NC))
ky, kx = np.meshgrid(np.fft.fftfreq(NR), np.fft.fftfreq(NC), indexing="ij")
KR = np.sqrt(kx ** 2 + ky ** 2)                        # cycles per cell
BINS = np.linspace(1.0 / NC, 0.5, 30)
KC = 0.5 * (BINS[1:] + BINS[:-1])


def load(name):
    z = np.load(DATA / f"{name}.npz")
    return {int(d): i for i, d in enumerate(z["days"])}, z["pred"].astype(np.float32)


tdays, truth = load("truth")
land = np.isfinite(truth[0, 0, 0, R0:R0 + NR, C0:C0 + NC])


def psd(field):
    a = np.where(land, field, np.nan)
    a = np.nan_to_num(a - np.nanmean(a)) * WIN
    p = np.abs(np.fft.fft2(a)) ** 2
    idx = np.digitize(KR.ravel(), BINS) - 1
    ok = (idx >= 0) & (idx < len(KC))
    return np.bincount(idx[ok], p.ravel()[ok], len(KC)) / np.maximum(np.bincount(idx[ok], minlength=len(KC)), 1)


def spectrum(name, lead_i, mode):
    days, pred = load(name)
    acc_m, acc_t = np.zeros(len(KC)), np.zeros(len(KC))
    for d, i in days.items():
        if d not in tdays:
            continue
        t = truth[tdays[d], lead_i, 0, R0:R0 + NR, C0:C0 + NC]
        E = pred[i, lead_i, :, R0:R0 + NR, C0:C0 + NC]
        fields = [E.mean(0)] if mode == "mean" else list(E)
        acc_m += np.mean([psd(f) for f in fields], 0)
        acc_t += psd(t)
    return acc_m / acc_t


CURVES = [("reg", "member", "Same network, regression", "#8a8985", ":"), ("convlstm", "member", None, None, None),
          ("earthformer", "member", None, None, None), ("simvp", "member", None, None, None), ("topoflow", "member", None, None, None),
          ("nofp", "member", None, None, None), ("foehn", "mean", "FOEHN, ensemble mean", "#8b1414", ":"), ("foehn", "member", None, None, None)]
fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.62, 1.9), sharey=True, constrained_layout=True)
wl = 25.0 / KC                                            # wavelength (km), 0.25 deg ~ 25 km at these latitudes (N-S)
for ax, (li, title) in zip(axes, ((0, "(a) Lead 1 day"), (1, "(b) Lead 3 days"))):
    ax.axhline(1.0, color="#52514e", lw=0.7)
    for name, mode, lab, col, ls in CURVES:
        st = STYLE.get(name, {})
        r = spectrum(name, li, mode)
        ax.plot(wl, r, color=col or st["color"], ls=ls or st["ls"], lw=1.5 if (name == "foehn" and mode == "member") else 1.0,
                label=lab or (st["label"] + (", member" if name in ("foehn", "nofp") else "")), zorder=10 if name == "foehn" else 3)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("Wavelength (km)")
    ax.set_title(title, loc="left")
axes[0].set_ylabel("Power ratio, forecast / GHAP")
axes[0].set_ylim(0, 1.3)
h, l = axes[1].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="outside right center", fontsize=5.8)
save(fig, "fig_spectra")
for li in (0, 1):
    r = {n + m: spectrum(n, li, m)[-8:].mean() for n, m, *_ in CURVES}
    print(f"lead index {li}: small-scale (<~60 km) power ratio", {k: round(v, 2) for k, v in r.items()})
