#!/usr/bin/env python3
"""Proof that the re-rendered Extended Data Fig. 4 differs from the submitted one only in panels g and h.
Rasterises ref_nature_final/figs/fig_exposure.pdf and ../figs/fig_exposure_eqiss.pdf at 300 dpi (PyMuPDF), compares the
pixels and reports the bounding boxes of the differences in mm from the top-left corner of the page.
-> ../figs/diff_fig_exposure.png (differences in red over the new figure), printed boxes."""
import sys
from pathlib import Path

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent.parent
REF = HERE.parent.parent / "ref_nature_final" / "figs" / "fig_exposure.pdf"
NEW = HERE / "figs" / (sys.argv[1] if len(sys.argv) > 1 else "fig_exposure_eqiss.pdf")
DPI = 300


def raster(p):
    d = fitz.open(p); pg = d[0]; pm = pg.get_pixmap(dpi=DPI, alpha=False)
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)[:, :, :3].astype(np.int16)
    return a, pg.rect.width / 72 * 25.4, pg.rect.height / 72 * 25.4


a, wa, ha = raster(REF); b, wb, hb = raster(NEW)
print(f"submitted page {wa:.1f} x {ha:.1f} mm ({a.shape[1]} x {a.shape[0]} px); new page {wb:.1f} x {hb:.1f} mm ({b.shape[1]} x {b.shape[0]} px)")
H, W = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
d = np.abs(a[:H, :W] - b[:H, :W]).max(-1) > 40
print(f"pixels that differ: {d.sum()} of {H * W} ({100 * d.mean():.3f} %)")
ys, xs = np.nonzero(d)
if len(ys):
    mm = 25.4 / DPI
    print(f"bounding box of all differences: x {xs.min() * mm:.1f}-{xs.max() * mm:.1f} mm, y {ys.min() * mm:.1f}-{ys.max() * mm:.1f} mm (page height {H * mm:.1f} mm)")
    # split along x into clusters separated by more than 5 mm
    cols = np.flatnonzero(d.any(0)); cut = np.flatnonzero(np.diff(cols) > 5 / mm)
    for seg in np.split(cols, cut + 1):
        sub = d[:, seg[0]:seg[-1] + 1]; yy = np.flatnonzero(sub.any(1))
        print(f"  cluster x {seg[0] * mm:6.1f}-{seg[-1] * mm:6.1f} mm, y {yy.min() * mm:6.1f}-{yy.max() * mm:6.1f} mm, {int(sub.sum())} px")
out = b[:H, :W].astype(np.uint8).copy(); out[d] = (255, 0, 0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.imsave(HERE / "figs" / "diff_fig_exposure.png", out)
print("saved", HERE / "figs" / "diff_fig_exposure.png")
