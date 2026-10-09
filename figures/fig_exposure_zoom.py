#!/usr/bin/env python3
"""Panel 'what a 40 km grid hides': one day of GHAP at 1 km over a box, the same field averaged over the 40 km cells of the
CAMS / Aurora grid (0.4 deg, aligned on 84 N / 180 W), the residents (GHS-POP 2020), and the residents above a WHO
threshold at 1 km whom the 40 km cells place below it.
    python fig_exposure_zoom.py <doy 2022> <N> <S> <W> <E> [threshold 75] [name]   (stats only: add --stats)"""
import os
import sys

import numpy as np

G = "/scratch/project_462001140/ammar/eccv/data/world/global1km"
I2022, T, NTC = 1826, 512, 71


def hdr(path):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); sh, _, dt = np.lib.format.read_array_header_1_0(f); return f.tell(), sh, dt


def ghap_box(d, r0, r1, c0, c1):
    """GHAP 1 km of global day d, rows r0:r1, cols c0:c1, from the (D, 29, 71, 512, 512) tile store (only the tiles needed)"""
    path = f"{G}/ghap1km_tiles.npy"; h0, sh, dt = hdr(path)
    out = np.zeros((r1 - r0, c1 - c0), np.float32)
    fd = os.open(path, os.O_RDONLY)
    try:
        for tr in range(r0 // T, (r1 - 1) // T + 1):
            for tc in range(c0 // T, (c1 - 1) // T + 1):
                off = h0 + ((((d * 29 + tr) * NTC + tc) * T * T) * dt.itemsize)
                b = np.frombuffer(os.pread(fd, T * T * dt.itemsize, off), dt).reshape(T, T)
                rr0, rr1 = max(r0, tr * T), min(r1, (tr + 1) * T); cc0, cc1 = max(c0, tc * T), min(c1, (tc + 1) * T)
                out[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = b[rr0 - tr * T:rr1 - tr * T, cc0 - tc * T:cc1 - tc * T]
    finally:
        os.close(fd)
    return out


def pop_box(r0, r1, c0, c1):
    path = f"{G}/pop1km_global.npy"; h0, sh, dt = hdr(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        rows = [np.frombuffer(os.pread(fd, (c1 - c0) * 4, h0 + (r * sh[1] + c0) * 4), np.float32) for r in range(r0, r1)]
    finally:
        os.close(fd)
    return np.stack(rows)


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    doy, N, S, W, E = int(a[0]), *map(float, a[1:5])
    thr = float(a[5]) if len(a) > 5 else 75.0
    name = a[6] if len(a) > 6 else "box"
    # box snapped to whole 40 km cells (0.4 deg = 40 px), so that every cell is complete
    r0 = int(np.floor((84.0 - N) / 0.4)) * 40; r1 = int(np.ceil((84.0 - S) / 0.4)) * 40
    c0 = int(np.floor((W + 180.0) / 0.4)) * 40; c1 = int(np.ceil((E + 180.0) / 0.4)) * 40
    x = ghap_box(I2022 + doy, r0, r1, c0, c1); pop = pop_box(r0, r1, c0, c1)
    ok = x > 0
    nr, nc = (r1 - r0) // 40, (c1 - c0) // 40
    xb, okb = x.reshape(nr, 40, nc, 40), ok.reshape(nr, 40, nc, 40)
    with np.errstate(all="ignore"):
        cell = np.where(okb, xb, 0).sum((1, 3)) / okb.sum((1, 3))
    x40 = np.repeat(np.repeat(cell, 40, 0), 40, 1)
    w = np.where(ok & np.isfinite(x40), pop, 0.0)
    a1, a4 = x > thr, x40 > thr
    tot, e1, e4, hid = w.sum(), (w * a1).sum(), (w * a4).sum(), (w * (a1 & ~a4)).sum()
    print(f"{name} doy {doy}: {tot / 1e6:.1f} M residents with data | above {thr:g}: 1 km {e1 / 1e6:.2f} M, 40 km {e4 / 1e6:.2f} M, "
          f"hidden {hid / 1e6:.2f} M ({100 * hid / max(e1, 1):.0f} % of those exposed) | PM 1 km p5-p95 "
          f"{np.nanpercentile(np.where(ok, x, np.nan), 5):.0f}-{np.nanpercentile(np.where(ok, x, np.nan), 95):.0f}")
    if "--stats" in sys.argv:
        return
    from nature_style import DOUBLE, MM, letter, plt, pm_cmap, save
    from matplotlib.colors import LogNorm
    cmap, norm = pm_cmap()
    ext = (-180 + c0 * 0.01, -180 + c1 * 0.01, 84 - r1 * 0.01, 84 - r0 * 0.01)
    fig, axs = plt.subplots(1, 4, figsize=(DOUBLE, 52 * MM), constrained_layout=True)
    show = lambda ax, f, **k: ax.imshow(np.ma.masked_invalid(f), extent=ext, origin="upper", interpolation="nearest", **k)
    im = show(axs[0], np.where(ok, x, np.nan), cmap=cmap, norm=norm); axs[0].set_title("PM$_{2.5}$ at 1 km", fontsize=7)
    show(axs[1], np.where(ok, x40, np.nan), cmap=cmap, norm=norm); axs[1].set_title("Same field on 40 km cells", fontsize=7)
    show(axs[2], np.where(pop > 0, pop, np.nan), cmap="Greys", norm=LogNorm(10, max(pop.max(), 100)))
    axs[2].set_title("Residents per km$^2$", fontsize=7)
    hidden = np.where(a1 & ~a4 & (pop > 0), pop, np.nan)
    show(axs[3], np.where(ok, 0.0, np.nan), cmap="Greys", vmin=0, vmax=8)
    show(axs[3], hidden, cmap="Reds", norm=LogNorm(10, max(np.nanmax(hidden) if np.isfinite(hidden).any() else 100, 100)))
    axs[3].set_title(f"Above {thr:g} µg m$^{{-3}}$ at 1 km,\nbelow it on the 40 km cells", fontsize=7)
    for k, ax in enumerate(axs):
        ax.set_xticks([]); ax.set_yticks([]); letter(ax, "abcd"[k], x=0.0, y=1.02)
        for c in range(1, nc):
            ax.axvline(ext[0] + 0.4 * c, color="white", lw=0.25, alpha=0.6)
        for r in range(1, nr):
            ax.axhline(ext[3] - 0.4 * r, color="white", lw=0.25, alpha=0.6)
    cb = fig.colorbar(im, ax=axs[:2], orientation="horizontal", fraction=0.06, pad=0.02, ticks=[0, 15, 37.5, 75, 150, 250])
    cb.set_label("Daily PM$_{2.5}$ (µg m$^{-3}$)", fontsize=6); cb.ax.tick_params(labelsize=5)
    fig.suptitle(f"{name}, {np.datetime64('2022-01-01') + np.timedelta64(doy, 'D')}: {hid / 1e6:.1f} million residents above "
                 f"{thr:g} µg m$^{{-3}}$ are invisible on the 40 km grid ({100 * hid / max(e1, 1):.0f} % of those exposed)", fontsize=7)
    save(fig, f"fig_exposure_zoom_{name}_{doy}")


if __name__ == "__main__":
    main()
