#!/usr/bin/env python3
"""Zooms of the exposure figure: GHAP 1 km PM2.5, the same field averaged over the 40 km cells (0.4 deg, aligned on the
global cell grid of figdata_exposure.py) and the residents, around a city on one day of 2022.
    python figdata_exposure_zoom.py   -> data/exposure_zoom.npz"""
from pathlib import Path

import os

import numpy as np

from figdata_exposure import F, G, GH, GW, I2022, ghap_day


def pop_rows(r0, r1):
    """rows r0:r1 of pop1km_global.npy (pread: no memmap of the 2 GB file on the login node)"""
    path = G / "pop1km_global.npy"
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); sh, _, dt = np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    fd = os.open(path, os.O_RDONLY)
    try:
        b = os.pread(fd, (r1 - r0) * sh[1] * dt.itemsize, hdr + r0 * sh[1] * dt.itemsize)
    finally:
        os.close(fd)
    return np.frombuffer(b, dt).reshape(r1 - r0, sh[1])

HERE = Path(__file__).resolve().parent
CASES = {"Karachi": (89, 25.0, 67.0), "Delhi": (53, 28.6, 77.2)}             # day of year (0-based), centre lat, lon
HALF = 160                                                                  # pixels each side (1.6 deg, 4 x 4 cells)


def main():
    out = {}
    for name, (doy, la, lo) in CASES.items():
        x = ghap_day(I2022 + doy).astype(np.float32)
        r = int(round((84.0 - la) / 0.01)); c = int(round((lo + 180.0) / 0.01))
        r0 = (r - HALF) // F * F; c0 = (c - HALF) // F * F                      # snap to the 40 km cell grid
        r1, c1 = r0 + 2 * HALF + F, c0 + 2 * HALF + F
        sub = x[r0:r1, c0:c1]; ok = sub > 0
        cell = np.where(ok, sub, 0).reshape((r1 - r0) // F, F, (c1 - c0) // F, F).sum((1, 3)) / np.maximum(ok.reshape((r1 - r0) // F, F, (c1 - c0) // F, F).sum((1, 3)), 1)
        x40 = np.where(ok, np.repeat(np.repeat(cell, F, 0), F, 1), np.nan)
        p = pop_rows(r0, r1)[:, c0:c1].astype(np.float32)
        out[name] = dict(doy=doy, x1=np.where(ok, sub, np.nan).astype(np.float32), x40=x40.astype(np.float32), pop=p,
                         lat0=84.0 - 0.01 * r0, lon0=-180.0 + 0.01 * c0)
        hid = (p * (sub > 75) * (x40 <= 75)).sum()
        print(f"{name} doy {doy}: residents above 75 at 1 km {(p * (sub > 75)).sum() / 1e6:.1f} M, at 40 km {(p * (x40 > 75)).sum() / 1e6:.1f} M, "
              f"hidden {hid / 1e6:.1f} M | max 1 km {np.nanmax(sub[ok]):.0f}, cell means {np.round(cell.max(), 0)}", flush=True)
    np.savez(HERE / "data" / "exposure_zoom.npz", **{f"{k}_{n}": v for n, d in out.items() for k, v in d.items()})


if __name__ == "__main__":
    main()
