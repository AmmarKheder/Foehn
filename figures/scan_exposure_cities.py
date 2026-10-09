#!/usr/bin/env python3
"""For candidate cities, every day of 2022 (GHAP 1 km, GHS-POP): residents above the WHO 24-h interim targets 37.5 and
75 ug m-3 at 1 km, and those hidden by the 40 km cells (0.4 deg, aligned on 84 N / 180 W) -> data/exposure_cities.npz,
and the most telling day per city and threshold.
    python scan_exposure_cities.py"""
import numpy as np

from fig_exposure_zoom import ghap_box, pop_box
from nature_style import HERE

CITIES = {   # name: (lat, lon); box = city +- 0.6 deg lat, +- 0.8 deg lon, snapped to whole 40 km cells
    "Beijing": (39.90, 116.40), "Lahore": (31.55, 74.34), "Delhi": (28.61, 77.21), "Dhaka": (23.81, 90.41),
    "Kolkata": (22.57, 88.36), "Ulaanbaatar": (47.92, 106.92), "Chengdu": (30.57, 104.07), "Xi'an": (34.34, 108.94),
    "Seoul": (37.57, 126.98), "Bangkok": (13.76, 100.50), "Jakarta": (-6.21, 106.85), "Kathmandu": (27.72, 85.32),
    "Mexico City": (19.43, -99.13), "Santiago": (-33.45, -70.67), "Lima": (-12.05, -77.04), "Bogota": (4.71, -74.07),
    "Lagos": (6.52, 3.38), "Kano": (12.00, 8.52), "Cairo": (30.04, 31.24), "Riyadh": (24.71, 46.68), "Tehran": (35.69, 51.39),
    "Johannesburg": (-26.20, 28.05), "Milan": (45.46, 9.19), "Los Angeles": (34.05, -118.24), "Kinshasa": (-4.44, 15.27),
}
THR = (37.5, 75.0)


def box(lat, lon):
    r0 = int(np.floor((84.0 - (lat + 0.6)) / 0.4)) * 40; r1 = int(np.ceil((84.0 - (lat - 0.6)) / 0.4)) * 40
    c0 = int(np.floor((lon - 0.8 + 180.0) / 0.4)) * 40; c1 = int(np.ceil((lon + 0.8 + 180.0) / 0.4)) * 40
    return r0, r1, c0, c1


def main():
    names = list(CITIES)
    res = np.zeros((len(names), 365, len(THR), 3))                            # above 1 km, above 40 km, hidden (residents)
    for i, (nm, (la, lo)) in enumerate(CITIES.items()):
        r0, r1, c0, c1 = box(la, lo)
        pop = pop_box(r0, r1, c0, c1)
        nr, nc = (r1 - r0) // 40, (c1 - c0) // 40
        for d in range(365):
            x = ghap_box(1826 + d, r0, r1, c0, c1); ok = x > 0
            if not ok.any():
                continue
            with np.errstate(all="ignore"):
                cell = np.where(ok, x, 0).reshape(nr, 40, nc, 40).sum((1, 3)) / ok.reshape(nr, 40, nc, 40).sum((1, 3))
            x40 = np.repeat(np.repeat(cell, 40, 0), 40, 1)
            w = np.where(ok & np.isfinite(x40), pop, 0.0)
            for k, t in enumerate(THR):
                a1, a4 = x > t, x40 > t
                res[i, d, k] = [(w * a1).sum(), (w * a4).sum(), (w * (a1 & ~a4)).sum()]
        k75 = res[i, :, 1, 2]; k37 = res[i, :, 0, 2]
        print(f"{nm:13s} hidden residents: >37.5 max {k37.max() / 1e6:5.2f} M (doy {k37.argmax()}), annual {k37.sum() / 1e6:7.1f} M person-days"
              f" | >75 max {k75.max() / 1e6:5.2f} M (doy {k75.argmax()}), annual {k75.sum() / 1e6:6.1f} M", flush=True)
    np.savez(HERE / "data" / "exposure_cities.npz", names=np.array(names), coords=np.array(list(CITIES.values())), thr=np.array(THR), res=res)


if __name__ == "__main__":
    main()
