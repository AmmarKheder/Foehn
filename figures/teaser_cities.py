#!/usr/bin/env python3
"""Choose the six call-out cities of the teaser (rule fixed before looking at forecasts): in each of six regions, the
city with the most person-days of 2022 above the threshold hidden by 40 km cells (scan_exposure_cities.py); its day =
the day with the most hidden residents. Prints the choice and the eval_global_green.py crop command (Foehn J+1, start day
= target - 1) -> data/teaser_cities.json
    python teaser_cities.py [threshold 37.5]"""
import json
import sys

import numpy as np

from nature_style import HERE
from scan_exposure_cities import CITIES, box

THR = float(sys.argv[1]) if len(sys.argv) > 1 else 37.5
GROUPS = {"South Asia": ["Delhi", "Dhaka", "Lahore", "Kolkata", "Kathmandu"],
          "East Asia": ["Beijing", "Chengdu", "Xi'an", "Seoul", "Ulaanbaatar"],
          "Southeast Asia": ["Bangkok", "Jakarta"],
          "Middle East and North Africa": ["Cairo", "Riyadh", "Tehran"],
          "Sub-Saharan Africa": ["Lagos", "Kano", "Kinshasa", "Johannesburg"],
          "Americas": ["Mexico City", "Santiago", "Lima", "Bogota", "Los Angeles"]}
z = np.load(HERE / "data" / "exposure_cities.npz")
names, thr, res = list(z["names"]), list(z["thr"]), z["res"]
k = thr.index(THR)
pick = []
for g, cands in GROUPS.items():
    annual = {c: res[names.index(c), :, k, 2].sum() for c in cands}
    best = max(annual, key=annual.get)
    day = int(res[names.index(best), :, k, 2].argmax())
    hid = res[names.index(best), day, k, 2] / 1e6
    r0, r1, c0, c1 = box(*CITIES[best])
    pick.append(dict(region=g, city=best, doy=day, hidden_M=round(float(hid), 2), annual_hidden_M=round(float(annual[best] / 1e6), 1),
                     box=[84 - r0 / 100, 84 - r1 / 100, c0 / 100 - 180, c1 / 100 - 180]))
    print(f"{g:30s} {best:12s} day {day} ({np.datetime64('2022-01-01') + np.timedelta64(day, 'D')}): {hid:.1f} M hidden; "
          f"annual {annual[best] / 1e6:.0f} M person-days | " + ", ".join(f"{c} {v / 1e6:.0f}" for c, v in annual.items()))
json.dump(dict(threshold=THR, cities=pick), open(HERE / "data" / "teaser_cities.json", "w"), indent=1)
crops = " ".join(f"--crop {b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f}" for b in (p["box"] for p in pick))
days = " ".join(str(p["doy"] - 1) for p in pick)
print(f"\neval args: {crops} --crop-only --crop-days {days} --doys {days} --leads 1")
