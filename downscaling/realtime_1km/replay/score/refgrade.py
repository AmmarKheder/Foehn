#!/usr/bin/env python3
"""WP1b, question 5: which catalogue monitors sit at a site of an official reference network?

No file on disk carries the sensor type of the OpenAQ locations (coupling/build_st.py l.18). What is on disk are the
registries of two official networks (downloaded on 2026-10-26 for the 2024 station test):
  data/stations/eea/PanEuropean_metadata.csv   sampling points of the European network (PM2.5 = pollutant code 6001)
  data/stations/epa_aqs/daily_88101_2024.zip   United States, parameter 88101 (PM2.5 by reference or equivalent methods)
  data/stations/epa_aqs/daily_88502_2024.zip   United States, parameter 88502 (PM2.5, other accepted methods)
PROXY, not a sensor-type flag: a catalogue monitor is called 'at an official site' when it lies within TOL metres of a
PM2.5 site of one of these registries. Evaluation only; no network request.
  python refgrade.py --out S/work"""
import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

W = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
TOL = 150.0


def eea_sites():
    pts = set()
    with open(W / "data/stations/eea/PanEuropean_metadata.csv", newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if row["AirPollutantCode"].rstrip("/").endswith("/6001"):
                try:
                    pts.add((round(float(row["Latitude"]), 5), round(float(row["Longitude"]), 5)))
                except ValueError:
                    pass
    return np.array(sorted(pts))


def epa_sites(name):
    pts = set()
    with zipfile.ZipFile(W / "data/stations/epa_aqs" / name) as z:
        with z.open(z.namelist()[0]) as f:
            r = csv.reader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
            h = next(r)
            i, j = h.index("Latitude"), h.index("Longitude")
            for row in r:
                try:
                    pts.add((round(float(row[i]), 5), round(float(row[j]), 5)))
                except (ValueError, IndexError):
                    pass
    return np.array(sorted(pts))


def nearest_m(lat, lon, P):
    """distance in metres from each (lat, lon) to the nearest point of P (n, 2), by latitude-sorted search"""
    o = np.argsort(P[:, 0])
    P = P[o]
    out = np.full(lat.size, np.inf)
    lo = np.searchsorted(P[:, 0], lat - 0.05)
    hi = np.searchsorted(P[:, 0], lat + 0.05)
    for k in range(lat.size):
        if hi[k] > lo[k]:
            q = P[lo[k]:hi[k]]
            dy = (q[:, 0] - lat[k]) * 111320.0
            dx = (q[:, 1] - lon[k]) * 111320.0 * np.cos(np.radians(lat[k]))
            out[k] = np.sqrt(dx * dx + dy * dy).min()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cat = np.load(W / "realtime_1km/verif/catalogue.npz")
    lat, lon, reg = cat["lat"].astype(np.float64), cat["lon"].astype(np.float64), cat["region"].astype(str)
    E = eea_sites()
    U1 = epa_sites("daily_88101_2024.zip")
    U2 = epa_sites("daily_88502_2024.zip")
    d_eea, d_u1, d_u2 = nearest_m(lat, lon, E), nearest_m(lat, lon, U1), nearest_m(lat, lon, U2)
    usa = (reg == "North America") & (lat >= 24.5) & (lat <= 49.5) & (lon >= -125.0) & (lon <= -66.5)     # conterminous box (the registry has no Canada or Mexico)
    info = dict(tolerance_m=TOL, eea_pm25_sites=int(E.shape[0]), epa_88101_sites=int(U1.shape[0]), epa_88502_sites=int(U2.shape[0]),
                europe=dict(monitors=int((reg == "Europe").sum()), at_official_site=int(((reg == "Europe") & (d_eea <= TOL)).sum()),
                            hist_m={f"<= {t}": int(((reg == "Europe") & (d_eea <= t)).sum()) for t in (10, 25, 50, 100, 150, 300, 500, 1000, 2000)}),
                usa_box=dict(monitors=int(usa.sum()), at_88101_site=int((usa & (d_u1 <= TOL)).sum()), at_88502_only=int((usa & (d_u1 > TOL) & (d_u2 <= TOL)).sum()),
                             hist_m={f"<= {t}": int((usa & (np.minimum(d_u1, d_u2) <= t)).sum()) for t in (10, 25, 50, 100, 150, 300, 500, 1000, 2000)}))
    np.savez(Path(a.out) / "refgrade.npz", id=cat["id"], d_eea=d_eea.astype(np.float32), d_epa88101=d_u1.astype(np.float32), d_epa88502=d_u2.astype(np.float32),
             usa_box=usa, tol=TOL)
    (Path(a.out) / "refgrade.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    sys.exit(main())
