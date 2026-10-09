#!/usr/bin/env python3
"""How wrong are forecast trajectories? For the training years, the Green's functions driven by the HRES run of day d
and by ERA5 (the analysed truth) of the same receptor, start day and lead: great-circle distance between the mean
particle origins (forecast transport error), and the spread of each particle cloud (turbulent dispersion), by lead,
latitude band and season. Uses only 2017-2020 (no test data). -> data/traj_error.json
    python traj_error.py [--every 5]"""
import argparse
import json
from pathlib import Path

import numpy as np

G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
NR, NC = 288, 720
LAT = 84.0 - 0.5 * np.arange(NR) - 0.125


def gc_km(lon1, lat1, lon2, lat2):
    p1, p2, dl = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def mean_origin(o):
    """o (..., P, 4) lon, lat, z/h, survival -> circular mean lon, mean lat over particles"""
    lon, lat = np.deg2rad(o[..., 0].astype(np.float32)), o[..., 1].astype(np.float32)
    return np.rad2deg(np.arctan2(np.sin(lon).mean(-1), np.cos(lon).mean(-1))), lat.mean(-1)


def main(a):
    bands = [(60, 84), (30, 60), (0, 30), (-30, 0), (-60, -30)]
    res = {}
    for y in (2017, 2018, 2019, 2020):
        dh, de = G / "green" / "hres" / f"done_{y}", G / "green" / "era5" / f"done_{y}"
        if not (dh.exists() and de.exists()):
            continue
        days = sorted({int(p.name) for p in dh.glob("*")} & {int(p.name) for p in de.glob("*")})[::a.every]
        OH = np.load(G / "green" / "hres" / f"origins_{y}.npy", mmap_mode="r")
        OE = np.load(G / "green" / "era5" / f"origins_{y}.npy", mmap_mode="r")
        for d in days:
            season = ["DJF", "DJF", "MAM", "MAM", "MAM", "JJA", "JJA", "JJA", "SON", "SON", "SON", "DJF"][
                int(str(np.datetime64(f"{y}-01-01") + np.timedelta64(d, "D"))[5:7]) - 1]
            for h in range(1, 8):
                oh, oe = np.asarray(OH[d, h - 1]), np.asarray(OE[d, h - 1])          # (288, 720, P, 4)
                valid = (np.abs(oh[..., 0]).sum(-1) > 0) & (np.abs(oe[..., 0]).sum(-1) > 0)
                lo_h, la_h = mean_origin(oh); lo_e, la_e = mean_origin(oe)
                err = gc_km(lo_h, la_h, lo_e, la_e)
                lo0, la0 = np.meshgrid(-180.0 + 0.5 * np.arange(NC) + 0.125, LAT)
                dist = gc_km(lo0, la0, lo_e, la_e)                                   # travel distance (ERA5)
                sp = gc_km(oe[..., 0], oe[..., 1], lo_e[..., None], la_e[..., None]).mean(-1)   # cloud radius (ERA5)
                for b0, b1 in bands:
                    m = valid & (LAT[:, None] >= b0) & (LAT[:, None] < b1)
                    if m.sum() < 50:
                        continue
                    key = f"{h}|{b0}..{b1}|{season}"
                    r = res.setdefault(key, {"err": [], "dist": [], "spread": []})
                    r["err"].append(float(np.median(err[m]))); r["dist"].append(float(np.median(dist[m])))
                    r["spread"].append(float(np.median(sp[m])))
            print(y, d, flush=True)
    out = {k: {kk: float(np.mean(v)) for kk, v in r.items()} | {"n_days": len(r["err"])} for k, r in res.items()}
    Path(__file__).resolve().parents[1].joinpath("figures", "data", "traj_error.json").write_text(json.dumps(out, indent=1))
    for h in range(1, 8):
        ks = [k for k in out if k.startswith(f"{h}|")]
        if ks:
            e = np.mean([out[k]["err"] for k in ks]); dst = np.mean([out[k]["dist"] for k in ks]); s = np.mean([out[k]["spread"] for k in ks])
            print(f"lead {h}: forecast-trajectory error {e:6.0f} km | travel {dst:6.0f} km ({100 * e / max(dst, 1):4.1f} %) | cloud radius {s:5.0f} km")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=5)
    main(ap.parse_args())
