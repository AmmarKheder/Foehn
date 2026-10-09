#!/usr/bin/env python3
"""Episode day of each teaser zone (fig1_teaser_v4.py, row f), chosen without any forecast: the valid day t = 1..364 of 2022
with the largest exposure above 37.5 ug/m3 at 1 km (GHAP) hidden by the 40 km cell (84N 180W grid, mean of the valid pixels),
sum of residents x (PM2.5 - 37.5) where the pixel is above and its cell below (marginal days, all near 37.5, score low). Saves for that day GHAP and the residents at 1 km over the zone, and the CAMS (run 00 UTC of t, lead day 0) and Aurora
(initialised 00 UTC of t) daily fields at every pixel (their 0.4 deg cell); Foehn is added from the crops of the eval
(eval_global_green.py --crop ... --crop-days, forecast issued at the end of day t - 1).
    python figdata_teaser_episodes.py                 -> data/teaser_episodes.npz (days, GHAP, residents, CAMS, Aurora)
    python figdata_teaser_episodes.py foehn <eval>    -> adds the Foehn ensemble mean of those days"""
import sys
from pathlib import Path

import numpy as np
import zarr

from fig1_teaser_v4 import ZONES, nearest
from fig_exposure_zoom import ghap_box, pop_box

HERE = Path(__file__).resolve().parent
AURD = HERE.parent / "benchmark" / "aurora"
OUT = HERE / "data" / "teaser_episodes.npz"
THR, I2022 = 37.5, 1826


def box_px(N, S, W, E):
    return int(round((84 - N) * 100)), int(round((84 - S) * 100)), int(round((W + 180) * 100)), int(round((E + 180) * 100))


def episode(N, S, W, E):
    r0, r1, c0, c1 = box_px(N, S, W, E)
    R0, R1, C0, C1 = r0 // 40 * 40, -(-r1 // 40) * 40, c0 // 40 * 40, -(-c1 // 40) * 40     # whole 40 km cells around it
    pop = pop_box(R0, R1, C0, C1)
    best = (-1.0, None)
    for t in range(1, 365):
        x = ghap_box(I2022 + t, R0, R1, C0, C1); ok = x > 0
        n = ok.reshape((R1 - R0) // 40, 40, (C1 - C0) // 40, 40).sum((1, 3))
        s = np.where(ok, x, 0).reshape((R1 - R0) // 40, 40, (C1 - C0) // 40, 40).sum((1, 3))
        with np.errstate(all="ignore"):
            x40 = np.repeat(np.repeat(np.where(n > 0, s / np.maximum(n, 1), np.nan), 40, 0), 40, 1)
        hid = (pop * np.where(ok & ~(x40 > THR), np.maximum(x - THR, 0), 0))[r0 - R0:r1 - R0, c0 - C0:c1 - C0].sum()
        if hid > best[0]:
            best = (hid, t, np.where(ok, x, np.nan)[r0 - R0:r1 - R0, c0 - C0:c1 - C0], x40[r0 - R0:r1 - R0, c0 - C0:c1 - C0])
    return best, pop[r0 - R0:r1 - R0, c0 - C0:c1 - C0]


def main():
    cams = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
    aur = {}
    for p in sorted((AURD / "out_12h_all").glob("part_*.npz")):
        z = np.load(p)
        aur.update({int(d): (p, k) for k, d in enumerate(z["days"])})
    res = {}
    for j, (zn, (N, S, W, E)) in enumerate(ZONES.items()):
        (hid, t, x1, x40), pop = episode(N, S, W, E)
        r0, r1, c0, c1 = box_px(N, S, W, E)
        lats, lons = 84.0 - 0.01 * (np.arange(r0, r1) + 0.5), -180.0 + 0.01 * (np.arange(c0, c1) + 0.5)
        p, k = aur[t]; A = np.load(p)["pred"][k, :2].astype(np.float32)
        an = np.load(AURD / "raw" / f"{t:03d}.npz")["surf"][0, 5, :450].astype(np.float32) * 1e9
        grid = (90.0, 0.4, 0.0, 0.4)
        res.update({f"day_{j}": t, f"ghap_{j}": x1.astype(np.float32), f"ghap40_{j}": x40.astype(np.float32), f"pop_{j}": pop,
                    f"cams_{j}": nearest(np.asarray(cams[t, 0, 0], np.float32), grid, lats, lons),
                    f"aurora_{j}": nearest(0.25 * an + 0.5 * A[0] + 0.25 * A[1], grid, lats, lons)})
        above = np.nan_to_num(x1) > THR
        date = (np.datetime64("2022-01-01") + np.timedelta64(t, "D")).astype(object)
        hidden = (pop * above * ~(x40 > THR)).sum()
        print(f"{zn}: day {t} ({date:%d %b}) | GHAP max {np.nanmax(x1):.0f} | residents above {THR:g}: {(pop * above).sum() / 1e6:.2f} M, hidden at 40 km {hidden / 1e6:.2f} M"
              f" | warned by CAMS {(pop * above * (res[f'cams_{j}'] > THR)).sum() / 1e6:.2f} M, Aurora {(pop * above * (res[f'aurora_{j}'] > THR)).sum() / 1e6:.2f} M",
              flush=True)
    np.savez(OUT, zones=np.array(list(ZONES)), thr=THR, **res)


def add_foehn(ev):
    z = dict(np.load(OUT))
    for j, (zn, box) in enumerate(ZONES.items()):
        t = int(z[f"day_{j}"])
        c = np.load(ev / f"crop{j}_d{t - 1}_h1.npz")
        assert np.allclose(c["box"], box), (c["box"], box)
        f = c["members"].astype(np.float32).mean(0)
        z[f"foehn_{j}"] = np.where(np.isfinite(z[f"ghap_{j}"]), f, np.nan)
        above = np.nan_to_num(z[f"ghap_{j}"]) > THR; pop = z[f"pop_{j}"]
        print(f"{zn}: day {t}: warned by Foehn {(pop * above * (f > THR)).sum() / 1e6:.2f} M of {(pop * above).sum() / 1e6:.2f} M")
    np.savez(OUT, **z)


if __name__ == "__main__":
    add_foehn(Path(sys.argv[2])) if len(sys.argv) > 1 and sys.argv[1] == "foehn" else main()
