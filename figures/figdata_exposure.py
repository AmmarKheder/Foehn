#!/usr/bin/env python3
"""What a 40 km grid hides from the people exposed (data only, no forecast): every day of 2022, the daily PM2.5 of GHAP
at 1 km (pixels with a retrieval) against the same field averaged over 40 km cells (0.4 deg, the CAMS / Aurora grid: 40 x 40
pixels; every resident of a cell gets the cell mean, as a 40 km forecast would give), weighted by the residents of every
1 km pixel (GHS-POP 2020, pop1km_global.npy). For the WHO 2021 24-h guideline (15 ug m-3) and interim targets (25, 37.5,
50, 75): residents above the threshold at 1 km, at 40 km, above at 1 km but not at 40 km (hidden), and the reverse
(spurious), globally and by region; and, for the map, the region-day with the most hidden residents above 75.
One rank per process (srun); every rank takes days rank, rank + world, ...  -> data/exposure/part_<rank>.npz
    srun -n 8 python figdata_exposure.py ;  python figdata_exposure.py merge"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
OUT = HERE / "data" / "exposure"
GH, GW, T, NTR, NTC = 14400, 36000, 512, 29, 71
I2022 = 1826
THR = np.array([15.0, 25.0, 37.5, 50.0, 75.0], np.float32)
REGIONS = {"World": None, "South Asia": (37, 5, 60, 97), "East Asia": (54, 18, 97, 146), "Africa": (37, -35, -20, 52),
           "Middle East": (42, 12, 35, 63), "Southeast Asia": (18, -11, 92, 142), "Europe": (72, 35, -25, 45),
           "North America": (72, 15, -170, -50), "South America": (15, -56, -95, -30)}
F = 40                                                                       # 1 km pixels per 40 km cell side (0.4 deg)


def read_rows(path, n):
    """row n of a .npy (pread, in 1 GB pieces)"""
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); sh, _, dt = np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    size = int(np.prod(sh[1:])) * dt.itemsize
    fd = os.open(path, os.O_RDONLY); parts, got = [], 0
    try:
        while got < size:
            b = os.pread(fd, min(size - got, 1 << 30), hdr + n * size + got); parts.append(b); got += len(b)
    finally:
        os.close(fd)
    return np.frombuffer(b"".join(parts), dt).reshape(sh[1:])


def ghap_day(i):
    a = read_rows(G / "ghap1km_tiles.npy", i)                                 # (29, 71, 512, 512) float16
    return a.transpose(0, 2, 1, 3).reshape(NTR * T, NTC * T)[:GH, :GW]


def region_masks():
    """per region, (rows, cols) slices of the 40 km cell grid and a cell mask (cells whose centre is inside the box)"""
    lat = 84.0 - 0.4 * (np.arange(GH // F) + 0.5); lon = -180.0 + 0.4 * (np.arange(GW // F) + 0.5)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    return {r: (np.ones(LA.shape, bool) if b is None else (LA <= b[0]) & (LA > b[1]) & (LO >= b[2]) & (LO < b[3])) for r, b in REGIONS.items()}


def run():
    rank, world = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    OUT.mkdir(parents=True, exist_ok=True)
    pop = np.load(G / "pop1km_global.npy", mmap_mode="r")
    pop = np.asarray(pop, np.float32).reshape(GH // F, F, GW // F, F)            # residents per 1 km pixel, cell blocks
    masks = region_masks(); R = list(REGIONS)
    days = list(range(365))[rank::world]
    res = np.zeros((len(days), len(R), 4, len(THR)))                            # above 1 km, above 40 km, hidden, spurious
    pop_valid = np.zeros((len(days), len(R)))
    best = []                                                                    # (hidden above 75, day, region cell idx)
    maps = np.zeros((2, len(THR), GH // F, GW // F))                            # annual person-days per 40 km cell: above at 1 km, hidden
    for n, d in enumerate(days):
        x = ghap_day(I2022 + d).astype(np.float32).reshape(GH // F, F, GW // F, F)
        ok = x > 0
        with np.errstate(all="ignore"):
            cell = (np.where(ok, x, 0).sum((1, 3)) / ok.sum((1, 3)))             # 40 km cell means over valid pixels
        x40 = np.broadcast_to(cell[:, None, :, None], x.shape)
        w = np.where(ok & np.isfinite(x40), pop, 0.0)
        for k, thr in enumerate(THR):
            a1, a4 = x > thr, x40 > thr
            terms = [(w * a1).sum((1, 3)), (w * a4).sum((1, 3)), (w * (a1 & ~a4)).sum((1, 3)), (w * (~a1 & a4)).sum((1, 3))]
            for g, r in enumerate(R):
                m = masks[r]
                res[n, g, :, k] = [t_[m].sum() for t_ in terms]
            maps[0, k] += terms[0]; maps[1, k] += terms[2]
            if thr == 75.0:
                hid = terms[2]
                best.append((float(hid.max()), d, int(np.argmax(hid))))
        pop_valid[n] = [w.sum((1, 3))[masks[r]].sum() for r in R]
        print(f"rank {rank}: day {d} world above 75: 1 km {res[n, 0, 0, 4] / 1e6:.0f} M, 40 km {res[n, 0, 1, 4] / 1e6:.0f} M, "
              f"hidden {res[n, 0, 2, 4] / 1e6:.0f} M", flush=True)
    np.savez(OUT / f"part_{rank}.npz", days=np.array(days), res=res, pop_valid=pop_valid, regions=np.array(R), thr=THR,
             best=np.array(best), maps=maps.astype(np.float32))


def merge():
    P = [np.load(p) for p in sorted(OUT.glob("part_*.npz"))]
    days = np.concatenate([p["days"] for p in P]); o = np.argsort(days)
    res = np.concatenate([p["res"] for p in P])[o]; pv = np.concatenate([p["pop_valid"] for p in P])[o]
    best = np.concatenate([p["best"] for p in P]); R = [str(r) for r in P[0]["regions"]]
    maps = sum(p["maps"].astype(np.float64) for p in P)
    np.savez(HERE / "data" / "exposure_2022.npz", days=days[o], res=res, pop_valid=pv, regions=np.array(R), thr=THR, best=best,
             maps=maps.astype(np.float32))
    print(f"{len(days)} days of 2022 | person-days (billions) above each daily threshold, 1 km vs 40 km cells")
    for g, r in enumerate(R):
        row = []
        for k, thr in enumerate(THR):
            a1, a4, hid, spu = (res[:, g, j, k].sum() / 1e9 for j in range(4))
            row.append(f">{thr:g}: 1 km {a1:6.1f} | 40 km {a4:6.1f} | hidden {hid:5.1f} ({100 * hid / max(a1, 1e-9):.0f} %) | spurious {spu:5.1f}")
        print(f"\n{r}:\n  " + "\n  ".join(row))
    b = best[np.argsort(-best[:, 0])[:5]]
    print("\ndays with the most hidden residents above 75 in one 40 km cell (residents, day of year, cell index):")
    for h, d, c in b:
        print(f"  {h / 1e6:.2f} M  doy {int(d)}  cell row {int(c) // 900} col {int(c) % 900} (lat {84 - 0.4 * (int(c) // 900 + 0.5):.1f}, "
              f"lon {-180 + 0.4 * (int(c) % 900 + 0.5):.1f})")


if __name__ == "__main__":
    merge() if len(sys.argv) > 1 and sys.argv[1] == "merge" else run()
