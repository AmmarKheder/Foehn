#!/usr/bin/env python3
"""Alerts for the next day, 2022 (teaser, Fig. 1f): of the residents (GHS-POP 2020) whose 1 km pixel is above a WHO 24-h
threshold on day t (GHAP, the reference), how many were warned the day before, and how many warned residents were not above
it. Population-weighted person-days over the 8.1 M pixels where >= 100 people live (eval_global_green.py --pop), valid
days t = 1..364, every forecast issued at 00 UTC of day t (equal issuance):
  * Foehn 1 km: ensemble mean at the resident's pixel; Foehn 40 km: its mean over the 40 km cell (resolution control);
  * CAMS: run of 00 UTC of day t, lead day 0, 0.4 deg cell of the pixel; Aurora: initialised at 00 UTC of day t (as
    figdata_teaser_fc.py), 0.4 deg cell of the pixel;
  * a perfect 40 km forecast: GHAP of day t averaged over the 40 km cell (the limit of any 40 km forecast).
Per region (figdata_exposure.py) and threshold: exposed E, hits H and false alarms F (person-days), and the world per day.
    python figdata_alerts.py <eval dir> [output name]  -> data/alerts_2022_h1.npz"""
import glob
import sys
from pathlib import Path

import numpy as np
import zarr

from figdata_exposure import REGIONS

HERE = Path(__file__).resolve().parent
AURD = HERE.parent / "benchmark" / "aurora"
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
GW = 36000
THRS = (15.0, 25.0, 37.5, 50.0, 75.0)
SYSTEMS = ["Foehn (1 km)", "Foehn (40 km cells)", "CAMS (40 km)", "Aurora (40 km)", "perfect 40 km forecast"]


def main(ev):
    idx = np.load(ev / "pop_idx.npy")
    pop = np.load(G / "pop1km_global.npy", mmap_mode="r").reshape(-1)[idx].astype(np.float64)
    r, c = idx // GW, idx % GW
    lat, lon = 84.0 - 0.01 * (r + 0.5), -180.0 + 0.01 * (c + 0.5)
    ci, cj = np.round((90.0 - lat) / 0.4).astype(int), np.round((lon % 360.0) / 0.4).astype(int) % 900      # CAMS / Aurora cell
    c40 = (r // 40) * 900 + c // 40                                                                          # 40 km cell (84N 180W)
    masks = {k: (np.ones(len(idx), bool) if b is None else (lat <= b[0]) & (lat > b[1]) & (lon >= b[2]) & (lon < b[3]))
             for k, b in REGIONS.items()}
    R = list(REGIONS)
    combos, label = np.unique(np.stack(list(masks.values()), 1), axis=0, return_inverse=True)   # overlapping regions: their
    label = label.reshape(-1); M = combos.astype(np.float64); K = len(M)                        # combinations, summed by M
    cams = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
    aur = {}
    for p in sorted(glob.glob(str(AURD / "out_12h_all" / "part_*.npz"))):
        z = np.load(p)
        for d, pr in zip(z["days"], z["pred"]):
            aur[int(d)] = pr[:2]
    S = np.zeros((len(SYSTEMS), len(R), len(THRS), 3))                       # E, H, F (person-days)
    days = []
    WD = np.zeros((364, len(SYSTEMS), len(THRS), 3))                          # world, per valid day
    for p in sorted(ev.glob("part_*.npz")):
        rk = p.stem.split("_")[1]
        keys = np.load(p)["keys"]
        ens = np.load(ev / f"pop_{rk}.npy", mmap_mode="r"); gh = np.load(ev / f"pop_ghap_{rk}.npy", mmap_mode="r")
        cl = np.load(ev / f"cell40_{rk}.npy", mmap_mode="r")
        assert len(keys) == len(ens) == len(gh) == len(cl), (p, len(keys), len(ens), len(gh), len(cl))
        for n, (d0, h, t) in enumerate(keys):
            assert h == 1 and t == d0 + 1
            days.append(int(t))
            g = gh[n].astype(np.float32); ok = np.isfinite(g); w = np.where(ok, pop, 0.0)
            an = np.load(AURD / "raw" / f"{t:03d}.npz")["surf"][0, 5, :450].astype(np.float32) * 1e9
            au = 0.25 * an + 0.5 * aur[int(t)][0].astype(np.float32) + 0.25 * aur[int(t)][1].astype(np.float32)
            F = [ens[n].astype(np.float32).mean(-1), cl[n, 0].astype(np.float32).reshape(-1)[c40],
                 np.asarray(cams[int(t), 0, 0], np.float32)[ci, cj], au[np.minimum(ci, 449), cj], cl[n, 1].astype(np.float32).reshape(-1)[c40]]
            for k, thr in enumerate(THRS):
                above = ok & (g > thr)
                e = np.bincount(label, weights=w * above, minlength=K)
                for s, f in enumerate(F):
                    warn = f > thr                                                   # NaN (no 40 km cell value): no alert
                    v = np.stack([e, np.bincount(label, weights=w * (above & warn), minlength=K),
                                  np.bincount(label, weights=w * (ok & ~above & warn), minlength=K)])      # (3, K)
                    S[s, :, k] += (v @ M).T
                    WD[int(t) - 1, s, k] = v.sum(1)
        print(p.name, len(keys), "forecasts", flush=True)
    days = sorted(days)
    print(f"{len(days)} days ({days[0]}..{days[-1]}), {len(set(days))} distinct")
    np.savez(HERE / "data" / (sys.argv[2] if len(sys.argv) > 2 else "alerts_2022_h1.npz"), S=S, WD=WD, systems=np.array(SYSTEMS), regions=np.array(R), thr=np.array(THRS),
             days=np.array(days), cols=np.array(["exposed", "hits", "false_alarms"]))
    k = THRS.index(37.5)
    print(f"world, 37.5: exposed {S[0, 0, k, 0] / len(days) / 1e6:.0f} M people per day")
    for s, nm in enumerate(SYSTEMS):
        E, H, Fa = S[s, 0, k]
        print(f"  {nm:24s} warned {100 * H / E:5.1f}% | false alerts {100 * Fa / max(H + Fa, 1):5.1f}% of the warned | CSI {H / (E + Fa):.3f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
