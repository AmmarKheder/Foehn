#!/usr/bin/env python3
"""Annual means 2022 of the forecasts for the next day, for the teaser (fig1_teaser_v4.py). Valid days t = 1..364 (2 January
to 31 December 2022), every system issued at the end of day t-1 (00 UTC of day t):
  * Foehn: ensemble mean at 1 km (eval_global_green.py --leads 1 --annual), and GHAP on the same pixel-days;
  * CAMS: run of 00 UTC of day t, lead day 0 (daily mean), 0.4 deg;
  * Aurora: initialised at 00 UTC of day t from the CAMS analyses (out_12h_all), daily mean 0.25 x analysis of 00 UTC
    + 0.5 x 12 UTC + 0.25 x 00 UTC of day t + 1 (as compare_fair.py), 0.4 deg.
And the days above 37.5 ug/m3 (WHO 2021 24-h interim target 3) of each.
    python figdata_teaser_fc.py coarse            -> data/fc_annual_2022/coarse.npz
    python figdata_teaser_fc.py foehn <eval dir>  -> data/fc_annual_2022/{foehn,ghap}_{mean,days37}.npy, n.npy (1 km),
                                                     world01.npz (0.1 deg means over the valid 1 km pixels)
    (--partial: a smoke test with a few days, to data/fc_annual_2022_partial)"""
import glob
import sys
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "fc_annual_2022"
AURD = HERE.parent / "benchmark" / "aurora"
GH, GW, THR = 14400, 36000, 37.5
DAYS = list(range(1, 365))


def coarse():
    cams = zarr.open("/scratch/project_462001140/ammar/eccv/data/world/cams_forecast_daily/2022.zarr", mode="r")
    aur = {}
    for p in sorted(glob.glob(str(AURD / "out_12h_all" / "part_*.npz"))):
        z = np.load(p)
        assert float(z["lat"][0]) == 90.0 and float(z["lon"][0]) == 0.0 and list(z["leads"][:2]) == [12, 24], p
        for d, pr in zip(z["days"], z["pred"]):
            aur[int(d)] = pr[:2].astype(np.float32)                      # 12 and 24 h after 00 UTC of day d
    sc, sa = np.zeros((451, 900)), np.zeros((450, 900))
    xc, xa = np.zeros((451, 900), np.int32), np.zeros((450, 900), np.int32)
    for t in DAYS:
        c = np.asarray(cams[t, 0, 0], np.float64); sc += c; xc += c > THR
        an = np.load(AURD / "raw" / f"{t:03d}.npz")["surf"][0, 5, :450].astype(np.float64) * 1e9   # analysis 00 UTC, ug m-3
        a = 0.25 * an + 0.5 * aur[t][0] + 0.25 * aur[t][1]; sa += a; xa += a > THR
        if t % 60 == 0:
            print("day", t, flush=True)
    np.savez(OUT / "coarse.npz", cams_mean=(sc / len(DAYS)).astype(np.float32), cams_days37=xc,
             aurora_mean=(sa / len(DAYS)).astype(np.float32), aurora_days37=xa, days=np.array(DAYS), grid=np.array([90.0, 0.4, 0.0, 0.4]))
    print("saved coarse: CAMS / Aurora global mean", sc.mean() / len(DAYS), sa.mean() / len(DAYS))


def foehn(ev):
    idx = np.load(ev / "annual_idx.npy")
    S = {k: np.zeros(len(idx), np.float64 if k in ("f", "g") else np.int32) for k in ("f", "g", "n", "xf", "xg")}
    days = []
    for p in sorted(ev.glob("annual_*.npz")):
        z = np.load(p); assert list(z["leads"]) == [1] and float(z["thr"]) == THR, p
        for k in S:
            S[k] += z[f"{k}_1"]
        days += [int(t) for d, h, t in np.load(ev / f"part_{p.stem.split('_')[1]}.npz")["keys"]]
    assert len(days) == len(set(days)) and (PARTIAL or sorted(days) == DAYS), (len(days), sorted(set(DAYS) - set(days))[:10])
    n = S["n"]; ok = n > 0
    assert n.max() <= len(DAYS)
    print(f"{len(days)} days; valid land pixels {ok.mean():.4f}, days per pixel: median {np.median(n[ok]):.0f}, min {n[ok].min()}", flush=True)

    def full(v, dtype, fill):
        m = np.full(GH * GW, fill, dtype); m[idx] = v
        return m.reshape(GH, GW)
    with np.errstate(all="ignore"):
        V = {"foehn_mean": S["f"] / n, "ghap_mean": S["g"] / n, "foehn_frac37": S["xf"] / n, "ghap_frac37": S["xg"] / n}
    for k in ("foehn_mean", "ghap_mean"):
        np.save(OUT / f"{k}.npy", full(np.where(ok, V[k], np.nan).astype(np.float16), np.float16, np.nan))
    for k in ("foehn", "ghap"):
        np.save(OUT / f"{k}_days37.npy", full(S[f"x{k[0]}"].astype(np.uint16), np.uint16, 0))
    np.save(OUT / "n.npy", full(n.astype(np.uint16), np.uint16, 0))
    b = (idx // GW // 10) * (GW // 10) + (idx % GW) // 10                  # 0.1 deg block of every land pixel
    cnt = np.bincount(b[ok], minlength=GH * GW // 100)
    w01 = {}
    for k, v in V.items():
        s = np.bincount(b[ok], weights=v[ok], minlength=GH * GW // 100)
        w01[k] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan).reshape(GH // 10, GW // 10).astype(np.float32)
    np.savez(OUT / "world01.npz", days=np.array(sorted(days)), **w01)
    with np.errstate(all="ignore"):
        print("saved: land mean Foehn %.2f GHAP %.2f; share of days above 37.5: Foehn %.3f GHAP %.3f" % (
            np.nanmean(V["foehn_mean"][ok]), np.nanmean(V["ghap_mean"][ok]), np.mean(V["foehn_frac37"][ok]), np.mean(V["ghap_frac37"][ok])))


if __name__ == "__main__":
    PARTIAL = "--partial" in sys.argv                                         # smoke test: a few days, written aside
    if PARTIAL:
        OUT = OUT.with_name(OUT.name + "_partial")
    OUT.mkdir(parents=True, exist_ok=True)
    coarse() if sys.argv[1] == "coarse" else foehn(Path(sys.argv[2]))
