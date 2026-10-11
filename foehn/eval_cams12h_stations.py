#!/usr/bin/env python3
"""'Foehn trained as Aurora', judged against the real air: the forecasts of the CAMS protocol (initialised 00 UTC, 1 May -
30 Nov 2022, 12-hour steps, 0.4 deg) scored at the stations of the six official networks (evaluation only).
Daily means of day d+n (n = 1..4) from the instantaneous fields as for Aurora before: 0.25 x (00 UTC d+n) + 0.5 x
(12 UTC d+n) + 0.25 x (00 UTC d+n+1), i.e. leads (24n, 24n+12, 24n+24) h; read at the stations bilinearly. Systems on
identical station-days: Foehn-CAMS (8 members: ensemble mean, fair CRPS), Aurora (same CAMS analyses), the CAMS
operational forecast (00 UTC runs) and persistence of the station's own mean of day d-1 (the last complete day at t0).
RMSE, bias, CRPS (= absolute error for single forecasts), per network and lead; 7-day block bootstrap of RMSE ratios.
    python eval_cams12h_stations.py [--run cams04_flow_shift2.6_emis]   -> ../figures/data/cams12h_stations.npz"""
import argparse
import glob
from pathlib import Path

import numpy as np

from eval_cams12h import AUR, CFC, LAT, LON, NL, NX, NY, to_domain

HERE = Path(__file__).resolve().parent
G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km")
NETS = {0: "Europe (EEA)", 1: "China (CNEMC)", 3: "USA (EPA)", 4: "Korea (AirKorea)", 5: "Japan (NIES)", 6: "India (CPCB)"}
W3 = (0.25, 0.5, 0.25)


def main(a):
    import os
    STATIONS = os.environ.get("STATIONS", "stations_2022.npz")              # UTC-day means: stations_2022_utc.npz
    st = np.load(G / STATIONS)
    keep = np.isin(st["source"], list(NETS)) & (st["lat"] < 84.0) & (st["lat"] > -60.0)
    slat, slon, src, obs = st["lat"][keep], st["lon"][keep], st["source"][keep], st["daily"][keep].astype(np.float32)
    fi = (84.0 - slat) / 0.4; fj = ((slon + 180.0) % 360.0) / 0.4                      # domain grid: 84 - 0.4 i, -180 + 0.4 j
    i0 = np.clip(np.floor(fi).astype(int), 0, NY - 2); wi = (fi - i0).astype(np.float32)
    j0 = np.floor(fj).astype(int) % NX; j1 = (j0 + 1) % NX; wj = (fj - np.floor(fj)).astype(np.float32)
    at_st = lambda f: ((1 - wi) * ((1 - wj) * f[..., i0, j0] + wj * f[..., i0, j1]) + wi * ((1 - wj) * f[..., i0 + 1, j0] + wj * f[..., i0 + 1, j1]))
    fdir = HERE / "runs" / a.run / a.out
    fdoys = list(np.load(fdir / "doys.npy")); E_all = np.load(fdir / "pred.npy", mmap_mode="r")
    aur = {}
    for p in sorted(glob.glob(str(AUR / "part_*.npz"))):
        z = np.load(p)
        for d, pr in zip(z["days"], z["pred"]):
            aur[int(d)] = at_st(to_domain(pr.astype(np.float32), z["lat"], z["lon"]))           # (10, n_st)
    cfc = {}
    for p in sorted(CFC.glob("2022_*.npz")):
        z = np.load(p); m = int(p.name[5:7])
        f = to_domain(z["pm25"].astype(np.float32), z["lat"], z["lon"])                         # (days, 10, NY, NX)
        first = int((np.datetime64(f"2022-{m:02d}-01") - np.datetime64("2022-01-01")).astype(int))
        for k in range(f.shape[0]):
            cfc[first + k] = at_st(f[k])
        print("CAMS month", m, flush=True)
    doys = [d for d in fdoys if d in aur and d in cfc]
    SYS = ["Foehn-CAMS (mean)", "Aurora", "CAMS forecast", "persistence (d-1)"]
    # per system, lead, network: sums over station-days of se, err, |err| (CRPS for Foehn), n; and per-init RMSE parts
    acc = np.zeros((len(SYS), 4, len(NETS) + 1, 4))
    per = np.full((len(SYS), len(doys), 4, 2), np.nan)                                          # (sse, n) per init, all nets
    nets = list(NETS)
    for c_, d in enumerate(doys):
        E = at_st(np.asarray(E_all[fdoys.index(d)], np.float32))                                # (M, 10, n_st)
        A, C = aur[d], cfc[d]
        for n in range(1, 5):
            ks = (2 * n - 1, 2 * n, 2 * n + 1)                                                  # leads 24n, 24n+12, 24n+24 h
            if d + n >= 365:
                continue
            y = obs[:, d + n]
            em = sum(w * E[:, k] for w, k in zip(W3, ks))                                       # (M, n_st) daily-mean members
            fc = [em.mean(0), sum(w * A[k] for w, k in zip(W3, ks)), sum(w * C[k] for w, k in zip(W3, ks)), obs[:, d - 1]]
            ok = np.isfinite(y) & np.all([np.isfinite(f) for f in fc], 0)
            M = em.shape[0]
            es = np.sort(em, 0)
            crps_f = np.abs(em - y).mean(0) - (2 * (2 * np.arange(M) - M + 1)[:, None] * es).sum(0) / (2 * M * (M - 1))
            for s, f in enumerate(fc):
                e = f - y
                cr = crps_f if s == 0 else np.abs(e)
                for g, net in enumerate(nets + [None]):
                    m = ok & (src == net) if net is not None else ok
                    acc[s, n - 1, g] += [np.nansum(e[m] ** 2), np.nansum(e[m]), np.nansum(cr[m]), m.sum()]
                per[s, c_, n - 1] = [np.nansum(e[ok] ** 2), ok.sum()]
        if c_ % 40 == 0:
            print(f"{c_}/{len(doys)}", flush=True)
    names = [NETS[k] for k in nets] + ["All six networks"]
    np.savez(HERE.parent / "figures" / "data" / f"cams12h_stations{__import__('os').environ.get('TAG', '')}.npz", acc=acc, per=per, systems=np.array(SYS), networks=np.array(names),
             doys=np.array(doys))
    for g, nm in reversed(list(enumerate(names))):
        print(f"\n=== {nm}: RMSE / bias / CRPS (ug m-3), daily means, identical station-days")
        for s, sn in enumerate(SYS):
            print(f"{sn:20s}" + "".join(f"   J+{n + 1} {np.sqrt(acc[s, n, g, 0] / acc[s, n, g, 3]):5.1f} {acc[s, n, g, 1] / acc[s, n, g, 3]:+5.1f} "
                                        f"{acc[s, n, g, 2] / acc[s, n, g, 3]:5.2f}" for n in range(4)))
        print(f"{'station-days':20s}" + "".join(f"   J+{n + 1} {int(acc[0, n, g, 3]):>16d}" for n in range(4)))
    rng = np.random.default_rng(0)
    nd, B = len(doys), 7
    idx = (rng.integers(0, nd - B + 1, size=(10000, -(-nd // B)))[:, :, None] + np.arange(B)).reshape(10000, -1)[:, :nd]
    for j in (1, 2):
        r = []
        for n in range(4):
            a0, a1 = per[0, :, n], per[j, :, n]
            ok = np.isfinite(a0[:, 0]) & np.isfinite(a1[:, 0])
            ratio = lambda ix: np.sqrt(np.nansum(a0[ix, 0], 1) / np.nansum(a0[ix, 1], 1)) / np.sqrt(np.nansum(a1[ix, 0], 1) / np.nansum(a1[ix, 1], 1)) - 1
            bs = ratio(idx)
            pt = np.sqrt(np.nansum(a0[ok, 0]) / np.nansum(a0[ok, 1])) / np.sqrt(np.nansum(a1[ok, 0]) / np.nansum(a1[ok, 1])) - 1
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            r.append(f"J+{n + 1} {100 * pt:+.0f}% [{100 * lo:+.0f},{100 * hi:+.0f}]")
        print(f"Foehn-CAMS vs {SYS[j]} (RMSE, all networks, 95% CI): " + " ".join(r))
    print("saved")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cams04_flow_shift2.6_emis")
    ap.add_argument("--out", default="cams12h")
    main(ap.parse_args())
