#!/usr/bin/env python3
"""Rehearsal of the calibrated 2022 pass on SELECTION-YEAR data only (2021; no value of 2022, no station file).
  positions   mech2021/st_ring_2022.npz + st_pix_2022.npz: PSEUDO-monitors = 400 mask pixels drawn at random (seed 0) with
              their 1/3/10 km rings, in the format test2022_ghap/stations_2022.py writes (the ring construction is copied
              from it). The file names are the ones the scorers expect; nothing in them comes from a station.
  truth       after the predictions: for each system dir (mech2021/frozen, mech2021/cal) the scorer's test layout:
              preds/<arm>/2022-01-<k>.npz -> symlinks to the k-th predicted 2021 day (the scorers count days from
              2022-01-01; in the rehearsal "2022-01-01 + k" is only the name of the k-th test day), and in mech2021:
              truth.npy (K, N_SUB) = the stored 2021 hold-out field at every 8th mask pixel (NaN = hole), truth_st.npy
              (K, n positions) = the same at the pseudo-monitor and ring positions, st_truth_2022.npz = pseudo-observations
              = the 1 km estimate of 2021 at the pseudo-monitor pixel (network "PSEUDO"), days.json.
  python -I test2022_cal/mech_setup.py positions | truth"""
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/project_462001804/ammar/global_model_forecasting_air_pollution")
D = ROOT / "test2022_cal/mech2021"
GH = ROOT / "data/processed/ghap"
RINGS = (1, 3, 10)
NPSEUDO = 400


def read_row(path, i):
    a = np.load(path, mmap_mode="r")
    return np.asarray(a[i])


def positions():
    D.mkdir(parents=True, exist_ok=True)
    mask = np.load(GH / "grid/ghap_mask.npy").astype(np.int64)
    pos = np.sort(np.random.default_rng(0).choice(mask.size, NPSEUDO, replace=False)).astype(np.int64)
    mgid = mask[pos]
    r0, c0 = mgid // 36000, mgid % 36000
    ring_g = {}
    for dd in RINGS:                                   # stations_2022.py, A2(e) ring construction
        off = np.array([(u, v) for u in range(-dd, dd + 1) for v in range(-dd, dd + 1) if max(abs(u), abs(v)) == dd], np.int64)
        rr = r0[:, None] + off[None, :, 0]
        cc = (c0[:, None] + off[None, :, 1]) % 36000
        g = np.where((rr >= 0) & (rr < 14400), rr * 36000 + cc, -1)
        q = np.searchsorted(mask, g)
        ok = (g >= 0) & (q < mask.size) & (mask[np.minimum(q, mask.size - 1)] == g)
        ring_g[dd] = np.where(ok, q, -1)
    allpos = np.unique(np.concatenate([pos] + [v[v >= 0] for v in ring_g.values()]))
    ring_idx = {f"ring{dd}": np.where(v >= 0, np.searchsorted(allpos, np.maximum(v, 0)), -1) for dd, v in ring_g.items()}
    np.savez(D / "st_pix_2022.npz", pos=pos)
    np.savez(D / "st_ring_2022.npz", pos=allpos.astype(np.int64), mon_idx=np.searchsorted(allpos, pos), rings=np.array(RINGS), **ring_idx)
    print(f"[positions] {NPSEUDO} pseudo-monitor pixels, {allpos.size} positions with rings -> {D}/st_ring_2022.npz, st_pix_2022.npz")


def truth():
    hold = GH / "ghap_eval/full_2021.npy"              # the stored 2021 hold-out field (selection year), mask order
    systems = [s for s in ("frozen", "cal") if (D / s / "preds" / "cdffm").is_dir()]
    days = sorted(p.name[:-4] for p in (D / systems[0] / "preds" / "cdffm").glob("2021-*.npz"))
    for s in systems:
        assert sorted(p.name[:-4] for p in (D / s / "preds" / "cdffm").glob("2021-*.npz")) == days, f"{s}: other days"
    K = len(days)
    for s in systems:                                  # the scorers' day names
        for sub in ("preds", "preds_st"):
            for arm in sorted(p.name for p in (D / s / sub).iterdir() if p.is_dir()):
                for k, d in enumerate(days):
                    ln = D / s / sub / arm / f"{dt.date(2022, 1, 1) + dt.timedelta(k)}.npz"
                    if ln.is_symlink() or ln.exists():
                        ln.unlink()
                    os.symlink(f"{d}.npz", ln)
    Z = np.load(D / "st_ring_2022.npz")
    allpos, mon = Z["pos"], np.load(D / "st_pix_2022.npz")["pos"]
    tr, ts = [], []
    for d in days:
        row = read_row(hold, (dt.date.fromisoformat(d) - dt.date(2021, 1, 1)).days).astype(np.float32)
        row[row < 0] = np.nan                           # -1 = hole
        tr.append(row[::8])
        ts.append(row[allpos])
    np.save(D / "truth.npy", np.stack(tr).astype(np.float16))
    np.save(D / "truth_st.npy", np.stack(ts).astype(np.float16))
    daily = np.full((NPSEUDO, 365), np.nan, np.float32)
    daily[:, :K] = np.stack(ts)[:, np.searchsorted(allpos, mon)].T
    mask = np.load(GH / "grid/ghap_mask.npy", mmap_mode="r")
    flat = np.asarray(mask[mon]).astype(np.int64)
    np.savez(D / "st_truth_2022.npz", id=np.arange(NPSEUDO), network=np.array(["PSEUDO"] * NPSEUDO), lat=(83.995 - 0.01 * (flat // 36000)).astype(np.float32),
             lon=(-179.995 + 0.01 * (flat % 36000)).astype(np.float32), pix=np.arange(NPSEUDO), daily=daily, daily12=daily)
    (D / "days.json").write_text(json.dumps({"days_2021": days, "named": [str(dt.date(2022, 1, 1) + dt.timedelta(k)) for k in range(K)], "systems": systems,
                                             "note": "rehearsal: 2021 selection days under the scorers' day names; pseudo-monitors, no station"}, indent=1))
    print(f"[truth] {K} days {days[0]}..{days[-1]} for {systems}: truth.npy {np.stack(tr).shape}, truth_st.npy {np.stack(ts).shape}, pseudo-observations at {NPSEUDO} pixels")


if __name__ == "__main__":
    {"positions": positions, "truth": truth}[sys.argv[1]]()
