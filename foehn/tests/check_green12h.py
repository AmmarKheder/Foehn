#!/usr/bin/env python3
"""Integrity of the FLEXGREEN store of 'Foehn trained as Aurora' (green12h: --instant --all-cells, 32 particles, emis),
day by day, and the split of foehn_cams.py. Per finished day, three receptor rows (60 N, 20 N, 20 S) of the 10 leads:
no NaN, positions on the sphere, share of receptors whose air survived (>= 0.8 per row), mean survival in (0.05, 0.95), origins
moving away from the receptors with lead; fire and emission terms finite and >= 0. Then: required days missing, and
the train / test samples of foehn_cams.CamsData (no target at or after 00 UTC 1 May 2022 in training, 214 x 10 test).
    python tests/check_green12h.py [--split]      (--split needs the training container: tests/cpu_container.sh)"""
import os
import sys
from pathlib import Path

import numpy as np

G = Path("/scratch/project_462001140/ammar/eccv/data/world/global1km/green12h/hres")
NR, NC, NL = 288, 720, 10
ROWS = (48, 128, 208)                                    # receptor rows at 59.9 N, 19.9 N, 20.1 S
REQ = {2017: range(274, 365), 2018: range(365), 2019: range(365), 2020: range(366), 2021: range(365), 2022: range(334)}
RLAT = 84.0 - 0.125 - 0.5 * np.arange(NR)
RLON = -180.0 + 0.125 + 0.5 * np.arange(NC)


def header(path):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); sh, _, dt = np.lib.format.read_array_header_1_0(f); return f.tell(), sh, dt


def pread(path, hdr, offset_items, count, dt):
    fd = os.open(path, os.O_RDONLY)
    try:
        b = os.pread(fd, count * dt.itemsize, hdr + offset_items * dt.itemsize)
    finally:
        os.close(fd)
    assert len(b) == count * dt.itemsize, (path, offset_items)
    return np.frombuffer(b, dt)


def check_year(y):
    po, pf, pe = G / f"origins_{y}.npy", G / f"fire_{y}.npy", G / f"emis_{y}.npy"
    (ho, so, do), (hf, sf, df), (he, se, de) = header(po), header(pf), header(pe)
    P = so[4]
    done = {int(p.name) for p in (G / f"done_{y}").glob("*")}
    missing = sorted(set(REQ[y]) - done)
    bad, stats = [], []
    for d in sorted(done & set(REQ[y])):
        surv_share, msurv, disp = [], [], []
        for li in range(NL):
            for r in ROWS:
                o = pread(po, ho, (((d * NL + li) * NR + r) * NC) * P * 4, NC * P * 4, do).reshape(NC, P, 4).astype(np.float32)
                if not np.isfinite(o).all():
                    bad.append((d, li, r, f"nan in {int((~np.isfinite(o)).any(-1).any(-1).sum())} receptors"))
                    o = np.nan_to_num(o)
                lon, lat, zr, sv = np.moveaxis(o, -1, 0)
                if np.abs(lat).max() > 90.01 or np.abs(lon).max() > 180.01 or sv.min() < 0 or sv.max() > 1.001:
                    bad.append((d, li, r, "range"))
                surv_share.append((sv.sum(-1) > 0).mean()); msurv.append(sv.mean())
                dl = (lon - RLON[:, None] + 180.0) % 360.0 - 180.0
                disp.append(np.hypot(dl * np.cos(np.deg2rad(RLAT[r])), lat - RLAT[r]).mean())
        ss = np.array(surv_share).reshape(NL, len(ROWS)); ms = np.array(msurv).reshape(NL, len(ROWS))
        dp = np.array(disp).reshape(NL, len(ROWS)).mean(1)
        if ss.min() < 0.8:                               # wet season at 20 S: up to 11 % of cells lose all aerosol by 108-120 h
            bad.append((d, int(np.argmin(ss.min(1))), -1, f"survivors {ss.min():.2f}"))
        if not (0.05 < ms.mean() < 0.95):
            bad.append((d, -1, -1, f"mean survival {ms.mean():.2f}"))
        if not dp[-1] > dp[0]:
            bad.append((d, -1, -1, f"displacement {dp[0]:.1f} -> {dp[-1]:.1f}"))
        f = pread(pf, hf, (d * NL) * 3 * NR * NC, NL * 3 * NR * NC, df)
        e = pread(pe, he, (d * NL) * 6 * NR * NC, NL * 6 * NR * NC, de)
        if not (np.isfinite(f).all() and np.isfinite(e).all() and f.min() >= 0 and e.min() >= 0 and e.max() > 0):
            bad.append((d, -1, -1, "fire/emis"))
        stats.append((ss.mean(), ms[0].mean(), ms[-1].mean(), dp[0], dp[-1]))
    s = np.array(stats) if stats else np.full((1, 5), np.nan)
    print(f"{y}: {len(done & set(REQ[y]))}/{len(REQ[y])} days done, {len(missing)} missing, {len(bad)} anomalies | survivors "
          f"{s[:, 0].min():.3f}..{s[:, 0].max():.3f} | survival lead 12 h {s[:, 1].mean():.2f}, 120 h {s[:, 2].mean():.2f} | "
          f"displacement {s[:, 3].mean():.1f} -> {s[:, 4].mean():.1f} deg", flush=True)
    for b in bad[:20]:
        print("   anomaly (day, lead, row, what):", b)
    return len(missing), len(bad)


def check_split():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import foehn_cams as fc
    tr = fc.CamsData("train", psub=16, emis=True)
    te = fc.CamsData("test", psub=16, emis=True, train=False)
    first = lambda gi: np.datetime64("2017-01-01") + np.timedelta64(int(gi), "D")
    t_max = max(2 * (gi - fc.GI_FIRST) + k for gi, k in tr.samples)
    print(f"train: {len(tr)} samples, {len({g for g, _ in tr.samples})} initial days {first(min(g for g, _ in tr.samples))} .. "
          f"{first(max(g for g, _ in tr.samples))}; last target series index {t_max} < test start {fc.T_TEST}: {t_max < fc.T_TEST}")
    print(f"test: {len(te)} samples, {len({g for g, _ in te.samples})} initial days {first(min(g for g, _ in te.samples))} .. "
          f"{first(max(g for g, _ in te.samples))}")
    assert t_max < fc.T_TEST and min(g for g, _ in tr.samples) - fc.GI_FIRST >= 1


if __name__ == "__main__":
    if "--split" in sys.argv:
        check_split()
    else:
        years = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(REQ)
        tot = [check_year(y) for y in years]
        print(f"total: {sum(m for m, _ in tot)} required days missing, {sum(b for _, b in tot)} anomalies")
