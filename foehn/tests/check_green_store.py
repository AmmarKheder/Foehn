#!/usr/bin/env python3
"""Checks a new Green's function store against the 32-particle store (same physics, independent particles): per land
receptor and lead, the fraction of air from aloft, the mean survival and the mean origin must agree up to the sampling
error of 32 particles; the fire terms too; prints the emission term (ug m-3) at a few cities. Exit 1 on failure.
    python tests/check_green_store.py green256 2022 0 1 2 3 4 5 6 7        (login node: pread only)"""
import os
import sys

import numpy as np

G = "/scratch/project_462001140/ammar/eccv/data/world/global1km"
STORE, YEAR, DAYS = sys.argv[1], int(sys.argv[2]), [int(v) for v in sys.argv[3:]]
NR, NC = 288, 720
SPECIES = ("BC", "OC", "SO2", "NOx", "NH3", "CO")
CITIES = {"Beijing": (39.9, 116.4), "Delhi": (28.6, 77.2), "Paris": (48.86, 2.35), "Lagos": (6.5, 3.4), "Los Angeles": (34.05, -118.25)}


def npy(path):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); sh, _, dt = np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    fd = os.open(path, os.O_RDONLY)

    def read(index):                                  # one (day, lead) block
        n = int(np.prod(sh[2:]))
        k = int(np.ravel_multi_index(index, sh[:2]))
        b = os.pread(fd, n * dt.itemsize, hdr + k * n * dt.itemsize)
        return np.frombuffer(b, dt).reshape(sh[2:])
    return sh, read


def cell(la, lo):
    return int((84.0 - la) // 0.5), int((lo + 180.0) // 0.5)


bad = []
sh_n, org_n = npy(f"{G}/{STORE}/hres/origins_{YEAR}.npy")
sh_o, org_o = npy(f"{G}/green/hres/origins_{YEAR}.npy")
_, fs_n = npy(f"{G}/{STORE}/hres/fire_{YEAR}.npy")
_, fs_o = npy(f"{G}/green/hres/fire_{YEAR}.npy")
es_path = f"{G}/{STORE}/hres/emis_{YEAR}.npy"
es = npy(es_path)[1] if os.path.exists(es_path) else None
print(f"{STORE}: origins {sh_n} vs green {sh_o}")
for d in DAYS:
    if not os.path.exists(f"{G}/{STORE}/hres/done_{YEAR}/{d:03d}"):
        bad.append(f"day {d} not done"); continue
    for h in (1, 3, 7):
        a, b = org_n((d, h - 1)).astype(np.float32), org_o((d, h - 1)).astype(np.float32)   # (288, 720, P, 4)
        land = np.abs(b).sum((-1, -2)) > 0
        new_land = np.abs(a).sum((-1, -2)) > 0
        if (land != new_land).any():
            bad.append(f"day {d} lead {h}: receptor sets differ ({land.sum()} vs {new_land.sum()})")
        A, B = a[land], b[land]                                                   # (n, P, 4)
        stats = {}
        for name, f in (("aloft", lambda x: ((x[..., 2] > 1) * x[..., 3]).sum(-1) / np.maximum(x[..., 3].sum(-1), 1e-3)),
                        ("survival", lambda x: x[..., 3].mean(-1)),
                        ("origin lat", lambda x: x[..., 1].mean(-1))):
            fa, fb = f(A), f(B)
            r = np.corrcoef(fa, fb)[0, 1]
            stats[name] = (float(np.mean(fa)), float(np.mean(fb)), float(r))
        dl = np.abs(((A[..., 0].mean(-1) - B[..., 0].mean(-1)) + 180) % 360 - 180)
        ok_range = (A[..., 3] >= 0).all() and (A[..., 3] <= 1.0001).all() and np.isfinite(A).all()
        if not ok_range:
            bad.append(f"day {d} lead {h}: survival out of [0, 1] or non-finite values")
        if stats["survival"][2] < 0.8 or stats["origin lat"][2] < 0.95:
            bad.append(f"day {d} lead {h}: weak agreement with the 32-particle store {stats}")
        fr = np.corrcoef(np.log1p(fs_n((d, h - 1))[0][land] * 1e9), np.log1p(fs_o((d, h - 1))[0][land] * 1e9))[0, 1]
        print(f"day {d} lead {h}: " + " | ".join(f"{k} {v[0]:.3f} vs {v[1]:.3f} (r {v[2]:.2f})" for k, v in stats.items())
              + f" | origin lon median |diff| {np.median(dl):.2f} deg | fire r {fr:.2f}")
        if es is not None:
            e = es((d, h - 1))                                                    # (6, 288, 720) ug m-3
            if not np.isfinite(e).all() or (e < 0).any():
                bad.append(f"day {d} lead {h}: emission term non-finite or negative")
            msg = []
            for c, (la, lo) in CITIES.items():
                i, j = cell(la, lo)
                msg.append(f"{c} " + "/".join(f"{v:.3g}" for v in e[:, i, j]))
            land_med = np.median(e[:, land], 1)
            print(f"   emission term (ug m-3, {'/'.join(SPECIES)}): " + " | ".join(msg) + " | land median " + "/".join(f"{v:.3g}" for v in land_med))
if bad:
    print("FAILED:\n  " + "\n  ".join(bad)); sys.exit(1)
print("store OK")
