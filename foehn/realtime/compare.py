#!/usr/bin/env python3
"""RT step 4a, the non-regression test: the forecast of a 2022 day by the real-time chain (inputs copied from the
training archive by build.py --from-archive, Green's functions recomputed by flexgreen.py --rt-dir, forecast by
predict.py) against the paper's cams12h/pred.npy of the same day. Prints the maximal and mean absolute difference, per
lead, and the same for the Green's functions against the green12h store.
    python compare.py 2022-09-01"""
import sys
from pathlib import Path

import numpy as np

import common as C


def npy_shape(p):
    with open(p, "rb") as f:
        np.lib.format.read_magic(f); sh, _, dt = np.lib.format.read_array_header_1_0(f); hdr = f.tell()
    return sh, dt, hdr


def block(p, i):
    sh, dt, hdr = npy_shape(p)
    n = int(np.prod(sh[1:]))
    return np.fromfile(p, dtype=dt, count=n, offset=hdr + i * n * dt.itemsize).reshape(sh[1:])


def main():
    d = C.parse_date(sys.argv[1]); doy = C.doy_of(d)
    ref_dir = C.RUN / "cams12h"
    doys = list(np.load(ref_dir / "doys.npy"))
    ref = block(ref_dir / "pred.npy", doys.index(doy)).astype(np.float32)           # (8, 10, 361, 900)
    new = np.load(C.out_dir(d) / "pred.npy").astype(np.float32)
    assert ref.shape == new.shape, (ref.shape, new.shape)
    diff = np.abs(new - ref)
    print(f"forecast {d}: members {new.shape[0]}, leads {new.shape[1]}")
    print(f"  max |diff| {diff.max():.4f} ug m-3, mean |diff| {diff.mean():.6f}, cells with |diff| > 0.1: "
          f"{(diff > 0.1).mean():.2%}; reference mean {ref.mean():.3f}, new mean {new.mean():.3f}")
    for k in range(new.shape[1]):
        print(f"  lead {12 * (k + 1):3d} h: max {diff[:, k].max():.4f} mean {diff[:, k].mean():.6f} | ensemble mean field: "
              f"ref {ref[:, k].mean():.3f} new {new[:, k].mean():.3f}")
    # the Green's functions themselves
    rt = C.day_dir(d) / "green"
    for name in ("origins", "fire", "emis"):
        p_new = rt / f"{name}_{d.year}.npy"; p_ref = C.G / "green12h" / "hres" / f"{name}_{d.year}.npy"
        if not (p_new.exists() and p_ref.exists()):
            print(f"  {name}: not compared ({'missing' if not p_new.exists() else 'no reference'})"); continue
        a, b = block(p_new, 0).astype(np.float32), block(p_ref, doy).astype(np.float32)
        ok = np.isfinite(a) & np.isfinite(b)
        print(f"  {name}: shape {a.shape}, max |diff| {np.abs(a - b)[ok].max():.4g}, equal {(a[ok] == b[ok]).mean():.4%}, "
              f"non-finite new {(~np.isfinite(a)).mean():.2e} ref {(~np.isfinite(b)).mean():.2e}")


if __name__ == "__main__":
    main()
