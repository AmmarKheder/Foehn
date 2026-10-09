#!/usr/bin/env python3
"""PM2.5 today at 5 km (mean of valid 1 km pixels per 5 x 5 block, NaN if none): the source field of the 1 km footprint
layer, whole Europe, aligned with ../fullimage/data/ghap1km.npy rows. data/pm5km.npy (D, 838, 1398) float16.
    python build_pm5km.py init | <year>"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fullimage"))
from build_ghap1km import D, H, W, ND, YEARS, F as SRC  # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "pm5km.npy"
S, H5, W5 = 5, 838, 1398
if sys.argv[1] == "init":
    with open(OUT, "wb") as f:
        np.lib.format.write_array_header_1_0(f, {"descr": "<f2", "fortran_order": False, "shape": (D, H5, W5)})
        hdr = f.tell(); f.truncate(hdr + D * H5 * W5 * 2)
    print("created", D); sys.exit()


def header(path):
    with open(path, "rb") as f:
        np.lib.format.read_magic(f); np.lib.format.read_array_header_1_0(f); return f.tell()


y = int(sys.argv[1]); off = sum(ND[k] for k in YEARS if k < y)
src = np.memmap(SRC, dtype=np.float16, mode="r", offset=header(SRC) + off * H * W * 2, shape=(ND[y], H, W))
dst = np.memmap(OUT, dtype=np.float16, mode="r+", offset=header(OUT) + off * H5 * W5 * 2, shape=(ND[y], H5, W5))
for d in range(ND[y]):
    a = np.asarray(src[d, :H5 * S, :W5 * S], np.float32).reshape(H5, S, W5, S)
    ok = np.isfinite(a)
    n = ok.sum((1, 3))
    with np.errstate(invalid="ignore"):
        m = np.where(ok, a, 0).sum((1, 3)) / n
    dst[d] = np.where(n > 0, m, np.nan).astype(np.float16)
    if d % 60 == 0:
        print(y, d, flush=True)
dst.flush(); print("done", y)
